# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import tempfile
import shutil
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model


# DAVIS 2017 palette for mask visualization
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def xywh_to_xyxy(box):
    """Convert box from [x, y, w, h] to [x1, y1, x2, y2] format."""
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def load_sequence_data(seq_dir):
    """
    Load frame indices and bounding boxes for a sequence.
    
    Args:
        seq_dir: Path to the sequence directory
        
    Returns:
        frame_indices: List of frame indices (integers)
        boxes: List of boxes in xywh format (one per line in groundtruth_rect.txt)
    """
    frames_file = os.path.join(seq_dir, "frames.txt")
    gt_file = os.path.join(seq_dir, "groundtruth_rect.txt")
    
    if not os.path.exists(frames_file):
        raise FileNotFoundError(f"Missing frames.txt in {seq_dir}")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Missing groundtruth_rect.txt in {seq_dir}")
    
    # Load frame indices
    frame_indices = []
    with open(frames_file, "r") as f:
        for line in f:
            try:
                frame_idx = int(line.strip())
                frame_indices.append(frame_idx)
            except ValueError:
                continue
    
    # Load bounding boxes
    boxes = []
    with open(gt_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                boxes.append(None)
                continue
            try:
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                w = float(parts[2].strip())
                h = float(parts[3].strip())
                if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
                    boxes.append(None)
                else:
                    boxes.append([x, y, w, h])
            except (ValueError, IndexError):
                boxes.append(None)
    
    # Make sure we have the same number of frames and boxes
    if len(frame_indices) != len(boxes):
        raise ValueError(
            f"Mismatch between number of frames ({len(frame_indices)}) "
            f"and boxes ({len(boxes)}) in {seq_dir}"
        )
    
    return frame_indices, boxes


def prepare_frame_directory(img_dir, frame_indices, tmp_dir):
    """
    Create a temporary directory with symlinks to the actual frame files.
    Frame naming convention: frame_NNNNNNNNNN.jpg where NNNNNNNNNN is the frame index.
    
    Args:
        img_dir: Directory containing frame_*.jpg files
        frame_indices: List of frame indices to link
        tmp_dir: Temporary directory to create symlinks in
        
    Returns:
        frame_names: List of frame names (without extension) in order
    """
    os.makedirs(tmp_dir, exist_ok=True)
    
    frame_names = []
    for i, frame_idx in enumerate(frame_indices):
        # Look for frame with the given index
        frame_filename = f"frame_{frame_idx:010d}.jpg"
        src_path = os.path.join(img_dir, frame_filename)
        
        if not os.path.exists(src_path):
            # Try alternative formats
            found = False
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
                alt_path = os.path.join(img_dir, f"frame_{frame_idx:010d}{ext}")
                if os.path.exists(alt_path):
                    src_path = alt_path
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"Frame {frame_idx} not found in {img_dir}. "
                    f"Expected: {frame_filename}"
                )
        
        # Create symlink with sequential naming for SAM3 to load correctly
        link_name = f"{i:06d}.jpg"
        link_path = os.path.join(tmp_dir, link_name)
        if not os.path.exists(link_path):
            try:
                os.symlink(os.path.abspath(src_path), link_path)
            except OSError:
                # Fallback to copy if symlink not allowed
                shutil.copy(src_path, link_path)
        
        frame_names.append(os.path.splitext(link_name)[0])
    
    return frame_names


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def trek150_inference(
    predictor,
    seq_dir,
    output_mask_dir,
    seq_name,
    score_thresh=0.0,
    offload_video_to_cpu=False,
):
    """Run SAM3 inference on a single TREK-150 sequence."""
    
    # Load frame indices and bounding boxes
    frame_indices, boxes = load_sequence_data(seq_dir)
    
    # Find the first valid bounding box
    first_valid_idx = None
    for i, box in enumerate(boxes):
        if box is not None:
            first_valid_idx = i
            break
    
    if first_valid_idx is None:
        print(f"  No valid bounding box found in {seq_name}, skipping...")
        return
    
    # Use frames starting from the first valid annotation
    valid_frame_indices = frame_indices[first_valid_idx:]
    valid_boxes = boxes[first_valid_idx:]
    first_box = valid_boxes[0]
    
    # Prepare temporary directory with symlinks to frames
    img_dir = os.path.join(seq_dir, "img")
    if not os.path.isdir(img_dir):
        print(f"  Image directory not found: {img_dir}, skipping...")
        return
    
    tmp_dir = tempfile.mkdtemp(prefix=f"trek150_{seq_name.replace('/', '_')}_")
    try:
        print(f"  Preparing {len(valid_frame_indices)} frames...")
        frame_names = prepare_frame_directory(img_dir, valid_frame_indices, tmp_dir)
        
        # Initialize inference state
        print(f"  Initializing state with {len(frame_names)} frames...")
        inference_state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        predictor.clear_all_points_in_video(inference_state)
        
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        
        print(f"  Video dimensions: {width}x{height}")
        
        # Convert box from xywh to xyxy format
        box_xyxy = xywh_to_xyxy(first_box)
        
        # Add box prompt on the first frame (frame 0 of the prepared sequence)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box_xyxy,
            rel_coordinates=False,
        )
        
        # Create output directory
        output_seq_dir = os.path.join(output_mask_dir, seq_name)
        os.makedirs(output_seq_dir, exist_ok=True)
        
        # Run propagation throughout the video
        print(f"  Running propagation on {seq_name}...")
        output_palette = DAVIS_PALETTE
        frame_count = 0
        
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frame_names),
            reverse=False,
            propagate_preflight=True,
        ):
            # Get the actual frame index (from frame_indices)
            actual_frame_idx = valid_frame_indices[frame_idx]
            frame_filename = f"frame_{actual_frame_idx:010d}"
            
            # Create per-object mask dictionary
            per_obj_output_mask = {}
            for i, obj_id in enumerate(obj_ids):
                mask = video_res_masks[i]
                # Convert torch tensor to numpy if needed
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                # Threshold mask
                mask_binary = (mask > score_thresh).astype(np.uint8)
                per_obj_output_mask[int(obj_id)] = mask_binary
            
            # Combine into single mask (0 background, object_id foreground)
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            for obj_id, obj_mask in per_obj_output_mask.items():
                if obj_mask.shape != (height, width):
                    obj_mask = obj_mask.reshape(height, width)
                combined_mask[obj_mask > 0] = obj_id
            
            # Save output mask
            output_mask_path = os.path.join(output_seq_dir, f"{frame_filename}.png")
            save_ann_png(output_mask_path, combined_mask, output_palette)
            frame_count += 1
        
        print(f"  Saved {frame_count} masks to {output_seq_dir}")
        
        # Clean up inference state
        try:
            if hasattr(predictor, "reset_state"):
                predictor.reset_state(inference_state)
        except Exception:
            pass
        try:
            del inference_state
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    finally:
        # Clean up temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 inference on TREK-150 dataset"
    )
    parser.add_argument(
        "--trek150_dir",
        type=str,
        required=True,
        help="Root directory of TREK-150 dataset (containing sequences.txt)",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Threshold for mask confidence (default: 0.0)",
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action="store_true",
        help="Whether to offload video frames to CPU memory to save GPU memory",
    )
    parser.add_argument(
        "--seq_list_file",
        type=str,
        default=None,
        help="Text file containing sequence names to process (one per line, optional)",
    )
    args = parser.parse_args()
    
    # Load SAM3 model
    print("Loading SAM3 model...")
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print("SAM3 model loaded successfully")
    
    # Read sequences to process
    sequences_file = os.path.join(args.trek150_dir, "sequences.txt")
    if not os.path.exists(sequences_file):
        print(f"ERROR: sequences.txt not found in {args.trek150_dir}")
        return
    
    # Load all sequences
    all_sequences = []
    with open(sequences_file, "r") as f:
        for line in f:
            seq_name = line.strip()
            if seq_name:
                all_sequences.append(seq_name)
    
    print(f"Found {len(all_sequences)} sequences in {sequences_file}")
    
    # Filter sequences if a list file is provided
    if args.seq_list_file is not None:
        if not os.path.exists(args.seq_list_file):
            print(f"ERROR: Sequence list file not found: {args.seq_list_file}")
            return
        
        seq_to_process = set()
        with open(args.seq_list_file, "r") as f:
            for line in f:
                seq_name = line.strip()
                if seq_name:
                    seq_to_process.add(seq_name)
        
        sequences = [s for s in all_sequences if s in seq_to_process]
        print(f"Filtered to {len(sequences)} sequences from {args.seq_list_file}")
    else:
        sequences = all_sequences
    
    os.makedirs(args.output_mask_dir, exist_ok=True)
    
    # Process each sequence
    for idx, seq_name in enumerate(sequences):
        print(f"\n[{idx + 1}/{len(sequences)}] Processing {seq_name}...")
        
        seq_dir = os.path.join(args.trek150_dir, seq_name)
        if not os.path.isdir(seq_dir):
            print(f"  ERROR: Sequence directory not found: {seq_dir}")
            continue
        
        # Check if already processed
        output_seq_dir = os.path.join(args.output_mask_dir, seq_name)
        if os.path.isdir(output_seq_dir):
            masks = os.listdir(output_seq_dir)
            if len(masks) > 0:
                print(f"  Already processed ({len(masks)} masks found), skipping...")
                continue
        
        try:
            trek150_inference(
                predictor=predictor,
                seq_dir=seq_dir,
                output_mask_dir=args.output_mask_dir,
                seq_name=seq_name,
                score_thresh=args.score_thresh,
                offload_video_to_cpu=args.offload_video_to_cpu,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Force garbage collection between sequences
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\nCompleted processing. Output masks saved to {args.output_mask_dir}")


if __name__ == "__main__":
    main()
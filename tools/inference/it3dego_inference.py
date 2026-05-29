# -*- coding: utf-8 -*-
"""
Memory-efficient SAM3 inference on IT3DEgo dataset.

For each sequence and each entity:
  - Loads bounding boxes from /annotations/{sequence}/2d_bbox_annot/{entity}.txt
  - Loads corresponding frame images from /raw_videos/{sequence}/pv/{frame_id}.jpg
  - Runs SAM3 mask propagation
  - Saves masks to output directory

Expected dataset structure:
    /media/TBDataNAS/Egocentric Vision/IT3DEgo/
        annotations/
            video_1_scene_1/
                2d_bbox_annot/
                    entity_01.txt
                    entity_02.txt
        raw_videos/
            video_1_scene_1/
                pv/
                    609784323354.jpg
                    609814310907.jpg
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add tools directory to path to import IT3DEgo loader
sys.path.insert(0, os.path.dirname(__file__))
from IT3DEgo import IT3DEgo

from sam3.model_builder import build_sam3_video_model

# DAVIS 2017 palette for mask visualization
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

logger = logging.getLogger(__name__)


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    # Normalize to numpy array
    mask = np.asarray(mask)

    # Convert boolean masks to uint8
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8)

    # Remove singleton dimensions if present
    if mask.ndim > 2:
        mask = np.squeeze(mask)

    # If still multi-channel (e.g., [C,H,W] or [H,W,C]), collapse channels
    if mask.ndim > 2:
        # take the max across the first axis (channels) and binarize
        mask = (np.max(mask, axis=0) > 0).astype(np.uint8)

    if mask.ndim != 2:
        raise ValueError(f"Unable to convert mask to 2D array, got shape {mask.shape}")

    # Ensure uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def xywh_to_xyxy(box):
    """Convert box from (x, y, w, h) format to (x1, y1, x2, y2)."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def it3dego_entity_inference(
    predictor,
    image_paths,
    boxes,
    output_dir,
    sequence_name,
    entity_name,
    score_thresh=0.0,
    offload_video_to_cpu=False,
):
    """Run SAM3 inference on a single entity track.
    
    Parameters:
    - predictor: SAM3 video tracker
    - image_paths: list of image file paths for this track
    - boxes: (N, 4) array of bounding boxes in (x, y, w, h) format
    - output_dir: where to save output masks
    - sequence_name: name of the sequence (for organizing output)
    - entity_name: name of the entity being tracked
    - score_thresh: confidence threshold for masks
    - offload_video_to_cpu: whether to offload frames to CPU
    """
    if len(image_paths) == 0:
        logger.warning(f"No frames found for {sequence_name}/{entity_name}")
        return
    
    # Validate inputs
    if len(boxes) != len(image_paths):
        logger.warning(
            f"Mismatch: {len(boxes)} boxes vs {len(image_paths)} images "
            f"for {sequence_name}/{entity_name}"
        )
        # Pad boxes with NaN if needed
        if len(boxes) < len(image_paths):
            boxes = np.vstack([boxes, np.full((len(image_paths) - len(boxes), 4), np.nan)])
        else:
            boxes = boxes[:len(image_paths)]
    
    # Create temporary directory to hold symlinks to frames for SAM3
    import tempfile
    tmp_video_dir = tempfile.mkdtemp()
    try:
        # Create numbered symlinks for SAM3 (0.jpg, 1.jpg, ...)
        for i, img_path in enumerate(image_paths):
            link_path = os.path.join(tmp_video_dir, f"{i:06d}.jpg")
            if os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(img_path, link_path)
        
        # Initialize SAM3 inference state on temporary directory
        inference_state = predictor.init_state(
            video_path=tmp_video_dir,
            offload_video_to_cpu=offload_video_to_cpu
        )
        predictor.clear_all_points_in_video(inference_state)
        
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        logger.info(f"Video dimensions: {width}x{height}, {len(image_paths)} frames")
        
        # Find first frame with a valid box annotation
        first_frame_with_box = None
        first_box = None
        for frame_idx, box in enumerate(boxes):
            if not np.isnan(box).any():
                first_frame_with_box = frame_idx
                first_box = box
                break
        
        if first_frame_with_box is None:
            logger.warning(f"No valid boxes found for {sequence_name}/{entity_name}")
            return
        
        logger.info(
            f"Using frame {first_frame_with_box} as prompt for {sequence_name}/{entity_name}"
        )
        
        # Convert first box from (x, y, w, h) to (x1, y1, x2, y2)
        x1, y1, x2, y2 = xywh_to_xyxy(first_box)
        
        # Clamp box to image bounds
        x1 = max(0, min(int(x1), width - 1))
        y1 = max(0, min(int(y1), height - 1))
        x2 = max(0, min(int(x2), width - 1))
        y2 = max(0, min(int(y2), height - 1))
        
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid box bounds for {sequence_name}/{entity_name}: {(x1, y1, x2, y2)}")
            return
        
        # Add bounding box prompt to SAM3
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=first_frame_with_box,
            obj_id=1,
            box=[x1, y1, x2, y2],
            rel_coordinates=False,
        )
        
        # Run propagation and collect masks
        logger.info(f"Running propagation for {sequence_name}/{entity_name}...")
        output_palette = DAVIS_PALETTE
        
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in \
                predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(image_paths),
                    reverse=False,
                    propagate_preflight=True,
                ):
            # Extract mask for object 1 (our tracked entity)
            if 1 in obj_ids:
                obj_idx = list(obj_ids).index(1)
                output_mask = (video_res_masks[obj_idx] > score_thresh).cpu().numpy().astype(np.uint8)
            else:
                output_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Save output mask as PNG
            frame_id = Path(image_paths[frame_idx]).stem
            output_path = os.path.join(
                output_dir, sequence_name, entity_name, f"{frame_id}.png"
            )
            save_ann_png(output_path, output_mask, output_palette)
        
        logger.info(f"Completed {sequence_name}/{entity_name}")
        
    finally:
        # Cleanup temporary directory
        import shutil
        shutil.rmtree(tmp_video_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 inference on IT3DEgo dataset entities"
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="Path to raw_videos root directory",
    )
    parser.add_argument(
        "--ann_root",
        type=str,
        required=True,
        help="Path to annotations root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Specific sequence to run on (default: all sequences)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Specific entity to run on (default: all entities in sequence)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Confidence threshold for output masks",
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action="store_true",
        help="Offload video frames to CPU to save GPU memory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load dataset
    logger.info("Loading IT3DEgo dataset...")
    dataset = IT3DEgo(
        frames_root=args.frames_root,
        ann_root=args.ann_root,
        seq_names=args.sequence,
    )
    logger.info(f"Dataset: {dataset}")

    # Load SAM3 model
    logger.info("Loading SAM3 model...")
    sam3_model = build_sam3_video_model(device=args.device)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    logger.info("SAM3 model loaded")

    # Determine which entries to process
    if args.sequence and args.entity:
        entries_to_process = [
            e for e in dataset.entries
            if e["sequence"] == args.sequence and e["entity"] == args.entity
        ]
    elif args.sequence:
        entries_to_process = [
            e for e in dataset.entries
            if e["sequence"] == args.sequence
        ]
    else:
        entries_to_process = dataset.entries

    logger.info(f"Will process {len(entries_to_process)} entities")

    # Process each entity
    for n_entity, entry in enumerate(entries_to_process):
        seq_name = entry["sequence"]
        entity_name = entry["entity"]
        
        logger.info(f"\n[{n_entity + 1}/{len(entries_to_process)}] Processing {seq_name}/{entity_name}")
        
        # Get data for this entity
        try:
            image_paths, boxes, meta = dataset[(seq_name, entity_name)]
        except (IndexError, FileNotFoundError) as e:
            logger.error(f"Failed to load {seq_name}/{entity_name}: {e}")
            continue
        
        if len(image_paths) == 0:
            logger.warning(f"No images for {seq_name}/{entity_name}")
            continue
        
        logger.info(f"  Found {len(image_paths)} frames, {np.sum(~np.isnan(boxes[:, 0]))} annotated")
        
        # Run inference
        try:
            it3dego_entity_inference(
                predictor=predictor,
                image_paths=image_paths,
                boxes=boxes,
                output_dir=args.output_dir,
                sequence_name=seq_name,
                entity_name=entity_name,
                score_thresh=args.score_thresh,
                offload_video_to_cpu=args.offload_video_to_cpu,
            )
        except Exception as e:
            logger.error(f"Inference failed for {seq_name}/{entity_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\nCompleted inference on {len(entries_to_process)} entities")
    logger.info(f"Output masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()

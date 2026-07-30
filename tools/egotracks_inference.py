"""
Memory-efficient SAM3 inference on EgoTracks dataset.

Loads annotations from directory structure like:
    {video_uuid}*{object_id}*{object_name}/
        ├── boxes.txt (one bbox per line: x,y,w,h)
        └── frames.txt (one frame index per line)

Frames are located at:
    /media/TBDataNAS/Egocentric Vision/Ego4D/v2/clips_frames_val/frames/{video_uuid}/

Saves masks frame-by-frame and frees GPU memory incrementally.
"""

import argparse
import csv
import gc
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_model
from egotracks_episode_loader import build_episode_annotations

# DAVIS 2017 palette for mask visualization
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def has_existing_output_dir(path):
    """Return True when an output directory already contains generated artifacts."""
    if not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file() or entry.is_dir():
                    return True
    except Exception:
        return False
    return False


def xywh_to_xyxy(box):
    """Convert box from [x, y, w, h] to [x1, y1, x2, y2] format."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def parse_annotation_dir(anno_dir):
    """
    Parse annotation directory name to extract video_id, object_id, and object_name.
    Format: {video_uuid}*{object_id}*{object_name}
    Example: 0bfda958-2a25-4198-8de6-9348b4f483fb*1*blue container
    """
    dirname = os.path.basename(anno_dir.rstrip("/"))
    parts = dirname.split("*")
    if len(parts) < 3:
        raise ValueError(f"Invalid annotation directory name: {dirname}")
    video_id = parts[0]
    object_id = int(parts[1])
    object_name = "*".join(parts[2:])  # Handle object names with asterisks
    return video_id, object_id, object_name


def load_annotation(anno_dir, start_line=None, end_line=None):
    """
    Load frames and boxes from annotation directory.
    Returns lists of frame indices and their corresponding boxes.

    Args:
        start_line: 1-based inclusive line number in frames.txt/boxes.txt to start from.
        end_line: 1-based inclusive line number in frames.txt/boxes.txt to stop at.
    """
    frames_file = os.path.join(anno_dir, "frames.txt")
    boxes_file = os.path.join(anno_dir, "boxes.txt")

    if not os.path.exists(frames_file) or not os.path.exists(boxes_file):
        raise FileNotFoundError(
            f"Missing frames.txt or boxes.txt in {anno_dir}"
        )

    frame_lines = []
    with open(frames_file, "r") as f:
        for line in f:
            frame_lines.append(line.strip())

    boxes_lines = []
    with open(boxes_file, "r") as f:
        for line in f:
            boxes_lines.append(line.strip())

    if len(frame_lines) != len(boxes_lines):
        raise ValueError(
            f"Mismatch between number of frames ({len(frame_lines)}) "
            f"and boxes ({len(boxes_lines)}) in {anno_dir}"
        )

    if start_line is not None or end_line is not None:
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = len(frame_lines)
        if start_line < 1 or end_line < 1:
            raise ValueError("start_line and end_line must be >= 1")
        if end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if end_line > len(frame_lines):
            raise ValueError(
                f"end_line={end_line} exceeds available lines={len(frame_lines)}"
            )
        frame_lines = frame_lines[start_line - 1 : end_line]
        boxes_lines = boxes_lines[start_line - 1 : end_line]

    # Load frame indices
    frame_indices = []
    for line in frame_lines:
        try:
            frame_idx = int(float(line))
            frame_indices.append(frame_idx)
        except ValueError:
            continue

    # Load boxes
    boxes = []
    for line in boxes_lines:
        parts = line.split(",")
        if len(parts) != 4:
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
        except ValueError:
            boxes.append(None)

    if len(frame_indices) != len(boxes):
        raise ValueError(
            f"Mismatch between number of frames ({len(frame_indices)}) "
            f"and boxes ({len(boxes)}) in {anno_dir}"
        )

    return frame_indices, boxes


def prepare_frame_directory(frames_dir, frame_indices, tmp_dir):
    """
    Create a temporary directory with symlinks to the actual frame files.
    This lets init_state load only the frames we need, in the correct order.
    
    Args:
        frames_dir: Directory containing {video_id}/{frame_idx}.jpg files
        frame_indices: List of frame indices to link
        tmp_dir: Temporary directory to create symlinks in
    
    Returns:
        List of frame names (without extension) in order
    """
    os.makedirs(tmp_dir, exist_ok=True)

    frame_names = []
    for i, frame_idx in enumerate(frame_indices):
        # Try common image extensions
        frame_found = False
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
            src_path = os.path.join(frames_dir, f"{frame_idx}{ext}")
            if os.path.exists(src_path):
                # Create symlink with sequential naming for SAM2 to load correctly
                link_name = f"{i:06d}{ext}"
                link_path = os.path.join(tmp_dir, link_name)
                if not os.path.exists(link_path):
                    os.symlink(os.path.abspath(src_path), link_path)
                frame_names.append(os.path.splitext(link_name)[0])
                frame_found = True
                break
        
        if not frame_found:
            raise FileNotFoundError(
                f"Frame {frame_idx} not found in {frames_dir}"
            )

    return frame_names


def save_score_evolution(save_path, score_hist, sequence_name):
    """Save score history as CSV and a plot image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    csv_path = os.path.join(os.path.dirname(save_path), "score_evolution.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "source_frame_idx", "frame_name", "object_score_logits", "iou_pred", "eff_iou_score"],
        )
        writer.writeheader()
        for row in score_hist:
            writer.writerow(row)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        if score_hist:
            frame_ids = [row["frame_idx"] for row in score_hist]
            object_scores = [row["object_score_logits"] for row in score_hist]
            iou_preds = [row["iou_pred"] for row in score_hist]
            eff_iou_scores = [row["eff_iou_score"] for row in score_hist]
            ax.plot(frame_ids, object_scores, label="object_score_logits", linewidth=1.5)
            ax.plot(frame_ids, iou_preds, label="iou_pred", linewidth=1.2)
            ax.plot(frame_ids, eff_iou_scores, label="eff_iou_score", linewidth=1.2)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Score")
            ax.set_title(f"{sequence_name} score evolution")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def free_inference_memory(predictor, inference_state):
    """Try to reset predictor state (if supported) and aggressively free GPU memory."""
    try:
        if hasattr(predictor, "reset_state"):
            predictor.reset_state(inference_state)
    except Exception:
        pass
    # Clear cached features
    if isinstance(inference_state, dict) and "cached_features" in inference_state:
        inference_state["cached_features"].clear()
    # Clear image tensors
    if isinstance(inference_state, dict) and "images" in inference_state:
        try:
            del inference_state["images"]
        except Exception:
            pass
    try:
        del inference_state
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_inference_single_track(
    predictor,
    frames_dir,
    frame_indices,
    boxes,
    output_dir,
    video_id,
    object_id,
    object_name,
    score_thresh=0.0,
    offload_video_to_cpu=True,
    output_subdir=None,
    source_anno_dir=None,
):
    """
    Run SAM3 inference on a single EgoTracks track.
    Saves masks frame-by-frame DURING propagation to avoid GPU memory buildup.

    Args:
        source_anno_dir: if set (episode mode), the directory holding the
            episode's frames.txt/boxes.txt/visibility.txt/episode_info.json.
            These are copied into the output folder for traceability.
    """
    # Find the first frame with a valid box annotation
    first_valid_idx = None
    for i, box in enumerate(boxes):
        if box is not None:
            first_valid_idx = i
            break

    if first_valid_idx is None:
        print(f"  [{object_name}] No valid bbox annotation found, skipping...")
        return

    # Use frames starting from the first valid annotation
    valid_frame_indices = frame_indices[first_valid_idx:]
    valid_boxes = boxes[first_valid_idx:]
    first_box = valid_boxes[0]

    # Create temporary directory with symlinks to the frames
    tmp_dir = tempfile.mkdtemp(
        prefix=f"sam3_{video_id}_{object_id}_{object_name.replace(' ', '_')}_"
    )
    try:
        print(
            f"  [{object_name}] Preparing {len(valid_frame_indices)} frames "
            f"(starting from frame {valid_frame_indices[0]})..."
        )
        frame_names = prepare_frame_directory(frames_dir, valid_frame_indices, tmp_dir)

        print(f"  [{object_name}] Initializing state with {len(frame_names)} frames...")
        inference_state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        height = inference_state["video_height"]
        width = inference_state["video_width"]

        # Convert box from xywh to xyxy (absolute pixel coords expected)
        box_xyxy = xywh_to_xyxy(first_box)

        # Add box prompt on the first frame using the annotation's object id
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            box=box_xyxy,
            rel_coordinates=False,
        )

        # Create output directory
        save_dir = os.path.join(output_dir, video_id, f"{object_id}_{object_name}")
        if output_subdir:
            save_dir = os.path.join(save_dir, output_subdir)
        if has_existing_output_dir(save_dir):
            print(f"  [{object_name}] [SKIP] Output already exists at {save_dir}")
            return
        os.makedirs(save_dir, exist_ok=True)

        # Run propagation and collect masks in memory; save once at the end
        print(f"  [{object_name}] Running propagation and collecting frames...")
        frame_count = 0
        max_frames_to_track = len(frame_names)
        video_segments = {}
        score_history = []
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=max_frames_to_track,
            reverse=False,
            propagate_preflight=True,
        ):
            # Normalize frame name
            frame_name = frame_names[frame_idx]

            # Normalize obj_ids to a python list
            try:
                obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
            except Exception:
                try:
                    obj_ids_list = list(obj_ids)
                except Exception:
                    obj_ids_list = []

            if object_id in obj_ids_list:
                idx = obj_ids_list.index(object_id)
                mask_tensor = video_res_masks[idx]
                # Convert to numpy and threshold
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np = (mask_tensor > score_thresh).cpu().numpy()
                else:
                    mask_np = (mask_tensor > score_thresh).astype(np.uint8)
            else:
                # object not present in this frame -> empty mask
                mask_np = np.zeros((height, width), dtype=np.uint8)

            # Ensure mask is 2D uint8
            mask_np = np.asarray(mask_np)
            if mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np)
            if mask_np.ndim > 2:
                # fallback: take last two dims
                mask_np = mask_np.reshape(mask_np.shape[-2], mask_np.shape[-1])
            mask_np = mask_np.astype(np.uint8)

            video_segments[frame_idx] = mask_np
            frame_count += 1

            # Collect score evolution for the current object
            object_score_value = None
            iou_pred_value = None
            eff_iou_score_value = None
            try:
                object_score_value = float(obj_scores[idx].reshape(-1)[0].detach().cpu())
            except Exception:
                object_score_value = None

            try:
                obj_output_dict = inference_state["output_dict_per_obj"][idx]
                frame_out = None
                for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    if frame_idx in obj_output_dict[storage_key]:
                        frame_out = obj_output_dict[storage_key][frame_idx]
                        break
                if frame_out is not None:
                    iou_tensor = frame_out.get("iou_score")
                    if iou_tensor is not None:
                        iou_pred_value = float(iou_tensor.reshape(-1)[0].detach().cpu())
                    eff_tensor = frame_out.get("eff_iou_score")
                    if eff_tensor is not None:
                        eff_iou_score_value = float(eff_tensor.reshape(-1)[0].detach().cpu())
            except Exception:
                pass

            if object_score_value is None:
                object_score_value = float("nan")
            if iou_pred_value is None:
                iou_pred_value = float("nan")
            if eff_iou_score_value is None:
                eff_iou_score_value = float("nan")

            score_history.append(
                {
                    "frame_idx": frame_idx,
                    "source_frame_idx": valid_frame_indices[frame_idx],
                    "frame_name": frame_names[frame_idx],
                    "object_score_logits": object_score_value,
                    "iou_pred": iou_pred_value,
                    "eff_iou_score": eff_iou_score_value,
                }
            )

            # Periodically clear GPU cache
            if frame_count % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save all collected masks (use DAVIS palette)
        print(f"  [{object_name}] Saving {len(video_segments)} masks to disk...")
        for fidx in sorted(video_segments.keys()):
            fname = frame_names[fidx]
            save_path = os.path.join(save_dir, f"{fname}.png")
            save_ann_png(save_path, video_segments[fidx], DAVIS_PALETTE)

        score_plot_path = os.path.join(save_dir, "score_evolution.png")
        save_score_evolution(score_plot_path, score_history, object_name)

        # If this run was for an episode, copy the episode's source
        # annotation files (and the traceability metadata) alongside the
        # generated masks, so the output folder is self-contained and it's
        # always possible to tell which original frames/lines were used.
        if source_anno_dir is not None:
            for fname in ("frames.txt", "boxes.txt", "visibility.txt", "episode_info.json"):
                src = os.path.join(source_anno_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_dir, fname))

        # Free all inference state memory
        free_inference_memory(predictor, inference_state)
        print(f"  [{object_name}] Done. {frame_count} masks saved to {save_dir}")

    finally:
        # Clean up temp directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient SAM3 inference on EgoTracks dataset"
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="Path to SAM 3 model checkpoint (optional; will download if not provided)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/media/TBDataNAS/Egocentric Vision/Ego4D/v2/clips_frames_val/frames",
        help="Root directory of EgoTracks frames",
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        default="/home/zaira/Projects/sam2/TREK-150-toolkit/toolkit/datasets/egotracks-annotations",
        help="Root directory of EgoTracks annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="Threshold for mask logits",
    )
    parser.add_argument(
        "--offload_video_to_cpu",
        action="store_true",
        help="Whether to offload video frames to CPU memory to save GPU memory",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="Process a single sequence by its annotation directory name",
    )
    parser.add_argument(
        "--seq_names_file",
        type=str,
        default=None,
        help="Text file containing sequence names to process (one per line)",
    )
    parser.add_argument(
        "--start_line",
        type=int,
        default=None,
        help="1-based starting line in frames.txt/boxes.txt to include",
    )
    parser.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="1-based ending line in frames.txt/boxes.txt to include",
    )
    parser.add_argument(
        "--split_into_episodes",
        action="store_true",
        help="Split each annotation into visibility-based episode directories before inference",
    )
    parser.add_argument(
        "--visible_frames",
        type=int,
        default=6,
        help="Number of consecutive visible frames required for an episode",
    )
    parser.add_argument(
        "--missing_frames",
        type=int,
        default=10,
        help="Minimum number of consecutive missing frames required to form an episode "
        "(the FULL gap is used, this is only a lower bound, not a fixed window size)",
    )
    parser.add_argument(
        "--reappearance_frames",
        type=int,
        default=1,
        help="Number of consecutive visible frames to include after the missing segment as the reappearance window",
    )
    parser.add_argument(
        "--episodes_dir",
        type=str,
        default=None,
        help="Root directory where generated episode annotations are stored",
    )
    args = parser.parse_args()
    # Build SAM3 model and predictor
    print("Loading SAM 3 model...")
    sam3_model = build_sam3_video_model(checkpoint_path=args.sam3_checkpoint)
    predictor = sam3_model.tracker
    # attach detector backbone (used by some utilities)
    predictor.backbone = sam3_model.detector.backbone
    print("SAM 3 model loaded successfully")

    # Collect all annotation directories
    anno_dirs = []
    for entry in os.listdir(args.anno_dir):
        entry_path = os.path.join(args.anno_dir, entry)
        if os.path.isdir(entry_path) and "*" in entry:
            anno_dirs.append(entry_path)

    print(f"Found {len(anno_dirs)} annotation directories in {args.anno_dir}")

    if args.seq_name is not None:
        filtered_anno_dirs = []
        for anno_dir in anno_dirs:
            dirname = os.path.basename(anno_dir.rstrip("/"))
            if dirname == args.seq_name:
                filtered_anno_dirs.append(anno_dir)
        print(f"Filtered to {len(filtered_anno_dirs)} sequence(s) matching {args.seq_name}")
        anno_dirs = filtered_anno_dirs

    # Filter by seq_names_file if provided
    if args.seq_names_file is not None:
        if not os.path.exists(args.seq_names_file):
            print(f"[ERROR] Sequence names file not found: {args.seq_names_file}")
            return
        
        # Read sequence names from file
        seq_names_to_process = set()
        with open(args.seq_names_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    seq_names_to_process.add(line)
        
        print(f"Loaded {len(seq_names_to_process)} sequence names from {args.seq_names_file}")
        
        # Filter anno_dirs to match seq_names
        filtered_anno_dirs = []
        for anno_dir in anno_dirs:
            dirname = os.path.basename(anno_dir.rstrip("/"))
            if dirname in seq_names_to_process:
                filtered_anno_dirs.append(anno_dir)
        
        print(f"Filtered to {len(filtered_anno_dirs)} matching sequences")
        anno_dirs = filtered_anno_dirs
    
    os.makedirs(args.output_dir, exist_ok=True)
    episodes_root = args.episodes_dir or os.path.join(args.output_dir, "episodes")
    os.makedirs(episodes_root, exist_ok=True)

    for idx, anno_dir in enumerate(sorted(anno_dirs)):
        try:
            video_id, object_id, object_name = parse_annotation_dir(anno_dir)
        except ValueError as e:
            print(f"[{idx + 1}/{len(anno_dirs)}] Skipping {anno_dir}: {e}")
            continue

        print(
            f"\n[{idx + 1}/{len(anno_dirs)}] Video: {video_id}, Object {object_id}: {object_name}"
        )

        # Check if already processed for the full-sequence path
        if not args.split_into_episodes:
            output_object_dir = os.path.join(
                args.output_dir, video_id, f"{object_id}_{object_name}"
            )
            if has_existing_output_dir(output_object_dir):
                print(f"  [SKIP] Already processed output found at {output_object_dir}")
                continue

        try:
            if args.split_into_episodes:
                print(
                    f"  Splitting sequence into episodes with {args.visible_frames} visible frames and {args.missing_frames} missing frames..."
                )
                episode_infos = build_episode_annotations(
                    anno_dir=anno_dir,
                    output_dir=episodes_root,
                    visible_frames=args.visible_frames,
                    missing_frames=args.missing_frames,
                    reappearance_frames=args.reappearance_frames,
                )
                if not episode_infos:
                    print("  [INFO] No episodes found for this sequence; skipping")
                    continue

                pending_episodes = []
                for episode in episode_infos:
                    episode_dir = episode["episode_dir"]
                    episode_output_subdir = os.path.basename(episode_dir)
                    expected_output_dir = os.path.join(
                        args.output_dir,
                        video_id,
                        f"{object_id}_{object_name}",
                        episode_output_subdir,
                    )
                    if has_existing_output_dir(expected_output_dir):
                        print(f"  [SKIP] Episode already processed: {episode_output_subdir}")
                        continue
                    pending_episodes.append(episode)

                if not pending_episodes:
                    print("  [SKIP] All episode outputs already exist for this sequence")
                    continue

                for episode_idx, episode in enumerate(pending_episodes, start=1):
                    episode_dir = episode["episode_dir"]
                    print(f"  Running episode {episode_idx}/{len(pending_episodes)}: {os.path.basename(episode_dir)}")
                    episode_frame_indices, episode_boxes = load_annotation(episode_dir)
                    episode_output_subdir = os.path.basename(episode_dir)

                    run_inference_single_track(
                        predictor=predictor,
                        frames_dir=os.path.join(args.frames_dir, video_id),
                        frame_indices=episode_frame_indices,
                        boxes=episode_boxes,
                        output_dir=args.output_dir,
                        video_id=video_id,
                        object_id=object_id,
                        object_name=object_name,
                        score_thresh=args.score_thresh,
                        offload_video_to_cpu=args.offload_video_to_cpu,
                        output_subdir=episode_output_subdir,
                        source_anno_dir=episode_dir,
                    )
            else:
                # Load frame indices and boxes from annotation
                frame_indices, boxes = load_annotation(
                    anno_dir,
                    start_line=args.start_line,
                    end_line=args.end_line,
                )
                print(
                    f"  Loaded {len(frame_indices)} frame indices from lines {args.start_line or 1} to {args.end_line or len(frame_indices)}"
                )

                # Get actual video frames directory
                video_frames_dir = os.path.join(args.frames_dir, video_id)
                if not os.path.isdir(video_frames_dir):
                    print(f"  [ERROR] Video frames directory not found: {video_frames_dir}")
                    continue

                # Run inference for this object track
                run_inference_single_track(
                    predictor=predictor,
                    frames_dir=video_frames_dir,
                    frame_indices=frame_indices,
                    boxes=boxes,
                    output_dir=args.output_dir,
                    video_id=video_id,
                    object_id=object_id,
                    object_name=object_name,
                    score_thresh=args.score_thresh,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                )

        except Exception as e:
            print(f"  [ERROR] Failed to process: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Force garbage collection between tracks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nCompleted. Output masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()
"""
Memory-efficient SAM3 inference on EgoExo4D sequences.

Collects masks during propagation and saves them at the end of
the sequence to reduce per-frame I/O and avoid shape/assert issues.
"""

import argparse
import fnmatch
import gc
import os
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image

import sys

from sam3.model_builder import build_sam3_video_model
from vos_inference import save_masks_to_dir

# Import EgoExo4D dataset
sys.path.insert(0, os.path.dirname(__file__))
from EgoExo4D import EgoExo4D

from huggingface_hub import login
login()

# DAVIS 2017 palette for mask visualization
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


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
    return [x, y, x + w, y + h]


def expand_seq_patterns(patterns, all_seq_names):
    """
    Expand wildcard patterns to match against available sequence names.
    
    Args:
        patterns: List of pattern strings (may contain * wildcards)
        all_seq_names: List of all available sequence names
    
    Returns:
        List of matched sequence names
    """
    if patterns is None:
        return None
    
    matched_names = []
    for pattern in patterns:
        # Use fnmatch to find matching sequence names
        matches = fnmatch.filter(all_seq_names, pattern)
        if matches:
            matched_names.extend(matches)
        else:
            # If no matches found, try the pattern as-is (exact match)
            if pattern in all_seq_names:
                matched_names.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in matched_names:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)
    
    return unique_names if unique_names else None


def prepare_frame_directory(image_paths, tmp_dir):
    """
    Create a temporary directory with symlinks to the given image paths.
    This lets init_state load only the frames we need.
    Returns the tmp directory path and ordered frame names (without extension).
    """
    os.makedirs(tmp_dir, exist_ok=True)

    frame_names = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        basename = os.path.basename(img_path)
        link_path = os.path.join(tmp_dir, basename)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(img_path), link_path)
        frame_names.append(os.path.splitext(basename)[0])

    frame_names.sort(key=lambda p: int(p))
    return frame_names


def free_inference_memory(predictor, inference_state):
    """Reset predictor state and aggressively free GPU memory (robust)."""
    try:
        predictor.reset_state(inference_state)
    except Exception:
        pass
    try:
        if "cached_features" in inference_state:
            inference_state["cached_features"].clear()
    except Exception:
        pass
    try:
        if "images" in inference_state:
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
def run_inference_single_view(
    predictor,
    image_paths,
    box_first_frame,
    output_dir,
    seq_name,
    view_name,
    score_thresh=0.0,
    num_pathway=3,
    iou_thre=0.3,
    uncertainty=1,
):
    """
    Run SAM2Long inference on a single view (ego or exo) of an EgoExo4D sequence.
    Saves masks frame-by-frame and frees memory incrementally.
    """
    print(f"  [{view_name}] Input image paths count: {len(image_paths)}")
    
    # Filter out frames that don't exist on disk
    valid_image_paths = [p for p in image_paths if os.path.exists(p)]
    print(f"  [{view_name}] Valid frames after filtering: {len(valid_image_paths)}")
    
    if len(valid_image_paths) != len(image_paths):
        # Show first few missing paths for debugging
        missing = [p for p in image_paths if not os.path.exists(p)]
        print(f"  [{view_name}] Sample of missing paths (first 3):")
        for p in missing[:3]:
            print(f"      - {p}")
    
    if len(valid_image_paths) == 0:
        print(f"  [SKIP] No valid frames for {view_name}")
        return

    # Create temporary directory with symlinks to the frames
    tmp_dir = tempfile.mkdtemp(prefix=f"sam2_{seq_name}_{view_name}_")
    try:
        frame_names = prepare_frame_directory(valid_image_paths, tmp_dir)
        print(f"  [{view_name}] Frame names linked: {len(frame_names)}")
        if len(frame_names) == 0:
            print(f"  [SKIP] No frames linked for {view_name}")
            return

        print(f"  [{view_name}] Initializing state with {len(frame_names)} frames...")
        inference_state = predictor.init_state(
            video_path=tmp_dir,
            offload_video_to_cpu=True,
        )
        try:
            predictor.clear_all_points_in_video(inference_state)
        except Exception:
            pass

        height = inference_state["video_height"]
        width = inference_state["video_width"]
        print(f"  [{view_name}] Video initialized: {width}x{height}")

        # Convert box from xywh to xyxy (dataset boxes are xywh)
        box_xyxy = xywh_to_xyxy(box_first_frame)
        print(f"  [{view_name}] Box (xyxy): {box_xyxy}")

        # Add box prompt on the first frame (use absolute coordinates)
        obj_id = 1
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=box_xyxy,
            rel_coordinates=False,
        )

        # Run propagation and collect per-frame results
        print(f"  [{view_name}] Running propagation...")
        video_segments = {}
        for (
            frame_idx,
            obj_ids,
            low_res_masks,
            video_res_masks,
            obj_scores,
        ) in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frame_names),
            reverse=False,
            propagate_preflight=True,
        ):
            per_obj_output_mask = {
                obj_id: (video_res_masks[i] > score_thresh).cpu().numpy()
                for i, obj_id in enumerate(obj_ids)
            }
            video_segments[frame_idx] = per_obj_output_mask

        # Save all masks at the end (same layout as before)
        print(f"  [{view_name}] Saving output masks...")
        output_palette = DAVIS_PALETTE
        for out_frame_idx, per_obj_output_mask in video_segments.items():
            save_masks_to_dir(
                output_mask_dir=output_dir,
                video_name=os.path.join(seq_name, view_name),
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=output_palette,
            )

        # Free all inference state memory
        free_inference_memory(predictor, inference_state)
        print(f"  [{view_name}] Done. Masks saved to {os.path.join(output_dir, seq_name, view_name)}")

    finally:
        # Clean up temp directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient SAM2Long inference on EgoExo4D sequences"
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=None,
        help="Optional SAM3 checkpoint path",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["both", "ego", "exo"],
        default="both",
        help="Which view(s) to run inference on",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Root directory of EgoExo4D frames",
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        required=True,
        help="Root directory of EgoExo4D annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output masks",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        choices=[1, 5, 10, 30],
        help="FPS for frame subsampling (default: 5)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lt",
        choices=["lt", "st"],
        help="EgoExo4D tracking mode: lt (long-term) or st (short-term)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--num_pathway",
        type=int,
        default=3,
        help="Number of SAM2Long pathways (default: 3)",
    )
    parser.add_argument(
        "--iou_thre",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--uncertainty",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--seq_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequence names to process (default: all)",
    )
    parser.add_argument(
        "--use_frames_file",
        action="store_true",
        help="If set, use the sequence's frames.txt to select frames instead of fps subsampling",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
    )
    args = parser.parse_args()

    # Build SAM3 model and predictor
    print("Loading SAM3 model...")
    if args.sam3_checkpoint:
        sam3_model = build_sam3_video_model(checkpoint_path=args.sam3_checkpoint)
    else:
        sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    try:
        predictor.backbone = sam3_model.detector.backbone
    except Exception:
        pass
    print("SAM3 model loaded successfully")

    # Load EgoExo4D dataset - first without seq_names to get all available sequences
    print("\nLoading all available sequences...")
    dataset_all = EgoExo4D(
        frames_dir=args.frames_dir,
        anno_dir=args.anno_dir,
        fps=args.fps,
        resolution=args.resolution,
        mode=args.mode,
        seq_names=None,
    )
    print(f"Total available sequences: {len(dataset_all.seq_names)}")
    
    # Expand patterns if seq_names provided
    if args.seq_names is not None:
        print(f"\nPattern matching:")
        for pattern in args.seq_names:
            print(f"  Pattern: '{pattern}'")
        expanded_names = expand_seq_patterns(args.seq_names, dataset_all.seq_names)
        if expanded_names:
            print(f"Expanded {len(args.seq_names)} pattern(s) to {len(expanded_names)} sequence(s):")
            for name in expanded_names:
                print(f"  - {name}")
            seq_names_to_use = expanded_names
        else:
            print(f"WARNING: No sequences matched the provided patterns")
            print(f"Available sequences (first 10): {dataset_all.seq_names[:10]}")
            seq_names_to_use = None
    else:
        seq_names_to_use = None
    
    # Load EgoExo4D dataset with expanded sequence names
    dataset = EgoExo4D(
        frames_dir=args.frames_dir,
        anno_dir=args.anno_dir,
        fps=args.fps,
        resolution=args.resolution,
        mode=args.mode,
        seq_names=seq_names_to_use,
    )

    print(f"\nDataset loaded: {len(dataset)} sequences at {args.fps} fps")
    if seq_names_to_use:
        print(f"Processing filtered sequences:")
        for sn in dataset.seq_names[:5]:
            print(f"  - {sn}")
        if len(dataset.seq_names) > 5:
            print(f"  ... and {len(dataset.seq_names) - 5} more")
    os.makedirs(args.output_dir, exist_ok=True)

    for idx in range(len(dataset)):
        seq_name = dataset.seq_names[idx]
        print(f"\n[{idx + 1}/{len(dataset)}] Sequence: {seq_name}")

        # Check if already processed
        ego_done = os.path.isdir(os.path.join(args.output_dir, seq_name, "ego"))
        exo_done = os.path.isdir(os.path.join(args.output_dir, seq_name, "exo"))
        if ego_done and exo_done:
            # Check if there are already output masks
            ego_masks = os.listdir(os.path.join(args.output_dir, seq_name, "ego"))
            exo_masks = os.listdir(os.path.join(args.output_dir, seq_name, "exo"))
            if len(ego_masks) > 0 and len(exo_masks) > 0:
                print(f"  [SKIP] Already processed")
                continue

        # Load sequence images/boxes. If requested, prefer frames.txt from
        # the annotation folder instead of the dataset's fps-based subsampling.
        try:
            if args.use_frames_file:
                take_name = seq_name.split('*')[0]
                resol_str = f"{args.resolution}" if args.resolution > 0 else ""
                if args.mode == 'st':
                    seq_name_parts = seq_name.split('$')
                    st_sq_idx = seq_name_parts[-1]
                    st_seq_name = seq_name_parts[0]
                    keys_dir = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos'
                    )
                else:
                    keys_dir = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos'
                    )

                keys = sorted(os.listdir(keys_dir))
                ego_key = keys[0]
                exo_key = keys[1]

                if args.mode == 'st':
                    files_dir_ego = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{ego_key}', st_sq_idx
                    )
                    files_dir_exo = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos', f'{exo_key}', st_sq_idx
                    )
                else:
                    files_dir_ego = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{ego_key}'
                    )
                    files_dir_exo = os.path.join(
                        args.anno_dir, args.mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos', f'{exo_key}'
                    )

                frames_file_ego = os.path.join(files_dir_ego, 'frames.txt')
                frames_file_exo = os.path.join(files_dir_exo, 'frames.txt')

                if not os.path.exists(frames_file_ego) or not os.path.exists(frames_file_exo):
                    raise FileNotFoundError(f"frames.txt not found: {frames_file_ego} or {frames_file_exo}")

                frame_idxs_ego = np.genfromtxt(frames_file_ego, delimiter='\n', dtype=np.int64)
                frame_idxs_exo = np.genfromtxt(frames_file_exo, delimiter='\n', dtype=np.int64)
                frame_idxs_ego = np.atleast_1d(frame_idxs_ego).astype(np.int64)
                frame_idxs_exo = np.atleast_1d(frame_idxs_exo).astype(np.int64)

                boxes_ego = np.genfromtxt(os.path.join(files_dir_ego, 'boxes.txt'), delimiter=',')
                boxes_exo = np.genfromtxt(os.path.join(files_dir_exo, 'boxes.txt'), delimiter=',')
                boxes_ego = np.atleast_2d(boxes_ego)
                boxes_exo = np.atleast_2d(boxes_exo)

                images_ego = [
                    os.path.join(args.frames_dir, f'{args.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{ego_key}', f'{fi}.jpg')
                    for fi in frame_idxs_ego
                ]
                images_exo = [
                    os.path.join(args.frames_dir, f'{args.resolution}', 'takes', take_name, 'frame_aligned_videos', f'{exo_key}', f'{fi}.jpg')
                    for fi in frame_idxs_exo
                ]

                print(f"  Using frames.txt: ego frames={len(images_ego)}, exo frames={len(images_exo)}")
            else:
                images_ego, images_exo, boxes_ego, boxes_exo = dataset[idx]
        except Exception as e:
            print(f"  [ERROR] Failed to load sequence (frames_file={args.use_frames_file}): {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"  Loaded: {len(images_ego)} ego images, {len(images_exo)} exo images")
        print(f"  Ego boxes shape: {boxes_ego.shape}, Exo boxes shape: {boxes_exo.shape}")
        
        # Check for NaN boxes
        ego_valid_count = sum(1 for box in boxes_ego if not np.any(np.isnan(box)))
        exo_valid_count = sum(1 for box in boxes_exo if not np.any(np.isnan(box)))
        print(f"  Valid boxes - Ego: {ego_valid_count}/{len(boxes_ego)}, Exo: {exo_valid_count}/{len(boxes_exo)}")

        # Find the first frame with a valid box annotation
        # (non-NaN box for initialization)
        first_valid_ego = None
        for i, box in enumerate(boxes_ego):
            if not np.any(np.isnan(box)):
                first_valid_ego = i
                break

        first_valid_exo = None
        for i, box in enumerate(boxes_exo):
            if not np.any(np.isnan(box)):
                first_valid_exo = i
                break

        # Run selected view(s)
        if args.view in ("both", "ego"):
            if first_valid_ego is not None:
                ego_images_subset = images_ego[first_valid_ego:]
                ego_box = boxes_ego[first_valid_ego]
                run_inference_single_view(
                    predictor=predictor,
                    image_paths=ego_images_subset,
                    box_first_frame=ego_box,
                    output_dir=args.output_dir,
                    seq_name=seq_name,
                    view_name="ego",
                    score_thresh=args.score_thresh,
                    num_pathway=args.num_pathway,
                    iou_thre=args.iou_thre,
                    uncertainty=args.uncertainty,
                )
            else:
                print(f"  [SKIP] No valid ego box annotation found")

        if args.view in ("both", "exo"):
            if first_valid_exo is not None:
                exo_images_subset = images_exo[first_valid_exo:]
                exo_box = boxes_exo[first_valid_exo]
                run_inference_single_view(
                    predictor=predictor,
                    image_paths=exo_images_subset,
                    box_first_frame=exo_box,
                    output_dir=args.output_dir,
                    seq_name=seq_name,
                    view_name="exo",
                    score_thresh=args.score_thresh,
                    num_pathway=args.num_pathway,
                    iou_thre=args.iou_thre,
                    uncertainty=args.uncertainty,
                )
            else:
                print(f"  [SKIP] No valid exo box annotation found")

        # Force garbage collection between sequences
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nCompleted. Output masks saved to {args.output_dir}")


if __name__ == "__main__":
    main()

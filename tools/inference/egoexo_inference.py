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
    *,
    box_first_frame=None,
    mask_first_frame=None,
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

        # Try to initialize with a mask if provided, otherwise fallback to box
        obj_id = 1
        used_mask = False
        if mask_first_frame is not None and os.path.exists(mask_first_frame):
            try:
                mask_img = Image.open(mask_first_frame).convert('L')
                mask_arr = np.array(mask_img)
                mask_bin = (mask_arr > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_bin)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=mask_tensor,
                    add_mask_to_memory=False,
                )
                used_mask = True
                print(f"  [{view_name}] Initialized with mask: {mask_first_frame}")
            except Exception as e:
                print(f"  [{view_name}] Failed to load/use mask {mask_first_frame}: {e}")

        if not used_mask:
            # Convert box from xywh to xyxy (dataset boxes are xywh) and fallback to box init
            if box_first_frame is None:
                raise RuntimeError("No initialization box or mask provided for inference")
            box_xyxy = xywh_to_xyxy(box_first_frame)
            print(f"  [{view_name}] Box (xyxy): {box_xyxy}")
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
                video_name=os.path.join(view_name, seq_name),
                frame_name=frame_names[out_frame_idx],
                per_obj_output_mask=per_obj_output_mask,
                height=height,
                width=width,
                per_obj_png_file=False,
                output_palette=output_palette,
            )

        # Free all inference state memory
        free_inference_memory(predictor, inference_state)
        print(f"  [{view_name}] Done. Masks saved to {os.path.join(output_dir, view_name, seq_name)}")

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
        "--masks_root",
        type=str,
        default=None,
        help="Root directory containing exported masks (contains 'single-obj/<res>/takes/...'). If provided, masks will be used to initialize tracking; otherwise bbox init is used.",
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
    
    # Expand patterns if seq_names provided (initial candidate set)
    if args.seq_names is not None:
        print(f"\nPattern matching:")
        for pattern in args.seq_names:
            print(f"  Pattern: '{pattern}'")
        expanded_names = expand_seq_patterns(args.seq_names, dataset_all.seq_names)
        if expanded_names:
            print(f"Expanded {len(args.seq_names)} pattern(s) to {len(expanded_names)} sequence(s):")
            for name in expanded_names:
                print(f"  - {name}")
            candidate_seq_names = expanded_names
        else:
            print(f"WARNING: No sequences matched the provided patterns")
            print(f"Available sequences (first 10): {dataset_all.seq_names[:10]}")
            candidate_seq_names = None
    else:
        candidate_seq_names = None

    # If masks_root is provided, prefer sequences that have exported masks there.
    # We still rely on the dataset's annotation list for frame selection,
    # so we intersect masks-derived sequences with `dataset_all.seq_names`.
    if args.masks_root:
        print("\nDetecting sequences available in masks_root...")
        resol_str = f"{args.resolution}" if args.resolution > 0 else ""
        # Prefer an explicit sequences.txt exported with the masks, if present
        masks_seq_file = "/media/TBDataNAS/Egocentric Vision/EgoExo4D/v2/annotations/vot_ego_exo/sot/val/st/720/sequences.txt"
        mask_seq_list = None
        if os.path.exists(masks_seq_file):
            try:
                mask_seq_list = np.genfromtxt(masks_seq_file, delimiter='\n', dtype=str).tolist()
                if isinstance(mask_seq_list, str):
                    mask_seq_list = [mask_seq_list]
                print(f"  Found {len(mask_seq_list)} sequences in masks sequences.txt")
            except Exception:
                mask_seq_list = None

        # Fallback: scan the masks folder structure if no sequences.txt is available
        if mask_seq_list is None:
            masks_base = os.path.join(args.masks_root, 'single-obj', resol_str, 'takes')
            mask_seq_candidates = []
            if os.path.isdir(masks_base):
                for take_name in sorted(os.listdir(masks_base)):
                    take_path = os.path.join(masks_base, take_name)
                    if not os.path.isdir(take_path):
                        continue
                    for seq_dir in sorted(os.listdir(take_path)):
                        seq_dir_path = os.path.join(take_path, seq_dir)
                        if not os.path.isdir(seq_dir_path):
                            continue
                        if args.mode == 'st':
                            # For short-term, match by prefix to dataset seq naming
                            for dsn in dataset_all.seq_names:
                                if dsn.split('$')[0] == seq_dir:
                                    mask_seq_candidates.append(dsn)
                        else:
                            mask_seq_candidates.append(seq_dir)

            # Deduplicate while preserving order
            seen = set()
            mask_seq_list = []
            for s in mask_seq_candidates:
                if s not in seen:
                    mask_seq_list.append(s)
                    seen.add(s)
            print(f"  Found {len(mask_seq_list)} sequences by scanning masks directory")

        if len(mask_seq_list) == 0:
            print(f"  WARNING: No sequences found under masks_root for resolution '{resol_str}'")

        # Choose final seq_names_to_use depending on whether the user provided patterns
        if candidate_seq_names is None:
            seq_names_to_use = mask_seq_list if len(mask_seq_list) > 0 else None
        else:
            # Intersect pattern-based candidates with mask-available sequences (prefer mask ordering)
            seq_names_to_use = [s for s in mask_seq_list if s in candidate_seq_names]
            if not seq_names_to_use:
                print("  WARNING: No sequences matched both provided patterns and masks_root content; falling back to mask-based selection")
                seq_names_to_use = mask_seq_list
    else:
        seq_names_to_use = candidate_seq_names
    
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
        ego_done = os.path.isdir(os.path.join(args.output_dir, "ego", seq_name))
        exo_done = os.path.isdir(os.path.join(args.output_dir, "exo", seq_name))
        if ego_done and exo_done:
            # Check if there are already output masks
            ego_masks = os.listdir(os.path.join(args.output_dir, "ego", seq_name))
            exo_masks = os.listdir(os.path.join(args.output_dir, "exo", seq_name))
            if len(ego_masks) > 0 and len(exo_masks) > 0:
                print(f"  [SKIP] Already processed")
                continue

        # Load sequence images/boxes. For ST mode, always use the manual
        # frames.txt/boxes.txt path construction, because EgoExo4D.__getitem__
        # does not handle the nested ST subdirectory structure correctly.
        # For LT mode, use frames.txt only when --use_frames_file is set.
        try:
            if args.use_frames_file or args.mode == 'st':
                take_name = seq_name.split('*')[0]
                resol_str = f"{args.resolution}" if args.resolution > 0 else ""
                if args.mode == 'st':
                    # ST seq_name may come in two formats:
                    #   (a) "<take>*<seq_name>$<st_idx>"  – from sequences.txt via vos_inference
                    #   (b) "<take>*<seq_name>"           – from EgoExo4D dataset (no '$')
                    # In case (b) we discover st_sq_idx by scanning the camera subdirectory.
                    if '$' in seq_name:
                        seq_name_parts = seq_name.split('$')
                        st_sq_idx = seq_name_parts[-1]
                        st_seq_name = seq_name_parts[0]
                    else:
                        st_seq_name = seq_name
                        st_sq_idx = None  # will be resolved below after we know ego_key
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
                    # Resolve st_sq_idx by scanning subdirs if not known from '$' split
                    if st_sq_idx is None:
                        ego_cam_dir = os.path.join(
                            args.anno_dir, args.mode, resol_str, 'takes', take_name,
                            st_seq_name, 'frame_aligned_videos', ego_key
                        )
                        sub_dirs = sorted([
                            d for d in os.listdir(ego_cam_dir)
                            if os.path.isdir(os.path.join(ego_cam_dir, d))
                        ])
                        if len(sub_dirs) == 0:
                            raise FileNotFoundError(
                                f"No ST subdirectory found under {ego_cam_dir}"
                            )
                        if len(sub_dirs) > 1:
                            print(f"  [WARNING] Multiple ST subdirs found under {ego_cam_dir}: {sub_dirs}; using first: {sub_dirs[0]}")
                        st_sq_idx = sub_dirs[0]
                        print(f"  [ST] Resolved st_sq_idx='{st_sq_idx}' by scanning {ego_cam_dir}")

                    # ST annotation structure:
                    #   .../frame_aligned_videos/{camera_key}/{st_sq_idx}/frames.txt
                    #   .../frame_aligned_videos/{camera_key}/{st_sq_idx}/boxes.txt
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
                # try to find corresponding mask for the initialization frame
                ego_mask_path = None
                if args.masks_root is not None:
                    try:
                        resol_str = f"{args.resolution}" if args.resolution > 0 else ""
                        take_name = seq_name.split('*')[0]
                        if args.mode == 'st':
                            st_seq_name = seq_name.split('$')[0]
                            seq_dir = st_seq_name
                        else:
                            seq_dir = seq_name
                        keys_dir = os.path.join(args.anno_dir, args.mode, resol_str, 'takes', take_name, seq_dir, 'frame_aligned_videos')
                        keys = sorted(os.listdir(keys_dir))
                        ego_key = keys[0]
                        frame_name = os.path.splitext(os.path.basename(ego_images_subset[0]))[0]
                        maybe_path = os.path.join(args.masks_root, 'single-obj', resol_str, 'takes', take_name, seq_dir, 'frame_aligned_videos', ego_key, f"{frame_name}.png")
                        if os.path.exists(maybe_path):
                            ego_mask_path = maybe_path
                    except Exception:
                        ego_mask_path = None
                run_inference_single_view(
                    predictor=predictor,
                    image_paths=ego_images_subset,
                    box_first_frame=ego_box,
                    mask_first_frame=ego_mask_path,
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
                exo_mask_path = None
                if args.masks_root is not None:
                    try:
                        resol_str = f"{args.resolution}" if args.resolution > 0 else ""
                        take_name = seq_name.split('*')[0]
                        if args.mode == 'st':
                            st_seq_name = seq_name.split('$')[0]
                            seq_dir = st_seq_name
                        else:
                            seq_dir = seq_name
                        keys_dir = os.path.join(args.anno_dir, args.mode, resol_str, 'takes', take_name, seq_dir, 'frame_aligned_videos')
                        keys = sorted(os.listdir(keys_dir))
                        exo_key = keys[1]
                        frame_name = os.path.splitext(os.path.basename(exo_images_subset[0]))[0]
                        maybe_path = os.path.join(args.masks_root, 'single-obj', resol_str, 'takes', take_name, seq_dir, 'frame_aligned_videos', exo_key, f"{frame_name}.png")
                        if os.path.exists(maybe_path):
                            exo_mask_path = maybe_path
                    except Exception:
                        exo_mask_path = None
                run_inference_single_view(
                    predictor=predictor,
                    image_paths=exo_images_subset,
                    box_first_frame=exo_box,
                    mask_first_frame=exo_mask_path,
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
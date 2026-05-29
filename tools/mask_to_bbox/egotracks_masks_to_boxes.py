"""
Convert SAM2Long mask outputs to bounding boxes for EgoTracks dataset.

Reads PNG masks from egotracks_inference output directory:
  /media/TBData/marco/Projects/SAM2Long/EgoTracks/masks/{video_uuid}/{object_id}_{object_name}/

Converts masks to bounding boxes and saves them in matching GT structure:
  {output_dir}/{video_uuid}*{object_id}*{object_name}/
    ├── boxes.txt (x, y, w, h)
    └── frames.txt (frame indices)
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def mask_to_bbox(mask):
    """
    Convert a binary mask to bounding box in xywh format.
    
    Args:
        mask: 2D binary numpy array (H x W)
    
    Returns:
        [x, y, w, h] or None if mask is empty
    """
    mask = mask > 0
    if not np.any(mask):
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not (np.any(rows) and np.any(cols)):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    return [float(x_min), float(y_min), float(w), float(h)]


def parse_mask_path(mask_dir):
    """
    Parse mask directory path to extract video_id and object_id_name.
    Expected format: masks/{video_uuid}/{object_id}_{object_name}/
    """
    parts = Path(mask_dir).parts
    if len(parts) < 2:
        return None, None
    
    # Get parent (masks dir) and grandparent structure
    object_name_dir = parts[-1]  # e.g. "1_blue container"
    video_id = parts[-2]  # e.g. "0bfda958-2a25-4198-8de6-9348b4f483fb"
    
    # Parse object_id and object_name from "1_blue container"
    underscore_idx = object_name_dir.find('_')
    if underscore_idx == -1:
        return None, None
    
    try:
        object_id = int(object_name_dir[:underscore_idx])
        object_name = object_name_dir[underscore_idx + 1:]
    except ValueError:
        return None, None
    
    return video_id, object_id, object_name


def is_already_converted(output_dir, video_id, object_id, object_name):
    """
    Check if this object has already been converted and has valid annotations.
    
    Returns True if boxes.txt exists and is not empty (skip conversion).
    Returns False if boxes.txt doesn't exist or is empty (need conversion).
    """
    output_anno_dir = os.path.join(
        output_dir,
        f"{video_id}*{object_id}*{object_name}"
    )
    boxes_path = os.path.join(output_anno_dir, 'boxes.txt')
    
    if not os.path.exists(boxes_path):
        return False  # Not converted yet
    
    # Check if boxes.txt is empty
    try:
        with open(boxes_path, 'r') as f:
            content = f.read().strip()
            return len(content) > 0  # True if has content, False if empty
    except:
        return False  # If can't read, treat as not converted


def convert_mask_directory(mask_dir, output_dir):
    """
    Convert all masks in a single object track directory to bbox format.
    
    Args:
        mask_dir: Path to masks/{video_uuid}/{object_id}_{object_name}/
        output_dir: Output base directory
    
    Returns:
        (frames_saved, object_dir) or (0, None) on error
    """
    result = parse_mask_path(mask_dir)
    if result[0] is None:
        return 0, None
    
    video_id, object_id, object_name = result
    
    # Get all PNG files and sort by frame index
    png_files = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    if len(png_files) == 0:
        return 0, None
    
    # Extract frame indices from PNG filenames
    frame_indices = [int(os.path.splitext(f)[0]) for f in png_files]
    
    boxes = []
    frame_indices_list = []
    
    for png_file, frame_idx in zip(png_files, frame_indices):
        mask_path = os.path.join(mask_dir, png_file)
        
        try:
            # Load mask
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            
            # If mask is indexed (palette mode), convert to array
            if mask_img.mode == 'P':
                mask = np.array(mask_img.convert('RGB'))[:, :, 0]
            
            # Make binary mask (any non-zero pixel)
            mask = (mask > 0).astype(np.uint8)
            
            # Convert to bbox
            bbox = mask_to_bbox(mask)
            
            if bbox is not None:
                boxes.append(bbox)
            else:
                # Save NaN for empty masks
                boxes.append([np.nan, np.nan, np.nan, np.nan])
            
            frame_indices_list.append(frame_idx)
            
        except Exception as e:
            print(f"    [ERROR] Failed to process {png_file}: {e}")
            # Still add NaN bbox if we can get the frame index
            boxes.append([np.nan, np.nan, np.nan, np.nan])
            frame_indices_list.append(frame_idx)
    
    if len(boxes) == 0:
        return 0, None
    
    # Create output directory structure
    # {output_dir}/{video_uuid}*{object_id}*{object_name}/
    output_anno_dir = os.path.join(
        output_dir,
        f"{video_id}*{object_id}*{object_name}"
    )
    os.makedirs(output_anno_dir, exist_ok=True)
    
    # Save boxes.txt (x, y, w, h, comma-separated; NaN for empty masks)
    boxes_path = os.path.join(output_anno_dir, 'boxes.txt')
    with open(boxes_path, 'w') as f:
        for box in boxes:
            if np.any(np.isnan(box)):
                # Write NaN values
                f.write(f"nan,nan,nan,nan\n")
            else:
                f.write(f"{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}\n")
    
    # Save frames.txt (frame indices, newline-separated)
    frames_path = os.path.join(output_anno_dir, 'frames.txt')
    with open(frames_path, 'w') as f:
        for frame_idx in frame_indices_list:
            f.write(f"{frame_idx}\n")
    
    return len(boxes), output_anno_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM2Long masks to EgoTracks bbox format"
    )
    parser.add_argument(
        "--masks_dir",
        type=str,
        default="/media/TBData/marco/Projects/SAM2Long/EgoTracks/masks",
        help="Directory containing output masks from egotracks_inference.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for bbox annotations (will match GT structure)",
    )
    args = parser.parse_args()

    # Discover all video directories
    if not os.path.isdir(args.masks_dir):
        print(f"[ERROR] Masks directory not found: {args.masks_dir}")
        return

    video_dirs = sorted([
        d for d in os.listdir(args.masks_dir)
        if os.path.isdir(os.path.join(args.masks_dir, d))
    ])

    print(f"Found {len(video_dirs)} video directories")
    os.makedirs(args.output_dir, exist_ok=True)

    total_frames_processed = 0
    total_objects_processed = 0
    total_objects_skipped = 0
    failed_objects = []

    for video_idx, video_id in enumerate(video_dirs):
        video_mask_dir = os.path.join(args.masks_dir, video_id)
        
        # Discover all object directories within this video
        object_dirs = sorted([
            d for d in os.listdir(video_mask_dir)
            if os.path.isdir(os.path.join(video_mask_dir, d))
        ])
        
        for obj_idx, object_dir_name in enumerate(object_dirs):
            object_mask_dir = os.path.join(video_mask_dir, object_dir_name)
            
            # Parse object_id and object_name to check if already converted
            underscore_idx = object_dir_name.find('_')
            if underscore_idx != -1:
                try:
                    object_id = int(object_dir_name[:underscore_idx])
                    object_name = object_dir_name[underscore_idx + 1:]
                    
                    # Check if already converted with valid annotations
                    if is_already_converted(args.output_dir, video_id, object_id, object_name):
                        print(
                            f"[{video_idx + 1}/{len(video_dirs)}:{obj_idx + 1}/{len(object_dirs)}] "
                            f"{video_id} / {object_dir_name}...",
                            end=" "
                        )
                        print(f"✓ (already converted, skipping)")
                        total_objects_skipped += 1
                        continue
                except ValueError:
                    pass
            
            print(
                f"[{video_idx + 1}/{len(video_dirs)}:{obj_idx + 1}/{len(object_dirs)}] "
                f"{video_id} / {object_dir_name}...",
                end=" "
            )
            
            try:
                frames_saved, output_dir_created = convert_mask_directory(
                    object_mask_dir, args.output_dir
                )
                
                if frames_saved > 0:
                    print(f"✓ ({frames_saved} frames)")
                    total_frames_processed += frames_saved
                    total_objects_processed += 1
                else:
                    print(f"✗ (no valid masks)")
                    failed_objects.append(f"{video_id}/{object_dir_name}")
                    
            except Exception as e:
                print(f"✗ (ERROR: {e})")
                failed_objects.append(f"{video_id}/{object_dir_name}")

    print(f"\n{'='*60}")
    print(f"Conversion completed:")
    print(f"  Total objects processed: {total_objects_processed}")
    print(f"  Total objects skipped (already converted): {total_objects_skipped}")
    print(f"  Total frames converted: {total_frames_processed}")
    print(f"  Failed objects: {len(failed_objects)}")
    print(f"  Output directory: {args.output_dir}")
    
    if failed_objects:
        failed_file = os.path.join(args.output_dir, "failed_conversions.txt")
        with open(failed_file, 'w') as f:
            for obj in failed_objects:
                f.write(f"{obj}\n")
        print(f"  Failed objects list: {failed_file}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

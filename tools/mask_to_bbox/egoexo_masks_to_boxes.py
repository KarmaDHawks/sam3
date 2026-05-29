"""
Convert SAM2Long mask outputs to bounding boxes in EgoExo4D GT format.

Reads PNG masks from egoexo_inference output, converts to bounding boxes,
and reorganizes files to match the GT directory structure:
    {output_dir}/{mode}/{resolution}/all/takes/{take_name}/{seq_name}/
        frame_aligned_videos/{view_key}/boxes.txt
        frame_aligned_videos/{view_key}/frames.txt
        frame_aligned_videos/{view_key}/visibilities.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from EgoExo4D import EgoExo4D


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


def get_view_keys_from_egoexo(anno_dir, mode, resolution, take_name, seq_name):
    """
    Extract ego and exo view keys from the original GT directory structure.
    
    Returns:
        (ego_key, exo_key) tuple
    """
    resol_str = f"{resolution}" if resolution > 0 else ""
    
    try:
        if mode == 'lt':
            video_dir = os.path.join(
                anno_dir, mode, resol_str, 'takes', take_name, seq_name, 'frame_aligned_videos'
            )
        else:  # 'st' mode
            seq_parts = seq_name.split('$')
            st_seq_name = seq_parts[0]
            video_dir = os.path.join(
                anno_dir, mode, resol_str, 'takes', take_name, st_seq_name, 'frame_aligned_videos'
            )
        
        keys = sorted(os.listdir(video_dir))
        ego_key = keys[0]  # aria key
        exo_key = keys[1]  # camera key
        
        assert 'aria' in ego_key, f"Expected 'aria' in ego_key, got {ego_key}"
        return ego_key, exo_key
    except Exception as e:
        print(f"  [ERROR] Failed to extract view keys: {e}")
        return None, None


def convert_sequence(
    seq_name,
    input_dir,
    anno_dir,
    output_dir,
    mode='lt',
    resolution=720,
):
    """
    Convert masks for a single sequence to GT format with restructured directories.
    """
    take_name = seq_name.split('*')[0]
    
    # Extract view keys from original GT structure
    ego_key, exo_key = get_view_keys_from_egoexo(anno_dir, mode, resolution, take_name, seq_name)
    if ego_key is None or exo_key is None:
        print(f"  [SKIP] Could not extract view keys for {seq_name}")
        return False
    
    seq_dir = os.path.join(input_dir, seq_name)
    if not os.path.isdir(seq_dir):
        print(f"  [SKIP] Directory not found: {seq_dir}")
        return False
    
    resol_str = f"{resolution}" if resolution > 0 else ""
    success = True
    
    for view_name, view_key in [("ego", ego_key), ("exo", exo_key)]:
        view_dir = os.path.join(seq_dir, view_name)
        if not os.path.isdir(view_dir):
            print(f"  [SKIP] View directory not found: {view_dir}")
            continue
        
        # Get PNG files (frame names)
        png_files = sorted(
            [f for f in os.listdir(view_dir) if f.endswith('.png')],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        
        if len(png_files) == 0:
            print(f"  [SKIP] No masks found in {view_dir}")
            continue
        
        # Extract frame indices from PNG filenames
        frame_indices = [int(os.path.splitext(f)[0]) for f in png_files]
        
        boxes = []
        visibilities = []
        
        for png_file, frame_idx in zip(png_files, frame_indices):
            mask_path = os.path.join(view_dir, png_file)
            
            # Load mask (first channel if RGB)
            mask_img = Image.open(mask_path)
            mask = np.array(mask_img)
            
            # If mask is indexed (palette mode), convert to array
            if mask_img.mode == 'P':
                mask = np.array(mask_img.convert('RGB'))[:, :, 0]
            
            # Make binary mask
            mask = (mask > 0).astype(np.uint8)
            
            # Convert to bbox
            bbox = mask_to_bbox(mask)
            
            if bbox is not None:
                boxes.append(bbox)
                visibilities.append(1.0)
            else:
                # If mask is empty, use NaN bbox
                boxes.append([np.nan, np.nan, np.nan, np.nan])
                visibilities.append(0.0)
        
        if len(boxes) == 0:
            print(f"  [SKIP] No valid boxes extracted for {view_name}")
            continue
        
        # Create output directory structure
        # Format: {output_dir}/{mode}/{resolution}/all/takes/{take_name}/{seq_name}/
        #         frame_aligned_videos/{view_key}/
        output_base = os.path.join(
            output_dir, #take_name, 
            seq_name, 'frame_aligned_videos', view_key
        )
        os.makedirs(output_base, exist_ok=True)
        
        # Save boxes.txt (x, y, w, h, comma-separated)
        boxes_path = os.path.join(output_base, 'boxes.txt')
        np.savetxt(boxes_path, boxes, fmt='%.1f', delimiter=',')
        
        # Save frames.txt (frame indices, newline-separated)
        #frames_path = os.path.join(output_base, 'frames.txt')
        #np.savetxt(frames_path, frame_indices, fmt='%d')
        
        # Save visibilities.txt (newline-separated)
        #visibilities_path = os.path.join(output_base, 'visibilities.txt')
        #np.savetxt(visibilities_path, visibilities, fmt='%.1f')
        
        print(f"  [{view_name}] Saved {len(boxes)} boxes to {output_base}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAM2Long masks to EgoExo4D GT format bounding boxes"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory from egoexo_inference.py output",
    )
    parser.add_argument(
        "--anno_dir",
        type=str,
        required=True,
        help="Original EgoExo4D annotations directory (for GT structure lookup)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for restructured annotations",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lt",
        choices=["lt", "st"],
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--seq_names",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequence names to process",
    )
    args = parser.parse_args()
    
    # Discover sequences
    if args.seq_names is not None:
        seq_names = args.seq_names
    else:
        # Discover from input_dir
        seq_names = sorted([
            d for d in os.listdir(args.input_dir)
            if os.path.isdir(os.path.join(args.input_dir, d)) and '*' in d
        ])
    
    print(f"Processing {len(seq_names)} sequences")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for idx, seq_name in enumerate(seq_names):
        print(f"\n[{idx + 1}/{len(seq_names)}] {seq_name}")
        
        success = convert_sequence(
            seq_name=seq_name,
            input_dir=args.input_dir,
            anno_dir=args.anno_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            resolution=args.resolution,
        )
        
        if not success:
            print(f"  [FAILED] Could not process {seq_name}")
    
    print(f"\nCompleted. Output saved to {args.output_dir}")


if __name__ == "__main__":
    main()

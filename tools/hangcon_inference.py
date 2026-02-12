# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def get_contours_from_mask(mask_np, min_area=0):
    """
    Extract contours/coordinates from a binary mask.
    
    Args:
        mask_np: numpy array with shape (H, W), binary mask
        min_area: minimum area threshold for contours
        
    Returns:
        list: list of polygons, each polygon is a list of [x, y] coordinates
    """
    mask_uint8 = (mask_np > 0).astype(np.uint8)
    
    if int(cv2.__version__[0]) > 3:
        contours, _ = cv2.findContours(
            mask_uint8.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        _, contours, _ = cv2.findContours(
            mask_uint8.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    
    polygons = []
    for contour in contours:
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Convert contour to polygon coordinates
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:  # At least 3 points (x, y pairs)
            polygons.append(polygon)
    
    return polygons


def get_bbox_from_mask(mask_np):
    """
    Get bounding box coordinates from a binary mask.
    
    Returns:
        dict: {'x': x_min, 'y': y_min, 'width': width, 'height': height, 'area': area}
    """
    y_indices, x_indices = np.where(mask_np > 0)
    
    if len(y_indices) == 0:
        return {'x': 0, 'y': 0, 'width': 0, 'height': 0, 'area': 0}
    
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = int(np.sum(mask_np > 0))
    
    return {
        'x': x_min,
        'y': y_min,
        'width': width,
        'height': height,
        'area': area
    }


def find_image_files(directory):
    """
    Find all image files in a directory.
    Supports: jpg, jpeg, png, JPG, JPEG, PNG
    """
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        image_files.extend([
            f for f in os.listdir(directory)
            if f.endswith(ext) and os.path.isfile(os.path.join(directory, f))
        ])
    return sorted(image_files)


def detect_dataset_structure(dataset_dir):
    """
    Detect if the dataset is flat (all images directly in folder) 
    or structured (images in subdirectories).
    
    Returns:
        tuple: (is_flat: bool, description: str)
    """
    items = os.listdir(dataset_dir)
    subdirs = [d for d in items if os.path.isdir(os.path.join(dataset_dir, d))]
    files = [f for f in items if os.path.isfile(os.path.join(dataset_dir, f))]
    
    # Find image files in root
    image_files_root = find_image_files(dataset_dir)
    
    # If there are image files in root and few/no subdirs, it's flat
    if len(image_files_root) > 0 and len(subdirs) <= 1:
        return True, f"Flat dataset detected: {len(image_files_root)} images in root directory"
    elif len(subdirs) > 0:
        return False, f"Structured dataset detected: {len(subdirs)} subdirectories found"
    else:
        raise ValueError("No images found in the dataset directory")


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_flat_dataset(
    processor,
    dataset_dir,
    output_mask_dir,
    output_json_dir,
    prompt,
    confidence_threshold=0.5,
    save_masks=True,
    save_json=True,
    use_gt=False,
    gt_dir=None,
    gt_class=1
):
    """
    Process a flat dataset (all images directly in the folder).
    """
    image_files = find_image_files(dataset_dir)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_dir}")
    
    print(f"Found {len(image_files)} images in flat dataset")
    print(f"Using prompt: '{prompt}'")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Create output directories
    if save_masks:
        os.makedirs(output_mask_dir, exist_ok=True)
    if save_json:
        os.makedirs(output_json_dir, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            image_path = os.path.join(dataset_dir, image_file)
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # Set image and prompt
            inference_state = processor.set_image(image)
            processor.reset_all_prompts(state=inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            # optionally add GT box prompt (YOLO format)
            if use_gt and gt_dir is not None:
                gt_filename = os.path.splitext(image_file)[0] + '.txt'
                gt_path = os.path.join(gt_dir, gt_filename)
                if os.path.exists(gt_path):
                    try:
                        with open(gt_path, 'r') as fg:
                            for line in fg:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                cls_id = int(parts[0])
                                if cls_id == int(gt_class):
                                    cx = float(parts[1])
                                    cy = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    # YOLO already normalized => use directly
                                    box_norm = [cx, cy, w, h]
                                    inference_state = processor.add_geometric_prompt(box=box_norm, label=True, state=inference_state)
                                    break
                    except Exception as e:
                        print(f"Warning: couldn't read GT {gt_path}: {e}")
            
            # Get segmentation mask
            masks = inference_state.get('masks', None)

            # Initialize with empty mask (all zeros)
            binary_mask = np.zeros((height, width), dtype=np.uint8)

            # If masks exist, use the first one
            if masks is not None and len(masks) > 0:
                mask = masks[0].cpu().numpy()
                
                # Ensure mask is 2D
                if mask.ndim == 3:
                    mask = mask[0]
                
                # Resize mask to image dimensions if necessary
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask, (width, height),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Binarize mask
                binary_mask = (mask > confidence_threshold).astype(np.uint8) * 255
            # else: binary_mask rimane tutta nera (inizializzata sopra)

            # Save PNG mask (always, even if empty)
            if save_masks:
                mask_filename = os.path.splitext(image_file)[0] + '.png'
                mask_path = os.path.join(output_mask_dir, mask_filename)
                Image.fromarray(binary_mask).save(mask_path)
            
            # Extract and save coordinates
            if save_json:
                mask_bool = binary_mask > 0
                
                # Get bounding box
                bbox = get_bbox_from_mask(mask_bool)
                
                # Get contours/polygons
                polygons = get_contours_from_mask(mask_bool)
                
                # Create JSON output
                json_data = {
                    'image_file': image_file,
                    'image_size': {'width': width, 'height': height},
                    'prompt': prompt,
                    'bounding_box': bbox,
                    'polygons': polygons,
                    'num_polygons': len(polygons),
                }
                
                # Save JSON
                json_filename = os.path.splitext(image_file)[0] + '.json'
                json_path = os.path.join(output_json_dir, json_filename)
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            
            stats['successful'] += 1
            stats['total_images'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            stats['total_images'] += 1
            stats['errors'].append({
                'image': image_file,
                'error': str(e)
            })
            print(f"Error processing {image_file}: {e}")
    
    return stats


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_structured_dataset(
    processor,
    dataset_dir,
    output_mask_dir,
    output_json_dir,
    prompt,
    confidence_threshold=0.5,
    save_masks=True,
    save_json=True,
    use_gt=False, 
    gt_dir=None, 
    gt_class=1
):
    """
    Process a structured dataset (images organized in subdirectories).
    """
    # Find all subdirectories in dataset
    subdirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    subdirs = sorted(subdirs)
    
    if len(subdirs) == 0:
        raise ValueError(f"No subdirectories found in {dataset_dir}")
    
    print(f"Found {len(subdirs)} subdirectories with images")
    print(f"Using prompt: '{prompt}'")
    print(f"Confidence threshold: {confidence_threshold}")
    
    # Create output directories
    if save_masks:
        os.makedirs(output_mask_dir, exist_ok=True)
    if save_json:
        os.makedirs(output_json_dir, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'errors': []
    }
    
    # Process each subdirectory
    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        subdir_path = os.path.join(dataset_dir, subdir)
        
        # Find image files
        image_files = find_image_files(subdir_path)
        
        if len(image_files) == 0:
            continue
        
        # Create output subdirectories
        if save_masks:
            mask_subdir = os.path.join(output_mask_dir, subdir)
            os.makedirs(mask_subdir, exist_ok=True)
        
        if save_json:
            json_subdir = os.path.join(output_json_dir, subdir)
            os.makedirs(json_subdir, exist_ok=True)
        
        # Process each image in the subdirectory
        for image_file in image_files:
            try:
                image_path = os.path.join(subdir_path, image_file)
                
                # Load and process image
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                
                # Set image and prompt
                inference_state = processor.set_image(image)
                processor.reset_all_prompts(state=inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )

                if use_gt and gt_dir is not None:
                    gt_subdir = os.path.join(gt_dir, subdir)
                    gt_filename = os.path.splitext(image_file)[0] + '.txt'
                    gt_path = os.path.join(gt_subdir, gt_filename)
                    if os.path.exists(gt_path):
                        try:
                            with open(gt_path, 'r') as fg:
                                for line in fg:
                                    parts = line.strip().split()
                                    if len(parts) < 5:
                                        continue
                                    cls_id = int(parts[0])
                                    if cls_id == int(gt_class):
                                        cx = float(parts[1])
                                        cy = float(parts[2])
                                        w = float(parts[3])
                                        h = float(parts[4])
                                        box_norm = [cx, cy, w, h]
                                        inference_state = processor.add_geometric_prompt(box=box_norm, label=True, state=inference_state)
                                        break
                        except Exception as e:
                            print(f"Warning: couldn't read GT {gt_path}: {e}")
                
                # Get segmentation mask
                masks = inference_state.get('masks', None)

                # Initialize with empty mask (all zeros)
                binary_mask = np.zeros((height, width), dtype=np.uint8)

                # If masks exist, use the first one
                if masks is not None and len(masks) > 0:
                    mask = masks[0].cpu().numpy()
                    
                    # Ensure mask is 2D
                    if mask.ndim == 3:
                        mask = mask[0]
                    
                    # Resize mask to image dimensions if necessary
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                            mask, (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    # Binarize mask
                    binary_mask = (mask > confidence_threshold).astype(np.uint8) * 255
                # else: binary_mask rimane tutta nera (inizializzata sopra)

                # Save PNG mask (always, even if empty)
                if save_masks:
                    mask_filename = os.path.splitext(image_file)[0] + '.png'
                    mask_path = os.path.join(output_mask_dir, mask_filename)
                    Image.fromarray(binary_mask).save(mask_path)
                
                # Extract and save coordinates
                if save_json:
                    mask_bool = binary_mask > 0
                    
                    # Get bounding box
                    bbox = get_bbox_from_mask(mask_bool)
                    
                    # Get contours/polygons
                    polygons = get_contours_from_mask(mask_bool)
                    
                    # Create JSON output
                    json_data = {
                        'image_file': image_file,
                        'image_size': {'width': width, 'height': height},
                        'prompt': prompt,
                        'bounding_box': bbox,
                        'polygons': polygons,
                        'num_polygons': len(polygons),
                    }
                    
                    # Save JSON
                    json_filename = os.path.splitext(image_file)[0] + '.json'
                    json_path = os.path.join(json_subdir, json_filename)
                    with open(json_path, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                stats['successful'] += 1
                stats['total_images'] += 1
                
            except Exception as e:
                stats['failed'] += 1
                stats['total_images'] += 1
                stats['errors'].append({
                    'image': f"{subdir}/{image_file}",
                    'error': str(e)
                })
                print(f"Error processing {subdir}/{image_file}: {e}")
    
    return stats


def print_summary(stats, output_mask_dir, output_json_dir, save_masks, save_json):
    """Print inference summary statistics."""
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    
    if stats['errors']:
        print(f"\nErrors:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error['image']}: {error['error']}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    if save_masks:
        print(f"\nMasks saved to: {output_mask_dir}")
    if save_json:
        print(f"Coordinates saved to: {output_json_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompt-based image segmentation inference (supports flat and structured datasets)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory (flat structure with images or structured with subdirectories)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for segmentation (e.g., 'person', 'dog', 'cup')"
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        default="./segmentation_masks",
        help="Directory to save segmentation masks as PNG files"
    )
    parser.add_argument(
        "--output_json_dir",
        type=str,
        default="./segmentation_coordinates",
        help="Directory to save segmentation coordinates as JSON files"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for mask binarization (default: 0.5)"
    )
    parser.add_argument(
        "--flat_dataset",
        action="store_true",
        help="Use this flag if your dataset is flat (all images directly in the folder). "
             "Without this flag, the script will auto-detect the structure."
    )
    parser.add_argument(
        "--structured_dataset",
        action="store_true",
        help="Use this flag if your dataset is structured (images in subdirectories). "
             "Without this flag, the script will auto-detect the structure."
    )
    parser.add_argument(
        "--no_masks",
        action="store_true",
        help="Don't save PNG masks (only save JSON coordinates)"
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Don't save JSON coordinates (only save PNG masks)"
    )
    parser.add_argument(
        "--use_gt",
        action="store_true",
        help="Use GT bbox (YOLO txt) as additional box prompt when class present"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Path to GT txt files (same names as images). If structured, mirrors dataset subdirs."
    )
    parser.add_argument(
        "--gt_class",
        type=int,
        default=1,
        help="GT class id to use (YOLO class index). Default: 1"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist")
        sys.exit(1)
    
    if args.no_masks and args.no_json:
        print("Error: At least one output format (masks or json) must be enabled")
        sys.exit(1)
    
    if args.flat_dataset and args.structured_dataset:
        print("Error: Cannot specify both --flat_dataset and --structured_dataset")
        sys.exit(1)
    
    # Detect or use specified dataset structure
    if args.flat_dataset:
        is_flat = True
        print("✓ Using flat dataset mode (images directly in folder)")
    elif args.structured_dataset:
        is_flat = False
        print("✓ Using structured dataset mode (images in subdirectories)")
    else:
        is_flat, description = detect_dataset_structure(args.dataset_dir)
        print(f"✓ {description}")
    
    # Load SAM3 model
    print("\nLoading SAM3 image model...")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    
    try:
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    print("\nStarting inference...")
    if is_flat:
        stats = process_flat_dataset(
            processor=processor,
            dataset_dir=args.dataset_dir,
            output_mask_dir=args.output_mask_dir,
            output_json_dir=args.output_json_dir,
            prompt=args.prompt,
            confidence_threshold=args.confidence_threshold,
            save_masks=not args.no_masks,
            save_json=not args.no_json,
            use_gt=args.use_gt,
            gt_dir=args.gt_dir,
            gt_class=args.gt_class,
        )
    else:
        stats = process_structured_dataset(
            processor=processor,
            dataset_dir=args.dataset_dir,
            output_mask_dir=args.output_mask_dir,
            output_json_dir=args.output_json_dir,
            prompt=args.prompt,
            confidence_threshold=args.confidence_threshold,
            save_masks=not args.no_masks,
            save_json=not args.no_json,
            use_gt=args.use_gt,
            gt_dir=args.gt_dir,
            gt_class=args.gt_class,
        )
    
    print_summary(stats, args.output_mask_dir, args.output_json_dir, not args.no_masks, not args.no_json)
    print("\n✓ Inference completed!")


if __name__ == "__main__":
    main()
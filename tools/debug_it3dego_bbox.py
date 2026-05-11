#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to visualize first frame and bbox from IT3DEgo dataset.
Shows bbox as parsed from .txt file and as configured for SAM3.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from IT3DEgo import IT3DEgo


def xywh_to_xyxy(box):
    """Convert box from (x, y, w, h) format to (x1, y1, x2, y2)."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def visualize_entity(dataset, seq_name, entity_name, output_path=None):
    """Visualize first frame with bbox for a given entity."""
    
    print(f"\n{'='*60}")
    print(f"Visualizing: {seq_name} / {entity_name}")
    print(f"{'='*60}")
    
    # Load data
    try:
        image_paths, boxes, meta = dataset[(seq_name, entity_name)]
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False
    
    print(f"Total frames: {len(image_paths)}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Boxes dtype: {boxes.dtype}")
    
    # Find first frame with valid box
    first_frame_idx = None
    first_box = None
    for i, box in enumerate(boxes):
        print(f"  Frame {i}: box = {box}")
        if not np.isnan(box).any():
            if first_frame_idx is None:
                first_frame_idx = i
                first_box = box
                print(f"    ^ FIRST VALID BOX")
    
    if first_frame_idx is None:
        print("ERROR: No valid boxes found!")
        return False
    
    # Load first frame image
    img_path = image_paths[first_frame_idx]
    print(f"\nFirst frame image: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"ERROR: Image not found: {img_path}")
        return False
    
    img = Image.open(img_path)
    img_width, img_height = img.size
    print(f"Image size: {img_width}x{img_height}")
    
    # Show original box
    print(f"\nOriginal box (from .txt, format x,y,w,h):")
    print(f"  {first_box}")
    x, y, w, h = first_box
    print(f"  x={x}, y={y}, w={w}, h={h}")
    
    # Convert to xyxy
    x1, y1, x2, y2 = xywh_to_xyxy(first_box)
    print(f"\nAfter xywh->xyxy conversion:")
    print(f"  ({x1}, {y1}, {x2}, {y2})")
    
    # Clamp to image bounds
    x1_clamped = max(0, min(int(x1), img_width - 1))
    y1_clamped = max(0, min(int(y1), img_height - 1))
    x2_clamped = max(0, min(int(x2), img_width - 1))
    y2_clamped = max(0, min(int(y2), img_height - 1))
    
    print(f"\nAfter clamping to image bounds:")
    print(f"  ({x1_clamped}, {y1_clamped}, {x2_clamped}, {y2_clamped})")
    
    if x1_clamped >= x2_clamped or y1_clamped >= y2_clamped:
        print(f"WARNING: Invalid clamped box (zero or negative size)!")
    else:
        box_area = (x2_clamped - x1_clamped) * (y2_clamped - y1_clamped)
        img_area = img_width * img_height
        print(f"  Box area: {box_area} pixels")
        print(f"  Image area: {img_area} pixels")
        print(f"  Box coverage: {100.0 * box_area / img_area:.2f}%")
    
    # Draw bbox on image (use original unclamped for visualization)
    img_viz = img.copy()
    draw = ImageDraw.Draw(img_viz)
    
    # Draw unclamped bbox in red
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    # Add label
    draw.text((int(x1), max(0, int(y1)-20)), "Unclamped (red)", fill="red")
    
    # Draw clamped bbox in green
    draw.rectangle([x1_clamped, y1_clamped, x2_clamped, y2_clamped], outline="green", width=3)
    # Add label
    draw.text((int(x1_clamped), max(0, int(y1_clamped)-40)), "Clamped (green)", fill="green")
    
    # Save visualization
    if output_path is None:
        output_path = f"debug_{seq_name.replace('/', '_')}_{entity_name}.png"
    
    img_viz.save(output_path)
    print(f"\nVisualization saved to: {output_path}")
    
    # Also try to display using matplotlib if available
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("on")
        
        axes[1].imshow(img_viz)
        axes[1].set_title(f"With BBox: {seq_name}/{entity_name}\n(Red=Unclamped, Green=Clamped)")
        axes[1].axis("on")
        
        plt.tight_layout()
        plt_path = output_path.replace(".png", "_plot.png")
        plt.savefig(plt_path, dpi=100, bbox_inches="tight")
        print(f"Matplotlib plot saved to: {plt_path}")
        plt.close()
    except ImportError:
        print("(matplotlib not available for interactive display)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Debug IT3DEgo bbox visualization")
    parser.add_argument("--frames_root", type=str, required=True, help="Path to raw_videos root")
    parser.add_argument("--ann_root", type=str, required=True, help="Path to annotations root")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence name to debug")
    parser.add_argument("--entity", type=str, required=True, help="Entity name to debug")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()
    
    # Load dataset
    print("Loading IT3DEgo dataset...")
    dataset = IT3DEgo(
        frames_root=args.frames_root,
        ann_root=args.ann_root,
        seq_names=args.sequence,
    )
    print(f"Dataset: {dataset}")
    
    # Visualize
    success = visualize_entity(dataset, args.sequence, args.entity, args.output)
    
    if success:
        print("\nDebug visualization complete!")
    else:
        print("\nDebug visualization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Convert IT3DEgo inference mask outputs to 2D bbox annotation files.

Expected input layout (as produced by `it3dego_inference.py`):
    {input_dir}/{sequence_name}/{entity_name}/{frame_id}.png

Output layout (compatible with the IT3DEgo loader):
    {output_dir}/{sequence_name}/2d_bbox_annot/{entity_name}.txt

Each line in `{entity_name}.txt` will be:
    {frame_id} x y w h
where x,y,w,h are floats (one decimal). If a mask is empty, NaN values
are written for the bbox.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def mask_to_bbox(mask):
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


def load_mask_as_binary(path):
    img = Image.open(path)
    # Handle palette/indexed images
    if img.mode == 'P':
        img = img.convert('RGB')
    arr = np.array(img)
    # If RGB, take the first channel
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def convert_entity_masks(entity_dir, output_annot_path):
    # entity_dir: path/to/{sequence}/{entity}/
    png_files = sorted([f for f in os.listdir(entity_dir) if f.lower().endswith('.png')])
    if len(png_files) == 0:
        return 0

    os.makedirs(os.path.dirname(output_annot_path), exist_ok=True)

    with open(output_annot_path, 'w') as out_f:
        for png in png_files:
            frame_name = os.path.splitext(png)[0]
            mask_path = os.path.join(entity_dir, png)
            try:
                mask = load_mask_as_binary(mask_path)
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    out_f.write(f"{frame_name} nan nan nan nan\n")
                else:
                    out_f.write(f"{frame_name} {bbox[0]:.1f} {bbox[1]:.1f} {bbox[2]:.1f} {bbox[3]:.1f}\n")
            except Exception as e:
                # On error write NaNs to keep alignment
                out_f.write(f"{frame_name} nan nan nan nan\n")
    return len(png_files)


def main():
    parser = argparse.ArgumentParser(description="Convert IT3DEgo masks to 2D bbox annotation files")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory with mask outputs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted annotation files')
    parser.add_argument('--sequence', type=str, default=None, help='Optional specific sequence to process')
    parser.add_argument('--entity', type=str, default=None, help='Optional specific entity to process')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    sequences = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    if args.sequence:
        sequences = [s for s in sequences if s == args.sequence]

    total = 0
    for seq in sequences:
        seq_dir = os.path.join(input_dir, seq)
        entities = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
        if args.entity:
            entities = [e for e in entities if e == args.entity]

        for ent in tqdm(entities, desc=f"Seq {seq}"):
            ent_dir = os.path.join(seq_dir, ent)
            output_annot_path = os.path.join(output_dir, seq, '2d_bbox_annot', f"{ent}.txt")
            n = convert_entity_masks(ent_dir, output_annot_path)
            if n > 0:
                total += n

    print(f"Converted {total} masks into annotation files under {output_dir}")


if __name__ == '__main__':
    main()

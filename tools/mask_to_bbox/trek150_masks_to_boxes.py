"""
Convert TREK-150 inference mask outputs to bbox (`frames.txt` + `boxes.txt`).

Expected input layout (as produced by `trek150_inference.py`):
    {input_dir}/{sequence_name}/frame_{frame_index:010d}.png

Output layout:
    {output_dir}/{sequence_name}/frames.txt  # list of frame indices
    {output_dir}/{sequence_name}/boxes.txt   # x,y,w,h per line (or nan)
"""

import argparse
import os
import re

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
    if img.mode == 'P':
        img = img.convert('RGB')
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def extract_frame_index(filename):
    m = re.search(r"(\d+)", filename)
    if not m:
        return None
    return int(m.group(1))


def convert_sequence(seq_dir, output_base):
    png_files = sorted([f for f in os.listdir(seq_dir) if f.lower().endswith('.png') and 'frame_' in f])
    if len(png_files) == 0:
        return 0

    frames = []
    boxes = []

    for png in tqdm(png_files, desc=f"Seq {os.path.basename(seq_dir)}"):
        idx = extract_frame_index(png)
        if idx is None:
            continue
        mask_path = os.path.join(seq_dir, png)
        try:
            mask = load_mask_as_binary(mask_path)
            bbox = mask_to_bbox(mask)
            if bbox is None:
                boxes.append([np.nan, np.nan, np.nan, np.nan])
            else:
                boxes.append(bbox)
            frames.append(idx)
        except Exception:
            frames.append(idx)
            boxes.append([np.nan, np.nan, np.nan, np.nan])

    os.makedirs(output_base, exist_ok=True)
    frames_path = os.path.join(output_base, 'frames.txt')
    boxes_path = os.path.join(output_base, 'boxes.txt')

    with open(frames_path, 'w') as f:
        for fr in frames:
            f.write(f"{fr}\n")

    with open(boxes_path, 'w') as f:
        for b in boxes:
            if np.any(np.isnan(b)):
                f.write('nan,nan,nan,nan\n')
            else:
                f.write(f"{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}\n")

    return len(frames)


def main():
    parser = argparse.ArgumentParser(description='Convert TREK-150 masks to frames+boxes files')
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory with trek150 mask outputs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save frames.txt and boxes.txt')
    parser.add_argument('--sequence', type=str, default=None, help='Optional specific sequence to process')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        return

    sequences = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    if args.sequence:
        sequences = [s for s in sequences if s == args.sequence]

    total = 0
    for seq in sequences:
        seq_dir = os.path.join(args.input_dir, seq)
        out_base = os.path.join(args.output_dir, seq)
        n = convert_sequence(seq_dir, out_base)
        if n > 0:
            total += n

    print(f"Converted {total} frames into {args.output_dir}")


if __name__ == '__main__':
    main()

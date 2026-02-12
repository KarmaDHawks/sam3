#!/usr/bin/env python3
"""
HangCon_eval_IoU.py

Calcola due tipi di IoU per ogni confronto maschera <-> GT:
  1) IoU_mask_bbox: confronto tra maschera binaria (foreground pixels) e la maschera rettangolare del GT (pixel-wise).
  2) IoU_bbox_bbox: la maschera viene convertita in bbox minima e confrontata con il bbox GT (bbox-vs-bbox).

Output:
  - mask_vs_gt_iou_detailed.csv (dettagli IoU mask_vs_bbox)
  - mask_bbox_vs_gt_bbox_iou_detailed.csv (dettagli IoU bbox_vs_bbox)
  - mask_iou_summary.json (riepilogo con entrambe le metriche)
  - per-class e per-view/class CSV separati per entrambe le metriche
"""

import os
import argparse
import csv
from PIL import Image
import numpy as np
from collections import defaultdict
import json

VIEWS = ['D1', 'D2', 'D3', 'D4', 'D8', 'S']


def read_list(csv_path):
    names = []
    with open(csv_path, 'r', newline='') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            base = os.path.splitext(os.path.basename(line))[0]
            names.append(base)
    return list(dict.fromkeys(names))


def find_mask_file(basename, masks_dir):
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    for e in exts:
        p = os.path.join(masks_dir, basename + e)
        if os.path.isfile(p):
            return p
    for fn in os.listdir(masks_dir):
        if os.path.splitext(fn)[0] == basename:
            return os.path.join(masks_dir, fn)
    return None


def read_ann_file(basename, ann_dir):
    path = os.path.join(ann_dir, basename + '.txt')
    if not os.path.isfile(path):
        return []
    anns = []
    with open(path, 'r') as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            parts = l.split()
            if len(parts) >= 5:
                cls = parts[0]
                try:
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                except ValueError:
                    continue
                anns.append((cls, x_c, y_c, w, h))
    return anns


def normalized_to_pixels(xc, yc, w, h, img_w, img_h):
    x_c_px = xc * img_w
    y_c_px = yc * img_h
    w_px = w * img_w
    h_px = h * img_h
    x_min = int(round(x_c_px - w_px / 2.0))
    y_min = int(round(y_c_px - h_px / 2.0))
    x_max = int(round(x_c_px + w_px / 2.0))
    y_max = int(round(y_c_px + h_px / 2.0))
    x_min = max(0, min(img_w - 1, x_min))
    y_min = max(0, min(img_h - 1, y_min))
    x_max = max(0, min(img_w - 1, x_max))
    y_max = max(0, min(img_h - 1, y_max))
    return x_min, y_min, x_max, y_max


def bbox_mask_array(xmin, ymin, xmax, ymax, img_w, img_h):
    if xmax < xmin or ymax < ymin:
        return np.zeros((img_h, img_w), dtype=bool)
    arr = np.zeros((img_h, img_w), dtype=bool)
    arr[ymin:(ymax + 1), xmin:(xmax + 1)] = True
    return arr


def mask_to_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if ys.size == 0 or xs.size == 0:
        return None
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return xmin, ymin, xmax, ymax


def bbox_area(xmin, ymin, xmax, ymax):
    if xmax < xmin or ymax < ymin:
        return 0
    return (xmax - xmin + 1) * (ymax - ymin + 1)


def bbox_iou(b1, b2):
    xmin1, ymin1, xmax1, ymax1 = b1
    xmin2, ymin2, xmax2, ymax2 = b2
    if xmax1 < xmin2 or xmax2 < xmin1 or ymax1 < ymin2 or ymax2 < ymin1:
        return 0.0
    xA = max(xmin1, xmin2)
    yA = max(ymin1, ymin2)
    xB = min(xmax1, xmax2)
    yB = min(ymax1, ymax2)
    inter_w = xB - xA + 1
    inter_h = yB - yA + 1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = bbox_area(xmin1, ymin1, xmax1, ymax1) + bbox_area(xmin2, ymin2, xmax2, ymax2) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def mask_to_bool_array(img_path):
    im = Image.open(img_path).convert('L')
    arr = np.array(im)
    return arr > 0, im.size  # (W,H)


def extract_view_from_name(basename):
    for v in VIEWS:
        if basename.startswith(v):
            return v
    return 'UNKNOWN'


def safe_mean(lst):
    return float(np.mean(lst)) if lst else 0.0


def main(args):
    os.makedirs(args.out, exist_ok=True)
    names = read_list(args.list)

    # Detailed lists and aggregators for both metrics
    detailed_maskbbox = []
    detailed_bboxbbox = []

    per_class_maskbbox = defaultdict(list)
    per_view_class_maskbbox = defaultdict(lambda: defaultdict(list))
    all_ious_maskbbox = []

    per_class_bboxbbox = defaultdict(list)
    per_view_class_bboxbbox = defaultdict(lambda: defaultdict(list))
    all_ious_bboxbbox = []

    for base in names:
        mask_file = find_mask_file(base, args.masks)
        if mask_file is None:
            print(f'WARN: mask not found for {base} in {args.masks}, skipping')
            continue
        mask_bool, size = mask_to_bool_array(mask_file)
        img_w, img_h = size
        if mask_bool.ndim != 2:
            mask_bool = mask_bool.squeeze()
        mask_bool = mask_bool.astype(bool)

        anns = read_ann_file(base, args.anns)
        if not anns:
            print(f'WARN: annotations not found or empty for {base} in {args.anns}, skipping')
            continue

        view = extract_view_from_name(base)

        # convert mask to bbox once (for bbox-vs-bbox)
        mask_bbox = mask_to_bbox(mask_bool)

        for (cls, x_c, y_c, w_n, h_n) in anns:
            gt_bbox = normalized_to_pixels(x_c, y_c, w_n, h_n, img_w, img_h)

            # 1) IoU mask vs bbox (pixel-wise): create GT bbox mask and compare with mask_bool
            gt_bbox_mask = bbox_mask_array(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], img_w, img_h)
            inter_pixels = np.logical_and(mask_bool, gt_bbox_mask).sum()
            union_pixels = np.logical_or(mask_bool, gt_bbox_mask).sum()
            iou_mask_bbox = float(inter_pixels) / float(union_pixels) if union_pixels > 0 else 0.0

            detailed_maskbbox.append({
                'file': base,
                'view': view,
                'class': cls,
                'iou_mask_bbox': iou_mask_bbox,
                'mask_path': os.path.abspath(mask_file),
                'ann_path': os.path.abspath(os.path.join(args.anns, base + '.txt')),
                'gt_bbox': gt_bbox
            })
            per_class_maskbbox[cls].append(iou_mask_bbox)
            per_view_class_maskbbox[view][cls].append(iou_mask_bbox)
            all_ious_maskbbox.append(iou_mask_bbox)

            # 2) IoU bbox-vs-bbox: only if mask_bbox exists
            if mask_bbox is None:
                iou_bbox_bbox = 0.0
            else:
                iou_bbox_bbox = bbox_iou(mask_bbox, gt_bbox)

            detailed_bboxbbox.append({
                'file': base,
                'view': view,
                'class': cls,
                'iou_bbox_bbox': iou_bbox_bbox,
                'mask_path': os.path.abspath(mask_file),
                'ann_path': os.path.abspath(os.path.join(args.anns, base + '.txt')),
                'mask_bbox': mask_bbox,
                'gt_bbox': gt_bbox
            })
            per_class_bboxbbox[cls].append(iou_bbox_bbox)
            per_view_class_bboxbbox[view][cls].append(iou_bbox_bbox)
            all_ious_bboxbbox.append(iou_bbox_bbox)

    # Write detailed CSVs
    detailed_mask_csv = os.path.join(args.out, 'mask_vs_gt_iou_detailed.csv')
    with open(detailed_mask_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'view', 'class', 'iou_mask_bbox', 'mask_path', 'ann_path', 'gt_bbox'])
        writer.writeheader()
        for r in detailed_maskbbox:
            writer.writerow(r)

    detailed_bbox_csv = os.path.join(args.out, 'mask_bbox_vs_gt_bbox_iou_detailed.csv')
    with open(detailed_bbox_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'view', 'class', 'iou_bbox_bbox', 'mask_path', 'ann_path', 'mask_bbox', 'gt_bbox'])
        writer.writeheader()
        for r in detailed_bboxbbox:
            writer.writerow(r)

    # Aggregates
    overall_miou_maskbbox = safe_mean(all_ious_maskbbox)
    per_class_miou_maskbbox = {cls: safe_mean(vals) for cls, vals in per_class_maskbbox.items()}
    per_view_class_miou_maskbbox = {view: {cls: safe_mean(vals) for cls, vals in clsdict.items()} for view, clsdict in per_view_class_maskbbox.items()}

    overall_miou_bboxbbox = safe_mean(all_ious_bboxbbox)
    per_class_miou_bboxbbox = {cls: safe_mean(vals) for cls, vals in per_class_bboxbbox.items()}
    per_view_class_miou_bboxbbox = {view: {cls: safe_mean(vals) for cls, vals in clsdict.items()} for view, clsdict in per_view_class_bboxbbox.items()}

    summary = {
        'overall_mIoU_mask_vs_bbox': overall_miou_maskbbox,
        'per_class_mIoU_mask_vs_bbox': per_class_miou_maskbbox,
        'per_view_class_mIoU_mask_vs_bbox': per_view_class_miou_maskbbox,
        'overall_mIoU_bbox_vs_bbox': overall_miou_bboxbbox,
        'per_class_mIoU_bbox_vs_bbox': per_class_miou_bboxbbox,
        'per_view_class_mIoU_bbox_vs_bbox': per_view_class_miou_bboxbbox,
        'n_comparisons_mask_vs_bbox': len(all_ious_maskbbox),
        'n_comparisons_bbox_vs_bbox': len(all_ious_bboxbbox),
        'n_files_processed': len(set(r['file'] for r in (detailed_maskbbox + detailed_bboxbbox)))
    }

    with open(os.path.join(args.out, 'mask_iou_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-class CSVs for both metrics
    per_class_mask_csv = os.path.join(args.out, 'mask_iou_per_class_mask_vs_bbox.csv')
    with open(per_class_mask_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'mIoU_mask_vs_bbox'])
        for cls, miou in sorted(per_class_miou_maskbbox.items()):
            writer.writerow([cls, miou])

    per_view_mask_csv = os.path.join(args.out, 'mask_iou_per_view_class_mask_vs_bbox.csv')
    with open(per_view_mask_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['view', 'class', 'mIoU_mask_vs_bbox'])
        for view, clsdict in sorted(per_view_class_miou_maskbbox.items()):
            for cls, miou in sorted(clsdict.items()):
                writer.writerow([view, cls, miou])

    per_class_bbox_csv = os.path.join(args.out, 'mask_iou_per_class_bbox_vs_bbox.csv')
    with open(per_class_bbox_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'mIoU_bbox_vs_bbox'])
        for cls, miou in sorted(per_class_miou_bboxbbox.items()):
            writer.writerow([cls, miou])

    per_view_bbox_csv = os.path.join(args.out, 'mask_iou_per_view_class_bbox_vs_bbox.csv')
    with open(per_view_bbox_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['view', 'class', 'mIoU_bbox_vs_bbox'])
        for view, clsdict in sorted(per_view_class_miou_bboxbbox.items()):
            for cls, miou in sorted(clsdict.items()):
                writer.writerow([view, cls, miou])

    print('Done.')
    print('Detailed mask-vs-bbox CSV:', detailed_mask_csv)
    print('Detailed bbox-vs-bbox CSV:', detailed_bbox_csv)
    print('Summary JSON:', os.path.join(args.out, 'mask_iou_summary.json'))
    print('Per-class mask-vs-bbox CSV:', per_class_mask_csv)
    print('Per-view/class mask-vs-bbox CSV:', per_view_mask_csv)
    print('Per-class bbox-vs-bbox CSV:', per_class_bbox_csv)
    print('Per-view/class bbox-vs-bbox CSV:', per_view_bbox_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--list', required=True, help='CSV or txt list, one filename per line (basename or filename)')
    p.add_argument('--masks', required=True, help='Folder with mask images')
    p.add_argument('--anns', required=True, help='Folder with annotation .txt files (YOLO normalized xcenter ycenter w h)')
    p.add_argument('--out', required=True, help='Output folder to store CSV/JSON results')
    args = p.parse_args()
    main(args)
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import csv

# IoU threshold for considering a prediction as a True Positive
IOU_THRESH = 0.5

LABEL_MAP = {
    "hanging object": 0,
    "rope": 1,
    "hanging object with rope": 2,
    "hanging_object": 0,
    "hangingobject": 0,
    "hangingobjectwithrope": 2,
    "hanging_object_with_rope": 2,
    "withrope": 2,
}

def parse_yolo_gt(path: Path):
    """
    Parse YOLO-format GT txt where tokens may be across lines or all in one line.
    Accepts repeated sequences: <class> <xc> <yc> <w> <h> ...
    Returns dict: class_id -> list of bboxes in normalized format [xc, yc, w, h]
    """
    out = {0: [], 1: [], 2: []}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return out
    tokens = text.split()
    i = 0
    while i + 4 < len(tokens):
        cls_token = tokens[i]
        try:
            cls = int(cls_token)
        except Exception:
            key = cls_token.lower()
            cls = LABEL_MAP.get(key, None)
            if cls is None:
                for k, v in LABEL_MAP.items():
                    if k in key or key in k:
                        cls = v
                        break
        if cls is None or cls not in (0, 1, 2):
            i += 1
            continue
        if i + 4 < len(tokens):
            try:
                xc = float(tokens[i + 1])
                yc = float(tokens[i + 2])
                w = float(tokens[i + 3])
                h = float(tokens[i + 4])
                out[cls].append([xc, yc, w, h])
                i += 5
            except Exception:
                i += 1
        else:
            break
    return out

def load_mask(path: Path):
    """
    Load mask as uint8 2D array (0/1). Return (mask_array, (height, width)).
    If file missing/unreadable, return (None, None).
    """
    if not path.exists():
        return None, None
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
            arr = np.asarray(arr)
        else:
            im = Image.open(str(path)).convert("L")
            arr = np.asarray(im)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = (arr > 0).astype(np.uint8)
        if mask.size == 0:
            return None, None
        h, w = mask.shape
        return mask, (h, w)
    except Exception:
        print(f"Warning: couldn't read mask {path}")
        return None, None

def find_pred_file_by_basename(base: str, pred_dir: Path):
    """
    Find files in pred_dir with given stem (base). Return list of Paths (non-recursive).
    """
    candidates = list(pred_dir.glob(f"{base}.*"))
    return sorted(candidates)

def select_prediction_mask(base: str, pred_dir: Path):
    """
    Return (mask, (h,w), path) choosing first non-empty mask among files with same basename.
    If no files at all, return (None,None,None).
    If files exist but all empty/unreadable, return (None,(None),first_path) to signal file present but empty.
    """
    files = find_pred_file_by_basename(base, pred_dir)
    if not files:
        return None, None, None
    first_path = files[0]
    # try to find first non-empty mask
    for p in files:
        mask, shape = load_mask(p)
        if mask is not None and mask.sum() > 0:
            return mask, shape, p
    # none non-empty, try to return first readable (even if empty) to indicate file exists
    for p in files:
        mask, shape = load_mask(p)
        if mask is not None:
            return mask, shape, p
    return None, None, first_path

def mask_to_bbox(mask):
    """
    mask: 2D uint8 array (0/1). Return bbox in pixel coords [xmin,ymin,xmax,ymax].
    If mask is None or empty -> return None.
    """
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    ymin = float(ys.min())
    ymax = float(ys.max())
    xmin = float(xs.min())
    xmax = float(xs.max())
    return [xmin, ymin, xmax, ymax]

def yolo_to_bbox_abs(yolo, img_w, img_h):
    """
    yolo: [xc, yc, w, h] normalized.
    return absolute bbox [xmin,ymin,xmax,ymax] in pixels (floats, clipped to image).
    """
    xc, yc, w, h = yolo
    bw = w * img_w
    bh = h * img_h
    cx = xc * img_w
    cy = yc * img_h
    xmin = cx - bw / 2.0
    ymin = cy - bh / 2.0
    xmax = cx + bw / 2.0
    ymax = cy + bh / 2.0
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(img_w, xmax)
    ymax = min(img_h, ymax)
    return [xmin, ymin, xmax, ymax]

def bbox_iou(boxA, boxB):
    """
    boxA, boxB: [xmin,ymin,xmax,ymax]
    returns IoU (float). If either is None -> 0.0
    """
    if boxA is None or boxB is None:
        return 0.0
    xa1, ya1, xa2, ya2 = map(float, boxA)
    xb1, yb1, xb2, yb2 = map(float, boxB)
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    areaB = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = areaA + areaB - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union

def evaluate(gt_dir: Path, pred_dir: Path, pred_ext: str = ".png", out_csv: Path = None, verbose=False, only_list=None):
    gt_files = sorted([p for p in gt_dir.glob("*.txt")])
    if only_list:
        gt_files = [p for p in gt_files if p.stem in only_list]
    TP = FP = FN = TN = 0
    per_image = []
    ious_by_class = {0: [], 1: [], 2: []}
    tp_ious_by_class = {0: [], 1: [], 2: []}  # Only for TP predictions

    for gt_path in gt_files:
        base = gt_path.stem
        gt_boxes = parse_yolo_gt(gt_path)  # dict class -> list of [xc,yc,w,h]
        # GT presence = any class 0 or 2 (rope-only class 1 is considered absence)
        gt_has_object = any(len(gt_boxes[c]) > 0 for c in (0, 2))

        pred_mask, shape, pred_path = select_prediction_mask(base, pred_dir)
        pred_present = False if pred_mask is None else (pred_mask.sum() > 0)

        iou0 = iou1 = iou2 = None
        best_iou = 0.0
        is_tp = False

        # Calculate IoU if prediction is present
        if pred_present and shape is not None:
            img_h, img_w = shape
            pred_bbox = mask_to_bbox(pred_mask)
            if pred_bbox is not None:
                def best_iou_for_class(cls):
                    boxes = gt_boxes.get(cls, [])
                    if not boxes:
                        return 0.0
                    best = 0.0
                    for yolo in boxes:
                        gt_abs = yolo_to_bbox_abs(yolo, img_w, img_h)
                        i = bbox_iou(pred_bbox, gt_abs)
                        if i > best:
                            best = i
                    return best
                iou0 = best_iou_for_class(0)
                iou1 = best_iou_for_class(1)
                iou2 = best_iou_for_class(2)
                # Best IoU is the maximum across all classes
                best_iou = max(iou0, iou1, iou2)

        # Apply new evaluation logic with IoU threshold
        if gt_has_object:
            if pred_present and best_iou >= IOU_THRESH:
                TP += 1
                is_tp = True
            elif pred_present:
                FP += 1
                FN += 1
            else:
                FN += 1
        else:
            if pred_present:
                FP += 1
            else:
                TN += 1

        # Store IoU values for statistics
        if pred_present:
            ious_by_class[0].append(iou0 if iou0 is not None else 0.0)
            ious_by_class[1].append(iou1 if iou1 is not None else 0.0)
            ious_by_class[2].append(iou2 if iou2 is not None else 0.0)
            
            # Store IoU only for TP predictions
            if is_tp:
                tp_ious_by_class[0].append(iou0 if iou0 is not None else 0.0)
                tp_ious_by_class[1].append(iou1 if iou1 is not None else 0.0)
                tp_ious_by_class[2].append(iou2 if iou2 is not None else 0.0)

        per_image.append((base, str(pred_path) if pred_path is not None else "", int(bool(pred_present)), int(bool(gt_has_object)),
                          "" if iou0 is None else f"{iou0:.6f}",
                          "" if iou1 is None else f"{iou1:.6f}",
                          "" if iou2 is None else f"{iou2:.6f}"))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # Calculate mIoU only over TP predictions
    mean_iou = {c: float(np.mean(tp_ious_by_class[c])) if len(tp_ious_by_class[c]) > 0 else 0.0 for c in (0, 1, 2)}

    N = len(per_image)
    images_with_object = TP + FN
    images_without_object = FP + TN

    tp_over_pos = TP / images_with_object if images_with_object > 0 else 0.0
    fn_over_pos = FN / images_with_object if images_with_object > 0 else 0.0
    fp_over_neg = FP / images_without_object if images_without_object > 0 else 0.0
    tn_over_neg = TN / images_without_object if images_without_object > 0 else 0.0

    print("Results:")
    print(f"  Images evaluated : {len(per_image)}")
    print(f"  TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    print(f"  Images with object: {images_with_object} (TP {TP} = {tp_over_pos:.4f}, FN {FN} = {fn_over_pos:.4f})")
    print(f"  Images without object: {images_without_object} (TN {TN} = {tn_over_neg:.4f}, FP {FP} = {fp_over_neg:.4f})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Mean IoU class0: {mean_iou[0]:.4f} (computed over predictions)")
    print(f"  Mean IoU class1: {mean_iou[1]:.4f} (computed over predictions)")
    print(f"  Mean IoU class2: {mean_iou[2]:.4f} (computed over predictions)")

    if verbose:
        print("\nPer-image (basename, pred_path, pred_present, gt_has_object, IoU0, IoU1, IoU2):")
        for row in per_image:
            show = (int(row[2]) != int(row[3]))
            if row[4] != "" and float(row[4]) < IOU_THRESH:
                show = True
            if show:
                print(" ", row)

    if out_csv is not None:
        out_csv_parent = out_csv.parent
        if not out_csv_parent.exists():
            out_csv_parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["basename", "pred_path", "pred_present", "gt_has_object", "iou_class0", "iou_class1", "iou_class2"])
            for row in per_image:
                writer.writerow(row)
            writer.writerow([])
            writer.writerow(["SUMMARY"])
            writer.writerow(["images", len(per_image)])
            writer.writerow(["TP", TP])
            writer.writerow(["FP", FP])
            writer.writerow(["FN", FN])
            writer.writerow(["TN", TN])
            writer.writerow(["images_with_object", images_with_object])
            writer.writerow(["images_without_object", images_without_object])
            writer.writerow(["precision", f"{precision:.6f}"])
            writer.writerow(["recall", f"{recall:.6f}"])
            writer.writerow(["f1", f"{f1:.6f}"])
            writer.writerow(["mean_iou_class0", f"{mean_iou[0]:.6f}"])
            writer.writerow(["mean_iou_class1", f"{mean_iou[1]:.6f}"])
            writer.writerow(["mean_iou_class2", f"{mean_iou[2]:.6f}"])
            writer.writerow([])
            writer.writerow(["fractions"])
            writer.writerow(["TP_over_images_with_object", f"{TP}/{images_with_object}" if images_with_object>0 else "0/0", f"{tp_over_pos:.6f}"])
            writer.writerow(["FN_over_images_with_object", f"{FN}/{images_with_object}" if images_with_object>0 else "0/0", f"{fn_over_pos:.6f}"])
            writer.writerow(["FP_over_images_without_object", f"{FP}/{images_without_object}" if images_without_object>0 else "0/0", f"{fp_over_neg:.6f}"])
            writer.writerow(["TN_over_images_without_object", f"{TN}/{images_without_object}" if images_without_object>0 else "0/0", f"{tn_over_neg:.6f}"])
        print(f"Wrote CSV to {out_csv}")

    return {
        "tp": TP, "fp": FP, "fn": FN, "tn": TN,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_iou": mean_iou,
        "images_with_object": images_with_object,
        "images_without_object": images_without_object,
        "tp_over_pos": tp_over_pos,
        "fn_over_pos": fn_over_pos,
        "fp_over_neg": fp_over_neg,
        "tn_over_neg": tn_over_neg,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate presence detection and bbox-IoU for Hanging-object task (YOLO GT).")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT .txt files (YOLO format).")
    parser.add_argument("--pred_dir", required=True, help="Directory with predicted masks; names must match GT basenames.")
    parser.add_argument("--pred_ext", default=".png", help="Preferred extension for predicted masks (not required).")
    parser.add_argument("--out_csv", default="/home/marco/Desktop/SAM3-exp/sam3/outputs/SAM3_HangCon_Prompt/hanging_object/score_0_5/hangcon_eval_exp3.csv", help="CSV output path for metrics and per-image IoUs.")
    parser.add_argument("--verbose", action="store_true", help="Print per-image mismatches.")
    parser.add_argument("--only", nargs="+", help="Optional list of basenames to evaluate (space separated).")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    if not gt_dir.is_dir():
        print("Error: gt-dir is not a directory.")
        return
    if not pred_dir.is_dir():
        print("Error: pred-dir is not a directory.")
        return

    evaluate(gt_dir, pred_dir, pred_ext=args.pred_ext if args.pred_ext.startswith(".") else "." + args.pred_ext,
             out_csv=Path(args.out_csv), verbose=args.verbose, only_list=args.only)

if __name__ == "__main__":
    main()
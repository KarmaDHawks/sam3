import argparse
import json
import os
import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


device = "cuda" if torch.cuda.is_available() else "cpu"

# build model (will download checkpoint if needed)
model = build_sam3_image_model(device=device, eval_mode=True)
processor = Sam3Processor(model, device=device)


def iou_xyxy(a, b):
    # a, b are tensors or lists in xyxy normalized (x0,y0,x1,y1)
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return [x0, y0, x1, y1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate candidate noun phrases for regions in a dataset or single image")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image to process")
    parser.add_argument("--bbox", type=str, default=None, help="BBox for single image as x0,y0,x1,y1 (pixel coords)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory to process (flat or structured)")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT YOLO txt folder (if structured, mirrors dataset)")
    parser.add_argument("--use_gt", action="store_true", help="Use GT YOLO boxes to select regions")
    parser.add_argument("--output_json_dir", type=str, default="./prompt_eval_results", help="Where to save per-image JSON results")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--gt_class", type=int, default=1)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB") if args.image is not None else None
    box_xyxy = None
    if args.bbox is not None:
        parts = [int(x) for x in args.bbox.split(",")]
        if len(parts) == 4:
            box_xyxy = parts

    def find_image_files(directory):
        exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        files = [f for f in os.listdir(directory) if os.path.splitext(f)[1] in exts]
        return sorted(files)

    def get_bbox_from_mask(mask_np):
        y_indices, x_indices = np.where(mask_np > 0)
        if len(y_indices) == 0:
            return None
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        return [x_min, y_min, x_max, y_max]

    def process_single(image, box_xyxy, out_json_path=None, gt_source=None):
        state = processor.set_image(image)
        backbone_out = state["backbone_out"]

        # convert input box to normalized cxcywh
        orig_h = state["original_height"]
        orig_w = state["original_width"]
        x0, y0, x1, y1 = box_xyxy
        cx = (x0 + x1) / 2.0 / orig_w
        cy = (y0 + y1) / 2.0 / orig_h
        w = (x1 - x0) / orig_w
        h = (y1 - y0) / orig_h
        box_norm = torch.tensor([cx, cy, w, h], device=device, dtype=torch.float32)

        # prepare geometric prompt
        from sam3.model.geometry_encoders import Prompt

        boxes = box_norm.view(1, 1, 4)
        labels = torch.ones(1, 1, dtype=torch.long, device=device)
        geometric_prompt = Prompt(box_embeddings=boxes, box_labels=labels, box_mask=torch.zeros(1, 1, dtype=torch.bool, device=device))

        # encode prompt (no text) -> run encoder and decoder to get decoder features
        find_input = processor.find_stage
        # Ensure backbone_out has language features (some flows expect them)
        if "language_features" not in backbone_out:
            text_outputs = model.backbone.forward_text(["visual"], device=device)
            backbone_out.update(text_outputs)
        prompt, prompt_mask, backbone_out = model._encode_prompt(
            backbone_out, find_input, geometric_prompt, encode_text=False
        )
        backbone_out, encoder_out, _ = model._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )
        out = {}
        out, hs = model._run_decoder(
            pos_embed=encoder_out["pos_embed"],
            memory=encoder_out["encoder_hidden_states"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        decoder_feats = hs[-1][0]
        pred_boxes = out["pred_boxes"][0]

        # compute IoU between each predicted box and input box
        ious = []
        box_xyxy_norm = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
        for pb in pred_boxes:
            pb_xy = cxcywh_to_xyxy(pb.tolist())
            ious.append(iou_xyxy(pb_xy, box_xyxy_norm))

        best_idx = int(torch.tensor(ious).argmax())
        region_feat = decoder_feats[best_idx]

        # candidate noun phrases
        candidates = [
            "hanging object with rope",
            "suspended load",
            "steel cable with hook",
            "chain sling",
            "bucket on a rope",
            "beam hanging from crane",
            "pipe suspended by strap",
            "netting bag hanging",
        ]

        # get text features from backbone
        text_outputs = model.backbone.forward_text(candidates, device=device)
        text_feats = text_outputs["language_features"]  # seq, batch, d
        text_mask = text_outputs["language_mask"]  # batch, seq (bool: True for padding)

        # apply prompt_mlp if present
        dps = model.dot_prod_scoring
        if dps.prompt_mlp is not None:
            text_feats = dps.prompt_mlp(text_feats)

        # mean-pool text
        pooled = dps.mean_pool_text(text_feats, text_mask)  # (batch, d_model)

        # project and score with region_feat
        proj_pooled = dps.prompt_proj(pooled)  # (N, d_proj)
        region_proj = dps.hs_proj(region_feat)  # (d_proj)
        scores = (proj_pooled @ region_proj.unsqueeze(-1)).squeeze(-1) * dps.scale

        scores_list = [float(s) for s in scores]
        ranking = torch.argsort(scores, descending=True)
        ranked = [candidates[int(i)] for i in ranking]

        result = {
            "bbox": box_xyxy,
            "candidates": candidates,
            "scores": scores_list,
            "ranking": ranked,
            "best": ranked[0] if len(ranked) > 0 else None,
            "gt_source": gt_source,
        }

        if out_json_path is not None:
            with open(out_json_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    # If dataset mode requested, iterate
    if args.dataset_dir is not None:
        os.makedirs(args.output_json_dir, exist_ok=True)
        # detect flat vs structured
        items = os.listdir(args.dataset_dir)
        subdirs = [d for d in items if os.path.isdir(os.path.join(args.dataset_dir, d))]
        image_files_root = find_image_files(args.dataset_dir)
        is_flat = len(image_files_root) > 0 and len(subdirs) <= 1

        if is_flat:
            files = find_image_files(args.dataset_dir)
            for fname in files:
                img_path = os.path.join(args.dataset_dir, fname)
                image = Image.open(img_path).convert("RGB")
                bbox = None
                gt_src = None
                if args.use_gt and args.gt_dir is not None:
                    gt_file = os.path.join(args.gt_dir, os.path.splitext(fname)[0] + ".txt")
                    if os.path.exists(gt_file):
                        with open(gt_file, "r") as fg:
                            for line in fg:
                                parts = line.strip().split()
                                if len(parts) < 5:
                                    continue
                                cls_id = int(parts[0])
                                if cls_id == int(args.gt_class):
                                    cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                                    # YOLO normalized -> convert to pixel bbox
                                    orig_h = image.height; orig_w = image.width
                                    x0 = int((cx - 0.5 * w) * orig_w)
                                    y0 = int((cy - 0.5 * h) * orig_h)
                                    x1 = int((cx + 0.5 * w) * orig_w)
                                    y1 = int((cy + 0.5 * h) * orig_h)
                                    bbox = [x0, y0, x1, y1]
                                    gt_src = gt_file
                                    break
                if bbox is None:
                    # Try to run text prompt and get mask bbox
                    state = processor.set_image(image)
                    state = processor.set_text_prompt(prompt="hanging object with rope", state=state)
                    masks = state.get("masks", None)
                    if masks is not None and len(masks) > 0:
                        mask = masks[0].cpu().numpy()
                        if mask.ndim == 3:
                            mask = mask[0]
                        bbox = get_bbox_from_mask(mask > args.confidence_threshold)
                out_json = os.path.join(args.output_json_dir, os.path.splitext(fname)[0] + ".json")
                if bbox is not None:
                    process_single(image, bbox, out_json_path=out_json, gt_source=gt_src)
                else:
                    # save empty
                    with open(out_json, "w") as f:
                        json.dump({"error": "no bbox found"}, f)
        else:
            # structured: iterate subdirs
            for subdir in sorted(subdirs):
                subdir_path = os.path.join(args.dataset_dir, subdir)
                files = find_image_files(subdir_path)
                out_subdir = os.path.join(args.output_json_dir, subdir)
                os.makedirs(out_subdir, exist_ok=True)
                for fname in files:
                    img_path = os.path.join(subdir_path, fname)
                    image = Image.open(img_path).convert("RGB")
                    bbox = None
                    gt_src = None
                    if args.use_gt and args.gt_dir is not None:
                        gt_file = os.path.join(args.gt_dir, subdir, os.path.splitext(fname)[0] + ".txt")
                        if os.path.exists(gt_file):
                            with open(gt_file, "r") as fg:
                                for line in fg:
                                    parts = line.strip().split()
                                    if len(parts) < 5:
                                        continue
                                    cls_id = int(parts[0])
                                    if cls_id == int(args.gt_class):
                                        cx = float(parts[1]); cy = float(parts[2]); w = float(parts[3]); h = float(parts[4])
                                        orig_h = image.height; orig_w = image.width
                                        x0 = int((cx - 0.5 * w) * orig_w)
                                        y0 = int((cy - 0.5 * h) * orig_h)
                                        x1 = int((cx + 0.5 * w) * orig_w)
                                        y1 = int((cy + 0.5 * h) * orig_h)
                                        bbox = [x0, y0, x1, y1]
                                        gt_src = gt_file
                                        break
                    if bbox is None:
                        state = processor.set_image(image)
                        state = processor.set_text_prompt(prompt="hanging object with rope", state=state)
                        masks = state.get("masks", None)
                        if masks is not None and len(masks) > 0:
                            mask = masks[0].cpu().numpy()
                            if mask.ndim == 3:
                                mask = mask[0]
                            bbox = get_bbox_from_mask(mask > args.confidence_threshold)
                    out_json = os.path.join(out_subdir, os.path.splitext(fname)[0] + ".json")
                    if bbox is not None:
                        process_single(image, bbox, out_json_path=out_json, gt_source=gt_src)
                    else:
                        with open(out_json, "w") as f:
                            json.dump({"error": "no bbox found"}, f)

        print("Done dataset processing")
        exit(0)

    if box_xyxy is None:
        raise SystemExit("No image/bbox provided. Use --image and --bbox or --dataset_dir with --use_gt")

    # process single image
    res = process_single(image, box_xyxy, out_json_path=None)
    print(res)
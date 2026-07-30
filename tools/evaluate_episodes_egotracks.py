"""
Evaluation of SAM3 on EgoTracks re-detection episodes, using the
Precision / Recall / F-score protocol from:

    Lukezic, Zajc, Vojir, Matas, Kristan. "A Novel Performance Evaluation
    Methodology for Long-Term Visual Object Tracking." IJCV, 2018.
    (the VOT-LT protocol)

The P/R/F formulas below are kept IDENTICAL to the original paper:

    At confidence threshold theta, the tracker "reports presence" at frame t
    iff its prediction is non-empty AND its confidence >= theta.

    Pr(theta) = (1 / N_p^theta) * sum_{t : reported presence} Omega_t
    Re(theta) = (1 / N_g)        * sum_{t : GT present}        Omega_t
    F(theta)  = 2 * Pr(theta) * Re(theta) / (Pr(theta) + Re(theta))

    where Omega_t is the overlap (IoU) between prediction and GT at frame t,
    and Omega_t = 0 whenever presence is reported but GT is absent (or vice
    versa). By convention Pr(theta) = 1 when N_p^theta = 0 (no presence
    reported -> trivially no wrong detections).

The only adaptation is domain-specific: GT is a box (not a mask), and only
available for the visible/init segment and the single reappearance frame of
each episode; predictions are DAVIS-palette masks, converted here to their
bounding box. Frames in the 10+ frame gap have GT = "absent" by construction
(no annotation needed: the episode loader guarantees it).

Usage
-----
    python evaluate_episodes.py \
        --results_dir /path/to/sam3_output_dir \
        --output_dir  /path/to/eval_output \
        [--confidence_key object_score_logits] \
        [--iou_tau 0.5] \
        [--include_init_frames]

`results_dir` is the --output_dir you passed to egotracks_inference.py
(the one containing {video_id}/{object_id}_{object_name}/{episode_name}/).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def xywh_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def box_iou(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def mask_bbox(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


# --------------------------------------------------------------------------
# Episode-level parsing (mirrors egotracks_inference.py / episode_loader.py)
# --------------------------------------------------------------------------

def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_boxes_file(path: Path) -> List[Optional[List[float]]]:
    boxes: List[Optional[List[float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            boxes.append(None)
            continue
        vals = [_to_float(p) for p in parts]
        boxes.append(None if any(np.isnan(v) for v in vals) else vals)
    return boxes


def parse_visibility_file(path: Path) -> List[bool]:
    vis = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vis.append(bool(int(float(line))))
    return vis


def load_score_csv(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    if not path.exists():
        return rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fidx = int(r["frame_idx"])
            rows[fidx] = {
                "object_score_logits": _to_float(r.get("object_score_logits")),
                "iou_pred": _to_float(r.get("iou_pred")),
                "eff_iou_score": _to_float(r.get("eff_iou_score")),
            }
    return rows


def load_episode(episode_dir: Path) -> Optional[Dict[str, Any]]:
    """Build the per-frame record list for one episode, correctly aligned
    with the mask filenames (000000.png, 000001.png, ...) exactly as
    produced by run_inference_single_track (which drops any leading frames
    whose box is None before naming masks 0, 1, 2, ...)."""
    info_path = episode_dir / "episode_info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))

    boxes = parse_boxes_file(episode_dir / "boxes.txt")
    visibility = parse_visibility_file(episode_dir / "visibility.txt")
    if len(boxes) != len(visibility):
        raise ValueError(f"boxes/visibility length mismatch in {episode_dir}")

    first_valid = next((i for i, b in enumerate(boxes) if b is not None), None)
    if first_valid is None:
        return None

    boxes = boxes[first_valid:]

    num_visible = int(info["num_visible_frames"])
    num_missing = int(info["num_missing_frames"])
    # `num_reappearance_frames` may differ from episode to episode (and even
    # be absent in episode_info.json files produced before the reappearance
    # window was made configurable, in which case it defaults to 1 frame,
    # matching the old single-frame behaviour). No fixed window length is
    # assumed anywhere else in the pipeline: everything downstream reads the
    # window length back out of the records themselves.
    num_reappear = int(info.get("num_reappearance_frames", 1))
    roles = (["init"] * num_visible) + (["gap"] * num_missing) + (["reappearance"] * num_reappear)
    roles = roles[first_valid:]

    if len(roles) != len(boxes):
        raise ValueError(
            f"Role/box length mismatch in {episode_dir}: "
            f"{len(roles)} roles vs {len(boxes)} boxes (first_valid_idx={first_valid})"
        )

    scores = load_score_csv(episode_dir / "score_evolution.csv")

    records = []
    for pos, (box, role) in enumerate(zip(boxes, roles)):
        mask_path = episode_dir / f"{pos:06d}.png"
        pred_bbox = mask_bbox(load_mask(mask_path)) if mask_path.exists() else None
        score_row = scores.get(pos, {})
        records.append({
            "pos": pos,
            "role": role,
            "gt_bbox": xywh_to_xyxy(box) if box is not None else None,
            "pred_bbox": pred_bbox,
            "object_score_logits": score_row.get("object_score_logits", float("nan")),
            "iou_pred": score_row.get("iou_pred", float("nan")),
            "eff_iou_score": score_row.get("eff_iou_score", float("nan")),
        })
    return {"info": info, "records": records}


# --------------------------------------------------------------------------
# VOT-LT Precision / Recall / F-score (formulas unchanged from the paper)
# --------------------------------------------------------------------------

def presence_at_threshold(record: Dict[str, Any], confidence_key: str, theta: float) -> bool:
    if record["pred_bbox"] is None:
        return False
    conf = record[confidence_key]
    if np.isnan(conf):
        # No confidence logged for this frame: fall back to raw mask presence.
        return True
    return conf >= theta


def pr_f_at_threshold(records: List[Dict[str, Any]], confidence_key: str, theta: float) -> Dict[str, float]:
    n_pred_present = 0
    n_gt_present = 0
    sum_overlap_pred = 0.0
    sum_overlap_gt = 0.0

    for r in records:
        present = presence_at_threshold(r, confidence_key, theta)
        gt_present = r["gt_bbox"] is not None
        overlap = box_iou(r["pred_bbox"], r["gt_bbox"]) if (present and gt_present) else 0.0

        if present:
            n_pred_present += 1
            sum_overlap_pred += overlap
        if gt_present:
            n_gt_present += 1
            sum_overlap_gt += overlap

    precision = 1.0 if n_pred_present == 0 else sum_overlap_pred / n_pred_present
    recall = 0.0 if n_gt_present == 0 else sum_overlap_gt / n_gt_present
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return {
        "theta": theta,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred_present": n_pred_present,
        "n_gt_present": n_gt_present,
    }


def sweep_thresholds(records: List[Dict[str, Any]], confidence_key: str) -> List[float]:
    values = sorted({
        r[confidence_key] for r in records
        if not np.isnan(r[confidence_key])
    })
    if not values:
        return [0.0]
    thetas = [values[0] - 1.0]
    thetas += [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    thetas.append(values[-1] + 1.0)
    return thetas


def pr_curve(records: List[Dict[str, Any]], confidence_key: str) -> List[Dict[str, float]]:
    return [pr_f_at_threshold(records, confidence_key, t) for t in sweep_thresholds(records, confidence_key)]


def best_operating_point(curve: List[Dict[str, float]]) -> Dict[str, float]:
    return max(curve, key=lambda p: p["f1"])


# --------------------------------------------------------------------------
# Diagnostics (supplementary, do NOT feed into Pr/Re/F): hallucination rate
# during the gap, and the recovery/miss/wrong-object breakdown at the
# reappearance frame.
# --------------------------------------------------------------------------

def diagnostics(records: List[Dict[str, Any]], confidence_key: str, theta: float, iou_tau: float) -> Dict[str, Any]:
    """Supplementary, human-readable breakdown, computed over the WHOLE
    reappearance window (which may be 1, 5, 10, or any other length --
    including a different length per episode within the same run: the
    window length is read per-episode from the records themselves, nothing
    here assumes a fixed value).

    Two INDEPENDENT axes are reported, each aggregated across every frame
    of the reappearance window:

    1. `reappearance_outcome` / `reappearance_iou` (confidence-gated, uses
       `theta`): whether SAM3 both produced a mask AND reported it with
       confidence >= theta, for the BEST frame in the window. This mixes
       "did it detect anything" with "did it trust that detection enough
       to report it", which is what the VOT-LT Pr/Re/F protocol wants, but
       is not the axis to use for failure-mode analysis.

    2. `raw_outcome` / `raw_iou` (confidence-independent): the best IoU
       reached anywhere in the window, regardless of confidence:
         - "no_detection"    : mask is empty in EVERY frame of the window
         - "wrong_object"     : some mask exists, but IoU == 0 everywhere
                                 (spatially disjoint from GT in every frame)
         - "partial_recovery" : best IoU in the window is in (0, iou_tau)
         - "recovery"          : best IoU in the window is >= iou_tau

    On top of that, when `raw_outcome == "recovery"`, two further fields
    characterize HOW the recovery happened across the window:
         - `first_recovery_offset` / `recovery_timing` ("immediate" if the
           very first reappearance frame already has IoU >= iou_tau,
           "delayed" if it takes one or more frames to get there)
         - `recovery_stability` ("stable" if, once IoU >= iou_tau is first
           reached, every subsequent frame in the window stays >= iou_tau;
           "intermittent" if the target is re-lost -- IoU drops back below
           iou_tau -- at some point after that, even though it is still
           physically present)
    `recovery_fraction` (n frames with IoU >= iou_tau / window length) is
    reported unconditionally as a continuous summary, useful when comparing
    episodes across DIFFERENT window lengths (a 2/2 and a 4/10 are both
    "recovery" categorically, but very different in this fraction).

    theta here should be a SINGLE FIXED threshold shared across all
    episodes (e.g. the dataset-level pooled optimum), not each episode's
    own local optimum -- see pass 2 in main().
    """
    gap_records = [r for r in records if r["role"] == "gap"]
    reap_records = [r for r in records if r["role"] == "reappearance"]  # already in temporal order

    n_gap = len(gap_records)
    n_halluc = sum(1 for r in gap_records if presence_at_threshold(r, confidence_key, theta))
    hallucination_rate = (n_halluc / n_gap) if n_gap else float("nan")

    window_len = len(reap_records)

    outcome = None
    reappearance_iou = float("nan")
    raw_outcome = None
    raw_iou = float("nan")
    first_recovery_offset = float("nan")
    recovery_timing = None
    recovery_stability = None
    recovery_fraction = float("nan")

    if reap_records:
        # -- confidence-independent (raw), per frame across the window --
        raw_ious = [
            box_iou(r["pred_bbox"], r["gt_bbox"]) if r["pred_bbox"] is not None else 0.0
            for r in reap_records
        ]
        any_pred = any(r["pred_bbox"] is not None for r in reap_records)
        max_raw_iou = max(raw_ious)

        if not any_pred:
            raw_outcome = "no_detection"
        elif max_raw_iou >= iou_tau:
            raw_outcome = "recovery"
        elif max_raw_iou > 0:
            raw_outcome = "partial_recovery"
        else:
            raw_outcome = "wrong_object"
        raw_iou = max_raw_iou

        recovered_flags = [iou >= iou_tau for iou in raw_ious]
        recovery_fraction = sum(recovered_flags) / window_len

        if raw_outcome == "recovery":
            first_idx = recovered_flags.index(True)
            first_recovery_offset = float(first_idx)
            recovery_timing = "immediate" if first_idx == 0 else "delayed"
            recovery_stability = "stable" if all(recovered_flags[first_idx:]) else "intermittent"

        # -- confidence-gated (VOT-LT-consistent), best frame in the window --
        gated_ious = []
        any_gated_pred = False
        for r in reap_records:
            present = presence_at_threshold(r, confidence_key, theta)
            if present:
                any_gated_pred = True
                gated_ious.append(box_iou(r["pred_bbox"], r["gt_bbox"]))
            else:
                gated_ious.append(0.0)
        max_gated_iou = max(gated_ious)

        if not any_gated_pred:
            outcome = "miss"
        else:
            reappearance_iou = max_gated_iou
            outcome = "recovery" if max_gated_iou >= iou_tau else "wrong_object"

    return {
        "n_gap_frames": n_gap,
        "n_hallucinated_frames": n_halluc,
        "hallucination_rate": hallucination_rate,
        "reappearance_window_len": window_len,
        "reappearance_iou": reappearance_iou,
        "reappearance_outcome": outcome,
        "raw_iou": raw_iou,
        "raw_outcome": raw_outcome,
        "first_recovery_offset": first_recovery_offset,
        "recovery_timing": recovery_timing,
        "recovery_stability": recovery_stability,
        "recovery_fraction": recovery_fraction,
        "global_theta_used": theta,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def collect_episodes(root_dir: Path) -> List[Path]:
    return sorted({p.parent for p in root_dir.rglob("episode_info.json")})


def main():
    parser = argparse.ArgumentParser(description="VOT-LT style evaluation for EgoTracks re-detection episodes")
    parser.add_argument("--results_dir", required=True, help="Output dir passed to egotracks_inference.py")
    parser.add_argument("--output_dir", required=True, help="Where to write evaluation results")
    parser.add_argument(
        "--confidence_key", default="object_score_logits",
        choices=["object_score_logits", "iou_pred", "eff_iou_score"],
        help="Which SAM3 score to use as the VOT-LT confidence for the theta sweep",
    )
    parser.add_argument("--iou_tau", type=float, default=0.5, help="IoU threshold for the recovery/wrong-object diagnostic")
    parser.add_argument("--include_init_frames", action="store_true", help="Include the 6 prompt frames in Pr/Re/F (excluded by default)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = collect_episodes(results_dir)
    print(f"Found {len(episode_dirs)} episodes under {results_dir}")

    # --- Pass 1: load every episode and build the pooled record list ---
    episodes = []
    pooled_records = []

    for ep_dir in episode_dirs:
        try:
            episode = load_episode(ep_dir)
        except Exception as e:
            print(f"[WARN] Skipping {ep_dir}: {e}")
            continue
        if episode is None:
            continue

        records = episode["records"]
        eval_records = records if args.include_init_frames else [r for r in records if r["role"] != "init"]
        if not eval_records:
            continue

        episodes.append((ep_dir, episode["info"], eval_records))
        pooled_records.extend(eval_records)

    if not episodes:
        print("No valid episodes found, nothing to evaluate.")
        return

    # --- dataset-level pooled Pr-Re-F curve (VOT-LT pools frames across all
    #     sequences before computing a single curve, rather than averaging
    #     per-sequence scores). This also gives us ONE global theta that is
    #     then used for every episode's diagnostics below, instead of each
    #     episode being scored at its own post-hoc optimal theta. ---
    pooled_curve = pr_curve(pooled_records, args.confidence_key)
    pooled_best = best_operating_point(pooled_curve)
    global_theta = pooled_best["theta"]

    # --- Pass 2: per-episode metrics. `local_*` is that episode's own
    #     best-F1 operating point (informational only -- it is NOT achievable
    #     with a single real-world threshold and tends to look artificially
    #     good, e.g. precision==recall==f1, whenever an episode has only one
    #     GT-positive frame). `reappearance_iou` / `reappearance_outcome` /
    #     `hallucination_rate` are evaluated at the single shared
    #     `global_theta` and are the numbers to actually trust and compare
    #     across episodes. ---
    per_episode_results = []
    for ep_dir, info, eval_records in episodes:
        local_curve = pr_curve(eval_records, args.confidence_key)
        local_best = best_operating_point(local_curve)
        diag = diagnostics(eval_records, args.confidence_key, global_theta, args.iou_tau)

        per_episode_results.append({
            "video_id": info.get("video_id"),
            "object_id": info.get("object_id"),
            "object_name": info.get("object_name"),
            "episode_name": info.get("episode_name"),
            "episode_dir": str(ep_dir),
            "num_missing_frames": info.get("num_missing_frames"),
            "local_optimal_theta": local_best["theta"],
            "local_precision": local_best["precision"],
            "local_recall": local_best["recall"],
            "local_f1": local_best["f1"],
            **diag,
        })

    # --- per-episode CSV ---
    per_episode_csv = out_dir / "per_episode_metrics.csv"
    with open(per_episode_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_episode_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_episode_results)
    print(f"Per-episode metrics written to {per_episode_csv}")

    curve_csv = out_dir / "pr_curve.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["theta", "precision", "recall", "f1", "n_pred_present", "n_gt_present"])
        writer.writeheader()
        writer.writerows(pooled_curve)
    print(f"Pooled PR curve written to {curve_csv}")

    fig, ax = plt.subplots(figsize=(5, 5))
    recalls = np.array([p["recall"] for p in pooled_curve])
    precisions = np.array([p["precision"] for p in pooled_curve])
    order = np.argsort(recalls)
    ax.plot(recalls[order], precisions[order], marker=".", linewidth=1)
    ax.scatter(
        [pooled_best["recall"]], [pooled_best["precision"]], color="red", zorder=5,
        label=f"best F1={pooled_best['f1']:.3f} (theta={pooled_best['theta']:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Dataset-level Precision-Recall (VOT-LT protocol)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)

    # --- aggregated diagnostics ---
    n_episodes = len(per_episode_results)
    outcomes = [r["reappearance_outcome"] for r in per_episode_results]
    n_recovery = outcomes.count("recovery")
    n_miss = outcomes.count("miss")
    n_wrong = outcomes.count("wrong_object")

    raw_outcomes = [r["raw_outcome"] for r in per_episode_results]
    n_raw_recovery = raw_outcomes.count("recovery")
    n_raw_partial = raw_outcomes.count("partial_recovery")
    n_raw_wrong = raw_outcomes.count("wrong_object")
    n_raw_no_detection = raw_outcomes.count("no_detection")

    mean_hallucination = float(np.nanmean([r["hallucination_rate"] for r in per_episode_results]))
    mean_reappearance_iou = float(np.nanmean([r["reappearance_iou"] for r in per_episode_results]))
    mean_raw_iou = float(np.nanmean([r["raw_iou"] for r in per_episode_results]))
    mean_recovery_fraction = float(np.nanmean([r["recovery_fraction"] for r in per_episode_results]))

    recovered = [r for r in per_episode_results if r["raw_outcome"] == "recovery"]
    n_recovered_episodes = len(recovered)
    timing_counts = {}
    stability_counts = {}
    if n_recovered_episodes:
        timings = [r["recovery_timing"] for r in recovered]
        stabilities = [r["recovery_stability"] for r in recovered]
        timing_counts = {
            "immediate_rate": timings.count("immediate") / n_recovered_episodes,
            "delayed_rate": timings.count("delayed") / n_recovered_episodes,
        }
        stability_counts = {
            "stable_rate": stabilities.count("stable") / n_recovered_episodes,
            "intermittent_rate": stabilities.count("intermittent") / n_recovered_episodes,
        }

    window_lens = [r["reappearance_window_len"] for r in per_episode_results]

    summary = {
        "num_episodes": n_episodes,
        "confidence_key": args.confidence_key,
        "iou_tau": args.iou_tau,
        "global_theta": global_theta,
        "mean_reappearance_iou": mean_reappearance_iou,
        "reappearance_window_len_stats": {
            "min": int(min(window_lens)),
            "max": int(max(window_lens)),
            "mean": float(np.mean(window_lens)),
            "note": "Per-episode window length, read from num_reappearance_frames -- "
                    "may legitimately vary across episodes/videos within the same run.",
        },
        "raw_reappearance_breakdown_confidence_independent": {
            "recovery_rate": n_raw_recovery / n_episodes,
            "partial_recovery_rate": n_raw_partial / n_episodes,
            "wrong_object_rate": n_raw_wrong / n_episodes,
            "no_detection_rate": n_raw_no_detection / n_episodes,
            "n_recovery": n_raw_recovery,
            "n_partial_recovery": n_raw_partial,
            "n_wrong_object": n_raw_wrong,
            "n_no_detection": n_raw_no_detection,
            "mean_iou_when_detected": mean_raw_iou,
            "mean_recovery_fraction": mean_recovery_fraction,
        },
        "recovery_timing_and_stability": {
            "n_recovered_episodes": n_recovered_episodes,
            "note": "Computed only among episodes with raw_outcome == 'recovery'. "
                    "timing: immediate = recovers on the very first reappearance frame, "
                    "delayed = takes 1+ frames within the window. "
                    "stability: stable = once recovered, stays recovered for the rest "
                    "of the window; intermittent = drops back below iou_tau again "
                    "while the target is still present.",
            **timing_counts,
            **stability_counts,
        },
        "dataset_level_vot_lt": {
            "precision": pooled_best["precision"],
            "recall": pooled_best["recall"],
            "f1": pooled_best["f1"],
            "theta": pooled_best["theta"],
        },
        "reappearance_breakdown": {
            "recovery_rate": n_recovery / n_episodes,
            "miss_rate": n_miss / n_episodes,
            "wrong_object_rate": n_wrong / n_episodes,
            "n_recovery": n_recovery,
            "n_miss": n_miss,
            "n_wrong_object": n_wrong,
        },
        "mean_gap_hallucination_rate": mean_hallucination,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
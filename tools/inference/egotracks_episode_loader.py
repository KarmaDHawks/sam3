"""Helpers to split EgoTracks annotations into visibility-based episodes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _parse_visibility_value(value: str) -> bool:
    """Parse a visibility token into a boolean value."""
    if value is None:
        return False
    raw = str(value).strip().lower()
    if raw in {"", "0", "false", "f", "no", "n", "none", "null"}:
        return False
    if raw in {"1", "true", "t", "yes", "y"}:
        return True
    try:
        return bool(float(raw))
    except ValueError:
        return bool(raw)


def _parse_box_line(line: str):
    """Parse a box line from boxes.txt into a list or None."""
    if not line or not line.strip():
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
    except ValueError:
        return None
    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        return None
    return [x, y, w, h]


def _resolve_visibility_file(anno_dir: Path) -> Path:
    """Resolve the visibility annotation file, accepting both singular and plural names."""
    for candidate in (anno_dir / "visibility.txt", anno_dir / "visibilities.txt"):
        if candidate.exists():
            return candidate
    return anno_dir / "visibility.txt"


def build_episode_annotations(
    anno_dir,
    output_dir,
    visible_frames: int = 6,
    missing_frames: int = 10,
    reappearance_frames: int = 1,
    episode_prefix: str = "episode",
) -> List[Dict[str, Any]]:
    """
    Split an EgoTracks annotation directory into episodes.

    Each episode is written as a directory containing:
        - frames.txt          (original frame indices used, in order)
        - boxes.txt           (matching boxes, "nan,nan,nan,nan" where absent)
        - visibility.txt      (matching 1/0 visibility flags)
        - episode_info.json   (mapping back to the source sequence: which
                                original lines/frames were used for each of
                                the three segments below)

    A valid episode is composed of three consecutive segments taken from the
    source sequence:
        - the LAST `visible_frames` frames of a run of consecutive visible
          frames (if the run is longer than `visible_frames`, only the tail
          closest to the disappearance is kept)
        - ALL of the consecutive invisible frames that follow (the gap),
          provided the gap is AT LEAST `missing_frames` long -- the full gap
          is kept, `missing_frames` is only a minimum-length threshold, not a
          fixed window size
        - a reappearance window: the first `reappearance_frames` visible
          frames after the gap

    Chaining: the reappearance frame of one episode may also be the first
    frame of the visible run used to build the NEXT episode (e.g. a 20-frame
    visible run, a 30-frame gap, another 20-frame visible run yields an
    episode ending on the first frame of the second run, and a following
    episode can still use frames from that same second run).
    """
    anno_dir = Path(anno_dir)
    output_dir = Path(output_dir)

    frames_file = anno_dir / "frames.txt"
    boxes_file = anno_dir / "boxes.txt"
    visibility_file = _resolve_visibility_file(anno_dir)

    if not frames_file.exists() or not boxes_file.exists() or not visibility_file.exists():
        raise FileNotFoundError(
            f"Expected frames.txt, boxes.txt and visibility.txt in {anno_dir}"
        )

    frame_lines = [line.strip() for line in frames_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    box_lines = [line.strip() for line in boxes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    visibility_lines = [line.strip() for line in visibility_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not frame_lines or not box_lines or not visibility_lines:
        raise ValueError(f"No data found in {anno_dir}")

    if len(frame_lines) != len(box_lines) or len(frame_lines) != len(visibility_lines):
        raise ValueError(
            f"Mismatch between frame, box and visibility lengths in {anno_dir}: "
            f"{len(frame_lines)} vs {len(box_lines)} vs {len(visibility_lines)}"
        )

    frame_indices = []
    boxes = []
    visibility = []

    for line in frame_lines:
        try:
            frame_indices.append(int(float(line)))
        except ValueError:
            frame_indices.append(-1)

    for line in box_lines:
        boxes.append(_parse_box_line(line))

    for line in visibility_lines:
        visibility.append(_parse_visibility_value(line))

    if len(frame_indices) != len(boxes) or len(frame_indices) != len(visibility):
        raise ValueError(
            f"Final frame/box/visibility lengths mismatch in {anno_dir}: "
            f"{len(frame_indices)} vs {len(boxes)} vs {len(visibility)}"
        )

    if visible_frames <= 0:
        raise ValueError("visible_frames must be > 0")
    if missing_frames <= 0:
        raise ValueError("missing_frames must be > 0")

    sequence_name = anno_dir.name
    name_parts = sequence_name.split("*")
    video_id = name_parts[0]
    object_id = int(name_parts[1]) if len(name_parts) > 1 else 0
    object_name = "*".join(name_parts[2:]) if len(name_parts) > 2 else sequence_name

    episode_infos: List[Dict[str, Any]] = []
    episode_idx = 0
    cursor = 0
    n = len(visibility)

    while cursor < n:
        if not visibility[cursor]:
            cursor += 1
            continue

        # Extend the run of consecutive visible frames starting at cursor.
        run_start = cursor
        run_end = run_start
        while run_end + 1 < n and visibility[run_end + 1]:
            run_end += 1
        run_len = run_end - run_start + 1

        if run_len < visible_frames:
            # Not enough consecutive visible frames in this run to seed an
            # episode. Skip past the whole run instead of re-scanning it
            # frame by frame.
            cursor = run_end + 1
            continue

        # Keep only the LAST `visible_frames` frames of the run.
        visible_start = run_end - visible_frames + 1
        gap_start = run_end + 1

        if gap_start >= n:
            # Nothing after this visible run: no gap, no episode possible.
            break

        # Extend the run of consecutive invisible frames starting at gap_start.
        gap_end = gap_start
        while gap_end + 1 < n and not visibility[gap_end + 1]:
            gap_end += 1
        gap_len = gap_end - gap_start + 1
        reappear_start = gap_end + 1
        reappear_end = reappear_start + reappearance_frames - 1

        if (
            gap_len < missing_frames
            or reappear_start >= n
            or reappear_end >= n
            or not all(visibility[idx] for idx in range(reappear_start, reappear_end + 1))
        ):
            # Gap shorter than the minimum required, or there aren't enough
            # consecutive visible frames immediately after the gap to form a
            # reappearance window -> no episode anchored on this run.
            cursor = run_end + 1
            continue

        episode_idx += 1
        episode_name = f"{episode_prefix}_{episode_idx:03d}"
        episode_dir = output_dir / sequence_name / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)

        episode_frame_indices = (
            frame_indices[visible_start : run_end + 1]
            + frame_indices[gap_start : gap_end + 1]
            + frame_indices[reappear_start : reappear_end + 1]
        )
        episode_boxes = (
            boxes[visible_start : run_end + 1]
            + boxes[gap_start : gap_end + 1]
            + boxes[reappear_start : reappear_end + 1]
        )
        episode_visibility = [True] * visible_frames + [False] * gap_len + [True] * reappearance_frames
        episode_visibility_output = [1 if value else 0 for value in episode_visibility]

        (episode_dir / "frames.txt").write_text(
            "\n".join(str(idx) for idx in episode_frame_indices) + "\n",
            encoding="utf-8",
        )
        (episode_dir / "boxes.txt").write_text(
            "\n".join(
                ",".join(str(v) for v in box) if box is not None else "nan,nan,nan,nan"
                for box in episode_boxes
            )
            + "\n",
            encoding="utf-8",
        )
        (episode_dir / "visibility.txt").write_text(
            "\n".join(str(int(v)) for v in episode_visibility_output) + "\n",
            encoding="utf-8",
        )

        # Traceability file: exactly which lines/frames of the ORIGINAL
        # sequence went into this episode, split by segment.
        episode_info = {
            "sequence_name": sequence_name,
            "video_id": video_id,
            "object_id": object_id,
            "object_name": object_name,
            "episode_name": episode_name,
            "source_anno_dir": str(anno_dir),
            "num_visible_frames": visible_frames,
            "num_missing_frames": gap_len,
            "num_reappearance_frames": reappearance_frames,
            "min_missing_frames_threshold": missing_frames,
            "segments": {
                "visible": {
                    "source_line_range_0based": [visible_start, run_end],
                    "source_frame_indices": frame_indices[visible_start : run_end + 1],
                },
                "missing": {
                    "source_line_range_0based": [gap_start, gap_end],
                    "source_frame_indices": frame_indices[gap_start : gap_end + 1],
                },
                "reappearance": {
                    "source_line_range_0based": [reappear_start, reappear_end],
                    "source_frame_indices": frame_indices[reappear_start : reappear_end + 1],
                },
            },
        }
        (episode_dir / "episode_info.json").write_text(
            json.dumps(episode_info, indent=2), encoding="utf-8"
        )

        episode_infos.append(
            {
                "episode_id": episode_idx,
                "episode_dir": str(episode_dir),
                "episode_name": episode_name,
                "video_id": video_id,
                "object_id": object_id,
                "object_name": object_name,
                "frame_indices": episode_frame_indices,
                "boxes": episode_boxes,
                "visibility": episode_visibility,
                "episode_info": episode_info,
            }
        )

        # Chaining: resume scanning FROM the first frame of the reappearance
        # window (not past it), since that frame is allowed to double as the
        # first frame of the next episode's visible run.
        cursor = reappear_start

    return episode_infos
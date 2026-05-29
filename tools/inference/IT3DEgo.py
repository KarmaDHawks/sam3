# -*- coding: utf-8 -*-
"""
IT3DEgo dataset loader

Expected layout (example):

/media/TBDataNAS/Egocentric Vision/IT3DEgo/
    annotations/
        video_1_scene_1/
            2d_bbox_annot/
                entity_01.txt
                entity_02.txt
    raw_videos/
        video_1_scene_1/
            pv/
                609784323354.jpg
                609814310907.jpg

Each entity .txt contains lines like:
609784323354 1211 213 68 378
609814310907 203 592 192 127

This loader exposes a simple index over (sequence, entity) entries
and returns (image_paths, boxes_array, meta) for each entity.
"""
import os
import glob
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class IT3DEgo:
    """Loader for the IT3DEgo dataset.

    Parameters
    - frames_root: path to raw_videos (e.g. "/media/TBDataNAS/.../raw_videos")
    - ann_root: path to annotations (e.g. "/media/TBDataNAS/.../annotations")
    - seq_names: optional list of sequence names to limit (folder names under annotations)
    - frame_subdir: typically 'pv' (subfolder under a sequence where images live)
    - img_exts: list of image extensions to try
    - skip_missing_frames: if True, missing images are skipped silently (default True)

    Usage:
        loader = IT3DEgo(frames_root, ann_root)
        imgs, boxes, meta = loader[0]  # first entity
    """

    def __init__(
        self,
        frames_root: str,
        ann_root: str,
        seq_names: Optional[List[str]] = None,
        frame_subdir: str = "pv",
        img_exts: Optional[List[str]] = None,
        skip_missing_frames: bool = True,
    ):
        self.frames_root = frames_root
        self.ann_root = ann_root
        self.frame_subdir = frame_subdir
        self.skip_missing_frames = skip_missing_frames
        self.img_exts = img_exts or [".jpg", ".jpeg", ".png", ".bmp"]

        if not os.path.isdir(self.ann_root):
            raise ValueError(f"Annotations root not found: {self.ann_root}")

        # discover sequences (folders) under annotations
        seqs = [d for d in os.listdir(self.ann_root) if os.path.isdir(os.path.join(self.ann_root, d))]
        seqs = sorted(seqs)
        if seq_names:
            if isinstance(seq_names, str):
                seq_names = [seq_names]
            seqs = [s for s in seqs if s in seq_names]

        self.sequences = seqs

        # collect all entity annotation files
        self.entries: List[Dict] = []
        for seq in self.sequences:
            anno_dir = os.path.join(self.ann_root, seq, "2d_bbox_annot")
            if not os.path.isdir(anno_dir):
                logger.warning(f"No 2d_bbox_annot folder for sequence {seq}: {anno_dir}")
                continue
            for fn in sorted(os.listdir(anno_dir)):
                if not fn.lower().endswith(".txt"):
                    continue
                entity_name = os.path.splitext(fn)[0]
                self.entries.append({
                    "sequence": seq,
                    "entity": entity_name,
                    "annot_path": os.path.join(anno_dir, fn),
                })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index):
        """Return (image_paths: List[str], boxes: np.ndarray (N,4), meta: dict)

        index may be an integer or a (sequence, entity) tuple.
        Boxes are returned as floats in the same format as the txt (x y w h).
        Only frames for which an image file is found are returned (unless
        skip_missing_frames=False, in which case a FileNotFoundError is raised).
        """
        if isinstance(index, int):
            entry = self.entries[index]
        elif isinstance(index, tuple) and len(index) == 2:
            seq, ent = index
            matches = [e for e in self.entries if e["sequence"] == seq and e["entity"] == ent]
            if not matches:
                raise IndexError(f"No entry for sequence={seq} entity={ent}")
            entry = matches[0]
        else:
            raise IndexError("Index must be int or (sequence, entity) tuple")

        seq_name = entry["sequence"]
        annot_path = entry["annot_path"]

        frame_names: List[str] = []
        boxes: List[List[float]] = []

        with open(annot_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Skipping malformed line in {annot_path}: {line}")
                    continue
                frame_name = parts[0]
                try:
                    coords = [float(x) for x in parts[1:5]]
                except ValueError:
                    logger.warning(f"Skipping non-numeric coords in {annot_path}: {line}")
                    continue
                frame_names.append(frame_name)
                boxes.append(coords)

        image_paths: List[str] = []
        boxes_filtered: List[List[float]] = []

        for fn, box in zip(frame_names, boxes):
            p = self._find_image(seq_name, fn)
            if p is None:
                msg = f"Image not found for frame {fn} in sequence {seq_name} (anno: {annot_path})"
                if self.skip_missing_frames:
                    logger.debug(msg + ", skipping")
                    continue
                else:
                    raise FileNotFoundError(msg)
            image_paths.append(p)
            boxes_filtered.append(box)

        if len(boxes_filtered) > 0:
            boxes_arr = np.asarray(boxes_filtered, dtype=float)
        else:
            boxes_arr = np.zeros((0, 4), dtype=float)

        meta = {
            "sequence": seq_name,
            "entity": entry["entity"],
            "annot_path": annot_path,
        }
        return image_paths, boxes_arr, meta

    def _find_image(self, sequence: str, frame_name: str) -> Optional[str]:
        """Try to locate an image file for the given sequence and frame_name.

        Looks in:
            frames_root/sequence/frame_subdir/
            frames_root/sequence/
        Tries exact extensions first, then glob for any extension, then fallback
        to files that start with frame_name.
        """
        candidates = []
        dir1 = os.path.join(self.frames_root, sequence, self.frame_subdir)
        dir2 = os.path.join(self.frames_root, sequence)
        if os.path.isdir(dir1):
            candidates.append(dir1)
        if os.path.isdir(dir2) and dir2 != dir1:
            candidates.append(dir2)

        for d in candidates:
            base = os.path.join(d, frame_name)
            for ext in self.img_exts:
                p = base + ext
                if os.path.exists(p):
                    return p
            # any extension
            gm = glob.glob(os.path.join(d, f"{frame_name}.*"))
            if gm:
                return gm[0]
            try:
                for fname in os.listdir(d):
                    if fname.startswith(frame_name):
                        return os.path.join(d, fname)
            except FileNotFoundError:
                continue

        return None

    def get_sequences(self) -> List[str]:
        return list(self.sequences)

    def get_entities(self, sequence: str) -> List[str]:
        return [e["entity"] for e in self.entries if e["sequence"] == sequence]

    def find_entry(self, sequence: str, entity: str) -> Optional[Dict]:
        matches = [e for e in self.entries if e["sequence"] == sequence and e["entity"] == entity]
        return matches[0] if matches else None

    def __repr__(self) -> str:
        return f"IT3DEgo(frames_root={self.frames_root!r}, ann_root={self.ann_root!r}, sequences={len(self.sequences)}, entries={len(self.entries)})"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick check for IT3DEgo loader")
    parser.add_argument("--frames", required=True, help="Path to raw_videos root")
    parser.add_argument("--ann", required=True, help="Path to annotations root")
    parser.add_argument("--list", action="store_true", help="List sequences and first few entities")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    loader = IT3DEgo(args.frames, args.ann)
    print(loader)
    if args.list:
        print("Sequences:")
        for s in loader.get_sequences():
            ents = loader.get_entities(s)
            print(f" - {s}: {len(ents)} entities (first 5): {ents[:5]}")
    else:
        if len(loader) > 0:
            imgs, boxes, meta = loader[0]
            print("First entry meta:", meta)
            print("Found images:", len(imgs))
            print("Boxes shape:", boxes.shape)
        else:
            print("No entries found. Check paths.")

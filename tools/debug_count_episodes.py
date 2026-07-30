"""Count how many episode annotations would be generated for a dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from sam3.tools.egotracks_episode_loader import build_episode_annotations
except ImportError:  # pragma: no cover - fallback for direct script execution
    from egotracks_episode_loader import build_episode_annotations


def _iter_annotation_dirs(anno_root: Path) -> List[Path]:
    """Return annotation directories containing the expected annotation files."""
    annotation_dirs: List[Path] = []
    for path in sorted(anno_root.rglob("*")):
        if not path.is_dir():
            continue
        if path == anno_root / "_debug_episodes":
            continue
        if path.is_relative_to(anno_root / "_debug_episodes"):
            continue
        if (path / "frames.txt").exists() and (path / "boxes.txt").exists():
            visibility_file = path / "visibility.txt"
            visibilities_file = path / "visibilities.txt"
            if visibility_file.exists() or visibilities_file.exists():
                annotation_dirs.append(path)
    return annotation_dirs


def count_episode_annotations(
    anno_dir: str | os.PathLike[str],
    visible_frames: int = 6,
    missing_frames: int = 10,
    reappearance_frames: int = 1,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Count episodes that would be generated for all annotation directories under anno_dir."""
    anno_root = Path(anno_dir)
    if not anno_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {anno_root}")

    sequence_dirs = _iter_annotation_dirs(anno_root)

    sequence_results: List[Dict[str, Any]] = []
    total_episodes = 0

    for sequence_dir in sequence_dirs:
        debug_output_dir = anno_root / "_debug_episodes"
        episode_infos = build_episode_annotations(
            anno_dir=sequence_dir,
            output_dir=debug_output_dir,
            visible_frames=visible_frames,
            missing_frames=missing_frames,
            reappearance_frames=reappearance_frames,
        )
        episode_count = len(episode_infos)
        total_episodes += episode_count
        sequence_results.append(
            {
                "sequence_name": sequence_dir.name,
                "sequence_dir": str(sequence_dir),
                "episode_count": episode_count,
            }
        )

    return total_episodes, sequence_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Count EgoTracks episodes without running inference")
    parser.add_argument("anno_dir", help="Directory containing annotation sequence folders")
    parser.add_argument("--visible_frames", type=int, default=6)
    parser.add_argument("--missing_frames", type=int, default=10)
    parser.add_argument("--reappearance_frames", type=int, default=1)
    args = parser.parse_args()

    total, sequence_results = count_episode_annotations(
        anno_dir=args.anno_dir,
        visible_frames=args.visible_frames,
        missing_frames=args.missing_frames,
        reappearance_frames=args.reappearance_frames,
    )

    
    for result in sequence_results:
        print(
            f"{result['sequence_name']}: {result['episode_count']}"
        )
    print(f"Total episodes: {total}")


if __name__ == "__main__":
    main()

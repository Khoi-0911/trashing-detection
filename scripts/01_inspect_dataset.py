"""Stage 0: inspect the dataset and report annotation problems.

Run this before anything else. It verifies the schema actually on disk, prints
per-split statistics, and lists the records that fail validation so that the
later stages do not silently train on them.

Example:
    python scripts/01_inspect_dataset.py --dataset-root DATASET
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iwdd.annotations import AnnotationStore  # noqa: E402
from iwdd.config import LabelingConfig, SamplingConfig  # noqa: E402
from iwdd.labeling import FrameLabeler  # noqa: E402
from iwdd.sampling import sampled_timestamps  # noqa: E402
from iwdd.utils import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inspect the MIVIA-IWDD dataset")
    parser.add_argument("--dataset-root", default="DATASET", help="Directory holding the splits")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["TRAINING_SET", "VALIDATION_SET"],
        help="Split directories to inspect",
    )
    parser.add_argument("--target-fps", type=float, default=5.0, help="Sampling rate to simulate")
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=2.0,
        help="Positive window length after the onset, in seconds",
    )
    parser.add_argument("--out", default="", help="Optional path for a JSON report")
    return parser.parse_args()


def label_statistics(
    store: AnnotationStore,
    target_fps: float,
    labeler: FrameLabeler,
) -> dict[str, float]:
    """Simulate frame labelling to predict the class balance.

    This is what determines whether the paper constant of ``w+ = 8`` is the
    right weight for the chosen window, without decoding a single video.

    Args:
        store: Annotations to simulate over.
        target_fps: Sampling rate in frames per second.
        labeler: Labeler carrying the window settings.

    Returns:
        Frame counts and the implied positive class weight.
    """
    total_frames = 0
    positive_frames = 0
    clamped = 0

    for annotation in store:
        timestamps = sampled_timestamps(annotation, target_fps)
        labels = labeler.label(annotation, timestamps)
        total_frames += labels.values.size
        positive_frames += labels.num_positive
        clamped += int(labels.clamped)

    negative_frames = total_frames - positive_frames
    return {
        "total_frames": total_frames,
        "positive_frames": positive_frames,
        "positive_ratio": round(positive_frames / total_frames, 4) if total_frames else 0.0,
        "implied_positive_weight": (
            round(negative_frames / positive_frames, 2) if positive_frames else 0.0
        ),
        "clamped_windows": clamped,
    }


def main() -> int:
    """Run the inspection and print a report.

    Returns:
        Process exit code: 0 on success.
    """
    args = parse_args()
    setup_logging("INFO")

    labeler = FrameLabeler(LabelingConfig(post_seconds=args.post_seconds))
    sampling = SamplingConfig(target_fps=args.target_fps)
    report: dict[str, object] = {"sampling": sampling.__dict__}

    for split in args.splits:
        split_dir = Path(args.dataset_root).expanduser().resolve() / split
        store = AnnotationStore.from_split(split_dir)

        print(f"\n{'=' * 70}\n{split}\n{'=' * 70}")
        description = store.describe()
        for key, value in description.items():
            print(f"  {key:26s} {value}")

        print("\n  -- annotation issues --")
        issues: dict[str, list[str]] = {}
        for annotation in store:
            found = annotation.validate(min_tail_seconds=1.0)
            if found:
                issues[annotation.video_id] = found
                print(f"  {annotation.video_id}: {'; '.join(found)}")
        if not issues:
            print("  none")

        stats = label_statistics(store, args.target_fps, labeler)
        print(f"\n  -- simulated frame labels (post_seconds={args.post_seconds}) --")
        for key, value in stats.items():
            print(f"  {key:26s} {value}")

        report[split] = {"description": description, "issues": issues, "labels": stats}

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

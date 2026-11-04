"""Stage 6: sweep every architecture against one cached feature set.

Reproduces one row of paper Table 1. Because the VLM is frozen and its output
is cached, the whole six-way comparison costs minutes rather than another pass
over the corpus.

Example:
    python scripts/04_benchmark.py --signature smolvlm-SmolVLM-500M-Instruct-multi-full-fps5-concat_text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iwdd.config import ExperimentConfig  # noqa: E402
from iwdd.models import PAPER_ARCHITECTURES, available_architectures  # noqa: E402
from iwdd.training import BenchmarkRunner  # noqa: E402
from iwdd.utils import set_seed, setup_logging  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_train_module = import_module("03_train")
load_sequence_dataset = _train_module.load_sequence_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark every temporal architecture")
    parser.add_argument("--dataset-root", default="DATASET")
    parser.add_argument("--train-split", default="TRAINING_SET")
    parser.add_argument("--cache-dir", default="artifacts/features")
    parser.add_argument("--signature", required=True, help="Feature cache subdirectory")
    parser.add_argument("--output-dir", default="artifacts/runs")
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(PAPER_ARCHITECTURES),
        choices=available_architectures(),
        help="Architectures to compare; defaults to the six from the paper",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """Run the architecture sweep and write the comparison table.

    Returns:
        Process exit code: 0 on success.
    """
    args = parse_args()
    setup_logging("INFO", log_file="artifacts/logs/benchmark.log")
    set_seed(args.seed)

    config = ExperimentConfig()
    config.paths.dataset_root = args.dataset_root
    config.paths.train_split = args.train_split
    config.paths.cache_dir = args.cache_dir
    config.paths.output_dir = args.output_dir
    config.train.num_folds = args.folds
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.positive_weight = args.positive_weight
    config.train.seed = args.seed
    config.train.subset_size = args.subset
    config.labeling.post_seconds = args.post_seconds
    config.run_name = args.run_name

    dataset = load_sequence_dataset(
        config, args.train_split, args.signature, args.subset
    )
    print(f"Benchmarking on {len(dataset)} videos, feature_dim={dataset.feature_dim}")

    table = BenchmarkRunner(config, dataset).run(args.architectures)

    print("\n{:<20s} {:>9s} {:>9s} {:>9s} {:>9s}".format("arch", "P", "R", "F1", "F1 std"))
    for name, metrics in sorted(table.items(), key=lambda kv: -kv[1].get("f1_mean", 0.0)):
        print(
            "{:<20s} {:9.4f} {:9.4f} {:9.4f} {:9.4f}".format(
                name,
                metrics.get("precision_mean", 0.0),
                metrics.get("recall_mean", 0.0),
                metrics.get("f1_mean", 0.0),
                metrics.get("f1_std", 0.0),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

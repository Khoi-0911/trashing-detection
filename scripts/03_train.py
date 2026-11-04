"""Stage 4-5: train a temporal classifier and score it with the contest metric.

Supports both input families. Sequence architectures read the cached
embeddings written by stage 2; clip architectures (conv3d, r3d18) read raw
video and ignore the cache entirely.

Examples:
    python scripts/03_train.py --arch cnn_gru --signature dummy-dummy-multi-full-fps5-concat_text
    python scripts/03_train.py --arch cnn_gru --cv --folds 5 --subset 50
    python scripts/03_train.py --arch conv3d --clip-len 16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iwdd.annotations import AnnotationStore  # noqa: E402
from iwdd.clip_training import run_clip_experiment  # noqa: E402
from iwdd.config import ExperimentConfig  # noqa: E402
from iwdd.datasets import SequenceDataset, build_sequence_loader  # noqa: E402
from iwdd.features import FeatureCache  # noqa: E402
from iwdd.models import available_architectures, build_model, input_kind_of  # noqa: E402
from iwdd.training import CrossValidator, Trainer  # noqa: E402
from iwdd.utils import set_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train an IWDD temporal classifier")
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset-root", default="DATASET")
    parser.add_argument("--train-split", default="TRAINING_SET")
    parser.add_argument("--test-split", default="VALIDATION_SET")
    parser.add_argument(
        "--arch",
        default="cnn_gru",
        choices=available_architectures(),
        help="Temporal architecture to train",
    )
    parser.add_argument("--cache-dir", default="artifacts/features")
    parser.add_argument(
        "--signature",
        default="",
        help="Feature cache subdirectory; defaults to the one implied by the config",
    )
    parser.add_argument("--output-dir", default="artifacts/runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--cv", action="store_true", help="Run k-fold cross-validation")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--subset",
        type=int,
        default=0,
        help="Use only N training videos; pass 50 to mirror the paper protocol",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--positive-weight", type=float, default=8.0)
    parser.add_argument(
        "--auto-positive-weight",
        action="store_true",
        help=(
            "Derive w+ from the observed frame balance instead of the paper constant. "
            "At post_seconds=2.0 this corpus implies w+ around 21, not 8."
        ),
    )
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Assemble an experiment configuration from the CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The configuration driving this training run.
    """
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    config.paths.dataset_root = args.dataset_root
    config.paths.train_split = args.train_split
    config.paths.test_split = args.test_split
    config.paths.cache_dir = args.cache_dir
    config.paths.output_dir = args.output_dir

    config.model.name = args.arch
    config.model.hidden_dim = args.hidden_dim
    config.model.dropout = args.dropout
    config.model.clip_len = args.clip_len
    config.model.clip_stride = args.clip_stride

    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.positive_weight = args.positive_weight
    config.train.num_folds = args.folds
    config.train.num_workers = args.num_workers
    config.train.seed = args.seed
    config.train.subset_size = args.subset

    config.labeling.post_seconds = args.post_seconds
    config.run_name = args.run_name or f"{args.arch}"
    return config


def load_sequence_dataset(
    config: ExperimentConfig,
    split: str,
    signature: str,
    subset: int = 0,
) -> SequenceDataset:
    """Load cached embeddings for one split.

    Args:
        config: Experiment configuration supplying paths.
        split: Split directory name.
        signature: Feature cache subdirectory.
        subset: Keep only the first N videos; 0 keeps all.

    Returns:
        A dataset over the cached videos.

    Raises:
        FileNotFoundError: If the cache directory does not exist.
    """
    cache_root = Path(config.paths.cache_dir) / signature
    if not cache_root.is_dir():
        raise FileNotFoundError(
            f"No feature cache at {cache_root}. Run scripts/02_extract_features.py first."
        )

    store = AnnotationStore.from_split(config.paths.split_dir(split))
    store = store.validated(
        min_tail_seconds=config.labeling.min_tail_seconds,
        drop=config.labeling.drop_invalid,
    )
    if subset:
        store = store.subset(subset)
    return SequenceDataset.from_cache(FeatureCache(cache_root), store)


def run_sequence(config: ExperimentConfig, args: argparse.Namespace) -> dict:
    """Train and evaluate a sequence architecture.

    Args:
        config: Experiment configuration.
        args: Parsed command-line arguments.

    Returns:
        A dictionary of result metrics.
    """
    signature = args.signature or config.feature_signature()
    train_dataset = load_sequence_dataset(
        config, config.paths.train_split, signature, config.train.subset_size
    )
    print(
        f"Loaded {len(train_dataset)} training videos, feature_dim={train_dataset.feature_dim}"
    )
    implied = train_dataset.positive_frame_weight()
    print(f"Implied positive frame weight: {implied:.2f} (configured: {config.train.positive_weight})")
    if args.auto_positive_weight:
        config.train.positive_weight = implied
        print(f"Using derived positive weight {implied:.2f}")

    if args.cv:
        results, aggregated = CrossValidator(config, train_dataset).run()
        return {"mode": "cross_validation", "aggregated": aggregated,
                "folds": [r.report.summary(config.metric) for r in results]}

    test_dataset = load_sequence_dataset(config, config.paths.test_split, signature)
    print(f"Loaded {len(test_dataset)} held-out videos")

    train_loader = build_sequence_loader(
        train_dataset, config.train.batch_size, shuffle=True, num_workers=config.train.num_workers
    )
    test_loader = build_sequence_loader(
        test_dataset, config.train.batch_size, shuffle=False, num_workers=config.train.num_workers
    )

    model = build_model(config.model, input_dim=train_dataset.feature_dim)
    print(f"Model {config.model.name}: {model.num_parameters:,} parameters")

    trainer = Trainer(config, model)
    history, best_epoch, best_state = trainer.fit(train_loader, test_loader)
    report, _ = trainer.evaluate(test_loader)

    output_dir = Path(config.paths.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(best_state, output_dir / "model.pt")

    return {
        "mode": "holdout",
        "best_epoch": best_epoch,
        "num_parameters": model.num_parameters,
        "metrics": report.summary(config.metric),
        "history": history,
        "per_video": report.per_video,
    }


def run_clip(config: ExperimentConfig) -> dict:
    """Train and evaluate a 3D CNN architecture.

    Args:
        config: Experiment configuration naming a clip architecture.

    Returns:
        A dictionary of result metrics.
    """
    train_store = AnnotationStore.from_split(
        config.paths.split_dir(config.paths.train_split)
    ).validated(drop=config.labeling.drop_invalid)
    test_store = AnnotationStore.from_split(
        config.paths.split_dir(config.paths.test_split)
    ).validated(drop=config.labeling.drop_invalid)

    if config.train.subset_size:
        train_store = train_store.subset(config.train.subset_size)

    report = run_clip_experiment(config, train_store, test_store)
    return {
        "mode": "clip_holdout",
        "metrics": report.summary(config.metric),
        "per_video": report.per_video,
    }


def main() -> int:
    """Dispatch to the sequence or clip training path.

    Returns:
        Process exit code: 0 on success.
    """
    args = parse_args()
    setup_logging("INFO", log_file="artifacts/logs/train.log")
    set_seed(args.seed)

    config = build_config(args)
    output_dir = Path(config.paths.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    if input_kind_of(config.model.name) == "clip":
        results = run_clip(config)
    else:
        results = run_sequence(config, args)

    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print("\n-- results --")
    summary = results.get("metrics") or results.get("aggregated") or {}
    for key, value in summary.items():
        print(f"  {key:22s} {value}")
    print(f"\nWrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

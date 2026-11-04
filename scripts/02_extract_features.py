"""Stage 1-3: sample frames, caption them with a VLM and cache embeddings.

This is the expensive stage and it only has to run once per VLM. At the
throughput the paper reports (0.85 FPS for five prompts on an A100) a full pass
over the 400 training videos is roughly 14 hours; everything downstream then
reads small .npz files and runs in minutes.

Start with --backend dummy to validate the plumbing without a GPU, then switch
to a real backend.

Examples:
    python scripts/02_extract_features.py --backend dummy --limit 4
    python scripts/02_extract_features.py --backend smolvlm --split TRAINING_SET
    python scripts/02_extract_features.py --backend qwen3vl --dtype bfloat16
    python scripts/02_extract_features.py --backend cosmos_nano --load-in-8bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iwdd.annotations import AnnotationStore  # noqa: E402
from iwdd.captioning import available_backends  # noqa: E402
from iwdd.config import ExperimentConfig  # noqa: E402
from iwdd.features import FeatureExtractionPipeline  # noqa: E402
from iwdd.utils import set_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract and cache frame features")
    parser.add_argument("--config", default="", help="Optional YAML config to start from")
    parser.add_argument("--dataset-root", default="DATASET")
    parser.add_argument("--splits", nargs="+", default=["TRAINING_SET", "VALIDATION_SET"])
    parser.add_argument(
        "--backend",
        default="smolvlm",
        choices=available_backends(),
        help="Vision-language captioning backend",
    )
    parser.add_argument("--model-id", default="", help="Override the backend default model id")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8, help="Frames per VLM forward pass")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--single-prompt",
        action="store_true",
        help="Use one generic prompt instead of five (paper Table 6 ablation)",
    )
    parser.add_argument(
        "--encoder-mode",
        default="concat_text",
        choices=["concat_text", "per_aspect", "vision"],
        help="How frames are turned into vectors",
    )
    parser.add_argument("--cache-dir", default="artifacts/features")
    parser.add_argument("--caption-dir", default="artifacts/captions")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N videos")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cached videos")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Assemble an experiment configuration from the CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The configuration driving this extraction run.
    """
    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()

    config.paths.dataset_root = args.dataset_root
    config.paths.cache_dir = args.cache_dir
    config.paths.caption_dir = args.caption_dir

    config.sampling.target_fps = args.target_fps

    config.caption.backend = args.backend
    config.caption.model_id = args.model_id
    config.caption.dtype = args.dtype
    config.caption.load_in_8bit = args.load_in_8bit
    config.caption.load_in_4bit = args.load_in_4bit
    config.caption.batch_size = args.batch_size
    config.caption.max_new_tokens = args.max_new_tokens
    config.caption.single_prompt = args.single_prompt

    config.encoder.mode = args.encoder_mode
    config.train.seed = args.seed
    return config


def main() -> int:
    """Run feature extraction over the requested splits.

    Returns:
        Process exit code: 0 on success.
    """
    args = parse_args()
    setup_logging("INFO", log_file="artifacts/logs/extract.log")
    set_seed(args.seed)

    config = build_config(args)
    pipeline = FeatureExtractionPipeline(config)
    print(f"Feature signature: {config.feature_signature()}")
    print(f"Cache directory:   {pipeline.cache.root}")

    for split in args.splits:
        store = AnnotationStore.from_split(config.paths.split_dir(split))
        store = store.validated(
            min_tail_seconds=config.labeling.min_tail_seconds,
            drop=config.labeling.drop_invalid,
        )
        if args.limit:
            store = store.subset(args.limit)
        pipeline.run(store, overwrite=args.overwrite)

    stats = pipeline.probe.as_dict()
    print("\n-- throughput --")
    for key, value in stats.items():
        print(f"  {key:18s} {value:.4f}" if isinstance(value, float) else f"  {key:18s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

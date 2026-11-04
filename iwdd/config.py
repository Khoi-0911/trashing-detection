"""Typed configuration objects for every stage of the IWDD pipeline.

All defaults reproduce the hyper-parameters reported in the reference paper
(Sections 3.2-3.4 and 4.2) so that a run with the stock config is a faithful
re-implementation; every value can be overridden from YAML or the CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class PathsConfig:
    """Filesystem locations used by the pipeline.

    Attributes:
        dataset_root: Directory containing TRAINING_SET and VALIDATION_SET.
        train_split: Name of the training split directory.
        test_split: Name of the held-out split directory.
        cache_dir: Where cached frame features (.npz) are written.
        caption_dir: Where raw generated captions (.jsonl) are written.
        output_dir: Where checkpoints, metrics and reports are written.
    """

    dataset_root: str = "DATASET"
    train_split: str = "TRAINING_SET"
    test_split: str = "VALIDATION_SET"
    cache_dir: str = "artifacts/features"
    caption_dir: str = "artifacts/captions"
    output_dir: str = "artifacts/runs"

    def split_dir(self, split: str) -> Path:
        """Return the absolute directory of a split.

        Args:
            split: Split directory name.

        Returns:
            Path to dataset_root/split.
        """
        return Path(self.dataset_root).expanduser().resolve() / split


@dataclass
class SamplingConfig:
    """Frame extraction settings (paper Section 3.2.1).

    Attributes:
        target_fps: Sampling rate in frames per second.
        max_frames: Hard cap on frames per video; 0 disables the cap. The corpus
            contains clips up to 109s, which is 548 frames at 5 FPS.
        resize_shorter_side: Shorter-side resize applied before captioning;
            0 keeps the native resolution.
        backend: Video decoding backend, currently only "opencv".
    """

    target_fps: float = 5.0
    max_frames: int = 0
    resize_shorter_side: int = 384
    backend: str = "opencv"


@dataclass
class LabelingConfig:
    """Rules converting a single onset timestamp into per-frame labels.

    The dataset ships one integer onset per positive video and explicitly no
    dense frame-level labels, so the positive window is synthesised here. The
    paper implies a short window: it describes dumping as a local temporal
    event of 5-15 frames at 5 FPS and uses a positive class weight of 8.

    Attributes:
        pre_seconds: Seconds before the onset still labelled positive.
        post_seconds: Seconds after the onset labelled positive.
        clamp_to_duration: Clip the window to the end of the video.
        drop_invalid: Drop videos whose onset falls outside their duration.
        min_tail_seconds: Minimum required seconds between onset and end of
            video; shorter positives are reported as suspicious.
    """

    pre_seconds: float = 0.0
    post_seconds: float = 2.0
    clamp_to_duration: bool = True
    drop_invalid: bool = True
    min_tail_seconds: float = 0.0


@dataclass
class CaptionConfig:
    """Vision-language captioning settings (paper Section 3.2.2).

    Attributes:
        backend: Captioner key in the registry (smolvlm, qwen3vl, cosmos_nano,
            blip, dummy).
        model_id: Hugging Face repository id; empty means the backend default.
        dtype: Torch dtype name, e.g. bfloat16 or float16.
        load_in_8bit: Load the model with bitsandbytes 8-bit quantisation.
        load_in_4bit: Load the model with bitsandbytes 4-bit quantisation.
        device: One of auto, cuda or cpu.
        max_new_tokens: Generation budget per caption.
        batch_size: Frames captioned per forward pass.
        single_prompt: Use one generic prompt instead of the five focused ones,
            reproducing the ablation in paper Table 6.
        save_captions: Persist raw caption text for inspection.
    """

    backend: str = "smolvlm"
    model_id: str = ""
    dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device: str = "auto"
    max_new_tokens: int = 48
    batch_size: int = 8
    single_prompt: bool = False
    save_captions: bool = True


@dataclass
class EncoderConfig:
    """Caption-to-vector encoding settings (paper Equation 6).

    Attributes:
        mode: concat_text joins the five captions and encodes once (the 384-d
            setting used in the paper); per_aspect encodes each aspect
            separately and concatenates the vectors (5x384); vision skips text
            entirely and uses pooled visual features from the VLM.
        model_id: Sentence-Transformers model id.
        device: One of auto, cuda or cpu.
        batch_size: Sentences encoded per forward pass.
        normalize: L2-normalise the embeddings.
    """

    mode: str = "concat_text"
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 128
    normalize: bool = False


@dataclass
class ModelConfig:
    """Temporal classifier settings (paper Sections 3.2.3 and 4.2).

    Attributes:
        name: Architecture key in the model registry.
        hidden_dim: Hidden width of the recurrent or attention core.
        dropout: Dropout probability.
        num_layers: Recurrent or transformer depth.
        bidirectional: Use bidirectional recurrence.
        conv_channels: Output channels of the Conv1D front-end in hybrids.
        conv_kernel_size: Kernel width of the Conv1D front-end.
        nhead: Attention heads for the transformer variant.
        clip_len: Frames per clip for the 3D CNN variant.
        clip_size: Spatial resolution for the 3D CNN variant.
        clip_stride: Stride between consecutive 3D CNN clips, in frames.
    """

    name: str = "cnn_gru"
    hidden_dim: int = 256
    dropout: float = 0.3
    num_layers: int = 2
    bidirectional: bool = True
    conv_channels: int = 256
    conv_kernel_size: int = 3
    nhead: int = 4
    clip_len: int = 16
    clip_size: int = 112
    clip_stride: int = 8


@dataclass
class TrainConfig:
    """Optimisation settings (paper Sections 3.3 and 4.2).

    Attributes:
        epochs: Training epochs.
        batch_size: Videos per batch.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        positive_weight: The w+ term in the weighted binary cross-entropy.
        negative_weight: The w- term in the weighted binary cross-entropy.
        grad_clip: Maximum gradient norm.
        scheduler_factor: ReduceLROnPlateau multiplicative factor.
        scheduler_patience: ReduceLROnPlateau patience in epochs.
        num_folds: Folds for cross-validation.
        seed: Global random seed.
        num_workers: DataLoader worker processes.
        early_stopping_patience: Epochs without improvement before stopping;
            0 disables early stopping.
        subset_size: Use only the first N videos (0 = all). The paper evaluated
            on a 50-video subset; set this to 50 to mirror that protocol.
    """

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    positive_weight: float = 8.0
    negative_weight: float = 1.0
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    num_folds: int = 5
    seed: int = 42
    num_workers: int = 0
    early_stopping_patience: int = 0
    subset_size: int = 0


@dataclass
class PostProcessConfig:
    """Score smoothing and event triggering (paper Section 3.2.4).

    Attributes:
        smoothing_window: Moving-average window w in frames.
        peak_threshold: Minimum smoothed peak confidence required to fire.
        run_threshold: Score a frame must exceed to extend a positive run.
        min_consecutive: Minimum run length in frames.
        onset_from: run_start reports the first frame of the qualifying run;
            peak reports the argmax frame.
    """

    smoothing_window: int = 9
    peak_threshold: float = 0.7
    run_threshold: float = 0.6
    min_consecutive: int = 5
    onset_from: str = "run_start"


@dataclass
class MetricConfig:
    """IWDD Contest 2026 scoring parameters (paper Section 3.4).

    A detection at time p for ground truth g is valid when
    g - tolerance_before <= p <= g + tolerance_after.

    Attributes:
        tolerance_before: Allowed early margin in seconds (delta_t).
        tolerance_after: Allowed late margin in seconds (T_max).
        target_fps: Minimum acceptable processing frame rate for the PFR score.
        target_memory_gb: Deployment memory budget for the memory score.
    """

    tolerance_before: float = 3.0
    tolerance_after: float = 10.0
    target_fps: float = 5.0
    target_memory_gb: float = 2.0


@dataclass
class ExperimentConfig:
    """Top-level configuration aggregating every stage.

    Attributes:
        paths: Filesystem locations.
        sampling: Frame extraction settings.
        labeling: Onset-to-frame-label rules.
        caption: Vision-language captioning settings.
        encoder: Caption encoding settings.
        model: Temporal classifier settings.
        train: Optimisation settings.
        postprocess: Event triggering settings.
        metric: Contest scoring parameters.
        run_name: Identifier used for output subdirectories.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    run_name: str = "default"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        """Build a configuration from a nested dictionary.

        Unknown keys raise an error rather than being silently ignored, so that
        typos in a YAML file surface immediately.

        Args:
            payload: Nested mapping mirroring the dataclass structure.

        Returns:
            A fully populated configuration.

        Raises:
            KeyError: If the payload contains an unknown field.
        """
        kwargs: dict[str, Any] = {}
        known = {f.name: f for f in fields(cls)}
        for key, value in payload.items():
            if key not in known:
                raise KeyError(f"Unknown config section or field: {key!r}")
            section_cls = _resolve(known[key].type)
            if isinstance(value, dict) and is_dataclass(section_cls):
                section_known = {f.name for f in fields(section_cls)}
                unknown = set(value) - section_known
                if unknown:
                    raise KeyError(f"Unknown fields in section {key!r}: {sorted(unknown)}")
                kwargs[key] = section_cls(**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load a configuration from a YAML file.

        Args:
            path: Path to the YAML document.

        Returns:
            A fully populated configuration.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a nested plain dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the configuration to disk as JSON.

        Args:
            path: Destination file path; parent directories are created.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def feature_signature(self) -> str:
        """Return a short key identifying the feature-extraction settings.

        Cached features are only reusable when sampling, captioning and encoding
        settings match, so the cache directory is keyed by this signature.

        Returns:
            A filesystem-safe signature string.
        """
        backend = self.caption.backend
        model = (self.caption.model_id or "default").rsplit("/", 1)[-1]
        prompts = "single" if self.caption.single_prompt else "multi"
        if self.caption.load_in_8bit:
            quant = "8bit"
        elif self.caption.load_in_4bit:
            quant = "4bit"
        else:
            quant = "full"
        return (
            f"{backend}-{model}-{prompts}-{quant}"
            f"-fps{self.sampling.target_fps:g}-{self.encoder.mode}"
        )


def _resolve(annotation: Any) -> Any:
    """Resolve a field annotation that may be a forward-reference string.

    Args:
        annotation: The raw annotation entry for a dataclass field.

    Returns:
        The referenced class when resolvable, otherwise the annotation itself.
    """
    if isinstance(annotation, str):
        return globals().get(annotation, annotation)
    return annotation

"""Temporal classifier registry and factory."""

from __future__ import annotations

from ..config import ModelConfig
from .base import BaseClipModel, BaseSequenceModel, BaseTemporalModel
from .cnn3d import Conv3DClassifier, PretrainedR3DClassifier
from .recurrent import (
    BiLSTMAttentionClassifier,
    CNNGRUClassifier,
    CNNLSTMClassifier,
    GRUClassifier,
    LSTMClassifier,
)
from .transformer import TransformerClassifier

#: Registry mapping config keys to model classes.
MODEL_REGISTRY: dict[str, type[BaseTemporalModel]] = {
    "lstm": LSTMClassifier,
    "cnn_lstm": CNNLSTMClassifier,
    "gru": GRUClassifier,
    "cnn_gru": CNNGRUClassifier,
    "bilstm_attention": BiLSTMAttentionClassifier,
    "transformer": TransformerClassifier,
    "conv3d": Conv3DClassifier,
    "r3d18": PretrainedR3DClassifier,
}

#: The six architectures compared in the reference paper, in table order.
PAPER_ARCHITECTURES: tuple[str, ...] = (
    "lstm",
    "cnn_lstm",
    "gru",
    "cnn_gru",
    "bilstm_attention",
    "transformer",
)

#: The pixel-space controls added on top of the paper comparison.
CLIP_ARCHITECTURES: tuple[str, ...] = ("conv3d", "r3d18")


def build_model(config: ModelConfig, input_dim: int = 0) -> BaseTemporalModel:
    """Instantiate the architecture named in a configuration.

    Args:
        config: Model settings naming the architecture and its hyper-parameters.
        input_dim: Width of the frame embeddings; required for sequence models
            and ignored by clip models.

    Returns:
        The constructed model.

    Raises:
        KeyError: If the architecture name is not registered.
        ValueError: If a sequence model is requested without ``input_dim``.
    """
    key = config.name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown architecture {config.name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )

    model_cls = MODEL_REGISTRY[key]
    if issubclass(model_cls, BaseClipModel):
        return model_cls(
            clip_len=config.clip_len,
            dropout=config.dropout,
        )

    if input_dim <= 0:
        raise ValueError(f"Architecture {key!r} needs a positive input_dim")

    return model_cls(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_layers=config.num_layers,
        bidirectional=config.bidirectional,
        conv_channels=config.conv_channels,
        conv_kernel_size=config.conv_kernel_size,
        nhead=config.nhead,
    )


def available_architectures() -> list[str]:
    """List the registered architecture keys.

    Returns:
        Sorted architecture names.
    """
    return sorted(MODEL_REGISTRY)


def input_kind_of(name: str) -> str:
    """Return the input family of an architecture without building it.

    Args:
        name: Architecture key.

    Returns:
        Either ``"sequence"`` or ``"clip"``.

    Raises:
        KeyError: If the architecture name is not registered.
    """
    return MODEL_REGISTRY[name.lower()].input_kind


__all__ = [
    "BaseClipModel",
    "BaseSequenceModel",
    "BaseTemporalModel",
    "BiLSTMAttentionClassifier",
    "CLIP_ARCHITECTURES",
    "CNNGRUClassifier",
    "CNNLSTMClassifier",
    "Conv3DClassifier",
    "GRUClassifier",
    "LSTMClassifier",
    "MODEL_REGISTRY",
    "PAPER_ARCHITECTURES",
    "PretrainedR3DClassifier",
    "TransformerClassifier",
    "available_architectures",
    "build_model",
    "input_kind_of",
]

"""Abstract bases shared by the sequence models and the 3D CNN.

Two input families exist and they are deliberately kept distinct rather than
forced behind one signature:

* :class:`BaseSequenceModel` consumes cached embedding sequences (B, T, D) and
  emits one logit per frame.
* :class:`BaseClipModel` consumes raw pixel clips (B, C, T, H, W) and emits one
  logit per clip.

The training loop dispatches on :attr:`BaseTemporalModel.input_kind`, so adding
a new architecture only means subclassing the right base and registering it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseTemporalModel(nn.Module, ABC):
    """Common interface for every classifier in this project.

    Attributes:
        input_kind: Either ``"sequence"`` or ``"clip"``; selects the dataset,
            collate function and loss reduction used during training.
    """

    input_kind: str = "sequence"

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters.

        The paper reports 2,220,673 for CNN+GRU and argues the parameter saving
        over CNN+LSTM is what makes it generalise on a small corpus, so this is
        a number worth reporting alongside accuracy.

        Returns:
            Count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict[str, object]:
        """Summarise the model for run reports.

        Returns:
            A dictionary with the class name, input kind and parameter count.
        """
        return {
            "class": type(self).__name__,
            "input_kind": self.input_kind,
            "num_parameters": self.num_parameters,
        }


class BaseSequenceModel(BaseTemporalModel):
    """Base class for models over per-frame embedding sequences.

    Attributes:
        input_dim: Width of the incoming frame embeddings.
        hidden_dim: Hidden width of the temporal core.
    """

    input_kind = "sequence"

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        """Store the common dimensions and build the output head.

        Args:
            input_dim: Width of the incoming frame embeddings.
            hidden_dim: Width of the representation entering the head.
            dropout: Dropout probability applied before the head.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.Identity()
        self.head = nn.Linear(hidden_dim, 1)

    def _set_head(self, output_dim: int) -> None:
        """Rebuild the normalisation and output head for a known width.

        Subclasses call this once their encoder width is known. The LayerNorm
        is what reconciles this implementation with the parameter counts the
        paper reports (a bare Linear head is 1024 parameters short).

        Args:
            output_dim: Width of the representation entering the head.
        """
        self.norm = nn.LayerNorm(output_dim)
        self.head = nn.Linear(output_dim, 1)

    @abstractmethod
    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map an embedding sequence to a per-frame hidden representation.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T); True marks a real frame.

        Returns:
            Float tensor of shape (B, T, hidden_dim).
        """

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Produce one logit per frame.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T); True marks a real frame.

        Returns:
            Float tensor of shape (B, T) with raw logits.
        """
        hidden = self.encode(features, mask)
        logits = self.head(self.dropout(self.norm(hidden))).squeeze(-1)
        # Padded positions must not leak into the loss or the detector.
        return logits.masked_fill(~mask, -30.0)


class BaseClipModel(BaseTemporalModel):
    """Base class for models over raw spatio-temporal clips.

    Attributes:
        clip_len: Number of frames per clip.
    """

    input_kind = "clip"

    def __init__(self, clip_len: int) -> None:
        """Store the clip length.

        Args:
            clip_len: Number of frames per clip.
        """
        super().__init__()
        self.clip_len = clip_len

    @abstractmethod
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Produce one logit per clip.

        Args:
            clips: Float tensor of shape (B, C, T, H, W).

        Returns:
            Float tensor of shape (B,) with raw logits.
        """


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over the time axis while ignoring padded positions.

    Args:
        values: Float tensor of shape (B, T, D).
        mask: Bool tensor of shape (B, T); True marks a real frame.

    Returns:
        Float tensor of shape (B, D).
    """
    weights = mask.unsqueeze(-1).to(values.dtype)
    total = (values * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return total / count


def lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert a validity mask into per-sequence lengths.

    Args:
        mask: Bool tensor of shape (B, T).

    Returns:
        Int64 tensor of shape (B,) with at least 1 for every row.
    """
    return mask.sum(dim=1).clamp_min(1).to(torch.int64)

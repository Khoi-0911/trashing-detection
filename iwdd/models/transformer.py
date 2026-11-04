"""Transformer encoder over caption embeddings (paper Section 3.2.3).

Included for completeness of the comparison. The paper finds it the weakest
architecture by a wide margin (F1 0.15-0.31) and attributes that to three
causes worth keeping in mind before tuning it: too little data for a
transformer, a mismatch between global attention and an event that is local in
time, and the difficulty of learning position-invariant patterns when the
onset can fall anywhere in the clip.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseSequenceModel


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding.

    Attributes:
        max_len: Longest sequence the table supports.
    """

    def __init__(self, dim: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        """Precompute the position table.

        Args:
            dim: Model width.
            max_len: Longest supported sequence; the corpus needs ~550 at 5 FPS.
            dropout: Dropout applied after adding the encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        table = torch.zeros(max_len, dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        table[:, 0::2] = torch.sin(position * divisor)
        table[:, 1::2] = torch.cos(position * divisor[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add positional information to a sequence.

        Args:
            features: Float tensor of shape (B, T, D).

        Returns:
            Float tensor of shape (B, T, D).

        Raises:
            ValueError: If the sequence is longer than the precomputed table.
        """
        length = features.shape[1]
        if length > self.table.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds positional table "
                f"({self.table.shape[1]}); raise max_len"
            )
        return self.dropout(features + self.table[:, :length])


class TransformerClassifier(BaseSequenceModel):
    """Transformer encoder producing one logit per frame."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        nhead: int = 4,
        max_len: int = 2048,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Model width inside the encoder.
            dropout: Dropout probability.
            num_layers: Encoder layers.
            nhead: Attention heads; ``hidden_dim`` is rounded to a multiple.
            max_len: Longest supported sequence.
            **_: Ignored extra configuration keys.
        """
        width = max(nhead * round(hidden_dim / nhead), nhead)
        super().__init__(input_dim, width, dropout)

        self.project = nn.Linear(input_dim, width)
        self.positions = SinusoidalPositionalEncoding(width, max_len, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=nhead,
            dim_feedforward=width * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self._set_head(width)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the sequence with the transformer.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T); True marks a real frame.

        Returns:
            Float tensor of shape (B, T, width).
        """
        hidden = self.positions(self.project(features))
        # PyTorch expects True to mark positions that should be ignored.
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        return torch.nan_to_num(hidden)

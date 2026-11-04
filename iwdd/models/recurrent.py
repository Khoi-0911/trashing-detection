"""Recurrent and convolutional-recurrent temporal classifiers.

These are the five non-transformer architectures compared in the paper
(Section 3.2.3 and Table 1):

    LSTM              h_t = BiLSTM(e_t, h_{t-1})
    CNN+LSTM          z_t = Conv1D(e_t);  h_t = BiLSTM(z_t, h_{t-1})
    GRU               h_t = BiGRU(e_t, h_{t-1})
    CNN+GRU           z_t = Conv1D(e_t);  h_t = BiGRU(z_t, h_{t-1})
    BiLSTM+Attention  h_t = BiLSTM(e_t);  a_t = softmax(W_a h_t);  h_t = a_t * h_t

CNN+GRU is the paper best (F1 0.7429 with BLIP captions). The Conv1D front-end
captures short-term changes between neighbouring captions, which matters here
because a dynamic dumping event can be over in a fraction of a second.

Sequences are packed before entering the recurrent core so that padded frames
never contribute to the hidden state.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import BaseSequenceModel, lengths_from_mask


class _RecurrentCore(nn.Module):
    """Shared packed-sequence wrapper around an LSTM or GRU.

    Attributes:
        rnn: The wrapped recurrent module.
        output_dim: Width of the returned hidden states.
    """

    def __init__(
        self,
        rnn_cls: type[nn.RNNBase],
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        """Build the recurrent core.

        Args:
            rnn_cls: Either ``nn.LSTM`` or ``nn.GRU``.
            input_dim: Width of the inputs.
            hidden_dim: Width of the hidden state per direction.
            num_layers: Stacked recurrent layers.
            dropout: Inter-layer dropout, ignored when ``num_layers`` is 1.
            bidirectional: Run the sequence in both directions.
        """
        super().__init__()
        directions = 2 if bidirectional else 1
        # hidden_dim is the width PER DIRECTION, matching nn.LSTM/nn.GRU and the
        # parameter counts the paper reports; the output is twice that.
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * directions

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the recurrent core over a padded batch.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, output_dim).
        """
        lengths = lengths_from_mask(mask).cpu()
        packed = pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.rnn(packed)
        hidden, _ = pad_packed_sequence(
            output, batch_first=True, total_length=features.shape[1]
        )
        return hidden


class _Conv1dFrontEnd(nn.Module):
    """Conv1D block extracting local temporal patterns before the RNN.

    Attributes:
        output_dim: Number of output channels.
    """

    def __init__(self, input_dim: int, channels: int, kernel_size: int, dropout: float) -> None:
        """Build the convolutional front-end.

        Args:
            input_dim: Width of the incoming embeddings.
            channels: Output channels.
            kernel_size: Temporal kernel width; padded to preserve length.
            dropout: Dropout applied after the activation.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_dim = channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the convolution along the time axis.

        Args:
            features: Float tensor of shape (B, T, D).

        Returns:
            Float tensor of shape (B, T, channels).
        """
        return self.block(features.transpose(1, 2)).transpose(1, 2)


class LSTMClassifier(BaseSequenceModel):
    """Bidirectional LSTM over caption embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Hidden width of the recurrent core.
            dropout: Dropout before the output head.
            num_layers: Stacked recurrent layers.
            bidirectional: Run the sequence in both directions.
            **_: Ignored extra configuration keys.
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.core = _RecurrentCore(
            nn.LSTM, input_dim, hidden_dim, num_layers, dropout, bidirectional
        )
        self._set_head(self.core.output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the sequence with the LSTM core.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, H).
        """
        return self.core(features, mask)


class GRUClassifier(BaseSequenceModel):
    """Bidirectional GRU over caption embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Hidden width of the recurrent core.
            dropout: Dropout before the output head.
            num_layers: Stacked recurrent layers.
            bidirectional: Run the sequence in both directions.
            **_: Ignored extra configuration keys.
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.core = _RecurrentCore(
            nn.GRU, input_dim, hidden_dim, num_layers, dropout, bidirectional
        )
        self._set_head(self.core.output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the sequence with the GRU core.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, H).
        """
        return self.core(features, mask)


class CNNLSTMClassifier(BaseSequenceModel):
    """Conv1D front-end followed by a bidirectional LSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Hidden width of the recurrent core.
            dropout: Dropout probability.
            num_layers: Stacked recurrent layers.
            bidirectional: Run the sequence in both directions.
            conv_channels: Output channels of the Conv1D front-end.
            conv_kernel_size: Temporal kernel width.
            **_: Ignored extra configuration keys.
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.conv = _Conv1dFrontEnd(input_dim, conv_channels, conv_kernel_size, dropout)
        self.core = _RecurrentCore(
            nn.LSTM, conv_channels, hidden_dim, num_layers, dropout, bidirectional
        )
        self._set_head(self.core.output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode with the convolutional front-end and the LSTM core.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, H).
        """
        return self.core(self.conv(features), mask)


class CNNGRUClassifier(BaseSequenceModel):
    """Conv1D front-end followed by a bidirectional GRU.

    The best configuration in the paper: F1 0.7429 with BLIP multi-aspect
    captions, at 2.22M parameters and an 8.49 MB checkpoint.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True,
        conv_channels: int = 256,
        conv_kernel_size: int = 3,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Hidden width of the recurrent core.
            dropout: Dropout probability.
            num_layers: Stacked recurrent layers.
            bidirectional: Run the sequence in both directions.
            conv_channels: Output channels of the Conv1D front-end.
            conv_kernel_size: Temporal kernel width.
            **_: Ignored extra configuration keys.
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.conv = _Conv1dFrontEnd(input_dim, conv_channels, conv_kernel_size, dropout)
        self.core = _RecurrentCore(
            nn.GRU, conv_channels, hidden_dim, num_layers, dropout, bidirectional
        )
        self._set_head(self.core.output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode with the convolutional front-end and the GRU core.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, H).
        """
        return self.core(self.conv(features), mask)


class BiLSTMAttentionClassifier(BaseSequenceModel):
    """Bidirectional LSTM with a learned per-frame attention gate.

    The paper reports this variant as the least stable, with an F1 standard
    deviation up to 0.378 across folds, which it attributes to overfitting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_layers: int = 2,
        **_: object,
    ) -> None:
        """Build the model.

        Args:
            input_dim: Width of the frame embeddings.
            hidden_dim: Hidden width of the recurrent core.
            dropout: Dropout probability.
            num_layers: Stacked recurrent layers.
            **_: Ignored extra configuration keys.
        """
        super().__init__(input_dim, hidden_dim, dropout)
        self.core = _RecurrentCore(
            nn.LSTM, input_dim, hidden_dim, num_layers, dropout, bidirectional=True
        )
        self.attention = nn.Linear(self.core.output_dim, 1)
        self._set_head(self.core.output_dim)

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode the sequence and re-weight frames by attention.

        Args:
            features: Float tensor of shape (B, T, D).
            mask: Bool tensor of shape (B, T).

        Returns:
            Float tensor of shape (B, T, H).
        """
        hidden = self.core(features, mask)
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        # Scaled by T so the gate re-weights frames without shrinking the
        # activations towards zero on long sequences.
        scale = mask.sum(dim=1).clamp_min(1).view(-1, 1, 1).to(hidden.dtype)
        return hidden * weights * scale

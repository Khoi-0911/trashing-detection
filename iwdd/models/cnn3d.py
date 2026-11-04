"""3D CNN baselines operating directly on pixels.

This family is not in the paper; it is the control the paper lacks. Every other
architecture here sees the video only through generated captions, so none of
them can tell you whether the vision-language detour helps or merely adds
1050ms per frame. A 3D CNN trained on the same clips answers that directly.

Two variants are provided:

* :class:`Conv3DClassifier` -- a compact R3D-style network trained from
  scratch. Honest, but 400 videos is very little data for it.
* :class:`PretrainedR3DClassifier` -- torchvision R3D-18 pretrained on Kinetics
  with a replaced head. On a corpus this small, the pretrained variant is the
  fairer comparison against a pretrained VLM.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseClipModel


class _Conv3DBlock(nn.Module):
    """Conv3D + BatchNorm + ReLU, optionally followed by pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: tuple[int, int, int] | None = (1, 2, 2),
    ) -> None:
        """Build the block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            pool: Pooling kernel and stride, or None to skip pooling.
        """
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool is not None:
            layers.append(nn.MaxPool3d(kernel_size=pool, stride=pool))
        self.block = nn.Sequential(*layers)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            clips: Float tensor of shape (B, C, T, H, W).

        Returns:
            Float tensor of shape (B, C_out, T', H', W').
        """
        return self.block(clips)


class Conv3DClassifier(BaseClipModel):
    """Compact 3D CNN trained from scratch on raw clips."""

    def __init__(
        self,
        clip_len: int = 16,
        dropout: float = 0.3,
        base_channels: int = 32,
        **_: object,
    ) -> None:
        """Build the network.

        Args:
            clip_len: Frames per clip.
            dropout: Dropout before the classification head.
            base_channels: Channel width of the first block; doubles per stage.
            **_: Ignored extra configuration keys.
        """
        super().__init__(clip_len)
        width = base_channels
        self.features = nn.Sequential(
            _Conv3DBlock(3, width, pool=(1, 2, 2)),
            _Conv3DBlock(width, width * 2, pool=(2, 2, 2)),
            _Conv3DBlock(width * 2, width * 4, pool=(2, 2, 2)),
            _Conv3DBlock(width * 4, width * 8, pool=(2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, 1),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Produce one logit per clip.

        Args:
            clips: Float tensor of shape (B, 3, T, H, W).

        Returns:
            Float tensor of shape (B,).
        """
        hidden = self.features(clips)
        pooled = self.pool(hidden)
        return self.classifier(pooled).squeeze(-1)


class PretrainedR3DClassifier(BaseClipModel):
    """Torchvision R3D-18 pretrained on Kinetics-400 with a new head."""

    def __init__(
        self,
        clip_len: int = 16,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        **_: object,
    ) -> None:
        """Build the network.

        Args:
            clip_len: Frames per clip.
            dropout: Dropout before the classification head.
            pretrained: Load Kinetics-400 weights.
            freeze_backbone: Train only the head, which is the sensible
                starting point on a 400-video corpus.
            **_: Ignored extra configuration keys.

        Raises:
            ImportError: If torchvision is not installed.
        """
        super().__init__(clip_len)
        try:
            from torchvision.models.video import R3D_18_Weights, r3d_18
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("torchvision is required for PretrainedR3DClassifier") from exc

        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = r3d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Produce one logit per clip.

        Args:
            clips: Float tensor of shape (B, 3, T, H, W).

        Returns:
            Float tensor of shape (B,).
        """
        return self.classifier(self.backbone(clips)).squeeze(-1)

"""Synthesis of frame-level supervision from a single onset timestamp.

MIVIA-IWDD-500 ships exactly one integer onset per positive video and states
that dense frame-level annotations are not available, yet the temporal
classifier is trained on a label per frame. The expansion rule therefore has to
be invented, and it is the single most influential free choice in the whole
pipeline -- it decides what the model is actually taught.

The default window is read off two clues in the paper:

* Section 6.1 calls dumping a "local temporal event (5-15 frames)", which at
  5 FPS is 1-3 seconds.
* Section 3.3 uses a positive class weight of ``w+ = 8``. With a median clip of
  18.6s a 2s window yields roughly 11% positive frames, which matches a weight
  of 8 closely.

Treat :attr:`LabelingConfig.post_seconds` as a hyper-parameter to sweep: the
corpus gives no event end time, so no value is verifiable against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .annotations import VideoAnnotation
from .config import LabelingConfig


@dataclass(frozen=True)
class FrameLabels:
    """Per-frame supervision for one video.

    Attributes:
        values: Binary labels of shape (N,), aligned with the sampled frames.
        window: The ``(start, end)`` positive interval in seconds, or None for
            negative videos.
        clamped: True when the window was shortened to fit inside the clip.
    """

    values: np.ndarray
    window: tuple[float, float] | None
    clamped: bool

    @property
    def num_positive(self) -> int:
        """Number of frames labelled positive."""
        return int(self.values.sum())

    @property
    def positive_ratio(self) -> float:
        """Fraction of frames labelled positive."""
        return float(self.values.mean()) if self.values.size else 0.0


class FrameLabeler:
    """Expands an onset timestamp into a binary label per sampled frame.

    Attributes:
        config: Rules controlling the window shape and validity handling.
    """

    def __init__(self, config: LabelingConfig | None = None) -> None:
        """Initialise the labeler.

        Args:
            config: Labeling settings; defaults are used when omitted.
        """
        self.config = config or LabelingConfig()

    def label(self, annotation: VideoAnnotation, timestamps: np.ndarray) -> FrameLabels:
        """Produce frame-level labels for one video.

        Args:
            annotation: Record carrying the onset and duration.
            timestamps: Timestamps in seconds of the sampled frames.

        Returns:
            The frame labels and the window that produced them.
        """
        timestamps = np.asarray(timestamps, dtype=np.float32)
        values = np.zeros(timestamps.shape[0], dtype=np.int64)

        if not annotation.is_positive or annotation.onset is None:
            return FrameLabels(values=values, window=None, clamped=False)

        start = annotation.onset - self.config.pre_seconds
        end = annotation.onset + self.config.post_seconds
        clamped = False

        if self.config.clamp_to_duration:
            clipped_start = max(start, 0.0)
            clipped_end = min(end, annotation.duration)
            clamped = (clipped_start != start) or (clipped_end != end)
            start, end = clipped_start, clipped_end

        if end <= start:
            # Degenerate window, e.g. vid0481 whose onset lies past the clip
            # end. Fall back to marking the final sampled frame so the video
            # still carries some positive signal instead of becoming a silent
            # false negative in training.
            if timestamps.size:
                values[-1] = 1
            return FrameLabels(values=values, window=None, clamped=True)

        values[(timestamps >= start) & (timestamps <= end)] = 1

        if values.sum() == 0 and timestamps.size:
            # The window fell between two sampled frames; snap to the nearest.
            nearest = int(np.argmin(np.abs(timestamps - annotation.onset)))
            values[nearest] = 1

        return FrameLabels(values=values, window=(start, end), clamped=clamped)

    def positive_weight(self, label_batches: list[np.ndarray]) -> float:
        """Derive the class weight implied by an observed label distribution.

        Lets a run replace the paper constant of 8.0 with the value the chosen
        window actually produces.

        Args:
            label_batches: Per-video label arrays.

        Returns:
            The ratio of negative to positive frames, or 1.0 when degenerate.
        """
        if not label_batches:
            return 1.0
        stacked = np.concatenate([np.asarray(batch).ravel() for batch in label_batches])
        positives = float(stacked.sum())
        negatives = float(stacked.size - positives)
        if positives <= 0:
            return 1.0
        return negatives / positives

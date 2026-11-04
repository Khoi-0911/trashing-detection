"""Turning per-frame scores into a single event decision (paper Section 3.2.4).

Smoothing with a moving average of width 9, then firing only when the smoothed
peak exceeds 0.7 and at least 5 consecutive frames exceed 0.6. The dual
threshold is what suppresses isolated spikes; at 5 FPS the run requirement of
5 frames means a detection must be supported by a full second of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PostProcessConfig


@dataclass(frozen=True)
class DetectionResult:
    """The decision taken for one video.

    Attributes:
        detected: True when the detector fired.
        onset: Predicted onset in seconds, or None when it did not fire.
        peak_score: Highest smoothed score observed.
        peak_time: Timestamp of the highest smoothed score.
        run_length: Length in frames of the qualifying run, 0 when none.
        smoothed: The smoothed score curve, kept for plotting.
    """

    detected: bool
    onset: float | None
    peak_score: float
    peak_time: float
    run_length: int
    smoothed: np.ndarray

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable summary without the score curve."""
        return {
            "detected": self.detected,
            "onset": self.onset,
            "peak_score": round(float(self.peak_score), 4),
            "peak_time": round(float(self.peak_time), 3),
            "run_length": self.run_length,
        }


class EventDetector:
    """Smooths frame scores and applies the dual-threshold trigger.

    Attributes:
        config: Smoothing window, thresholds and onset reporting rule.
    """

    def __init__(self, config: PostProcessConfig | None = None) -> None:
        """Initialise the detector.

        Args:
            config: Post-processing settings; defaults follow the paper.
        """
        self.config = config or PostProcessConfig()

    def smooth(self, scores: np.ndarray) -> np.ndarray:
        """Apply a centred moving average to a score curve.

        Edges are handled by reflecting the signal, which avoids the artificial
        dip a zero-padded average produces at the start of a clip -- important
        here because some videos have their onset at t=0.

        Args:
            scores: Float array of shape (N,) with per-frame probabilities.

        Returns:
            The smoothed curve, same shape as the input.
        """
        window = int(self.config.smoothing_window)
        if window <= 1 or scores.size == 0:
            return np.asarray(scores, dtype=np.float32)

        if window % 2 == 0:
            window += 1
        half = window // 2
        padded = np.pad(np.asarray(scores, dtype=np.float32), half, mode="reflect")
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(padded, kernel, mode="valid").astype(np.float32)

    def detect(self, scores: np.ndarray, timestamps: np.ndarray) -> DetectionResult:
        """Decide whether and when a dumping event occurred.

        Args:
            scores: Float array of shape (N,) with per-frame probabilities.
            timestamps: Float array of shape (N,) in seconds.

        Returns:
            The detection decision.
        """
        scores = np.asarray(scores, dtype=np.float32).ravel()
        timestamps = np.asarray(timestamps, dtype=np.float32).ravel()

        if scores.size == 0:
            return DetectionResult(False, None, 0.0, 0.0, 0, scores)

        smoothed = self.smooth(scores)
        peak_index = int(np.argmax(smoothed))
        peak_score = float(smoothed[peak_index])
        peak_time = float(timestamps[peak_index])

        if peak_score < self.config.peak_threshold:
            return DetectionResult(False, None, peak_score, peak_time, 0, smoothed)

        start, length = self._longest_run(smoothed >= self.config.run_threshold)
        if length < self.config.min_consecutive:
            return DetectionResult(False, None, peak_score, peak_time, length, smoothed)

        if self.config.onset_from == "peak":
            onset = peak_time
        else:
            onset = float(timestamps[start])

        return DetectionResult(True, onset, peak_score, peak_time, length, smoothed)

    def detect_batch(
        self,
        scores: list[np.ndarray],
        timestamps: list[np.ndarray],
    ) -> list[DetectionResult]:
        """Run :meth:`detect` over several videos.

        Args:
            scores: One score array per video.
            timestamps: One timestamp array per video, aligned with ``scores``.

        Returns:
            One detection result per video.
        """
        return [self.detect(s, t) for s, t in zip(scores, timestamps)]

    @staticmethod
    def _longest_run(flags: np.ndarray) -> tuple[int, int]:
        """Find the longest run of True values.

        Args:
            flags: Boolean array of shape (N,).

        Returns:
            A tuple of ``(start_index, length)``; ``(0, 0)`` when none.
        """
        best_start = 0
        best_length = 0
        current_start = 0
        current_length = 0

        for index, flag in enumerate(flags):
            if flag:
                if current_length == 0:
                    current_start = index
                current_length += 1
                if current_length > best_length:
                    best_length = current_length
                    best_start = current_start
            else:
                current_length = 0

        return best_start, best_length


def tune_thresholds(
    scores: list[np.ndarray],
    timestamps: list[np.ndarray],
    onsets: list[float],
    base_config: PostProcessConfig,
    peak_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    run_grid: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7),
    tolerance_before: float = 3.0,
    tolerance_after: float = 10.0,
) -> tuple[PostProcessConfig, float]:
    """Grid-search the trigger thresholds against contest F1.

    The paper fixes 0.7 and 0.6 without saying how they were chosen. Those
    values were tuned for BLIP captions, so a different VLM will almost
    certainly want different ones -- tune on the training folds only, never on
    the held-out split.

    Args:
        scores: Per-video score arrays.
        timestamps: Per-video timestamp arrays.
        onsets: Ground-truth onsets in seconds; -1 marks a negative video.
        base_config: Settings to copy the untuned fields from.
        peak_grid: Candidate peak thresholds.
        run_grid: Candidate run thresholds.
        tolerance_before: Allowed early margin in seconds.
        tolerance_after: Allowed late margin in seconds.

    Returns:
        A tuple of the best configuration and the F1 it achieved.
    """
    from dataclasses import replace

    from .metrics import IWDDEvaluator
    from .config import MetricConfig

    evaluator = IWDDEvaluator(
        MetricConfig(tolerance_before=tolerance_before, tolerance_after=tolerance_after)
    )

    best_config = base_config
    best_f1 = -1.0

    for peak in peak_grid:
        for run in run_grid:
            if run > peak:
                continue
            candidate = replace(base_config, peak_threshold=peak, run_threshold=run)
            detector = EventDetector(candidate)
            results = detector.detect_batch(scores, timestamps)
            report = evaluator.evaluate(results, onsets)
            if report.f1 > best_f1:
                best_f1 = report.f1
                best_config = candidate

    return best_config, best_f1

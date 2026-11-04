"""IWDD Contest 2026 scoring (paper Section 3.4).

Detection accuracy. With ground truth ``g`` and prediction ``p``, a detection
is valid when ``g - 3.0s <= p <= g + 10.0s``. Then:

    TP = valid detection inside a positive video
    FP = any detection in a negative video, or a detection outside the window
    FN = no valid detection in a positive video

Note the asymmetry this creates and that the paper does not spell out: a
positive video detected at the wrong time counts as *both* an FP and an FN. It
is the correct reading of the stated rule and it is also the right incentive --
an alarm at the wrong moment is a false alarm *and* a miss, not a half-credit.

Notification delay. ``D`` is the mean absolute error over true positives and
``D_norm = max(0, 1 - D / T_max)`` rewards faster detection. The paper reports
D = 1.13s and D_norm = 0.887 for BLIP+CNN+GRU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import MetricConfig
from .postprocess import DetectionResult


@dataclass
class ConfusionCounts:
    """Contest-rule confusion counts.

    Attributes:
        true_positive: Valid detections in positive videos.
        false_positive: Detections in negative videos or outside the window.
        false_negative: Positive videos with no valid detection.
        true_negative: Negative videos with no detection.
    """

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    @property
    def precision(self) -> float:
        """Precision over detections, 0 when nothing fired."""
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        """Recall over positive videos, 0 when there are none."""
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def accuracy(self) -> float:
        """Video-level accuracy over all scored videos."""
        total = (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )
        return (self.true_positive + self.true_negative) / total if total else 0.0

    @property
    def specificity(self) -> float:
        """True-negative rate over negative videos."""
        denominator = self.true_negative + self.false_positive
        return self.true_negative / denominator if denominator else 0.0


@dataclass
class EvaluationReport:
    """Full scoring output for one run.

    Attributes:
        counts: Contest-rule confusion counts.
        delays: Absolute onset errors over true positives, in seconds.
        num_videos: Number of videos scored.
        per_video: One record per video, for error analysis.
    """

    counts: ConfusionCounts
    delays: list[float] = field(default_factory=list)
    num_videos: int = 0
    per_video: list[dict[str, object]] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """Contest precision."""
        return self.counts.precision

    @property
    def recall(self) -> float:
        """Contest recall."""
        return self.counts.recall

    @property
    def f1(self) -> float:
        """Contest F1-score."""
        return self.counts.f1

    @property
    def mean_delay(self) -> float:
        """Mean absolute onset error over true positives, in seconds."""
        return float(np.mean(self.delays)) if self.delays else 0.0

    def normalized_delay(self, tolerance_after: float) -> float:
        """Compute the delay score that rewards faster detection.

        Args:
            tolerance_after: The ``T_max`` normaliser in seconds.

        Returns:
            ``max(0, 1 - D / T_max)``.
        """
        if not self.delays or tolerance_after <= 0:
            return 0.0
        return max(0.0, 1.0 - self.mean_delay / tolerance_after)

    def summary(self, config: MetricConfig) -> dict[str, float]:
        """Return the headline numbers as a flat dictionary.

        Args:
            config: Scoring parameters supplying ``T_max``.

        Returns:
            A dictionary of scalar metrics.
        """
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.counts.accuracy, 4),
            "specificity": round(self.counts.specificity, 4),
            "tp": self.counts.true_positive,
            "fp": self.counts.false_positive,
            "fn": self.counts.false_negative,
            "tn": self.counts.true_negative,
            "mean_delay_s": round(self.mean_delay, 4),
            "normalized_delay": round(self.normalized_delay(config.tolerance_after), 4),
            "num_videos": self.num_videos,
        }


class IWDDEvaluator:
    """Applies the contest matching rule and aggregates the scores.

    Attributes:
        config: Tolerance window and the PFR/memory targets.
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        """Initialise the evaluator.

        Args:
            config: Scoring parameters; defaults follow the paper.
        """
        self.config = config or MetricConfig()

    def is_valid_detection(self, predicted: float, ground_truth: float) -> bool:
        """Test a predicted onset against the tolerance window.

        Args:
            predicted: Predicted onset in seconds.
            ground_truth: True onset in seconds.

        Returns:
            True when the prediction falls inside the allowed window.
        """
        lower = ground_truth - self.config.tolerance_before
        upper = ground_truth + self.config.tolerance_after
        return lower <= predicted <= upper

    def evaluate(
        self,
        results: Sequence[DetectionResult],
        onsets: Sequence[float],
        video_ids: Sequence[str] | None = None,
    ) -> EvaluationReport:
        """Score a set of detections against ground-truth onsets.

        Args:
            results: One detection result per video.
            onsets: Ground-truth onsets in seconds; -1 marks a negative video.
            video_ids: Optional identifiers used in the per-video breakdown.

        Returns:
            The aggregated report.

        Raises:
            ValueError: If the inputs have mismatched lengths.
        """
        if len(results) != len(onsets):
            raise ValueError(
                f"Got {len(results)} detections for {len(onsets)} ground-truth onsets"
            )
        ids = list(video_ids) if video_ids is not None else [""] * len(results)

        counts = ConfusionCounts()
        delays: list[float] = []
        per_video: list[dict[str, object]] = []

        for result, onset, video_id in zip(results, onsets, ids):
            is_positive = onset is not None and onset >= 0
            outcome = self._score_one(result, onset, is_positive, counts, delays)
            per_video.append(
                {
                    "video_id": video_id,
                    "is_positive": is_positive,
                    "gt_onset": float(onset),
                    "outcome": outcome,
                    **result.as_dict(),
                }
            )

        return EvaluationReport(
            counts=counts,
            delays=delays,
            num_videos=len(results),
            per_video=per_video,
        )

    def processing_score(self, measured_fps: float) -> float:
        """Score throughput against the deployment target.

        Args:
            measured_fps: Observed processing frame rate.

        Returns:
            ``max(0, target/measured - 1)``; lower is better, 0 means the
            target was met or beaten.
        """
        if measured_fps <= 0:
            return float("inf")
        return max(0.0, self.config.target_fps / measured_fps - 1.0)

    def memory_score(self, measured_gb: float) -> float:
        """Score peak memory against the deployment budget.

        Args:
            measured_gb: Observed peak memory in gibibytes.

        Returns:
            ``max(0, measured/target - 1)``; lower is better, 0 means within
            budget.
        """
        if self.config.target_memory_gb <= 0:
            return 0.0
        return max(0.0, measured_gb / self.config.target_memory_gb - 1.0)

    def _score_one(
        self,
        result: DetectionResult,
        onset: float,
        is_positive: bool,
        counts: ConfusionCounts,
        delays: list[float],
    ) -> str:
        """Classify one video and update the running counts.

        Args:
            result: The detection decision for this video.
            onset: Ground-truth onset in seconds; -1 for negatives.
            is_positive: Whether the video contains an event.
            counts: Counts to update in place.
            delays: Delay list to append to on a true positive.

        Returns:
            One of ``TP``, ``FP``, ``FN``, ``TN`` or ``FP+FN``.
        """
        if not result.detected:
            if is_positive:
                counts.false_negative += 1
                return "FN"
            counts.true_negative += 1
            return "TN"

        if not is_positive:
            counts.false_positive += 1
            return "FP"

        predicted = float(result.onset if result.onset is not None else -1e9)
        if self.is_valid_detection(predicted, float(onset)):
            counts.true_positive += 1
            delays.append(abs(predicted - float(onset)))
            return "TP"

        # Fired in a positive video but at the wrong time: a false alarm and a
        # miss at the same time, per the contest definition.
        counts.false_positive += 1
        counts.false_negative += 1
        return "FP+FN"


def aggregate_folds(reports: Sequence[EvaluationReport], config: MetricConfig) -> dict[str, float]:
    """Average fold metrics and report their spread.

    The paper reports F1 0.7429 with a standard deviation of 0.191 over five
    folds of ten videos. The spread is as informative as the mean at that
    sample size, so both are always returned together.

    Args:
        reports: One report per fold.
        config: Scoring parameters.

    Returns:
        A dictionary of means and standard deviations.
    """
    if not reports:
        return {}

    keys = ("precision", "recall", "f1", "accuracy", "specificity", "mean_delay_s")
    summaries = [report.summary(config) for report in reports]

    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(summary[key]) for summary in summaries]
        aggregated[f"{key}_mean"] = round(float(np.mean(values)), 4)
        aggregated[f"{key}_std"] = round(float(np.std(values)), 4)

    for key in ("tp", "fp", "fn", "tn"):
        aggregated[f"{key}_total"] = int(sum(int(summary[key]) for summary in summaries))

    aggregated["num_folds"] = len(reports)
    return aggregated

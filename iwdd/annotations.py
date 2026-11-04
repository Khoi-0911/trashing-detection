"""Parsing, validation and querying of MIVIA-IWDD-500 JSON annotations.

Real schema observed on disk (500 files, verified):

    {
      "label": "true",                 # STRING, not a JSON boolean
      "timestamp": 4,                  # integer seconds; -1 for negatives
      "metadata": {
        "video_id": "vid0001.mp4",
        "resolution": "1280x720",      # "WxH" string
        "frame_rate": 30.0,
        "duration": 5.77,
        "event_type": "static dumping",
        "time_of_day": "day",
        "lighting": "high"
      }
    }

Two traps this module guards against:

* ``label`` is the string ``"true"``/``"false"``. A truthiness test such as
  ``if payload["label"]:`` is always True because ``"false"`` is a non-empty
  string. Parsing is therefore strict and explicit.
* A small number of records are internally inconsistent -- in TRAINING_SET,
  ``vid0481`` has ``timestamp=17`` with ``duration=14.06`` (onset after the end
  of the video), and several positives leave under one second after the onset.
  :meth:`VideoAnnotation.validate` surfaces these so they can be dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .utils import get_logger

LOGGER = get_logger()

_LABEL_TRUE = "true"
_LABEL_FALSE = "false"


class DisposalModality(str, Enum):
    """Disposal modality taxonomy used by the benchmark.

    Attributes:
        NORMAL: No dumping event (negative video).
        STATIC: Waste left at a fixed location.
        DYNAMIC: Waste released while the subject or vehicle is in motion.
    """

    NORMAL = "normal"
    STATIC = "static dumping"
    DYNAMIC = "dynamic dumping"

    @classmethod
    def parse(cls, raw: str) -> "DisposalModality":
        """Parse an ``event_type`` string into an enum member.

        Args:
            raw: Raw value from the annotation file.

        Returns:
            The matching modality.

        Raises:
            ValueError: If the value is not part of the taxonomy.
        """
        normalised = (raw or "").strip().lower()
        for member in cls:
            if member.value == normalised:
                return member
        raise ValueError(f"Unknown event_type: {raw!r}")


@dataclass(frozen=True)
class VideoAnnotation:
    """One annotated video: the ground truth plus its media metadata.

    Attributes:
        video_id: Stem of the video file, e.g. ``vid0001``.
        is_positive: True when the video contains a dumping event.
        onset: Event onset in seconds, or None for negative videos.
        modality: Disposal modality.
        width: Native frame width in pixels.
        height: Native frame height in pixels.
        frame_rate: Native frame rate reported by the annotation.
        duration: Clip duration in seconds.
        time_of_day: Either ``day`` or ``night``.
        lighting: Either ``high`` or ``low``.
        video_path: Absolute path to the ``.mp4`` file.
        label_path: Absolute path to the ``.json`` file.
        split: Split directory name this record came from.
    """

    video_id: str
    is_positive: bool
    onset: float | None
    modality: DisposalModality
    width: int
    height: int
    frame_rate: float
    duration: float
    time_of_day: str
    lighting: str
    video_path: Path
    label_path: Path
    split: str

    @classmethod
    def from_json(cls, label_path: Path, video_dir: Path, split: str) -> "VideoAnnotation":
        """Build an annotation from a single JSON file.

        Args:
            label_path: Path to the ``.json`` annotation.
            video_dir: Directory holding the matching ``.mp4`` files.
            split: Name of the split this record belongs to.

        Returns:
            The parsed annotation.

        Raises:
            ValueError: If ``label`` is not exactly ``"true"`` or ``"false"``.
        """
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})

        raw_label = str(payload["label"]).strip().lower()
        if raw_label not in (_LABEL_TRUE, _LABEL_FALSE):
            raise ValueError(f"{label_path.name}: unexpected label {payload['label']!r}")
        is_positive = raw_label == _LABEL_TRUE

        raw_timestamp = float(payload["timestamp"])
        onset = raw_timestamp if is_positive else None

        width, height = _parse_resolution(metadata.get("resolution", "0x0"))
        video_name = metadata.get("video_id") or f"{label_path.stem}.mp4"

        return cls(
            video_id=label_path.stem,
            is_positive=is_positive,
            onset=onset,
            modality=DisposalModality.parse(metadata.get("event_type", "normal")),
            width=width,
            height=height,
            frame_rate=float(metadata.get("frame_rate", 0.0)),
            duration=float(metadata.get("duration", 0.0)),
            time_of_day=str(metadata.get("time_of_day", "unknown")).lower(),
            lighting=str(metadata.get("lighting", "unknown")).lower(),
            video_path=video_dir / video_name,
            label_path=label_path,
            split=split,
        )

    @property
    def resolution(self) -> tuple[int, int]:
        """Native frame size as a ``(width, height)`` tuple."""
        return self.width, self.height

    @property
    def tail_seconds(self) -> float | None:
        """Seconds between the onset and the end of the clip, None if negative."""
        return None if self.onset is None else self.duration - self.onset

    def validate(self, min_tail_seconds: float = 0.0) -> list[str]:
        """Check the record for internal inconsistencies.

        Args:
            min_tail_seconds: Minimum acceptable gap between onset and the end
                of the clip before the record is flagged.

        Returns:
            A list of human-readable problems; empty when the record is sound.
        """
        issues: list[str] = []
        if self.duration <= 0:
            issues.append(f"non-positive duration ({self.duration})")
        if self.frame_rate <= 0:
            issues.append(f"non-positive frame_rate ({self.frame_rate})")
        if not self.video_path.exists():
            issues.append("video file missing")
        if self.is_positive:
            if self.onset is None:
                issues.append("positive video without onset")
            else:
                if self.onset < 0:
                    issues.append(f"negative onset ({self.onset})")
                if self.onset > self.duration:
                    issues.append(
                        f"onset {self.onset}s beyond duration {self.duration}s"
                    )
                elif (self.duration - self.onset) < min_tail_seconds:
                    issues.append(
                        f"only {self.duration - self.onset:.2f}s after onset"
                    )
            if self.modality is DisposalModality.NORMAL:
                issues.append("positive video labelled as normal")
        else:
            if self.modality is not DisposalModality.NORMAL:
                issues.append(f"negative video with modality {self.modality.value}")
        return issues

    def is_usable(self, min_tail_seconds: float = 0.0) -> bool:
        """Return True when the record passes validation.

        Args:
            min_tail_seconds: Forwarded to :meth:`validate`.

        Returns:
            True when no issues were found.
        """
        return not self.validate(min_tail_seconds=min_tail_seconds)


class AnnotationStore:
    """An in-memory, queryable collection of :class:`VideoAnnotation` records.

    Attributes:
        split: Name of the split that was loaded.
        annotations: The loaded records, ordered by ``video_id``.
    """

    def __init__(self, annotations: Sequence[VideoAnnotation], split: str = "") -> None:
        """Wrap an existing sequence of annotations.

        Args:
            annotations: Records to store.
            split: Optional split name for reporting.
        """
        self.annotations: list[VideoAnnotation] = sorted(annotations, key=lambda a: a.video_id)
        self.split = split
        self._by_id = {a.video_id: a for a in self.annotations}

    @classmethod
    def from_split(
        cls,
        split_dir: Path,
        *,
        labels_subdir: str = "labels",
        videos_subdir: str = "videos",
    ) -> "AnnotationStore":
        """Load every annotation in a split directory.

        Args:
            split_dir: Directory containing ``labels/`` and ``videos/``.
            labels_subdir: Name of the annotations subdirectory.
            videos_subdir: Name of the videos subdirectory.

        Returns:
            A store holding all parsed records.

        Raises:
            FileNotFoundError: If the expected subdirectories do not exist.
        """
        split_dir = Path(split_dir)
        label_dir = split_dir / labels_subdir
        video_dir = split_dir / videos_subdir
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {label_dir}")
        if not video_dir.is_dir():
            raise FileNotFoundError(f"Missing videos directory: {video_dir}")

        records = [
            VideoAnnotation.from_json(path, video_dir, split_dir.name)
            for path in sorted(label_dir.glob("*.json"))
        ]
        LOGGER.info("Loaded %d annotations from %s", len(records), split_dir.name)
        return cls(records, split=split_dir.name)

    def __len__(self) -> int:
        """Number of stored records."""
        return len(self.annotations)

    def __iter__(self) -> Iterator[VideoAnnotation]:
        """Iterate over the stored records."""
        return iter(self.annotations)

    def __getitem__(self, index: int) -> VideoAnnotation:
        """Return the record at a positional index."""
        return self.annotations[index]

    def get(self, video_id: str) -> VideoAnnotation:
        """Look up a record by video id.

        Args:
            video_id: Stem of the video, e.g. ``vid0012``.

        Returns:
            The matching annotation.

        Raises:
            KeyError: If no such record exists.
        """
        return self._by_id[video_id]

    @property
    def labels(self) -> list[int]:
        """Binary video-level labels, aligned with :attr:`annotations`."""
        return [int(a.is_positive) for a in self.annotations]

    def filter(self, predicate) -> "AnnotationStore":
        """Return a new store holding only the records matching a predicate.

        Args:
            predicate: Callable taking a record and returning a bool.

        Returns:
            A filtered store.
        """
        return AnnotationStore([a for a in self.annotations if predicate(a)], self.split)

    def subset(self, size: int, *, stratified: bool = True) -> "AnnotationStore":
        """Take a subset, optionally preserving the positive/negative balance.

        The paper evaluated on a 50-video subset of the 400 training videos;
        this reproduces that protocol in a label-balanced way.

        Args:
            size: Number of records to keep; 0 or negative keeps everything.
            stratified: Interleave positives and negatives to keep the balance.

        Returns:
            A store holding at most ``size`` records.
        """
        if size <= 0 or size >= len(self.annotations):
            return self
        if not stratified:
            return AnnotationStore(self.annotations[:size], self.split)

        positives = [a for a in self.annotations if a.is_positive]
        negatives = [a for a in self.annotations if not a.is_positive]
        picked: list[VideoAnnotation] = []
        for index in range(max(len(positives), len(negatives))):
            if index < len(positives):
                picked.append(positives[index])
            if index < len(negatives):
                picked.append(negatives[index])
            if len(picked) >= size:
                break
        return AnnotationStore(picked[:size], self.split)

    def validated(self, min_tail_seconds: float = 0.0, drop: bool = True) -> "AnnotationStore":
        """Report, and optionally remove, records that fail validation.

        Args:
            min_tail_seconds: Forwarded to :meth:`VideoAnnotation.validate`.
            drop: When True, invalid records are excluded from the result.

        Returns:
            Either a cleaned store or this store unchanged.
        """
        clean: list[VideoAnnotation] = []
        for record in self.annotations:
            issues = record.validate(min_tail_seconds=min_tail_seconds)
            if issues:
                LOGGER.warning("%s/%s: %s", record.split, record.video_id, "; ".join(issues))
                if drop:
                    continue
            clean.append(record)
        removed = len(self.annotations) - len(clean)
        if removed:
            LOGGER.warning("Dropped %d invalid annotation(s) from %s", removed, self.split)
        return AnnotationStore(clean, self.split)

    def describe(self) -> dict[str, object]:
        """Summarise the store for logging and dataset reports.

        Returns:
            A dictionary of counts and duration statistics.
        """
        durations = [a.duration for a in self.annotations]
        onsets = [a.onset for a in self.annotations if a.onset is not None]
        return {
            "split": self.split,
            "num_videos": len(self.annotations),
            "num_positive": sum(a.is_positive for a in self.annotations),
            "num_negative": sum(not a.is_positive for a in self.annotations),
            "modality": dict(Counter(a.modality.value for a in self.annotations)),
            "time_of_day": dict(Counter(a.time_of_day for a in self.annotations)),
            "lighting": dict(Counter(a.lighting for a in self.annotations)),
            "duration_total_min": round(sum(durations) / 60.0, 2) if durations else 0.0,
            "duration_min_s": round(min(durations), 2) if durations else 0.0,
            "duration_max_s": round(max(durations), 2) if durations else 0.0,
            "duration_mean_s": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "onset_min_s": round(min(onsets), 2) if onsets else None,
            "onset_max_s": round(max(onsets), 2) if onsets else None,
            "onset_mean_s": round(sum(onsets) / len(onsets), 2) if onsets else None,
            "distinct_frame_rates": len({a.frame_rate for a in self.annotations}),
            "distinct_resolutions": len({a.resolution for a in self.annotations}),
        }


def _parse_resolution(raw: str) -> tuple[int, int]:
    """Parse a ``"WxH"`` resolution string into integers.

    Args:
        raw: Resolution string such as ``"1920x1080"``.

    Returns:
        A ``(width, height)`` tuple; ``(0, 0)`` when unparseable.
    """
    try:
        width, height = str(raw).lower().split("x", 1)
        return int(width), int(height)
    except (ValueError, AttributeError):
        return 0, 0


def load_splits(
    dataset_root: str | Path,
    splits: Iterable[str],
) -> dict[str, AnnotationStore]:
    """Load several splits at once.

    Args:
        dataset_root: Directory containing the split subdirectories.
        splits: Split directory names to load.

    Returns:
        A mapping from split name to its store.
    """
    root = Path(dataset_root).expanduser().resolve()
    return {name: AnnotationStore.from_split(root / name) for name in splits}

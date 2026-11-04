"""Shared helpers: determinism, device selection, timing and logging."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LOGGER_NAME = "iwdd"


class Timer:
    """Context manager measuring wall-clock duration of a code block.

    Attributes:
        label: Human-readable name used when the timing is logged.
        elapsed: Duration in seconds, populated on exit.
    """

    def __init__(self, label: str = "block", logger: logging.Logger | None = None) -> None:
        """Initialise the timer.

        Args:
            label: Name of the measured block.
            logger: Optional logger; when given the result is logged on exit.
        """
        self.label = label
        self.elapsed: float = 0.0
        self._logger = logger
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the clock."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the clock and optionally log the duration."""
        self.elapsed = time.perf_counter() - self._start
        if self._logger is not None:
            self._logger.info("%s took %.3fs", self.label, self.elapsed)


@dataclass
class ResourceProbe:
    """Collects peak GPU memory and throughput statistics for a run.

    The IWDD Contest 2026 scores processing frame rate and memory usage, so these
    numbers are first-class outputs rather than debug information.

    Attributes:
        frames: Number of frames processed so far.
        seconds: Accumulated processing time in seconds.
        peak_memory_bytes: Peak CUDA memory observed, 0 on CPU-only runs.
    """

    frames: int = 0
    seconds: float = 0.0
    peak_memory_bytes: int = 0
    _per_video_fps: list[float] = field(default_factory=list)

    def add(self, frames: int, seconds: float) -> None:
        """Record one processed video.

        Args:
            frames: Frames processed for this video.
            seconds: Wall-clock seconds spent on this video.
        """
        self.frames += frames
        self.seconds += seconds
        if seconds > 0:
            self._per_video_fps.append(frames / seconds)

    def refresh_peak_memory(self) -> None:
        """Query the CUDA allocator for the peak memory watermark."""
        try:
            import torch

            if torch.cuda.is_available():
                self.peak_memory_bytes = max(
                    self.peak_memory_bytes, int(torch.cuda.max_memory_allocated())
                )
        except Exception:  # pragma: no cover - torch optional at import time
            pass

    @property
    def fps(self) -> float:
        """Overall processing frame rate in frames per second."""
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    @property
    def fps_std(self) -> float:
        """Standard deviation of the per-video frame rate."""
        return float(np.std(self._per_video_fps)) if self._per_video_fps else 0.0

    @property
    def peak_memory_gb(self) -> float:
        """Peak CUDA memory in gibibytes."""
        return self.peak_memory_bytes / (1024**3)

    def as_dict(self) -> dict[str, float]:
        """Return the collected statistics as a plain dictionary."""
        return {
            "frames": float(self.frames),
            "seconds": self.seconds,
            "fps": self.fps,
            "fps_std": self.fps_std,
            "peak_memory_gb": self.peak_memory_gb,
        }


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure and return the package logger.

    Args:
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        log_file: Optional path; when given, logs are mirrored to this file.

    Returns:
        The configured ``iwdd`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    """Return the package logger without reconfiguring it."""
    return logging.getLogger(LOGGER_NAME)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs.

    Args:
        seed: Seed value shared by every random number generator.
        deterministic: When True, request deterministic cuDNN kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch optional
        pass


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string, falling back to CPU when CUDA is unavailable.

    Args:
        requested: ``"auto"``, ``"cuda"``, ``"cuda:0"`` or ``"cpu"``.

    Returns:
        A concrete device string usable by PyTorch.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch optional
        return "cpu"


def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string.

    Args:
        num_bytes: Size in bytes.

    Returns:
        Formatted size such as ``"8.49 MB"``.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

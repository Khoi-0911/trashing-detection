"""Uniform frame sampling from surveillance clips (paper Section 3.2.1).

The paper samples every video at 5 FPS using a stride of ``fps_video / 5``. The
corpus makes that less trivial than it sounds: native frame rates span 6.0 to
59.94 FPS across 20 distinct values in the training split, so the stride is
fractional and integer accumulation would drift by seconds on long clips. This
module accumulates in floating point and rounds only at read time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Iterator

import numpy as np

from .annotations import VideoAnnotation
from .config import SamplingConfig
from .utils import get_logger

LOGGER = get_logger()


@dataclass(frozen=True)
class SampledFrame:
    """A single decoded frame together with its position on the timeline.

    Attributes:
        index: Position within the sampled sequence, starting at 0.
        source_index: Index of the frame in the original video.
        timestamp: Presentation time in seconds from the start of the clip.
        image: Decoded frame as an RGB ``uint8`` array of shape (H, W, 3).
    """

    index: int
    source_index: int
    timestamp: float
    image: np.ndarray


class VideoReader:
    """Context manager around an OpenCV video capture.

    Attributes:
        path: Path to the video file.
        frame_count: Total frames reported by the container, 0 when unknown.
        fps: Frame rate reported by the container.
    """

    def __init__(self, path: str | Path, fallback_fps: float = 0.0) -> None:
        """Open a video file.

        Args:
            path: Path to the ``.mp4`` file.
            fallback_fps: Frame rate to assume when the container reports none.

        Raises:
            ImportError: If OpenCV is not installed.
            IOError: If the file cannot be opened.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("opencv-python is required for video decoding") from exc

        self._cv2 = cv2
        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise IOError(f"Cannot open video: {self.path}")

        reported_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = reported_fps if reported_fps > 0 else float(fallback_fps)
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __enter__(self) -> "VideoReader":
        """Enter the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the capture."""
        self.release()

    def release(self) -> None:
        """Release the underlying capture handle."""
        if self._capture is not None:
            self._capture.release()

    def read_sequential(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield every frame in decode order.

        Sequential decoding is used rather than per-frame seeking because
        ``CAP_PROP_POS_FRAMES`` seeking is unreliable on variable-frame-rate
        files, of which this corpus has many.

        Yields:
            Tuples of ``(source_index, bgr_image)``.
        """
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield index, frame
            index += 1


class FrameSampler:
    """Samples frames from a video at a fixed target rate.

    Attributes:
        config: Sampling settings controlling rate, cap and resizing.
    """

    def __init__(self, config: SamplingConfig | None = None) -> None:
        """Initialise the sampler.

        Args:
            config: Sampling settings; defaults are used when omitted.
        """
        self.config = config or SamplingConfig()
        if self.config.backend != "opencv":
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def sample(self, annotation: VideoAnnotation) -> list[SampledFrame]:
        """Decode and return the sampled frames of one video.

        Args:
            annotation: Record identifying the video and its nominal frame rate.

        Returns:
            Sampled frames in temporal order; empty when decoding fails.
        """
        return list(self.iter_sample(annotation))

    def iter_sample(self, annotation: VideoAnnotation) -> Iterator[SampledFrame]:
        """Lazily yield sampled frames, keeping only one frame in memory.

        Args:
            annotation: Record identifying the video and its nominal frame rate.

        Yields:
            :class:`SampledFrame` instances in temporal order.
        """
        target_fps = float(self.config.target_fps)
        max_frames = int(self.config.max_frames)

        try:
            reader = VideoReader(annotation.video_path, fallback_fps=annotation.frame_rate)
        except (IOError, ImportError) as exc:
            LOGGER.error("%s: %s", annotation.video_id, exc)
            return

        with reader:
            native_fps = reader.fps if reader.fps > 0 else annotation.frame_rate
            if native_fps <= 0:
                LOGGER.error("%s: unknown frame rate, skipping", annotation.video_id)
                return

            # Fractional stride, accumulated in float to avoid cumulative drift.
            stride = max(native_fps / target_fps, 1.0)
            next_source_index = 0.0
            emitted = 0

            for source_index, bgr in reader.read_sequential():
                if source_index < int(round(next_source_index)):
                    continue
                next_source_index += stride

                image = self._to_rgb(bgr)
                image = self._resize(image)
                yield SampledFrame(
                    index=emitted,
                    source_index=source_index,
                    timestamp=source_index / native_fps,
                    image=image,
                )
                emitted += 1
                if max_frames and emitted >= max_frames:
                    LOGGER.debug("%s: hit max_frames=%d", annotation.video_id, max_frames)
                    return

    def expected_num_frames(self, annotation: VideoAnnotation) -> int:
        """Estimate how many frames :meth:`sample` will produce.

        Useful for progress bars and for budgeting captioning time without
        touching the video file.

        Args:
            annotation: Record carrying duration and frame rate.

        Returns:
            Estimated frame count after sampling.
        """
        count = int(annotation.duration * self.config.target_fps)
        if self.config.max_frames:
            count = min(count, self.config.max_frames)
        return max(count, 1)

    def _to_rgb(self, bgr: np.ndarray) -> np.ndarray:
        """Convert an OpenCV BGR frame to RGB.

        Args:
            bgr: Frame in BGR channel order.

        Returns:
            The same frame in RGB channel order.
        """
        return self._cv2_module().cvtColor(bgr, self._cv2_module().COLOR_BGR2RGB)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize so the shorter side matches the configured length.

        Frames are only ever downscaled: the corpus spans 320x240 to 4096x2160
        and upsampling a 320x240 crop adds cost without adding information.

        Args:
            image: RGB frame.

        Returns:
            The resized frame, or the input unchanged when no resize applies.
        """
        target = int(self.config.resize_shorter_side)
        if target <= 0:
            return image

        height, width = image.shape[:2]
        shorter = min(height, width)
        if shorter <= target:
            return image

        scale = target / float(shorter)
        new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
        cv2 = self._cv2_module()
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _cv2_module():
        """Import and return the OpenCV module.

        Returns:
            The imported ``cv2`` module.
        """
        import cv2

        return cv2


def sampled_timestamps(annotation: VideoAnnotation, target_fps: float) -> np.ndarray:
    """Predict the timestamps a sampler would emit, without decoding.

    Args:
        annotation: Record carrying duration and frame rate.
        target_fps: Sampling rate in frames per second.

    Returns:
        A float array of timestamps in seconds.
    """
    native_fps = annotation.frame_rate if annotation.frame_rate > 0 else target_fps
    stride = max(native_fps / target_fps, 1.0)
    total = int(annotation.duration * native_fps)
    indices = np.arange(0, total, stride)
    return indices / native_fps

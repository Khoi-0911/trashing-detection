"""PyTorch datasets for the two input families used by the models.

Sequence models (LSTM, GRU, their CNN hybrids, attention and transformer)
consume cached embedding sequences. The 3D CNN consumes raw pixel clips and so
bypasses the vision-language stage entirely, which is exactly what makes it a
useful control: it answers whether the captioning detour buys anything over
learning spatio-temporal features directly.

Clip lengths vary enormously in this corpus -- 2.9s to 109.5s, i.e. roughly 14
to 548 frames at 5 FPS -- so sequences are padded within a batch and every
model receives an explicit mask. Losses and metrics must respect that mask or
padding will dominate the long tail of short videos.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .annotations import AnnotationStore, VideoAnnotation
from .config import ModelConfig, SamplingConfig
from .features import FeatureCache, FrameFeatures
from .labeling import FrameLabeler
from .sampling import FrameSampler
from .utils import get_logger

LOGGER = get_logger()


@dataclass
class SequenceBatch:
    """A padded batch of embedding sequences.

    Attributes:
        features: Float tensor of shape (B, T, D).
        labels: Float tensor of shape (B, T) with synthesised frame labels.
        mask: Bool tensor of shape (B, T); True marks a real frame.
        timestamps: Float tensor of shape (B, T) in seconds.
        video_labels: Float tensor of shape (B,) with video-level labels.
        onsets: Float tensor of shape (B,); -1 for negative videos.
        video_ids: Source video identifiers, ordered like the batch.
    """

    features: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    timestamps: torch.Tensor
    video_labels: torch.Tensor
    onsets: torch.Tensor
    video_ids: list[str]

    def to(self, device: str | torch.device) -> "SequenceBatch":
        """Move every tensor in the batch to a device.

        Args:
            device: Target device.

        Returns:
            A new batch whose tensors live on ``device``.
        """
        return SequenceBatch(
            features=self.features.to(device),
            labels=self.labels.to(device),
            mask=self.mask.to(device),
            timestamps=self.timestamps.to(device),
            video_labels=self.video_labels.to(device),
            onsets=self.onsets.to(device),
            video_ids=self.video_ids,
        )

    def __len__(self) -> int:
        """Number of videos in the batch."""
        return len(self.video_ids)


class SequenceDataset(Dataset):
    """Serves one cached embedding sequence per video.

    Attributes:
        items: The cached features backing this dataset.
        feature_dim: Width of the frame embeddings.
    """

    def __init__(self, items: Sequence[FrameFeatures]) -> None:
        """Wrap a list of cached features.

        Args:
            items: Cached features, one per video.

        Raises:
            ValueError: If the list is empty or dimensions are inconsistent.
        """
        if not items:
            raise ValueError("SequenceDataset received no items")
        dims = {item.feature_dim for item in items}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent feature dimensions across videos: {sorted(dims)}")
        self.items = list(items)
        self.feature_dim = dims.pop()

    @classmethod
    def from_cache(
        cls,
        cache: FeatureCache,
        store: AnnotationStore,
    ) -> "SequenceDataset":
        """Build a dataset from the cached features of a split.

        Args:
            cache: Feature cache to read from.
            store: Annotations selecting which videos to load.

        Returns:
            A dataset over the cached videos.
        """
        return cls(cache.load_many(a.video_id for a in store))

    def __len__(self) -> int:
        """Number of videos."""
        return len(self.items)

    def __getitem__(self, index: int) -> FrameFeatures:
        """Return the cached features of one video.

        Args:
            index: Positional index.

        Returns:
            The cached features; collation turns these into tensors.
        """
        return self.items[index]

    @property
    def video_labels(self) -> list[int]:
        """Video-level labels, aligned with :attr:`items`."""
        return [item.video_label for item in self.items]

    def subset(self, indices: Sequence[int]) -> "SequenceDataset":
        """Return a dataset restricted to the given positions.

        Args:
            indices: Positional indices to keep.

        Returns:
            A dataset over the selected videos.
        """
        return SequenceDataset([self.items[i] for i in indices])

    def positive_frame_weight(self) -> float:
        """Compute the negative-to-positive frame ratio over the dataset.

        Returns:
            The implied ``w+``, or 1.0 when there are no positive frames.
        """
        total = sum(item.num_frames for item in self.items)
        positives = sum(int(item.frame_labels.sum()) for item in self.items)
        if positives == 0:
            return 1.0
        return float(total - positives) / float(positives)


class ClipDataset(Dataset):
    """Serves fixed-length raw pixel clips for the 3D CNN.

    Each item is a window of ``clip_len`` sampled frames. The window label is
    positive when any frame inside it falls in the synthesised event window,
    which matches how a 3D CNN is normally supervised for action detection.

    Attributes:
        annotations: Videos this dataset draws clips from.
        windows: Flat list of ``(video_position, start_frame)`` pairs.
    """

    def __init__(
        self,
        annotations: Sequence[VideoAnnotation],
        sampling: SamplingConfig,
        model: ModelConfig,
        labeler: FrameLabeler,
        cache_frames: bool = False,
    ) -> None:
        """Index every clip window across a set of videos.

        Args:
            annotations: Videos to draw clips from.
            sampling: Frame sampling settings.
            model: Model settings carrying clip length, size and stride.
            labeler: Expands onsets into frame labels.
            cache_frames: Keep decoded frames in RAM; fast but memory hungry.
        """
        self.annotations = list(annotations)
        self.sampling = sampling
        self.model = model
        self.labeler = labeler
        self.cache_frames = cache_frames
        self._sampler = FrameSampler(sampling)
        # Bounded LRU. Windows of one video are visited consecutively when the
        # loader is not shuffled, so even a small cache turns one decode per
        # window into one decode per video -- the difference between minutes
        # and hours on the 100-video test split.
        self._cache_capacity = max(len(self.annotations) if cache_frames else 2, 1)
        self._frame_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        self.windows: list[tuple[int, int]] = []
        for position, annotation in enumerate(self.annotations):
            num_frames = self._sampler.expected_num_frames(annotation)
            if num_frames < model.clip_len:
                self.windows.append((position, 0))
                continue
            for start in range(0, num_frames - model.clip_len + 1, model.clip_stride):
                self.windows.append((position, start))
        LOGGER.info(
            "ClipDataset: %d windows over %d videos (clip_len=%d, stride=%d)",
            len(self.windows),
            len(self.annotations),
            model.clip_len,
            model.clip_stride,
        )

    def __len__(self) -> int:
        """Number of clip windows."""
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Return one clip and its label.

        Args:
            index: Positional index into :attr:`windows`.

        Returns:
            A dict with ``clip`` of shape (C, T, H, W), ``label``, ``onset``,
            ``start_time`` and ``video_id``.
        """
        video_position, start = self.windows[index]
        annotation = self.annotations[video_position]
        frames, timestamps = self._frames_for(annotation)

        clip_len = self.model.clip_len
        window = frames[start : start + clip_len]
        times = timestamps[start : start + clip_len]
        if window.shape[0] < clip_len:
            pad = clip_len - window.shape[0]
            window = np.concatenate([window, np.repeat(window[-1:], pad, axis=0)], axis=0)
            times = np.concatenate([times, np.repeat(times[-1:], pad)])

        labels = self.labeler.label(annotation, times).values
        clip = torch.from_numpy(window).float().div_(255.0).permute(3, 0, 1, 2).contiguous()

        return {
            "clip": clip,
            "label": torch.tensor(float(labels.max()), dtype=torch.float32),
            "onset": torch.tensor(
                float(annotation.onset) if annotation.onset is not None else -1.0,
                dtype=torch.float32,
            ),
            "start_time": torch.tensor(float(times[0]), dtype=torch.float32),
            "end_time": torch.tensor(float(times[-1]), dtype=torch.float32),
            "video_id": annotation.video_id,
        }

    def _frames_for(self, annotation: VideoAnnotation) -> tuple[np.ndarray, np.ndarray]:
        """Decode and resize every sampled frame of one video.

        Args:
            annotation: The video to decode.

        Returns:
            A tuple of the frame stack (N, H, W, 3) and timestamps (N,).
        """
        cached = self._frame_cache.get(annotation.video_id)
        if cached is not None:
            self._frame_cache.move_to_end(annotation.video_id)
            return cached

        size = self.model.clip_size
        images: list[np.ndarray] = []
        times: list[float] = []
        for frame in self._sampler.iter_sample(annotation):
            images.append(_resize_square(frame.image, size))
            times.append(frame.timestamp)

        if not images:
            stack = np.zeros((self.model.clip_len, size, size, 3), dtype=np.uint8)
            time_array = np.zeros(self.model.clip_len, dtype=np.float32)
        else:
            stack = np.stack(images)
            time_array = np.asarray(times, dtype=np.float32)

        self._frame_cache[annotation.video_id] = (stack, time_array)
        while len(self._frame_cache) > self._cache_capacity:
            self._frame_cache.popitem(last=False)
        return stack, time_array


def collate_sequences(items: Sequence[FrameFeatures]) -> SequenceBatch:
    """Pad a list of variable-length sequences into one batch.

    Args:
        items: Cached features, one per video.

    Returns:
        A padded :class:`SequenceBatch` with a validity mask.
    """
    batch_size = len(items)
    max_len = max(item.num_frames for item in items)
    feature_dim = items[0].feature_dim

    features = torch.zeros(batch_size, max_len, feature_dim, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    timestamps = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for row, item in enumerate(items):
        length = item.num_frames
        features[row, :length] = torch.from_numpy(item.embeddings.astype(np.float32))
        labels[row, :length] = torch.from_numpy(item.frame_labels.astype(np.float32))
        timestamps[row, :length] = torch.from_numpy(item.timestamps.astype(np.float32))
        mask[row, :length] = True

    return SequenceBatch(
        features=features,
        labels=labels,
        mask=mask,
        timestamps=timestamps,
        video_labels=torch.tensor([float(i.video_label) for i in items]),
        onsets=torch.tensor([float(i.onset) for i in items]),
        video_ids=[item.video_id for item in items],
    )


def build_sequence_loader(
    dataset: SequenceDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over an embedding-sequence dataset.

    Args:
        dataset: The dataset to iterate.
        batch_size: Videos per batch.
        shuffle: Shuffle the order each epoch.
        num_workers: Worker processes.

    Returns:
        A configured DataLoader yielding :class:`SequenceBatch` objects.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sequences,
        drop_last=False,
    )


def build_clip_loader(
    dataset: ClipDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over a raw-pixel clip dataset.

    Args:
        dataset: The dataset to iterate.
        batch_size: Clips per batch.
        shuffle: Shuffle the order each epoch.
        num_workers: Worker processes.

    Returns:
        A configured DataLoader yielding dicts of tensors.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def stratified_folds(
    labels: Sequence[int],
    num_folds: int,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Split indices into label-balanced cross-validation folds.

    The corpus is exactly 50/50 positive to negative, and the paper uses 5-fold
    cross-validation on a 50-video subset where a single misallocated video
    shifts F1 by two points. Stratification keeps each fold balanced.

    Asking for more folds than the rarest class has members would leave some
    folds with no validation videos, so ``num_folds`` is capped at that count.
    This matters on small smoke-test subsets: 5 videos split 3/2 cannot support
    5 folds, only 2.

    Args:
        labels: Video-level binary labels.
        num_folds: Requested number of folds.
        seed: Shuffling seed.

    Returns:
        A list of ``(train_indices, validation_indices)`` pairs, none empty.

    Raises:
        ValueError: If fewer than two folds are possible.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    class_sizes = {int(v): int((labels == v).sum()) for v in np.unique(labels)}
    max_folds = min(class_sizes.values()) if class_sizes else 0
    if max_folds < 2:
        raise ValueError(
            f"Cannot cross-validate: class counts are {class_sizes}. "
            f"At least 2 videos per class are required."
        )
    if num_folds > max_folds:
        LOGGER.warning(
            "Reducing num_folds %d -> %d: class counts are %s",
            num_folds,
            max_folds,
            class_sizes,
        )
        num_folds = max_folds

    fold_assignment = np.empty(len(labels), dtype=int)
    for value in np.unique(labels):
        members = np.flatnonzero(labels == value)
        rng.shuffle(members)
        fold_assignment[members] = np.arange(len(members)) % num_folds

    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(num_folds):
        validation = np.flatnonzero(fold_assignment == fold).tolist()
        training = np.flatnonzero(fold_assignment != fold).tolist()
        if validation and training:
            folds.append((training, validation))
    return folds


def _resize_square(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a frame to a square of the given side length.

    Args:
        image: RGB ``uint8`` array of shape (H, W, 3).
        size: Target side length in pixels.

    Returns:
        The resized frame of shape (size, size, 3).
    """
    import cv2

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def discover_cache_dim(cache_root: str | Path) -> int:
    """Read the feature width from the first cached file in a directory.

    Args:
        cache_root: Directory holding ``.npz`` feature files.

    Returns:
        The embedding width, or 0 when the cache is empty.
    """
    for path in sorted(Path(cache_root).glob("*.npz")):
        return FrameFeatures.load(path).feature_dim
    return 0

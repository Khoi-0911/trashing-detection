"""Stage 1-3 orchestration: sample, caption, encode and cache to disk.

Caching is the single most important engineering decision in this pipeline.
The VLM is frozen, so captions and embeddings only ever need to be computed
once. At the throughput the paper reports (0.85 FPS on an A100) a full pass
over the 400 training videos is roughly 14 hours; without a cache, every
architecture and every hyper-parameter sweep would pay that again.

After one extraction pass, training reads only small ``.npz`` arrays and a grid
over all six architectures finishes in minutes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .annotations import AnnotationStore, VideoAnnotation
from .captioning import CaptionSet, PromptSet, build_captioner, get_prompt_set
from .config import ExperimentConfig
from .encoders import EncoderMode, build_encoder
from .labeling import FrameLabeler
from .sampling import FrameSampler
from .utils import ResourceProbe, Timer, get_logger

LOGGER = get_logger()


@dataclass
class FrameFeatures:
    """Cached per-frame features and supervision for one video.

    Attributes:
        video_id: Stem of the source video.
        embeddings: Float32 array of shape (N, D).
        timestamps: Float32 array of shape (N,) in seconds.
        frame_labels: Int64 array of shape (N,) with the synthesised labels.
        video_label: Video-level binary label.
        onset: Ground-truth onset in seconds, or -1 for negatives.
        duration: Clip duration in seconds.
        modality: Disposal modality string.
    """

    video_id: str
    embeddings: np.ndarray
    timestamps: np.ndarray
    frame_labels: np.ndarray
    video_label: int
    onset: float
    duration: float
    modality: str

    @property
    def num_frames(self) -> int:
        """Number of sampled frames."""
        return int(self.embeddings.shape[0])

    @property
    def feature_dim(self) -> int:
        """Width of each frame embedding."""
        return int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def save(self, path: Path) -> None:
        """Write the features to a compressed ``.npz`` file.

        Args:
            path: Destination file; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings.astype(np.float32),
            timestamps=self.timestamps.astype(np.float32),
            frame_labels=self.frame_labels.astype(np.int64),
            video_label=np.int64(self.video_label),
            onset=np.float32(self.onset),
            duration=np.float32(self.duration),
            modality=np.array(self.modality),
            video_id=np.array(self.video_id),
        )

    @classmethod
    def load(cls, path: Path) -> "FrameFeatures":
        """Read features back from a ``.npz`` file.

        Args:
            path: Source file written by :meth:`save`.

        Returns:
            The restored features.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                video_id=str(data["video_id"]),
                embeddings=data["embeddings"],
                timestamps=data["timestamps"],
                frame_labels=data["frame_labels"],
                video_label=int(data["video_label"]),
                onset=float(data["onset"]),
                duration=float(data["duration"]),
                modality=str(data["modality"]),
            )


class FeatureCache:
    """Filesystem cache keyed by the feature-extraction signature.

    Attributes:
        root: Directory holding one ``.npz`` per video plus a manifest.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(self, root: str | Path) -> None:
        """Point the cache at a directory, creating it if needed.

        Args:
            root: Cache directory.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, video_id: str) -> Path:
        """Return the cache file path of one video.

        Args:
            video_id: Stem of the video.

        Returns:
            Path to the ``.npz`` file, which may not exist yet.
        """
        return self.root / f"{video_id}.npz"

    def has(self, video_id: str) -> bool:
        """Check whether a video is already cached.

        Args:
            video_id: Stem of the video.

        Returns:
            True when the cache file exists.
        """
        return self.path_for(video_id).exists()

    def load(self, video_id: str) -> FrameFeatures:
        """Load one cached video.

        Args:
            video_id: Stem of the video.

        Returns:
            The cached features.
        """
        return FrameFeatures.load(self.path_for(video_id))

    def load_many(self, video_ids: Iterable[str]) -> list[FrameFeatures]:
        """Load several cached videos, skipping any that are missing.

        Args:
            video_ids: Stems of the videos to load.

        Returns:
            The features that were present in the cache.
        """
        items: list[FrameFeatures] = []
        for video_id in video_ids:
            if self.has(video_id):
                items.append(self.load(video_id))
            else:
                LOGGER.warning("Feature cache miss: %s", video_id)
        return items

    def write_manifest(self, payload: dict) -> None:
        """Record the settings that produced this cache.

        Args:
            payload: JSON-serialisable description of the extraction run.
        """
        path = self.root / self.MANIFEST_NAME
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def read_manifest(self) -> dict:
        """Read the cache manifest.

        Returns:
            The stored manifest, or an empty dict when absent.
        """
        path = self.root / self.MANIFEST_NAME
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))


class FeatureExtractionPipeline:
    """Runs sampling, captioning, encoding and labelling over a split.

    Attributes:
        config: The full experiment configuration.
        sampler: Frame sampler.
        labeler: Onset-to-frame-label expander.
        captioner: Frozen vision-language model.
        encoder: Caption or vision encoder.
        cache: Destination cache.
        probe: Throughput and memory statistics for the run.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Build every stage from a configuration.

        Args:
            config: The full experiment configuration.
        """
        self.config = config
        self.sampler = FrameSampler(config.sampling)
        self.labeler = FrameLabeler(config.labeling)
        self.prompt_set: PromptSet = get_prompt_set(config.caption.single_prompt)
        self.captioner = build_captioner(config.caption)
        self.encoder = build_encoder(config.encoder, self.captioner)
        self.cache = FeatureCache(Path(config.paths.cache_dir) / config.feature_signature())
        self.caption_dir = Path(config.paths.caption_dir) / config.feature_signature()
        self.probe = ResourceProbe()
        self._vision_only = EncoderMode.parse(config.encoder.mode) is EncoderMode.VISION

    def run(self, store: AnnotationStore, overwrite: bool = False) -> list[str]:
        """Extract features for every annotation in a store.

        Args:
            store: Videos to process.
            overwrite: Recompute videos that are already cached.

        Returns:
            Video ids successfully written to the cache.
        """
        pending = [a for a in store if overwrite or not self.cache.has(a.video_id)]
        LOGGER.info(
            "Feature extraction: %d/%d videos pending in %s",
            len(pending),
            len(store),
            self.cache.root,
        )
        if not pending:
            return [a.video_id for a in store]

        if self.config.caption.save_captions and not self._vision_only:
            self.caption_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []
        for position, annotation in enumerate(pending, start=1):
            try:
                features = self.process_video(annotation)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a 14h run
                LOGGER.exception("%s failed: %s", annotation.video_id, exc)
                continue
            if features is None:
                continue
            features.save(self.cache.path_for(annotation.video_id))
            written.append(annotation.video_id)
            LOGGER.info(
                "[%d/%d] %s: %d frames, dim=%d, pos_frames=%d",
                position,
                len(pending),
                annotation.video_id,
                features.num_frames,
                features.feature_dim,
                int(features.frame_labels.sum()),
            )

        self.probe.refresh_peak_memory()
        self.cache.write_manifest(
            {
                "signature": self.config.feature_signature(),
                "prompt_set": self.prompt_set.name,
                "num_prompts": len(self.prompt_set),
                "caption": self.config.caption.__dict__,
                "encoder": self.config.encoder.__dict__,
                "sampling": self.config.sampling.__dict__,
                "labeling": self.config.labeling.__dict__,
                "resources": self.probe.as_dict(),
                "num_videos": len(written),
            }
        )
        LOGGER.info(
            "Extraction done: %d videos, %.2f fps, peak GPU %.2f GB",
            len(written),
            self.probe.fps,
            self.probe.peak_memory_gb,
        )
        return written

    def process_video(self, annotation: VideoAnnotation) -> FrameFeatures | None:
        """Sample, caption, encode and label a single video.

        Args:
            annotation: The video to process.

        Returns:
            The extracted features, or None when the video yielded no frames.
        """
        with Timer(annotation.video_id) as timer:
            frames = self.sampler.sample(annotation)
            if not frames:
                LOGGER.error("%s: no frames decoded", annotation.video_id)
                return None

            images = [frame.image for frame in frames]
            timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float32)

            caption_sets: list[CaptionSet] = []
            if not self._vision_only:
                caption_sets = self.captioner.caption_frames(
                    images, self.prompt_set, timestamps=timestamps
                )
                if self.config.caption.save_captions:
                    self._write_captions(annotation.video_id, caption_sets)

            embeddings = self.encoder.encode(caption_sets, images, self.prompt_set)

        self.probe.add(frames=len(frames), seconds=timer.elapsed)
        self.probe.refresh_peak_memory()
        self._release_vram()

        labels = self.labeler.label(annotation, timestamps)
        return FrameFeatures(
            video_id=annotation.video_id,
            embeddings=embeddings,
            timestamps=timestamps,
            frame_labels=labels.values,
            video_label=int(annotation.is_positive),
            onset=float(annotation.onset) if annotation.onset is not None else -1.0,
            duration=float(annotation.duration),
            modality=annotation.modality.value,
        )

    @staticmethod
    def _release_vram() -> None:
        """Return cached blocks to CUDA between videos.

        Frame counts vary from 14 to 548, so allocation sizes swing wildly and
        the caching allocator fragments. The Colab OOM reported 3.49 GiB
        reserved but unallocated, which is exactly that.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch optional
            pass

    def _write_captions(self, video_id: str, caption_sets: list[CaptionSet]) -> None:
        """Persist raw captions as JSON Lines for later inspection.

        Args:
            video_id: Stem of the video.
            caption_sets: Captions to write, one record per frame.
        """
        path = self.caption_dir / f"{video_id}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for item in caption_sets:
                handle.write(
                    json.dumps(item.as_dict(), ensure_ascii=False, default=str) + "\n"
                )


def relabel_cache(cache: FeatureCache, store: AnnotationStore, labeler: FrameLabeler) -> int:
    """Rewrite frame labels in an existing cache without re-captioning.

    The positive-window length is a hyper-parameter with no ground truth behind
    it, so sweeping it must not cost another captioning pass. This rewrites the
    label arrays in place and leaves the embeddings untouched.

    Args:
        cache: Cache to update.
        store: Annotations supplying onsets and durations.
        labeler: Labeler carrying the new window settings.

    Returns:
        Number of cache entries updated.
    """
    updated = 0
    for annotation in store:
        if not cache.has(annotation.video_id):
            continue
        features = cache.load(annotation.video_id)
        labels = labeler.label(annotation, features.timestamps)
        features.frame_labels = labels.values
        features.save(cache.path_for(annotation.video_id))
        updated += 1
    LOGGER.info("Relabelled %d cached videos", updated)
    return updated

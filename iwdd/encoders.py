"""Turning per-frame captions into the embedding sequence (paper Equation 6).

The paper concatenates the five captions and encodes the result once with
Sentence-BERT, producing a 384-dimensional vector per frame. Two alternatives
are provided here because the paper never tests them and they change the
accuracy/latency trade-off substantially:

* ``per_aspect`` encodes each caption separately and concatenates the five
  vectors (1920-d). It keeps the aspects linearly separable instead of letting
  Sentence-BERT average them into one sentence-level vector.
* ``vision`` skips text entirely and pools the VLM vision tower. This removes
  the autoregressive generation that accounts for ~90% of the paper runtime,
  which is the only credible route to real-time inference on an edge device.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Sequence

import numpy as np

from .captioning import Aspect, BaseCaptioner, CaptionSet, PromptSet
from .config import EncoderConfig
from .utils import get_logger, resolve_device

LOGGER = get_logger()


class EncoderMode(str, Enum):
    """Available frame-representation strategies.

    Attributes:
        CONCAT_TEXT: Join all captions, encode once (the paper setting).
        PER_ASPECT: Encode each aspect separately, then concatenate.
        VISION: Pool the VLM vision tower and skip captioning.
    """

    CONCAT_TEXT = "concat_text"
    PER_ASPECT = "per_aspect"
    VISION = "vision"

    @classmethod
    def parse(cls, raw: str) -> "EncoderMode":
        """Parse a configuration string into a mode.

        Args:
            raw: Mode name from the configuration.

        Returns:
            The matching mode.

        Raises:
            ValueError: If the name is not recognised.
        """
        for member in cls:
            if member.value == str(raw).lower():
                return member
        raise ValueError(f"Unknown encoder mode: {raw!r}")


class BaseFrameEncoder(ABC):
    """Abstract interface producing one feature vector per frame.

    Attributes:
        config: Encoder settings.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        """Initialise the encoder without loading weights.

        Args:
            config: Encoder settings; defaults are used when omitted.
        """
        self.config = config or EncoderConfig()
        self.device = resolve_device(self.config.device)
        self._loaded = False

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the produced feature vectors."""

    @abstractmethod
    def load(self) -> None:
        """Load any weights the encoder needs."""

    @abstractmethod
    def encode(
        self,
        caption_sets: Sequence[CaptionSet],
        images: Sequence[np.ndarray],
        prompt_set: PromptSet,
    ) -> np.ndarray:
        """Encode one video worth of frames.

        Args:
            caption_sets: Captions per frame; may be empty in vision mode.
            images: Raw RGB frames, used only in vision mode.
            prompt_set: The aspects that were captioned.

        Returns:
            A float32 array of shape (num_frames, :attr:`output_dim`).
        """


class SentenceBertEncoder(BaseFrameEncoder):
    """Sentence-BERT encoder covering both text modes.

    Attributes:
        mode: Whether captions are concatenated or encoded per aspect.
        model: The loaded ``SentenceTransformer``.
    """

    def __init__(self, config: EncoderConfig | None = None) -> None:
        """Initialise the encoder.

        Args:
            config: Encoder settings.
        """
        super().__init__(config)
        self.mode = EncoderMode.parse(self.config.mode)
        if self.mode is EncoderMode.VISION:
            raise ValueError("SentenceBertEncoder cannot serve the vision mode")
        self.model = None
        self._base_dim = 0
        self._num_aspects = 1

    @property
    def output_dim(self) -> int:
        """Feature width, accounting for per-aspect concatenation."""
        if self.mode is EncoderMode.PER_ASPECT:
            return self._base_dim * self._num_aspects
        return self._base_dim

    def load(self) -> None:
        """Load the Sentence-Transformers model."""
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading sentence encoder %s", self.config.model_id)
        self.model = SentenceTransformer(self.config.model_id, device=self.device)
        self._base_dim = int(self.model.get_sentence_embedding_dimension())
        self._loaded = True

    def encode(
        self,
        caption_sets: Sequence[CaptionSet],
        images: Sequence[np.ndarray],
        prompt_set: PromptSet,
    ) -> np.ndarray:
        """Encode captions into per-frame vectors.

        Args:
            caption_sets: Captions per frame.
            images: Unused in this encoder.
            prompt_set: Aspect order used for per-aspect concatenation.

        Returns:
            A float32 array of shape (num_frames, :attr:`output_dim`).
        """
        if not self._loaded:
            self.load()
        self._num_aspects = max(len(prompt_set), 1)

        if not caption_sets:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        if self.mode is EncoderMode.CONCAT_TEXT:
            sentences = [item.concat() for item in caption_sets]
            return self._encode_sentences(sentences)

        aspects = prompt_set.aspects
        flat: list[str] = []
        for item in caption_sets:
            flat.extend(item.ordered(aspects))
        matrix = self._encode_sentences(flat)
        return matrix.reshape(len(caption_sets), len(aspects) * self._base_dim)

    def _encode_sentences(self, sentences: Sequence[str]) -> np.ndarray:
        """Run the sentence encoder over a flat list of strings.

        Args:
            sentences: Text to encode.

        Returns:
            A float32 array of shape (len(sentences), base_dim).
        """
        vectors = self.model.encode(
            list(sentences),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class VisionFeatureEncoder(BaseFrameEncoder):
    """Pools the VLM vision tower, bypassing caption generation.

    Attributes:
        captioner: The frozen VLM whose vision tower is reused.
    """

    def __init__(self, captioner: BaseCaptioner, config: EncoderConfig | None = None) -> None:
        """Initialise the encoder around an existing captioner.

        Args:
            captioner: A loaded or loadable captioning backend.
            config: Encoder settings.
        """
        super().__init__(config)
        self.captioner = captioner
        self._dim = 0

    @property
    def output_dim(self) -> int:
        """Feature width, discovered on the first encode call."""
        return self._dim

    def load(self) -> None:
        """Ensure the underlying VLM is resident."""
        if not self.captioner.is_loaded:
            self.captioner.load()
        self._loaded = True

    def encode(
        self,
        caption_sets: Sequence[CaptionSet],
        images: Sequence[np.ndarray],
        prompt_set: PromptSet,
    ) -> np.ndarray:
        """Encode raw frames into pooled visual features.

        Args:
            caption_sets: Unused in this encoder.
            images: RGB frames to encode.
            prompt_set: Unused in this encoder.

        Returns:
            A float32 array of shape (num_frames, :attr:`output_dim`).
        """
        if not self._loaded:
            self.load()
        if not images:
            return np.zeros((0, max(self._dim, 1)), dtype=np.float32)

        features = self.captioner.embed_images(images)
        if self.config.normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / np.clip(norms, 1e-8, None)
        self._dim = int(features.shape[1])
        return features.astype(np.float32)


def build_encoder(
    config: EncoderConfig,
    captioner: BaseCaptioner | None = None,
) -> BaseFrameEncoder:
    """Instantiate the encoder matching a configuration.

    Args:
        config: Encoder settings naming the mode.
        captioner: Required for vision mode, ignored otherwise.

    Returns:
        An unloaded encoder.

    Raises:
        ValueError: If vision mode is requested without a captioner.
    """
    mode = EncoderMode.parse(config.mode)
    if mode is EncoderMode.VISION:
        if captioner is None:
            raise ValueError("Vision encoder mode requires a captioner instance")
        return VisionFeatureEncoder(captioner, config)
    return SentenceBertEncoder(config)


def aspect_slice(aspect: Aspect, prompt_set: PromptSet, base_dim: int) -> slice:
    """Return the column slice of one aspect in a per-aspect embedding.

    Useful for ablations that drop or inspect individual aspects without
    recomputing the cached features.

    Args:
        aspect: The aspect of interest.
        prompt_set: The prompt set used during encoding.
        base_dim: Width of a single sentence embedding.

    Returns:
        A slice selecting that aspect from the concatenated vector.

    Raises:
        ValueError: If the aspect is not part of the prompt set.
    """
    aspects = prompt_set.aspects
    if aspect not in aspects:
        raise ValueError(f"{aspect} is not part of prompt set {prompt_set.name!r}")
    position = aspects.index(aspect)
    return slice(position * base_dim, (position + 1) * base_dim)

"""Captioner interfaces and the shared Hugging Face chat-template backend.

The vision-language model is frozen throughout: it is an inference-only feature
extractor, never trained. This is why the pipeline needs no caption ground
truth -- the captions are a generated intermediate representation, and the only
supervision used anywhere is the video-level label plus the onset timestamp.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..config import CaptionConfig
from ..utils import get_logger, resolve_device
from .prompts import Aspect, PromptSet

LOGGER = get_logger()

#: Generation classes tried in order when the declared one is unavailable.
#: transformers 5.x removed ``AutoModelForVision2Seq``; 4.x lacks
#: ``AutoModelForImageTextToText`` before 4.45, so both are kept.
FALLBACK_MODEL_CLASSES: tuple[str, ...] = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


@dataclass
class CaptionSet:
    """The captions generated for a single frame.

    Attributes:
        texts: Mapping from aspect to the generated caption.
        frame_index: Position of the frame within the sampled sequence.
        timestamp: Presentation time of the frame in seconds.
    """

    texts: dict[Aspect, str] = field(default_factory=dict)
    frame_index: int = 0
    timestamp: float = 0.0

    def concat(self, separator: str = " ") -> str:
        """Join every aspect caption into one string.

        This reproduces the concatenation inside Equation 6 of the paper, which
        feeds a single Sentence-BERT call.

        Args:
            separator: String inserted between consecutive captions.

        Returns:
            The concatenated caption text.
        """
        return separator.join(self.texts[aspect] for aspect in self.texts)

    def ordered(self, aspects: Sequence[Aspect]) -> list[str]:
        """Return the captions in an explicit aspect order.

        Args:
            aspects: Desired aspect order.

        Returns:
            Caption strings, empty for any missing aspect.
        """
        return [self.texts.get(aspect, "") for aspect in aspects]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "captions": {aspect.value: text for aspect, text in self.texts.items()},
        }


class BaseCaptioner(ABC):
    """Abstract interface every captioning backend implements.

    Attributes:
        config: Captioning settings.
        device: Resolved torch device string.
        model_id: Hugging Face repository id actually in use.
    """

    #: Repository id used when the config leaves ``model_id`` empty.
    default_model_id: str = ""

    def __init__(self, config: CaptionConfig | None = None) -> None:
        """Initialise the captioner without loading weights.

        Args:
            config: Captioning settings; defaults are used when omitted.
        """
        self.config = config or CaptionConfig()
        self.device = resolve_device(self.config.device)
        self.model_id = self.config.model_id or self.default_model_id
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """True once model weights are resident in memory."""
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights onto the configured device."""

    @abstractmethod
    def generate(self, images: Sequence[np.ndarray], prompt: str) -> list[str]:
        """Caption a batch of frames with one shared prompt.

        Args:
            images: RGB ``uint8`` arrays of shape (H, W, 3).
            prompt: Instruction applied to every image in the batch.

        Returns:
            One caption string per input image.
        """

    def unload(self) -> None:
        """Release model weights and free device memory."""
        self._loaded = False

    def caption_frames(
        self,
        images: Sequence[np.ndarray],
        prompt_set: PromptSet,
        timestamps: Sequence[float] | None = None,
    ) -> list[CaptionSet]:
        """Caption a batch of frames across every aspect of a prompt set.

        Prompts are looped on the outside and frames batched on the inside, so
        each aspect is one batched forward pass rather than one call per frame.

        Args:
            images: RGB frames to caption.
            prompt_set: Aspects and their instruction texts.
            timestamps: Optional timestamps aligned with ``images``.

        Returns:
            One :class:`CaptionSet` per input frame.
        """
        if not self._loaded:
            self.load()

        timestamps = list(timestamps) if timestamps is not None else [0.0] * len(images)
        results = [
            CaptionSet(frame_index=index, timestamp=timestamps[index])
            for index in range(len(images))
        ]

        for aspect, prompt in prompt_set.prompts.items():
            captions = self.generate(images, prompt)
            for index, caption in enumerate(captions):
                results[index].texts[aspect] = caption
        return results

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return pooled visual features, bypassing text generation.

        This supports the ``vision`` encoder mode, which skips the caption
        round-trip entirely. Backends that cannot expose a vision tower should
        leave this unimplemented.

        Args:
            images: RGB frames to encode.

        Returns:
            A float32 array of shape (len(images), D).

        Raises:
            NotImplementedError: If the backend has no vision-feature path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose pooled vision features"
        )

    def _torch_dtype(self) -> Any:
        """Resolve the configured dtype name to a torch dtype.

        Returns:
            The matching ``torch.dtype``, defaulting to float32 on CPU.
        """
        import torch

        if self.device == "cpu":
            return torch.float32
        return getattr(torch, self.config.dtype, torch.float32)

    def _quantization_config(self) -> Any | None:
        """Build a bitsandbytes config when quantisation is requested.

        Returns:
            A ``BitsAndBytesConfig`` instance, or None for full precision.
        """
        if not (self.config.load_in_8bit or self.config.load_in_4bit):
            return None
        from transformers import BitsAndBytesConfig

        if self.config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self._torch_dtype(),
            bnb_4bit_quant_type="nf4",
        )


class HFChatCaptioner(BaseCaptioner):
    """Shared implementation for Hugging Face chat-template vision models.

    SmolVLM, Qwen3-VL and Cosmos all expose the same contract: a processor with
    ``apply_chat_template`` and a conditional-generation model. Subclasses only
    declare which model class and repository id to use.

    Attributes:
        processor: The loaded Hugging Face processor.
        model: The loaded generation model.
    """

    #: Name of the ``transformers`` class to instantiate. When the installed
    #: version does not provide it, :data:`FALLBACK_MODEL_CLASSES` is tried.
    model_class_name: str = "AutoModelForImageTextToText"

    def __init__(self, config: CaptionConfig | None = None) -> None:
        """Initialise the backend without loading weights.

        Args:
            config: Captioning settings.
        """
        super().__init__(config)
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the processor and model onto the configured device."""
        import torch
        import transformers

        if not self.model_id:
            raise ValueError(
                f"{type(self).__name__} requires a model_id; set caption.model_id"
            )

        LOGGER.info("Loading %s (%s)", self.model_id, type(self).__name__)
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_id)

        model_cls = self._resolve_model_class()

        quantization = self._quantization_config()
        kwargs: dict[str, Any] = {"dtype": self._torch_dtype()}
        if quantization is not None:
            kwargs["quantization_config"] = quantization
            kwargs["device_map"] = "auto"

        self.model = model_cls.from_pretrained(self.model_id, **kwargs)
        if quantization is None:
            self.model = self.model.to(self.device)
        self.model.eval()

        if getattr(self.processor, "tokenizer", None) is not None:
            tokenizer = self.processor.tokenizer
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Left padding keeps generated continuations aligned across a batch.
            tokenizer.padding_side = "left"

        self._loaded = True
        total = sum(p.numel() for p in self.model.parameters())
        LOGGER.info("Loaded %s with %.1fM parameters", self.model_id, total / 1e6)
        del torch

    def unload(self) -> None:
        """Delete the model and empty the CUDA cache."""
        self.model = None
        self.processor = None
        super().unload()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch optional
            pass

    def generate(self, images: Sequence[np.ndarray], prompt: str) -> list[str]:
        """Caption a batch of frames with one shared prompt.

        Args:
            images: RGB ``uint8`` arrays of shape (H, W, 3).
            prompt: Instruction applied to every image.

        Returns:
            One cleaned caption per input image.
        """
        import torch

        if not images:
            return []
        if not self._loaded:
            self.load()

        pil_images = [self._to_pil(image) for image in images]
        outputs: list[str] = []
        batch_size = max(int(self.config.batch_size), 1)

        for start in range(0, len(pil_images), batch_size):
            chunk = pil_images[start : start + batch_size]
            texts = [self._build_chat_text(prompt) for _ in chunk]
            inputs = self.processor(
                text=texts,
                images=[[image] for image in chunk],
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: self._move(value) for key, value in inputs.items()}

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[1]
            trimmed = generated[:, prompt_len:]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            outputs.extend(self._clean(text) for text in decoded)

        return outputs

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return mean-pooled features from the vision tower.

        Args:
            images: RGB frames to encode.

        Returns:
            A float32 array of shape (len(images), D).

        Raises:
            NotImplementedError: If the model exposes no vision tower.
        """
        import torch

        if not self._loaded:
            self.load()

        tower = self._find_vision_tower()
        if tower is None:
            raise NotImplementedError(
                f"{self.model_id} does not expose a reachable vision tower"
            )

        pil_images = [self._to_pil(image) for image in images]
        vectors: list[np.ndarray] = []
        batch_size = max(int(self.config.batch_size), 1)

        for start in range(0, len(pil_images), batch_size):
            chunk = pil_images[start : start + batch_size]
            processed = self.processor(images=chunk, text=None, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            if pixel_values.dim() == 5:
                # Some processors emit (batch, num_tiles, C, H, W).
                pixel_values = pixel_values.flatten(0, 1)
            pixel_values = pixel_values.to(self.device, dtype=self._torch_dtype())

            with torch.inference_mode():
                hidden = tower(pixel_values).last_hidden_state
            pooled = hidden.mean(dim=1).float().cpu().numpy()
            if pooled.shape[0] != len(chunk):
                pooled = pooled.reshape(len(chunk), -1, pooled.shape[-1]).mean(axis=1)
            vectors.append(pooled)

        return np.concatenate(vectors, axis=0).astype(np.float32)

    def _resolve_model_class(self) -> Any:
        """Find a usable generation class across transformers versions.

        ``AutoModelForVision2Seq`` was the standard class up to transformers 4.x
        but was removed in 5.x in favour of ``AutoModelForImageTextToText``.
        Rather than pin a version, try the declared class first and fall back
        through the known aliases.

        Returns:
            The first ``transformers`` class that exists.

        Raises:
            ImportError: If none of the candidates are available.
        """
        import transformers

        candidates: list[str] = [self.model_class_name]
        candidates += [name for name in FALLBACK_MODEL_CLASSES if name not in candidates]

        for name in candidates:
            model_cls = getattr(transformers, name, None)
            if model_cls is not None:
                if name != self.model_class_name:
                    LOGGER.warning(
                        "transformers %s has no %s; using %s instead",
                        transformers.__version__,
                        self.model_class_name,
                        name,
                    )
                return model_cls

        raise ImportError(
            f"transformers {transformers.__version__} provides none of "
            f"{candidates}. Set caption.model_id and check the model card, "
            f"or pin a compatible transformers version."
        )

    def _build_chat_text(self, prompt: str) -> str:
        """Render one user turn containing an image and an instruction.

        Args:
            prompt: The instruction text.

        Returns:
            The templated prompt string ready for the processor.
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _find_vision_tower(self) -> Any | None:
        """Locate the vision encoder submodule across known layouts.

        Returns:
            The vision tower module, or None when not found.
        """
        candidates = (
            "vision_model",
            "vision_tower",
            "model.vision_model",
            "model.vision_tower",
            "model.visual",
            "visual",
        )
        for name in candidates:
            module: Any = self.model
            for part in name.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                return module
        return None

    def _move(self, value: Any) -> Any:
        """Move a tensor to the model device, leaving other values untouched.

        Args:
            value: A tensor or arbitrary processor output.

        Returns:
            The value on the configured device.
        """
        import torch

        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(self.device, dtype=self._torch_dtype())
            return value.to(self.device)
        return value

    @staticmethod
    def _to_pil(image: np.ndarray) -> Any:
        """Convert an RGB array to a PIL image.

        Args:
            image: RGB ``uint8`` array of shape (H, W, 3).

        Returns:
            The equivalent ``PIL.Image``.
        """
        from PIL import Image

        return Image.fromarray(np.ascontiguousarray(image))

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise whitespace and strip chat-template residue.

        Args:
            text: Raw decoded model output.

        Returns:
            A single-line caption.
        """
        cleaned = " ".join(str(text).split())
        for marker in ("Assistant:", "assistant:", "ASSISTANT:"):
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker) :].strip()
        return cleaned


class DummyCaptioner(BaseCaptioner):
    """Deterministic stand-in that needs no model weights.

    Used to smoke-test the pipeline end to end on a machine without a GPU or
    without the heavy dependencies installed. Captions are derived from trivial
    image statistics so they vary across frames but carry no real semantics.
    """

    default_model_id = "dummy"

    def load(self) -> None:
        """Mark the captioner as ready; nothing to load."""
        self._loaded = True

    def generate(self, images: Sequence[np.ndarray], prompt: str) -> list[str]:
        """Produce a synthetic caption per frame.

        Args:
            images: RGB frames.
            prompt: Instruction text, echoed into the caption.

        Returns:
            One synthetic caption per frame.
        """
        head = prompt.split(".")[0][:40]
        captions = []
        for image in images:
            brightness = float(np.mean(image)) if image.size else 0.0
            captions.append(f"{head} | mean brightness {brightness:.1f}")
        return captions

    def embed_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return a cheap colour-histogram embedding.

        Args:
            images: RGB frames.

        Returns:
            A float32 array of shape (len(images), 48).
        """
        vectors = []
        for image in images:
            channels = [
                np.histogram(image[..., c], bins=16, range=(0, 255))[0]
                for c in range(image.shape[-1])
            ]
            vector = np.concatenate(channels).astype(np.float32)
            total = vector.sum()
            vectors.append(vector / total if total else vector)
        return np.stack(vectors).astype(np.float32)

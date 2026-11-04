"""Concrete vision-language captioning backends.

Each class is a thin declaration on top of :class:`HFChatCaptioner`: it names a
default repository id and the ``transformers`` class to instantiate. Swapping
VLMs is therefore a config change, which is what lets the benchmark sweep the
VLM axis of the paper comparison.

Model sizes, for planning: SmolVLM-256M and SmolVLM-500M are the realistic edge
targets, Qwen3-VL-8B in bf16 needs roughly 17 GB of VRAM, and Cosmos in 8-bit
needs bitsandbytes. Captioning dominates runtime -- the paper measured 1050ms
per frame for five prompts, 89.7% of total processing -- so the backend choice
decides whether a full pass over the corpus takes hours or days.
"""

from __future__ import annotations

from .base import BaseCaptioner, DummyCaptioner, HFChatCaptioner


class SmolVLMCaptioner(HFChatCaptioner):
    """SmolVLM, the lightweight edge-oriented default.

    Available checkpoints range from 256M to 2.2B parameters. The 500M variant
    is the recommended starting point: small enough for consumer hardware, and
    still instruction-tuned enough to follow the five focused prompts.
    """

    default_model_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
    model_class_name = "AutoModelForImageTextToText"


class SmolVLM2Captioner(HFChatCaptioner):
    """SmolVLM2, the video-oriented successor to SmolVLM.

    Used here frame by frame rather than on video input, so that every backend
    sees exactly the same sampled frames and the comparison stays fair.
    """

    default_model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    model_class_name = "AutoModelForImageTextToText"


class Qwen3VLCaptioner(HFChatCaptioner):
    """Qwen3-VL, the highest-capacity backend in this project.

    Expect substantially richer captions than SmolVLM at a large throughput
    cost. Run it in bf16 on a 24 GB card, or enable 4-bit quantisation.
    """

    default_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model_class_name = "Qwen3VLForConditionalGeneration"


class CosmosNanoCaptioner(HFChatCaptioner):
    """NVIDIA Cosmos Reason, intended to run 8-bit quantised.

    The exact repository id depends on which Cosmos release is being targeted,
    so verify ``caption.model_id`` against the model card before a long run
    rather than trusting this default.
    """

    default_model_id = "nvidia/Cosmos-Reason1-7B"
    model_class_name = "AutoModelForImageTextToText"

    def load(self) -> None:
        """Load the model, defaulting to 8-bit when no precision was chosen."""
        if not (self.config.load_in_8bit or self.config.load_in_4bit):
            self.config.load_in_8bit = True
        super().load()


class BlipCaptioner(BaseCaptioner):
    """Original BLIP captioner, the best-performing VLM in the paper.

    BLIP has no chat template: it takes an optional text prefix and continues
    it. The focused prompts are therefore used as conditioning prefixes, which
    is how prompt-conditioned BLIP captioning works in practice. Included as
    the fidelity baseline to compare the edge backends against.
    """

    default_model_id = "Salesforce/blip-image-captioning-large"

    def __init__(self, config=None) -> None:
        """Initialise the backend without loading weights.

        Args:
            config: Captioning settings.
        """
        super().__init__(config)
        self.processor = None
        self.model = None

    def load(self) -> None:
        """Load the BLIP processor and conditional-generation model."""
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_id, dtype=self._torch_dtype()
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    def generate(self, images, prompt):
        """Caption a batch of frames using the prompt as a text prefix.

        Args:
            images: RGB ``uint8`` arrays of shape (H, W, 3).
            prompt: Conditioning prefix applied to every image.

        Returns:
            One caption per input image, with the prefix removed.
        """
        import numpy as np
        import torch
        from PIL import Image

        if not images:
            return []
        if not self._loaded:
            self.load()

        pil_images = [Image.fromarray(np.ascontiguousarray(img)) for img in images]
        captions: list[str] = []
        batch_size = max(int(self.config.batch_size), 1)

        for start in range(0, len(pil_images), batch_size):
            chunk = pil_images[start : start + batch_size]
            inputs = self.processor(
                images=chunk, text=[prompt] * len(chunk), return_tensors="pt", padding=True
            ).to(self.device, self._torch_dtype())
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False
                )
            decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
            for text in decoded:
                cleaned = " ".join(str(text).split())
                if cleaned.lower().startswith(prompt.lower()[:30]):
                    cleaned = cleaned[len(prompt) :].strip() or cleaned
                captions.append(cleaned)

        return captions


#: Registry mapping config keys to captioner classes.
CAPTIONER_REGISTRY: dict[str, type[BaseCaptioner]] = {
    "smolvlm": SmolVLMCaptioner,
    "smolvlm2": SmolVLM2Captioner,
    "qwen3vl": Qwen3VLCaptioner,
    "cosmos_nano": CosmosNanoCaptioner,
    "blip": BlipCaptioner,
    "dummy": DummyCaptioner,
}


def build_captioner(config) -> BaseCaptioner:
    """Instantiate the captioning backend named in a configuration.

    Args:
        config: A :class:`~iwdd.config.CaptionConfig` instance.

    Returns:
        An unloaded captioner ready for :meth:`BaseCaptioner.load`.

    Raises:
        KeyError: If the backend name is not registered.
    """
    key = config.backend.lower()
    if key not in CAPTIONER_REGISTRY:
        raise KeyError(
            f"Unknown captioning backend {config.backend!r}. "
            f"Available: {sorted(CAPTIONER_REGISTRY)}"
        )
    return CAPTIONER_REGISTRY[key](config)


def available_backends() -> list[str]:
    """List the registered captioning backend keys.

    Returns:
        Sorted backend names.
    """
    return sorted(CAPTIONER_REGISTRY)

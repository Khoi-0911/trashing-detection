"""Vision-language captioning: prompts, backends and the shared interface."""

from .backends import (
    CAPTIONER_REGISTRY,
    BlipCaptioner,
    CosmosNanoCaptioner,
    Qwen3VLCaptioner,
    SmolVLM2Captioner,
    SmolVLMCaptioner,
    available_backends,
    build_captioner,
)
from .base import BaseCaptioner, CaptionSet, DummyCaptioner, HFChatCaptioner
from .prompts import (
    MULTI_ASPECT_PROMPTS,
    SINGLE_PROMPT,
    Aspect,
    PromptSet,
    get_prompt_set,
)

__all__ = [
    "Aspect",
    "BaseCaptioner",
    "BlipCaptioner",
    "CAPTIONER_REGISTRY",
    "CaptionSet",
    "CosmosNanoCaptioner",
    "DummyCaptioner",
    "HFChatCaptioner",
    "MULTI_ASPECT_PROMPTS",
    "PromptSet",
    "Qwen3VLCaptioner",
    "SINGLE_PROMPT",
    "SmolVLM2Captioner",
    "SmolVLMCaptioner",
    "available_backends",
    "build_captioner",
    "get_prompt_set",
]

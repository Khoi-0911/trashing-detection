"""The multi-aspect prompt set from the reference paper (Section 3.2.2).

The five prompts are reproduced verbatim. They decompose scene understanding
into focused categories so that the sentence embedding carries explicit signal
about actions and waste rather than a single generic description.

The paper reports in Table 6 that this decomposition is what lifts CNN+GRU from
F1 0.5397 to 0.7429, while costing roughly 5x more inference than one prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Aspect(str, Enum):
    """The five semantic aspects captioned per frame.

    Attributes:
        SCENE: Overall scene description.
        ACTIONS: What people are physically doing.
        OBJECTS: What people hold, carry or touch.
        WASTE: Presence and appearance of waste-related items.
        DUMPING: Whether a disposal action is actively happening.
    """

    SCENE = "scene"
    ACTIONS = "actions"
    OBJECTS = "objects"
    WASTE = "waste"
    DUMPING = "dumping"


@dataclass(frozen=True)
class PromptSet:
    """An ordered collection of aspect prompts.

    Attributes:
        prompts: Mapping from aspect to its instruction text.
        name: Identifier used in cache keys and reports.
    """

    prompts: dict[Aspect, str]
    name: str = "multi_aspect"

    @property
    def aspects(self) -> list[Aspect]:
        """The aspects in their canonical order."""
        return list(self.prompts.keys())

    @property
    def texts(self) -> list[str]:
        """The prompt strings in their canonical order."""
        return [self.prompts[aspect] for aspect in self.prompts]

    def __len__(self) -> int:
        """Number of prompts in the set."""
        return len(self.prompts)


MULTI_ASPECT_PROMPTS = PromptSet(
    name="multi_aspect",
    prompts={
        Aspect.SCENE: "Describe this scene in one detailed sentence.",
        Aspect.ACTIONS: (
            "What specific actions are people performing? Describe their movements, "
            "gestures, and what they are doing with their hands and body."
        ),
        Aspect.OBJECTS: (
            "What objects, items, or materials are people holding, carrying, touching, "
            "or interacting with? Be specific."
        ),
        Aspect.WASTE: (
            "Look carefully: is there any trash, waste, garbage bags, bins, or "
            "waste-related items visible? Describe what you see."
        ),
        Aspect.DUMPING: (
            "Critical: Is anyone actively throwing, dumping, tossing, dropping, or "
            "disposing of any items or materials? If yes, describe exactly what they "
            "are doing and what they are dumping. If no, say "
            "'No dumping activity observed'."
        ),
    },
)
"""The five focused prompts used by the multi-aspect configuration."""


SINGLE_PROMPT = PromptSet(
    name="single",
    prompts={Aspect.SCENE: "a photo of"},
)
"""The generic single-prompt baseline used in the paper ablation (Table 6)."""


def get_prompt_set(single_prompt: bool = False) -> PromptSet:
    """Select the prompt set for a run.

    Args:
        single_prompt: When True, return the generic one-prompt baseline.

    Returns:
        The selected prompt set.
    """
    return SINGLE_PROMPT if single_prompt else MULTI_ASPECT_PROMPTS

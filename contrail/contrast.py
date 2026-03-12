"""
ContrastGenerator — generates semantically targeted contrast variants for a given input.

Produces 3 variants via a single structured LLM call:
  - similar:  same intent, different surface form
  - neutral:  urgency/sentiment stripped, core intent preserved
  - opposite: semantic inversion of the key dimension
"""
from __future__ import annotations

from dataclasses import dataclass


class ContrastGenerationError(Exception):
    """Raised when the LLM returns an unparseable or invalid contrast response."""


@dataclass
class ContrastSet:
    """Three semantically targeted contrast variants for a given input."""

    similar: str
    neutral: str
    opposite: str

    def as_list(self) -> list[str]:
        """Return variants as an ordered list: [similar, neutral, opposite]."""
        return [self.similar, self.neutral, self.opposite]

    def all_non_empty(self) -> bool:
        return all(len(v.strip()) >= 5 for v in self.as_list())


class ContrastGenerator:
    """
    Generates 3 contrast variants for any text input using a cheap LLM.

    Uses a single structured JSON call — all 3 variants in one response.
    The model used here is intentionally separate from the model being traced.

    Args:
        model: LLM model ID for contrast generation.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._client = None  # lazy-initialised in generate()

    async def generate(self, input_text: str) -> ContrastSet:
        """
        Generate 3 contrast variants for the given input text.

        Args:
            input_text: The original agent input to contrast.

        Returns:
            ContrastSet with similar, neutral, and opposite variants.

        Raises:
            ContrastGenerationError: If the LLM response cannot be parsed.
        """
        raise NotImplementedError("Phase 2")

    def _load_prompt_template(self) -> str:
        """Load the versioned prompt template from prompts/contrast_gen.txt."""
        raise NotImplementedError("Phase 2")

    def _parse_response(self, raw: str) -> ContrastSet:
        """Parse the LLM JSON response into a ContrastSet."""
        raise NotImplementedError("Phase 2")

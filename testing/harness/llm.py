"""
Provider-agnostic LLM factory for the Conntrail testing environment.

Detects available API keys and returns the appropriate ChatModel.
Priority: GROQ → ANTHROPIC → OPENAI

Usage:
    from testing.harness.llm import get_llm
    llm = get_llm()           # auto-detect
    llm = get_llm("groq")     # force provider
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Auto-load testing/.env if it exists — never required, never errors if missing
load_dotenv(Path(__file__).parent.parent / ".env")

Provider = Literal["groq", "anthropic", "openai"]

# Default models per provider — using 8b-instant to stay within Groq free-tier
# TPD limits (100K/day for 70b; much higher for 8b). The analyser runs each
# routing node 4× for contrast analysis, so token budget matters.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}

# Cheap/fast model for contrast generation per provider
CONTRAST_MODELS: dict[str, str] = {
    "groq": "llama-3.1-8b-instant",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}


def detect_provider() -> Provider:
    """Return the first provider with an available API key."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise EnvironmentError(
        "No LLM API key found. Set one of: GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY"
    )


def get_llm(
    provider: Provider | None = None,
    *,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
):
    """
    Return a LangChain ChatModel for the given provider.

    Args:
        provider: "groq", "anthropic", or "openai". Auto-detected if None.
        model: Override the default model for this provider.
        max_tokens: Maximum tokens in the response.
        temperature: Sampling temperature. Defaults to 0.0 for deterministic
            routing decisions — critical for stable entropy calibration tests.
    """
    resolved = provider or detect_provider()
    resolved_model = model or DEFAULT_MODELS[resolved]

    if resolved == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=resolved_model,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if resolved == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=resolved_model,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    if resolved == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=resolved_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(f"Unknown provider: {resolved!r}")


def get_contrast_llm(provider: Provider | None = None):
    """Return the cheapest/fastest model for contrast generation."""
    resolved = provider or detect_provider()
    return get_llm(resolved, model=CONTRAST_MODELS[resolved], max_tokens=512)

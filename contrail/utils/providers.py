"""
Provider resolution — maps model strings to LangChain chat model instances.

Supports Groq, Anthropic, and OpenAI. Provider is inferred from the model
name prefix. Falls back to whichever API key is available if the preferred
provider's key is missing.
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Model name prefixes → provider
_PREFIX_MAP = {
    "llama": "groq",
    "mixtral": "groq",
    "gemma": "groq",
    "qwen": "groq",
    "deepseek": "groq",
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
}

# Cheapest model per provider (used when falling back)
_FALLBACK_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}

_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _infer_provider(model: str) -> str | None:
    prefix = model.split("-")[0].lower()
    return _PREFIX_MAP.get(prefix)


def _available_provider() -> str:
    """Return the first provider with an available API key."""
    for provider in ("groq", "anthropic", "openai"):
        if os.getenv(_KEY_ENV[provider]):
            return provider
    raise EnvironmentError(
        "No LLM API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
    )


def get_chat_model(model: str, *, max_tokens: int = 512) -> BaseChatModel:
    """
    Resolve a model string to a LangChain BaseChatModel instance.

    Provider is inferred from the model name prefix. If the inferred
    provider's API key is not set, falls back to the first available
    provider and uses its default cheap model.

    Args:
        model: Model ID string e.g. "llama-3.1-8b-instant", "claude-haiku-4-5-20251001"
        max_tokens: Max tokens for the response.

    Returns:
        A configured LangChain BaseChatModel.
    """
    inferred = _infer_provider(model)

    # Check if inferred provider's key is available
    if inferred and os.getenv(_KEY_ENV[inferred]):
        resolved_provider = inferred
        resolved_model = model
    else:
        # Fall back to whatever key is available
        resolved_provider = _available_provider()
        resolved_model = _FALLBACK_MODELS[resolved_provider]

    return _build_model(resolved_provider, resolved_model, max_tokens=max_tokens)


def _build_model(provider: str, model: str, *, max_tokens: int) -> BaseChatModel:
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, max_tokens=max_tokens)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=max_tokens)  # type: ignore[call-arg]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=max_tokens)

    raise ValueError(f"Unknown provider: {provider!r}")

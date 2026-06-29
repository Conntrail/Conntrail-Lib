"""
Provider resolution — maps model strings to LangChain chat model instances.

Supports Groq, Anthropic, OpenAI, and local OpenAI-compatible servers
(vllm, unsloth, llama.cpp, etc.).

Local model usage:
    Pass model="local/<name>" or model="local" to route to the local server.
    Server URL is read from LOCAL_LLM_URL (default: http://127.0.0.1:8888/v1).
    Model name sent to the server is the part after "local/" or LOCAL_MODEL_NAME env var.
"""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Default local server (Unsloth Studio OpenAI-compatible endpoint)
_LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8888/v1")
_LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "unsloth/Qwen3.6-27B-MTP-GGUF")
_LOCAL_USERNAME = os.getenv("LOCAL_USERNAME", "unsloth")
_LOCAL_PASSWORD = os.getenv("LOCAL_PASSWORD", "")

def _get_local_token() -> str:
    """Exchange password for a fresh JWT via Unsloth Studio's login endpoint.

    Never cached — always re-authenticates so a server restart or session
    invalidation never causes a run to fail with stale 401 errors.
    """
    import urllib.request, urllib.error, json as _json

    base = _LOCAL_LLM_URL.rstrip("/v1").rstrip("/")
    password = os.getenv("LOCAL_PASSWORD") or _LOCAL_PASSWORD
    if not password:
        raise EnvironmentError(
            "LOCAL_PASSWORD env var is required for Unsloth Studio authentication."
        )

    payload = _json.dumps({"username": _LOCAL_USERNAME, "password": password}).encode()
    req = urllib.request.Request(
        f"{base}/api/auth/login",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise EnvironmentError(f"Unsloth Studio login failed ({e.code}): {e.read().decode()}") from e

    token = data.get("access_token")
    if not token:
        raise EnvironmentError(f"No access_token in login response: {data}")
    return token

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
    "local": _LOCAL_MODEL_NAME,
}

_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _infer_provider(model: str) -> str | None:
    # "local" or "local/<model-name>" always routes to local server
    if model == "local" or model.startswith("local/"):
        return "local"
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


def get_chat_model(
    model: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> BaseChatModel:
    """
    Resolve a model string to a LangChain BaseChatModel instance.

    Special prefix "local" or "local/<name>" routes to the local OpenAI-compatible
    server at LOCAL_LLM_URL (default: http://127.0.0.1:8888/v1).

    For cloud providers, provider is inferred from the model name prefix. If the
    inferred provider's API key is not set, falls back to the first available
    provider and its default cheap model.

    Args:
        model: Model ID string. Use "local/qwen3" for local inference.
        max_tokens: Max tokens for the response.
        temperature: Sampling temperature.

    Returns:
        A configured LangChain BaseChatModel.
    """
    inferred = _infer_provider(model)

    if inferred == "local":
        # Extract model name from "local/<name>" or fall back to LOCAL_MODEL_NAME
        local_name = model.split("/", 1)[1] if "/" in model else _LOCAL_MODEL_NAME
        return _build_model("local", local_name, max_tokens=max_tokens, temperature=temperature)

    # Check if inferred cloud provider's key is available
    if inferred and os.getenv(_KEY_ENV[inferred]):
        resolved_provider = inferred
        resolved_model = model
    else:
        resolved_provider = _available_provider()
        resolved_model = _FALLBACK_MODELS[resolved_provider]

    return _build_model(resolved_provider, resolved_model, max_tokens=max_tokens, temperature=temperature)


def _build_model(provider: str, model: str, *, max_tokens: int, temperature: float = 0.0) -> BaseChatModel:
    if provider == "local":
        from langchain_openai import ChatOpenAI
        token = _get_local_token()
        return ChatOpenAI(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=_LOCAL_LLM_URL,
            api_key=token,
            streaming=True,  # Unsloth Studio always returns SSE regardless of stream flag
            extra_body={"enable_thinking": False},  # disable Qwen3 chain-of-thought
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, max_tokens=max_tokens, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=max_tokens, temperature=temperature)  # type: ignore[call-arg]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=max_tokens, temperature=temperature)

    raise ValueError(f"Unknown provider: {provider!r}")

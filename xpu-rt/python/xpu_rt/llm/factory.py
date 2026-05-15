"""Factory helpers for selecting a CompGen LLM backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from compgen.llm._env import resolve_api_key
from compgen.llm.anthropic_client import AnthropicClient
from compgen.llm.base import CompGenLLMProtocol
from compgen.llm.cli_client import ClaudeCLIClient, CodexCLIClient
from compgen.llm.gemini_client import GeminiClient
from compgen.llm.openai_client import OpenAIClient

SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "gemini",
    "openai",
    "anthropic",
    "claude-cli",
    "codex-cli",
)

_ALIASES = {
    "gemmini": "gemini",
    "claude": "claude-cli",
    "claude_code": "claude-cli",
    "codex": "codex-cli",
}

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-5",
    "claude-cli": "sonnet",
    "codex-cli": "gpt-5.4-mini",
}


def create_llm_client(
    provider: str | None = None,
    *,
    model: str | None = None,
    working_dir: str | Path | None = None,
) -> CompGenLLMProtocol:
    """Create an LLM client from an explicit provider or environment defaults."""
    selected = resolve_provider_name(provider)
    cwd = Path(working_dir) if working_dir is not None else None

    if selected == "gemini":
        return GeminiClient(model=model or default_model_for_provider(selected))
    if selected == "openai":
        return OpenAIClient(model=model or default_model_for_provider(selected))
    if selected == "anthropic":
        return AnthropicClient(model=model or default_model_for_provider(selected))
    if selected == "claude-cli":
        return ClaudeCLIClient(model=model or default_model_for_provider(selected), working_dir=cwd)
    if selected == "codex-cli":
        return CodexCLIClient(model=model or default_model_for_provider(selected), working_dir=cwd)
    raise ValueError(f"Unknown LLM provider '{selected}'.")


def resolve_provider_name(provider: str | None = None) -> str:
    """Resolve an explicit or implicit provider name to a canonical backend id."""
    raw = (provider or os.environ.get("COMPGEN_LLM_BACKEND") or _detect_default_provider()).strip().lower()
    normalized = _ALIASES.get(raw, raw)
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{raw}'.")
    return normalized


def default_model_for_provider(provider: str) -> str:
    """Return the default model alias for a canonical provider id."""
    canonical = resolve_provider_name(provider)
    return _DEFAULT_MODELS[canonical]


def provider_transport(provider: str) -> str:
    """Return whether the provider is API-backed or CLI-backed."""
    canonical = resolve_provider_name(provider)
    return "cli" if canonical.endswith("-cli") else "api"


def _detect_default_provider() -> str:
    if resolve_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMMINI_API"):
        return "gemini"
    if resolve_api_key("OPENAI_API_KEY"):
        return "openai"
    if resolve_api_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    if shutil.which("claude"):
        return "claude-cli"
    if shutil.which("codex"):
        return "codex-cli"
    return "gemini"


__all__ = [
    "SUPPORTED_PROVIDERS",
    "create_llm_client",
    "default_model_for_provider",
    "provider_transport",
    "resolve_provider_name",
]

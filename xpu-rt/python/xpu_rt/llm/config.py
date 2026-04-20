"""High-level LLM selection and runtime-building helpers."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from compgen.llm._env import load_dotenv_map
from compgen.llm.base import CompGenLLMProtocol
from compgen.llm.factory import (
    SUPPORTED_PROVIDERS,
    create_llm_client,
    default_model_for_provider,
    provider_transport,
    resolve_provider_name,
)
from compgen.llm.recorder import LLMRecorder


@dataclass(frozen=True)
class LLMSelection:
    """Resolved project-level LLM configuration."""

    provider: str
    model: str
    record: bool
    record_dir: Path
    source: str
    transport: str


def resolve_llm_selection(
    provider: str | None = None,
    *,
    model: str | None = None,
    record: bool | None = None,
    record_dir: str | Path | None = None,
) -> LLMSelection:
    """Resolve global LLM configuration from CLI overrides, env, and defaults."""
    source = (
        "cli"
        if provider or model or record is not None or record_dir is not None
        else ("env" if os.environ.get("COMPGEN_LLM_BACKEND") or os.environ.get("COMPGEN_LLM_MODEL") else "auto")
    )
    selected_provider = resolve_provider_name(provider)
    selected_model = model or os.environ.get("COMPGEN_LLM_MODEL") or default_model_for_provider(selected_provider)
    selected_record = (
        record
        if record is not None
        else os.environ.get("COMPGEN_LLM_NO_RECORD", "")
        not in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
    )
    selected_record_dir = Path(record_dir or os.environ.get("COMPGEN_LLM_RECORD_DIR") or ".compgen_cache/llm_logs")
    return LLMSelection(
        provider=selected_provider,
        model=selected_model,
        record=selected_record,
        record_dir=selected_record_dir,
        source=source,
        transport=provider_transport(selected_provider),
    )


def build_llm_runtime(
    selection: LLMSelection,
    *,
    working_dir: str | Path | None = None,
) -> CompGenLLMProtocol:
    """Instantiate the selected client and wrap it with recording if enabled."""
    client = create_llm_client(selection.provider, model=selection.model, working_dir=working_dir)
    if selection.record:
        return LLMRecorder(client, log_dir=selection.record_dir)
    return client


def apply_selection_to_env(selection: LLMSelection) -> None:
    """Mirror a resolved selection into environment variables for downstream code."""
    os.environ["COMPGEN_LLM_BACKEND"] = selection.provider
    os.environ["COMPGEN_LLM_MODEL"] = selection.model
    os.environ["COMPGEN_LLM_RECORD_DIR"] = str(selection.record_dir)
    if selection.record:
        os.environ.pop("COMPGEN_LLM_NO_RECORD", None)
    else:
        os.environ["COMPGEN_LLM_NO_RECORD"] = "1"


def selection_status(selection: LLMSelection) -> dict[str, str]:
    """Return a lightweight readiness summary for the selected backend."""
    env_values = load_dotenv_map()
    if selection.provider == "gemini":
        available = any(
            env_values.get(name) or os.environ.get(name)
            for name in (
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "GEMMINI_API",
            )
        )
        detail = "API key present" if available else "Missing GOOGLE_API_KEY / GEMINI_API_KEY / GEMMINI_API"
    elif selection.provider == "openai":
        available = bool(env_values.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        detail = "API key present" if available else "Missing OPENAI_API_KEY"
    elif selection.provider == "anthropic":
        available = bool(env_values.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
        detail = "API key present" if available else "Missing ANTHROPIC_API_KEY"
    elif selection.provider == "claude-cli":
        path = shutil.which("claude")
        available = path is not None
        detail = path or "Claude CLI not found on PATH"
    elif selection.provider == "codex-cli":
        path = shutil.which("codex")
        available = path is not None
        detail = path or "Codex CLI not found on PATH"
    else:
        available = False
        detail = "Unknown provider"

    return {
        "provider": selection.provider,
        "model": selection.model,
        "transport": selection.transport,
        "source": selection.source,
        "recording": "enabled" if selection.record else "disabled",
        "record_dir": str(selection.record_dir),
        "available": "yes" if available else "no",
        "detail": detail,
    }


__all__ = [
    "LLMSelection",
    "SUPPORTED_PROVIDERS",
    "apply_selection_to_env",
    "build_llm_runtime",
    "resolve_llm_selection",
    "selection_status",
]

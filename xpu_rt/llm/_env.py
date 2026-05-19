"""Shared environment and .env helpers for LLM providers."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_map() -> dict[str, str]:
    """Load simple ``KEY=VALUE`` entries from the repo root ``.env``."""

    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def resolve_api_key(*candidate_names: str) -> str:
    """Resolve an API key from the environment or ``.env``.

    The first non-empty match wins and is mirrored back into ``os.environ``.
    """

    for name in candidate_names:
        value = os.environ.get(name, "")
        if value:
            return value

    dotenv_values = load_dotenv_map()
    for name in candidate_names:
        value = dotenv_values.get(name, "")
        if value:
            os.environ[name] = value
            return value
    return ""


__all__ = ["load_dotenv_map", "resolve_api_key"]

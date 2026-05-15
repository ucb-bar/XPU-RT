"""JSON schema definitions for the graph compilation artifact contract.

Schemas are versioned under ``v1/``. The :func:`load_schema` helper
returns the parsed JSON Schema for a given (version, name) pair.

These schemas are the authoritative on-disk contract. The dataclasses
in :mod:`compgen.graph_compilation.artifacts` mirror them; T18 in the test
suite enforces that the dataclass-derived shapes round-trip against
these schemas.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_schema(name: str, *, version: str = "v1") -> dict[str, Any]:
    """Load a versioned schema by name.

    ``name`` is the schema stem (e.g. ``"run_manifest"``). Raises
    ``FileNotFoundError`` if the schema is unknown.
    """
    pkg = f"compgen.graph_compilation.schemas.{version}"
    fname = f"{name}.schema.json"
    try:
        text = resources.files(pkg).joinpath(fname).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"unknown schema: {pkg}/{fname}") from exc
    parsed: dict[str, Any] = json.loads(text)
    return parsed


__all__ = ["load_schema"]

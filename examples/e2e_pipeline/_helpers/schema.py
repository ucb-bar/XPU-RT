"""JSON Schema loading and validation helpers.

Every script that writes a structured artifact routes its output through
`validate_or_raise(payload, schema_name)` to prove the artifact matches the
frozen schema before landing on disk.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"


@lru_cache(maxsize=32)
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by bare name (e.g., "kernel_contract").

    Args:
        name: Schema stem with or without trailing `.schema.yaml`.

    Returns:
        Parsed schema dict.
    """
    stem = name.removesuffix(".schema.yaml")
    path = SCHEMAS_DIR / f"{stem}.schema.yaml"
    if not path.exists():
        raise FileNotFoundError(f"schema not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=32)
def get_validator(name: str) -> Draft202012Validator:
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_or_raise(payload: dict[str, Any], schema_name: str) -> None:
    """Validate `payload` against schema; raise with aggregated errors."""
    validator = get_validator(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        msg_lines = [f"{schema_name} validation failed ({len(errors)} errors):"]
        for err in errors:
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            msg_lines.append(f"  at {loc}: {err.message}")
        raise ValueError("\n".join(msg_lines))

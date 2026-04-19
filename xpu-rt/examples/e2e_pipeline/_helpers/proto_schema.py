"""Prototype-schema loader. Companion to _helpers/schema.py.

Loads schemas from `user_perspective/prototypes/schemas/` for v2 drafts
and new artifact types (recipe_semantic_global, execution_runtime).
Keeps the existing v1 loader in `_helpers/schema.py` untouched so
Phase 0–5 scripts continue to use v1 unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

PROTO_SCHEMAS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "prototypes" / "schemas"
)


@lru_cache(maxsize=32)
def load_proto_schema(name: str) -> dict[str, Any]:
    """Load a v2 / prototype schema by bare name.

    Accepts:
        "target_resource.v2"          -> target_resource.v2.schema.yaml
        "kernel_contract.v2"          -> kernel_contract.v2.schema.yaml
        "recipe_semantic_global"      -> recipe_semantic_global.schema.yaml
        "execution_runtime"           -> execution_runtime.schema.yaml
    """
    stem = name.removesuffix(".schema.yaml")
    path = PROTO_SCHEMAS_DIR / f"{stem}.schema.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prototype schema not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=32)
def get_proto_validator(name: str) -> Draft202012Validator:
    schema = load_proto_schema(name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_or_raise_proto(payload: dict[str, Any], schema_name: str) -> None:
    """Validate payload against a prototype schema; raise with aggregated errors."""
    validator = get_proto_validator(schema_name)
    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if errors:
        msg_lines = [f"{schema_name} validation failed ({len(errors)} errors):"]
        for err in errors:
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            msg_lines.append(f"  at {loc}: {err.message}")
        raise ValueError("\n".join(msg_lines))

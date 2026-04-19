"""CompGen artifact JSON schemas.

v1 schemas are embedded resources under ``schemas/v1/``. Access via
:func:`load_schema` or :func:`schema_path`.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

_V1_DIR = "v1"
_V1_FILES = {
    "kernel_contract": "kernel_contract.schema.yaml",
    "recipe_ir": "recipe_ir.schema.yaml",
    "execution_plan": "execution_plan.schema.yaml",
    "target_resource": "target_resource.schema.yaml",
    "region_analysis": "region_analysis.schema.yaml",
}

__all__ = ["available_schemas", "load_schema", "schema_path"]


def available_schemas(version: str = "v1") -> list[str]:
    if version != "v1":
        raise ValueError(f"unknown schema version: {version}")
    return sorted(_V1_FILES)


def schema_path(name: str, version: str = "v1") -> Path:
    if version != "v1":
        raise ValueError(f"unknown schema version: {version}")
    if name not in _V1_FILES:
        raise KeyError(f"no schema named {name!r}; see available_schemas()")
    with resources.as_file(resources.files(__name__).joinpath(_V1_DIR, _V1_FILES[name])) as p:
        return Path(p)


def load_schema(name: str, version: str = "v1") -> dict[str, Any]:
    return yaml.safe_load(schema_path(name, version).read_text())

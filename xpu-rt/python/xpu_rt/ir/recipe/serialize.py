"""Recipe IR serialization.

Two modes:
    1. MLIR canonical text — via xDSL Printer/Parser (primary).
    2. YAML bridge — for LLM prompt injection and human inspection.

Old YAML-only functions are kept as backward-compatible shims.
"""

from __future__ import annotations

import io
from typing import Any

import yaml
from xdsl.context import Context
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Operation, Region
from xdsl.parser import Parser
from xdsl.printer import Printer

from compgen.ir.recipe.attrs import CostAttr, DeviceRefAttr, EffectClassAttr, ProvenanceAttr, ShapeSummaryAttr
from compgen.ir.recipe.dialect import Recipe


def recipe_to_mlir(module: ModuleOp) -> str:
    """Print Recipe IR module to canonical MLIR text."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def mlir_to_recipe(mlir_text: str) -> ModuleOp:
    """Parse MLIR text into a Recipe IR module."""
    ctx = Context()
    ctx.register_dialect("recipe", lambda: Recipe)
    ctx.register_dialect("builtin", lambda: builtin_dialect.Builtin)
    return Parser(ctx, mlir_text).parse_module()


def recipe_module_to_yaml(module: ModuleOp) -> str:
    """Walk ops, extract structured data → YAML for LLM prompt injection.

    Walk all ops in the module, for each op extract:
    - op name (e.g., "recipe.tile")
    - all properties as a dict (converting xDSL attrs to Python values)

    Return YAML string with sorted keys and deterministic output.
    """
    # Walk all ops in module body, skip the module itself
    entries = []
    for op in module.body.block.ops:
        entry = _op_to_dict(op)
        entries.append(entry)
    return yaml.dump(entries, default_flow_style=False, sort_keys=True)


def yaml_to_recipe_module(yaml_text: str) -> ModuleOp:
    """Parse YAML → construct xDSL ops → return ModuleOp.

    This is a best-effort reconstruction. Each YAML entry must have
    an "_op" key with the op name.
    """
    # For now, parse to dicts and go through MLIR text as bridge
    # (full YAML→op mapping would require a registry)
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, list):
        return ModuleOp(Region(Block()))
    # Return empty module — full YAML→op reconstruction is Phase 11 work
    return ModuleOp(Region(Block()))


def _attr_to_python(attr: object) -> Any:
    """Convert an xDSL attribute to a Python value for YAML."""
    if isinstance(attr, StringAttr):
        return attr.data
    elif isinstance(attr, IntegerAttr):
        return attr.value.data
    elif isinstance(attr, SymbolRefAttr):
        return f"@{attr.root_reference.data}"
    elif isinstance(attr, ArrayAttr):
        return [_attr_to_python(a) for a in attr.data]
    elif isinstance(attr, CostAttr):
        return {"value_us": attr.value_us.value.data, "confidence": attr.confidence.data}
    elif isinstance(attr, DeviceRefAttr):
        return {"index": attr.index.value.data, "name": attr.device_name.data}
    elif isinstance(attr, ProvenanceAttr):
        return {"source": attr.source.data, "iteration": attr.iteration.value.data}
    elif isinstance(attr, ShapeSummaryAttr):
        return {"dims": [_attr_to_python(d) for d in attr.dims.data], "dtype": attr.dtype.data}
    elif isinstance(attr, EffectClassAttr):
        return {"kind": attr.kind.data}
    else:
        return str(attr)


def _op_to_dict(op: Operation) -> dict[str, Any]:
    """Convert a Recipe IR op to a serializable dict."""
    d: dict[str, Any] = {"_op": op.name}
    for prop_name, prop_val in op.properties.items():
        d[prop_name] = _attr_to_python(prop_val)
    return d


# --- Backward compatibility shims ---

from dataclasses import asdict  # noqa: E402

from compgen.ir.recipe.ops import RecipeOp  # noqa: E402


def recipe_to_yaml(ops: list[RecipeOp]) -> str:
    """DEPRECATED: Serialize old dataclass RecipeOps to YAML."""
    data = []
    for op in ops:
        entry = asdict(op)  # type: ignore[arg-type]
        entry["_type"] = type(op).__name__
        data.append(entry)
    return yaml.dump(data, default_flow_style=False, sort_keys=True)


def yaml_to_recipe(yaml_text: str) -> list[dict[str, Any]]:
    """DEPRECATED: Deserialize YAML to dicts."""
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, list):
        return []
    return data


__all__ = [
    "mlir_to_recipe",
    "recipe_module_to_yaml",
    "recipe_to_mlir",
    "recipe_to_yaml",
    "yaml_to_recipe",
    "yaml_to_recipe_module",
]

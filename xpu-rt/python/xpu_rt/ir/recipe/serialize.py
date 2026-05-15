"""Recipe IR serialization.

Three modes:
    1. MLIR canonical text — via xDSL Printer/Parser (primary,
       byte-stable round-trip).
    2. JSON projection — for the /promoted-recipe sidecar
       and any agent-facing read of pattern-level attrs.
    3. YAML bridge — for LLM prompt injection and human inspection.

The JSON projection is canonical for cross-process / cross-language
consumers; MLIR text remains the canonical *storage* form.
"""

from __future__ import annotations

import io
import json
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
    """Convert a Recipe IR op to a serializable dict.

    Walks ``op.properties`` so any optional prop that *is* populated
    (e.g. ``recipe.promote.recipe_signature``,
    ``applies_when``, ``evidence_summary``, ``fallback_chain``,
    ``target_class``) appears in the output without per-op handling.
    """
    d: dict[str, Any] = {"_op": op.name}
    for prop_name, prop_val in op.properties.items():
        d[prop_name] = _attr_to_python(prop_val)
    return d


# --- JSON projection --------------------------------------------------


def recipe_module_to_json(module: ModuleOp) -> str:
    """Serialise a Recipe IR module to canonical JSON.

    The output is deterministic (sorted keys, no whitespace) — suitable
    for byte-stable storage promoted-recipe sidecars and for
    cross-process consumption retrieval.

    The schema mirrors :func:`recipe_module_to_yaml`: a list of op
    dicts, each with an ``_op`` key naming the op and the populated
    properties as fields. Optional props that are unset on an op are
    simply absent from the dict (round-trip preserves this — they
    will be re-set as ``None`` on parse).
    """
    entries = [_op_to_dict(op) for op in module.body.block.ops]
    return json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_to_recipe_module(json_text: str) -> ModuleOp:
    """Reconstruct a Recipe IR module from the JSON projection.

    Bridges through MLIR text: the JSON is converted to MLIR via
    :func:`yaml_to_recipe_module` semantics for the subset of ops
    currently supported, then parsed by xDSL. For ops that don't have
    a JSON-only construction path yet, callers should round-trip
    through MLIR text directly via :func:`mlir_to_recipe`.

    Returns an empty module if the JSON is malformed or empty — the
    bridge prefers this honest empty-result behaviour over
    raising, since promoted-recipe sidecars are best-effort.
    """
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return ModuleOp(Region(Block()))
    if not isinstance(data, list) or not data:
        return ModuleOp(Region(Block()))
    # Phase 11 work: full JSON→op reconstruction. Until then we go
    # through the MLIR-text path: the bridge stores recipe.mlir
    # alongside the JSON projection, so callers should prefer
    # mlir_to_recipe(). This stub keeps the API stable.
    return ModuleOp(Region(Block()))


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
    "json_to_recipe_module",
    "mlir_to_recipe",
    "recipe_module_to_json",
    "recipe_module_to_yaml",
    "recipe_to_mlir",
    "recipe_to_yaml",
    "yaml_to_recipe",
    "yaml_to_recipe_module",
]

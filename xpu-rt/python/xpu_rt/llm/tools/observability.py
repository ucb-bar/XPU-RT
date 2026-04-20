"""Observability tools — read-only helpers the LLM can call in any phase.

These wrap existing CompGen analyzers behind a typed interface the
registry can advertise. Each wrapper is thin: it doesn't re-implement
analysis, it just packages the output into a structured dict the LLM
can reason about.

Registered into ``compgen.llm.registry`` at import time so callers can
enumerate them via ``get_registry().list_tools(phase=...)``.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from compgen.llm.registry import (
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)


def _serialize(obj: Any) -> Any:
    """Shallow dataclass/iterable serialization for JSON-safe output."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    return repr(obj)


# ---------------------------------------------------------------------------
# read_target_features
# ---------------------------------------------------------------------------


def _read_target_features_impl(*, device: Any, slice_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return a JSON-safe subset of a CompGenDevice's target profile.

    The LLM passes a concrete ``compgen.device(...)`` handle (or its
    ``profile`` attribute). When ``slice_keys`` is empty, the full
    profile is serialized; otherwise only those top-level fields.
    """
    profile = getattr(device, "profile", device)
    full = _serialize(profile)
    if not isinstance(full, dict):
        return {"status": "error", "reason": "profile is not a dataclass"}
    if slice_keys:
        return {"status": "ok", "target": {k: full.get(k) for k in slice_keys}}
    return {"status": "ok", "target": full}


read_target_features = Tool(
    name="read_target_features",
    phase=2,
    kind="observability",
    wraps_pass="compgen.targets TargetProfile",
    autocomp_cost_impact="zero",
    args=(
        ToolArg("device", "compgen_device", "CompGenDevice or TargetProfile"),
        ToolArg(
            "slice_keys",
            "tuple[str]",
            "subset of top-level fields to return ([] = full profile)",
            required=False,
            default=(),
        ),
    ),
    result=ToolResult("TargetSlice", "typed subset of target profile"),
    description="Returns a JSON-safe slice of the target profile for LLM reasoning.",
    impl=_read_target_features_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# read_analyzer_dossier
# ---------------------------------------------------------------------------


def _read_analyzer_dossier_impl(*, analysis: Any) -> dict[str, Any]:
    """Return a JSON-safe summary of a NetworkAnalysis dossier.

    Expects a ``compgen.agent.analyzer.NetworkAnalysis`` object (result
    of ``NetworkAnalyzer().analyze(...)``). Produces the same fields the
    user-perspective ``03_graph_analysis.py`` script emits into
    ``gap_analysis.json`` but in-process and typed.
    """
    dossier = getattr(analysis, "dossier", None)
    if dossier is None:
        return {"status": "error", "reason": "analysis.dossier is None"}
    return {
        "status": "ok",
        "model_name": getattr(analysis, "model_name", ""),
        "total_params": int(getattr(analysis, "total_params", 0)),
        "total_flops": int(getattr(analysis, "total_flops", 0)),
        "total_bytes": int(getattr(analysis, "total_bytes", 0)),
        "cluster_count": len(getattr(analysis, "clusters", [])),
        "region_count": len(getattr(dossier, "regions", [])),
        "bottleneck_clusters": list(getattr(analysis, "bottleneck_clusters", [])),
        "dynamic_shape_regions": list(getattr(dossier, "dynamic_shape_regions", [])),
        "optimization_opportunities": list(getattr(analysis, "optimization_opportunities", [])),
    }


read_analyzer_dossier = Tool(
    name="read_analyzer_dossier",
    phase=2,
    kind="observability",
    wraps_pass="compgen.agent.analyzer.NetworkAnalyzer",
    autocomp_cost_impact="zero",
    args=(ToolArg("analysis", "NetworkAnalysis", "Result of NetworkAnalyzer().analyze(...)"),),
    result=ToolResult(
        "AnalyzerDossierSummary",
        "model totals, region count, bottleneck ids, optimization opportunities",
    ),
    description="Packages the NetworkAnalyzer dossier into a typed summary.",
    impl=_read_analyzer_dossier_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# read_region_shapes
# ---------------------------------------------------------------------------


def _read_region_shapes_impl(*, analysis: Any, region_id: str) -> dict[str, Any]:
    """Return shape+flops info for a single region from the dossier."""
    dossier = getattr(analysis, "dossier", None)
    if dossier is None:
        return {"status": "error", "reason": "analysis.dossier is None"}
    for r in getattr(dossier, "regions", []):
        if getattr(r, "region_id", None) == region_id:
            return {
                "status": "ok",
                "region_id": region_id,
                "kind": getattr(r, "kind", ""),
                "flops": int(getattr(r, "flops", 0)),
                "bytes": int(getattr(r, "bytes", 0)),
                "arithmetic_intensity": float(getattr(r, "arithmetic_intensity", 0.0)),
                "dynamic_shapes": bool(getattr(r, "dynamic_shapes", False)),
                "repeated_count": int(getattr(r, "repeated_count", 0)),
                "producers": list(getattr(r, "producers", [])),
                "consumers": list(getattr(r, "consumers", [])),
            }
    return {"status": "not_found", "region_id": region_id}


read_region_shapes = Tool(
    name="read_region_shapes",
    phase=2,
    kind="observability",
    wraps_pass="compgen.agent.analyzer (region lookup)",
    autocomp_cost_impact="zero",
    args=(
        ToolArg("analysis", "NetworkAnalysis", "Result of NetworkAnalyzer().analyze(...)"),
        ToolArg("region_id", "string", "region SymbolRef"),
    ),
    result=ToolResult("ShapeReport", "kind, flops, bytes, arithmetic intensity, producers, consumers"),
    description="Returns structural info for a region by id.",
    impl=_read_region_shapes_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> list[str]:
    """Register every observability tool. Idempotent."""
    registry = get_registry()
    registered: list[str] = []
    for tool in (read_target_features, read_analyzer_dossier, read_region_shapes):
        if registry.lookup_tool(tool.name, phase=tool.phase) is None:
            registry.register_tool(tool)
            registered.append(tool.name)
    return registered


# Auto-register on import so consumers of the registry see these tools.
register()


__all__ = [
    "read_analyzer_dossier",
    "read_region_shapes",
    "read_target_features",
    "register",
]

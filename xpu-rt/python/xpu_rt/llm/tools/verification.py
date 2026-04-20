"""Verification tools — mid-flight callable gates.

Wrap ``compgen.semantic.verify`` utilities behind a typed tool
interface the LLM can call between optimization decisions. The LLM
passes two callables (reference + candidate) for differential testing,
or an IR artifact for structural checks.

These are P11 from ``user_perspective/analysis/repo_patch_plan.md``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from compgen.llm.registry import (
    Tool,
    ToolArg,
    ToolResult,
    get_registry,
)
from compgen.semantic.verify.compare import (
    DTYPE_PRESETS,
    ComparisonConfig,
    compare_tensors,
)

# ---------------------------------------------------------------------------
# run_differential_test
# ---------------------------------------------------------------------------


def _run_differential_test_impl(
    *,
    ref_fn: Callable[[], Any],
    got_fn: Callable[[], Any],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Call both callables, compare outputs tensor-by-tensor.

    Returns a GateResult dict with ``status ∈ {accepted, rejected}``
    and per-output comparison details.
    """
    import torch

    t0 = time.perf_counter()
    try:
        ref_out = ref_fn()
    except Exception as e:  # noqa: BLE001
        return {
            "status": "rejected",
            "details": {"reason": "ref_fn raised", "error": f"{type(e).__name__}: {e}"},
        }
    ref_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    try:
        got_out = got_fn()
    except Exception as e:  # noqa: BLE001
        return {
            "status": "rejected",
            "details": {"reason": "got_fn raised", "error": f"{type(e).__name__}: {e}"},
        }
    got_ms = (time.perf_counter() - t0) * 1000.0

    def _to_list(x: Any) -> list[Any]:
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (list, tuple)):
            return [v for v in x if isinstance(v, torch.Tensor)]
        return []

    ref_list = _to_list(ref_out)
    got_list = _to_list(got_out)
    if len(ref_list) != len(got_list):
        return {
            "status": "rejected",
            "details": {
                "reason": "output tensor count mismatch",
                "ref_count": len(ref_list),
                "got_count": len(got_list),
            },
        }
    if not ref_list:
        return {
            "status": "rejected",
            "details": {"reason": "no tensor outputs to compare"},
        }

    comparisons: list[dict[str, Any]] = []
    all_passed = True
    for i, (r, g) in enumerate(zip(ref_list, got_list)):
        cmp = compare_tensors(r, g, atol=atol, rtol=rtol)
        comparisons.append({"index": i, **asdict(cmp)})
        if not cmp.passed:
            all_passed = False

    return {
        "status": "accepted" if all_passed else "rejected",
        "details": {
            "comparisons": comparisons,
            "latency_ref_ms": round(ref_ms, 3),
            "latency_got_ms": round(got_ms, 3),
            "atol": atol,
            "rtol": rtol,
        },
    }


run_differential_test = Tool(
    name="run_differential_test",
    phase=2,
    kind="verification",
    wraps_pass="compgen.semantic.verify.compare_tensors",
    autocomp_cost_impact="zero",
    args=(
        ToolArg("ref_fn", "callable", "zero-arg reference callable"),
        ToolArg("got_fn", "callable", "zero-arg candidate callable"),
        ToolArg("atol", "number", "absolute tolerance", required=False, default=1e-5),
        ToolArg("rtol", "number", "relative tolerance", required=False, default=1e-5),
    ),
    result=ToolResult("GateResult", "accepted/rejected + per-output NumericComparison"),
    description="Runs two callables, compares outputs, returns a gate result.",
    impl=_run_differential_test_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# run_structural_check
# ---------------------------------------------------------------------------


def _run_structural_check_impl(*, artifact: Any) -> dict[str, Any]:
    """Run xDSL verify() on a ModuleOp (or return a structured error).

    For non-ModuleOp artifacts, falls through to duck-typed checks
    (must be dict with required keys for schema-validated YAML
    artifacts).
    """
    # xDSL ModuleOp path
    try:
        from xdsl.dialects.builtin import ModuleOp
        from xdsl.ir import Operation
    except ImportError:  # pragma: no cover
        ModuleOp = object  # type: ignore
        Operation = object  # type: ignore

    if isinstance(artifact, ModuleOp) or isinstance(artifact, Operation):
        try:
            artifact.verify()
        except Exception as e:  # noqa: BLE001
            return {
                "status": "rejected",
                "details": {"reason": "xdsl verify failed", "error": str(e)},
            }
        return {"status": "accepted", "details": {"kind": "xdsl_module"}}

    # Dict/YAML-ish artifact path
    if isinstance(artifact, dict):
        missing: list[str] = []
        for req in ("schema_version",):
            if req not in artifact:
                missing.append(req)
        if missing:
            return {
                "status": "rejected",
                "details": {
                    "reason": "missing_required_fields",
                    "missing": missing,
                },
            }
        return {
            "status": "accepted",
            "details": {"kind": "dict_artifact", "keys": sorted(artifact)},
        }

    return {
        "status": "rejected",
        "details": {"reason": f"unsupported artifact type: {type(artifact).__name__}"},
    }


run_structural_check = Tool(
    name="run_structural_check",
    phase=2,
    kind="verification",
    wraps_pass="xdsl verify + schema check",
    autocomp_cost_impact="zero",
    args=(ToolArg("artifact", "artifact_ref", "xDSL Operation/ModuleOp or dict-shaped artifact"),),
    result=ToolResult("GateResult", "accepted/rejected + error list"),
    description="Runs IR verifier or dict-schema structural check.",
    impl=_run_structural_check_impl,
    stub=False,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register() -> list[str]:
    """Register every verification tool. Idempotent."""
    registry = get_registry()
    registered: list[str] = []
    for tool in (run_differential_test, run_structural_check):
        if registry.lookup_tool(tool.name, phase=tool.phase) is None:
            registry.register_tool(tool)
            registered.append(tool.name)
    return registered


# Auto-register on import so callers of the registry see these tools.
register()


__all__ = [
    "DTYPE_PRESETS",
    "ComparisonConfig",
    "register",
    "run_differential_test",
    "run_structural_check",
]

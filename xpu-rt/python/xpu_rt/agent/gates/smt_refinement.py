"""SMT refinement gate — wraps compgen.ir.semantic.translation_validation.

Context::

    ctx = {
        "source_module": ModuleOp,     # required
        "target_module": ModuleOp,     # required
        "timeout_ms": 30_000,          # optional
        "require_smt": True,           # gate defers (skips) when False
    }

This gate is **opt-in**. Per the approved P7/P8 plan, the default
composite gate for ported-pass invent-slots uses structural +
differential only; SMT refinement is requested by the slot's
invent-slot metadata or by the caller setting ``require_smt=True``.

If ``require_smt`` is False (or absent), the gate returns
``deferred`` — telling the composite gate to skip it.
"""

from __future__ import annotations

from typing import Any


def smt_refinement_gate(proposal: dict[str, Any], **ctx: Any) -> dict[str, Any]:
    if not ctx.get("require_smt", False):
        return {
            "status": "deferred",
            "details": {"reason": "SMT refinement not requested (ctx.require_smt != True)"},
        }

    source_module = ctx.get("source_module")
    target_module = ctx.get("target_module")
    if source_module is None or target_module is None:
        return {
            "status": "deferred",
            "details": {
                "reason": "smt_refinement_gate requires ctx.source_module + ctx.target_module",
            },
        }

    timeout_ms = int(ctx.get("timeout_ms", 30_000))

    try:
        from compgen.ir.semantic.translation_validation import validate_translation
    except ImportError as e:   # pragma: no cover
        return {
            "status": "deferred",
            "details": {"reason": f"translation_validation unavailable: {e}"},
        }

    try:
        result = validate_translation(source_module, target_module, timeout_ms=timeout_ms)
    except Exception as e:   # noqa: BLE001
        return {
            "status": "rejected",
            "details": {
                "reason": "validate_translation raised",
                "error": f"{type(e).__name__}: {e}",
            },
        }

    if result.valid:
        return {
            "status": "accepted",
            "details": {
                "status": result.status,
                "solver_time_ms": result.solver_time_ms,
            },
        }

    return {
        "status": "rejected",
        "details": {
            "status": result.status,
            "solver_time_ms": result.solver_time_ms,
            "counterexample": result.counterexample,
        },
    }


__all__ = ["smt_refinement_gate"]

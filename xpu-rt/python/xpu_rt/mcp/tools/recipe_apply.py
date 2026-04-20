"""MCP tool: apply the accumulated Recipe IR to the live Payload IR.

After the agent calls :func:`propose_invent_slot` one or more times,
the recipe ModuleOp carries propose-ops appended by
:func:`compgen.agent.recipe_bridge_invent.proposal_to_recipe_op`. They
sit there as untaken decisions until the agent commits them by calling
``apply_recipe``.

The tool runs the deterministic two-step lowering pipeline that
existed (but wasn't exposed) before:

1. :func:`compgen.ir.recipe.lower.lower_recipe` — turn the recipe ops
   into transform scripts + kernel jobs + verification obligations.
2. :func:`compgen.ir.recipe.execute.RecipeExecutor.execute` — apply the
   transform scripts to the payload module (real Transform Dialect
   rewrites), dispatch kernel jobs, run verification.

The mutated payload module is then rebound on the env so the next
``view_recipe`` / ``bundle_export`` / ``compile`` call sees the
rewritten state.

This tool is the linchpin of "agent decisions actually change the
emitted bundle". Without calling it, the emitted artifacts ignore
every proposal.
"""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from compgen.ir.recipe.execute import RecipeExecutor
from compgen.ir.recipe.lower import lower_recipe
from compgen.ir.recipe.payload_mutators import apply_recipe_to_payload
from compgen.ir.recipe.serialize import recipe_to_mlir
from compgen.mcp.session import SessionManager

log = structlog.get_logger()


def _module_hash(module) -> str:
    """sha256-prefix of a payload module's MLIR text — for diffing."""
    if module is None:
        return "sha256:none"
    try:
        text = str(module)
    except Exception:   # noqa: BLE001
        text = repr(module)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def apply_recipe(
    sm: SessionManager,
    *,
    session_id: str,
    enable_transforms: bool = True,
    enable_eqsat: bool = True,
    enable_kernels: bool = False,
    enable_verification: bool = True,
) -> dict[str, Any]:
    """Lower the session's recipe and apply it to the live payload module.

    Returns a JSON-serialisable summary the agent can read without
    re-fetching the IR text:

    ``{ok, payload_hash_before, payload_hash_after, recipe_hash,
       transforms_applied, transforms_failed, eqsat_runs, kernel_jobs,
       verification: {total, passed, failed, skipped}, diagnostics}``

    ``enable_kernels`` defaults to False because the kernel-search path
    is heavy + currently only relevant when an event-tensor megakernel
    job lives in the recipe; the agent flips it on explicitly.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    env = driver.env

    if env.recipe is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": "No Recipe IR on this session — recipe tracking off.",
        }

    payload = env.payload_module
    if payload is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": "No Payload IR on this session — env not reset yet.",
        }

    payload_before = _module_hash(payload)
    recipe_hash = _module_hash(env.recipe)

    # Direct payload mutation pass: apply the recipe's FuseOp /
    # ProposeFusionOp / TileOp / PlaceOnDeviceOp / ProposeMegakernelSynthesisOp
    # directly to the payload module's op attributes. This is what makes the
    # agent's proposal observable in the emitted artifacts (e.g. forward.c)
    # without needing an xDSL Transform Dialect interpreter.
    mutation_report = apply_recipe_to_payload(env.recipe, payload)

    try:
        lowered = lower_recipe(env.recipe)
    except Exception as exc:   # noqa: BLE001
        log.exception("apply_recipe.lower_failed")
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"lower_recipe failed: {type(exc).__name__}: {exc}",
            "payload_hash_before": payload_before,
            "recipe_hash": recipe_hash,
            "mutation_report": mutation_report.to_dict(),
        }

    executor = RecipeExecutor(
        enable_transforms=enable_transforms,
        enable_eqsat=enable_eqsat,
        enable_kernels=enable_kernels,
        enable_verification=enable_verification,
    )
    try:
        result = executor.execute(payload, lowered, env._target)
    except Exception as exc:   # noqa: BLE001
        log.exception("apply_recipe.execute_failed")
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"RecipeExecutor.execute failed: {type(exc).__name__}: {exc}",
            "payload_hash_before": payload_before,
            "recipe_hash": recipe_hash,
        }

    # Rebind the mutated payload on the env so subsequent tool calls
    # see the rewritten state.
    env.set_payload_module(result.module)
    payload_after = _module_hash(result.module)

    # Bump the driver's last_view since the recipe (count of facts /
    # candidates) hasn't changed but mutation count has.
    driver._last_view = driver._compute_view()

    ver_results = result.verification_results or []
    ver_passed = sum(1 for v in ver_results if getattr(v, "passed", False))
    ver_skipped = sum(
        1 for v in ver_results if getattr(v, "status", "") == "skipped"
    )
    ver_failed = len(ver_results) - ver_passed - ver_skipped

    # Per-obligation surface (P7.2): every VerificationResult expanded
    # so the agent can read region_id + status + counterexample summary
    # + solver wall-clock without paging through the underlying
    # dataclass. Source data is already in memory; we just stop
    # discarding it.
    per_obligation: list[dict[str, Any]] = []
    for v in ver_results:
        cex = getattr(v, "counterexample", None)
        per_obligation.append({
            "region_id": getattr(v, "region_id", ""),
            "type": getattr(v, "obligation_type", ""),
            "status": getattr(v, "status", ""),
            "passed": bool(getattr(v, "passed", False)),
            "solver_time_ms": float(getattr(v, "solver_time_ms", 0.0)),
            "counterexample_summary": getattr(cex, "summary", None) if cex else None,
            "details": dict(getattr(v, "details", {}) or {}),
        })

    # Per-script transform error surface (P7.6): _apply_transforms emits
    # diagnostics shaped ``transform(recipe_transform_<i>): <level> — <msg>``.
    # Parse them back out so the agent gets one entry per script with
    # status + error text instead of an opaque ``transforms_failed: 56``.
    per_script: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    import re as _re
    diag_pat = _re.compile(
        r"transform\(recipe_transform_(\d+)\):\s*(\w+)\s*[—-]\s*(.+)"
    )
    for diag in result.diagnostics:
        m = diag_pat.match(diag)
        if not m:
            continue
        idx = int(m.group(1))
        level = m.group(2)
        msg = m.group(3).strip()
        per_script.append({
            "script_index": idx,
            "level": level,           # info / warning / error
            "error": msg if level.lower() != "info" else None,
            "message": msg,
        })
        seen_indices.add(idx)
    # Stash the per-obligation list on the driver so explain_verification
    # can recover it without re-running the executor.
    driver._last_verification = list(per_obligation)   # type: ignore[attr-defined]
    driver._last_transform_diagnostics = list(per_script)   # type: ignore[attr-defined]

    return {
        "ok": True,
        "session_id": session_id,
        "payload_hash_before": payload_before,
        "payload_hash_after": payload_after,
        "payload_changed": payload_before != payload_after,
        "recipe_hash": recipe_hash,
        "transforms": {
            "applied": result.transforms_applied,
            "failed": result.transforms_failed,
            "per_script": per_script,
        },
        # Back-compat: keep the flat counters so earlier callers still work.
        "transforms_applied": result.transforms_applied,
        "transforms_failed": result.transforms_failed,
        "eqsat_runs": result.eqsat_runs,
        "kernel_jobs_executed": len(result.kernels),
        "kernels_found": sum(1 for k in result.kernels if k.found),
        "verification": {
            "total": len(ver_results),
            "passed": ver_passed,
            "failed": ver_failed,
            "skipped": ver_skipped,
            "per_obligation": per_obligation,
        },
        "diagnostics": list(result.diagnostics),
        "mutation_report": mutation_report.to_dict(),
    }


APPLY_RECIPE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "apply_recipe",
        "description": (
            "Lower the session's accumulated Recipe IR (including any "
            "agent-proposed propose_* ops) and apply the resulting "
            "transform scripts + kernel jobs + verification obligations "
            "to the live Payload IR module. Required for agent proposals "
            "to influence bundle_export."
        ),
        "phase": "transform",
        "handler": apply_recipe,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "enable_transforms": {"type": "boolean", "default": True},
                "enable_eqsat": {"type": "boolean", "default": True},
                "enable_kernels": {"type": "boolean", "default": False},
                "enable_verification": {"type": "boolean", "default": True},
            },
            "required": ["session_id"],
        },
    },
]


__all__ = ["APPLY_RECIPE_TOOLS", "apply_recipe"]

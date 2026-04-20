"""MCP tool: explain the most recent verification + transform failures
in language an agent can act on.

Pulls from the per-obligation list and per-script transform diagnostics
that :func:`compgen.mcp.tools.recipe_apply.apply_recipe` writes through
to the driver. Each failure gets a typed ``remediation_hint`` keyed on
the obligation kind / transform level, plus a ``next_step`` action the
agent can take to recover (e.g., split a fused region, raise atol).

This is the symmetric counterpart to
:func:`compgen.agent.gates._remediation.add_remediation` for proposal-
time gate failures: that one fires when the structural / differential
gate rejects a proposal; this one fires when the post-apply
verification + transform stack reports failures.
"""

from __future__ import annotations

from typing import Any

from compgen.mcp.session import SessionManager


# Per-obligation hint dispatch. Keys match the ``obligation_type``
# strings produced by ``lower_recipe._lower_require_*`` handlers.
_VERIFICATION_HINTS: dict[str, dict[str, str]] = {
    "differential": {
        "default": (
            "Numerical drift between reference and candidate. Options: "
            "(1) raise atol/rtol via the next propose_invent_slot's "
            "gate_ctx if the rewrite is a known lossy fusion; "
            "(2) split the fused region into smaller pieces; "
            "(3) drop to a verified equivalent from the registry."
        ),
        "next_step": (
            "Call propose_invent_slot again with smaller grouped_regions "
            "OR increase atol via verify_proposal(gates=['differential'], atol=1e-3)"
        ),
    },
    "translation_validation": {
        "default": (
            "SMT couldn't prove the rewrite is sound. Likely the fused "
            "region exceeds what the TV backend can model symbolically. "
            "Either split the fused region or accept the differential "
            "gate alone for this proposal."
        ),
        "next_step": (
            "Submit the same proposal with verify_proposal(gates=['differential']) "
            "to bypass TV, OR shrink the fused region"
        ),
    },
    "layout_invariant": {
        "default": (
            "The proposed layout violates a stride / alignment invariant "
            "downstream. Pick a target-aligned layout (e.g. blocked_32x32 "
            "on Hexagon HVX) instead of the proposed one."
        ),
        "next_step": "Re-call suggest_proposals('propose_layout_plan') to get target-aligned candidates",
    },
    "memory_bound": {
        "default": (
            "The proposed allocation overflows the target's local memory. "
            "Either tile the offending region OR drop the proposal."
        ),
        "next_step": "Call suggest_proposals('propose_buffer_lifetime_plan') for tiled alternatives",
    },
    "check_file": {
        "default": (
            "A FileCheck-style assertion on the lowered IR failed. The "
            "structural change you proposed isn't visible in the lowered "
            "form — the lowering may have eaten it."
        ),
        "next_step": "Run apply_recipe with enable_transforms=False to inspect lowered IR",
    },
    "profile_budget": {
        "default": (
            "Profiled latency exceeded the budget the proposal claimed. "
            "Either lower the expected_impact in your next proposal OR "
            "pick a tile size that fits cache better."
        ),
        "next_step": "Use suggest_proposals to get cost-aware tile/fusion candidates",
    },
}


_TRANSFORM_LEVEL_HINTS: dict[str, str] = {
    "error": (
        "Transform-script execution error. The lowered transform.* IR "
        "didn't match the payload. Most often this means the targeted "
        "region was already rewritten by an earlier transform — re-fetch "
        "view_recipe and retry against the new region symbol."
    ),
    "warning": (
        "Transform applied with a warning. Usually safe; check whether "
        "the warning text mentions a missing operand the agent can fill in."
    ),
}


def _hint_for_obligation(entry: dict[str, Any]) -> tuple[str, str]:
    """Return ``(remediation_hint, next_step)`` for one obligation."""
    obtype = entry.get("type", "")
    table = _VERIFICATION_HINTS.get(obtype, {})
    return (
        table.get("default", "No registered hint for this obligation type."),
        table.get("next_step", "Call view_recipe + diff_recipe to inspect state."),
    )


def explain_verification(
    sm: SessionManager,
    *,
    session_id: str,
    n: int = 10,
    include_passed: bool = False,
) -> dict[str, Any]:
    """Return the latest verification + transform failures with hints.

    Reads from ``driver._last_verification`` and
    ``driver._last_transform_diagnostics``, both populated by the most
    recent ``apply_recipe`` call. Returns up to ``n`` failures per
    family, joined with a typed ``remediation_hint`` and ``next_step``.
    """
    session = sm.get(session_id)
    driver = session.require_driver()

    obligations = list(getattr(driver, "_last_verification", []) or [])
    transforms = list(getattr(driver, "_last_transform_diagnostics", []) or [])

    failed_obs = [
        o for o in obligations
        if include_passed or (
            not o.get("passed", False) and o.get("status") != "skipped"
        )
    ]
    failed_obs = failed_obs[:n]

    failed_transforms = [
        t for t in transforms
        if t.get("level", "").lower() == "error"
    ]
    failed_transforms = failed_transforms[:n]

    # Enrich each failure with hint + next_step.
    enriched_obligations: list[dict[str, Any]] = []
    for o in failed_obs:
        hint, next_step = _hint_for_obligation(o)
        enriched_obligations.append({
            **o,
            "remediation_hint": hint,
            "next_step": next_step,
        })

    enriched_transforms: list[dict[str, Any]] = []
    for t in failed_transforms:
        level = t.get("level", "").lower()
        enriched_transforms.append({
            **t,
            "remediation_hint": _TRANSFORM_LEVEL_HINTS.get(
                level, "Inspect the message for context."
            ),
            "next_step": (
                "Re-fetch view_recipe; the targeted region may have been "
                "rewritten by an earlier transform."
            ),
        })

    return {
        "ok": True,
        "session_id": session_id,
        "verification_failures": enriched_obligations,
        "transform_failures": enriched_transforms,
        "summary": {
            "obligations_total": len(obligations),
            "obligations_failed": sum(
                1 for o in obligations
                if not o.get("passed", False) and o.get("status") != "skipped"
            ),
            "transform_scripts_total": len(transforms),
            "transform_scripts_failed": sum(
                1 for t in transforms if t.get("level", "").lower() == "error"
            ),
        },
    }


EXPLAIN_TOOLS: list[dict[str, Any]] = [
    {
        "name": "explain_verification",
        "description": (
            "Return the latest apply_recipe's per-obligation + per-script "
            "failures with typed remediation hints + suggested next steps. "
            "Use after apply_recipe whenever transforms_failed > 0 or "
            "verification.failed > 0."
        ),
        "phase": "inspect",
        "handler": explain_verification,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "n": {"type": "integer", "default": 10},
                "include_passed": {"type": "boolean", "default": False},
            },
            "required": ["session_id"],
        },
    },
]


__all__ = ["EXPLAIN_TOOLS", "explain_verification"]

"""Remediation-hint library for gate rejections.

Gate rejections from :mod:`compgen.agent.gates.{structural,differential}`
carry a ``details.reason`` field. For an LLM-driven loop we want more
than a blunt ``reason``: we want a short, actionable hint the LLM can
use to fix the proposal on the next turn.

This module maps every known ``reason`` to a ``remediation_hint`` +
optional ``example_fix`` snippet. The composite gate calls
:func:`add_remediation` on every rejected result so LLM tool callers
can always surface a fix path.

Unknown reasons intentionally return ``remediation_hint = None`` —
misleading hints cause the LLM to thrash; no hint is better than a
wrong one.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Hint catalogue
# ---------------------------------------------------------------------------
#
# Keys match the ``reason`` strings populated by the structural and
# differential gates. Each entry supplies:
#   - hint: short imperative sentence for the LLM
#   - example_fix: optional copy-pasteable snippet
#   - applies_to: gate family(-ies) this reason can come from; used only
#     for sanity checks in tests.

_HINTS: dict[str, dict[str, Any]] = {
    # --- structural gate --------------------------------------------------
    "missing_required_keys": {
        "hint": (
            "Proposal is missing required top-level keys. "
            "Include 'chosen' (the picked candidate) and 'select_vs_invent' "
            "('select' when reusing a known kernel, 'invent' when proposing "
            "a new one)."
        ),
        "example_fix": (
            '{"chosen": "candidate_id", "select_vs_invent": "select", '
            '"candidates": [...]}'
        ),
        "applies_to": ("structural",),
    },
    "select_vs_invent must be 'select' or 'invent'": {
        "hint": (
            "'select_vs_invent' must be exactly 'select' or 'invent'. "
            "Use 'select' when picking a known kernel from the registry, "
            "'invent' when proposing a new one."
        ),
        "example_fix": '"select_vs_invent": "select"',
        "applies_to": ("structural",),
    },
    "xdsl verify failed": {
        "hint": (
            "The proposed IR failed xDSL structural verification. "
            "Check operand/result type parity, attribute shapes, and that "
            "any referenced symbols resolve. Call view_recipe(focus=...) "
            "to inspect the offending region before retrying."
        ),
        "applies_to": ("structural",),
    },
    # --- differential gate ------------------------------------------------
    "differential gate requires ctx.ref_fn + ctx.got_fn": {
        "hint": (
            "Differential gate needs runnable callables. Either supply "
            "ref_fn/got_fn in the context, or request a different gate "
            "(structural-only) for this proposal."
        ),
        "applies_to": ("differential",),
    },
    "ref_fn raised": {
        "hint": (
            "The reference function raised before producing tensors. "
            "Check input shapes/dtypes match the model's capture-artifact "
            "sample_inputs; the reference must be robust to the same input."
        ),
        "applies_to": ("differential",),
    },
    "got_fn raised": {
        "hint": (
            "The candidate function raised at execution time. "
            "Typical causes: shape broadcasting after a bad fusion, device "
            "mismatch, or missing kernel import. Run verify_proposal with "
            "only the structural gate to isolate the failure before rerunning."
        ),
        "applies_to": ("differential",),
    },
    "tensor count mismatch": {
        "hint": (
            "Candidate produced a different number of output tensors than "
            "the reference. Ensure the proposed transform preserves the "
            "full output tuple — e.g. a fusion that collapses two outputs "
            "into one violates this."
        ),
        "applies_to": ("differential",),
    },
    "no tensor outputs to compare": {
        "hint": (
            "Neither ref_fn nor got_fn returned tensors. The differential "
            "gate only compares torch.Tensor outputs; wrap scalar outputs "
            "in torch.as_tensor before returning."
        ),
        "applies_to": ("differential",),
    },
    # Numerical drift is signalled indirectly: differential gate returns
    # status=rejected with ``comparisons`` instead of a ``reason``. We
    # synthesise the reason in :func:`add_remediation`.
    "numerical_drift": {
        "hint": (
            "Candidate diverges from the reference beyond atol/rtol. "
            "Options: (1) raise ctx.atol/ctx.rtol if the transform is a "
            "known lossy rewrite (e.g. fp32->bf16 fusion); (2) split the "
            "fusion into two smaller regions; (3) fall back to a verified "
            "equivalent from the registry."
        ),
        "example_fix": 'ctx["atol"] = 1e-4; ctx["rtol"] = 1e-4',
        "applies_to": ("differential",),
    },
    # --- composite (wrapper) ---------------------------------------------
    "ctx.module is not an xDSL Operation": {
        "hint": (
            "ctx.module must be an xDSL Operation (ModuleOp or similar). "
            "If you serialized the IR for transport, parse it back with "
            "mlir_to_recipe() before passing it to the gate."
        ),
        "applies_to": ("structural",),
    },
}


# Reasons that prevent the gate from running entirely — signalled
# by status=deferred. These get a slightly different hint prefix.
_DEFERRED_HINTS: dict[str, str] = {
    "differential gate requires ctx.ref_fn + ctx.got_fn": (
        "Gate was skipped: please supply ref_fn and got_fn in ctx, or "
        "run verify_proposal with gates=[structural_gate] only."
    ),
    "missing dependency": (
        "Gate was skipped due to a missing runtime dependency. Install "
        "the requested package and retry."
    ),
}


def _extract_reason(details: dict[str, Any]) -> str | None:
    """Find the rejection reason in a gate-result details dict.

    Gates write the reason under ``details.reason`` or, for the
    differential gate, embed per-comparison info under ``comparisons``.
    """
    if "reason" in details:
        return str(details["reason"])
    # Differential gate: status=rejected + comparisons but no reason.
    comps = details.get("comparisons")
    if isinstance(comps, list) and any(not c.get("passed", True) for c in comps):
        return "numerical_drift"
    return None


def add_remediation(
    gate_result: dict[str, Any],
    *,
    slot_name: str | None = None,
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enrich a gate result with an LLM-actionable remediation hint.

    No-op when ``status == 'accepted'``. For rejected / deferred results,
    finds ``details.reason`` (or synthesises ``numerical_drift`` from
    the differential gate's comparison output), looks up a hint, and
    writes it back to ``gate_result['details']['remediation_hint']``
    plus optional ``example_fix``.

    Returns the (mutated) ``gate_result`` for call-chaining.

    Args:
        gate_result: A gate dict with ``{status, details}`` shape.
        slot_name: Name of the invent slot the gate was called for;
            included in the hint for context if present.
        ctx: The original gate context dict. Currently only used to
            look at ``ctx.atol`` / ``ctx.rtol`` for numerical-drift
            hints, but reserved for richer contextualisation later.
    """
    status = gate_result.get("status", "deferred")
    if status == "accepted":
        return gate_result

    details = gate_result.get("details")
    if not isinstance(details, dict):
        details = {}
        gate_result["details"] = details

    # If a composite gate nested several traces, walk them to find the
    # first rejected/deferred sub-trace and use *its* reason as the
    # primary one. Priority: rejected beats deferred.
    if "gate_trace" in details and not details.get("reason"):
        rejected_reason: str | None = None
        deferred_reason: str | None = None
        for entry in details["gate_trace"]:
            sub = entry.get("details") or {}
            sub_reason = _extract_reason(sub)
            if sub_reason is None:
                continue
            if entry.get("status") == "rejected" and rejected_reason is None:
                rejected_reason = sub_reason
            elif entry.get("status") == "deferred" and deferred_reason is None:
                deferred_reason = sub_reason
        chosen = rejected_reason or deferred_reason
        if chosen:
            details["reason"] = chosen

    reason = _extract_reason(details)
    if reason is None:
        # Unknown shape — leave as-is. Returning a fabricated hint would
        # mislead the LLM.
        details.setdefault("remediation_hint", None)
        return gate_result

    details["reason"] = reason  # normalise for downstream consumers

    entry = _HINTS.get(reason)
    if entry is None and status == "deferred":
        # Try the deferred-specific catalogue.
        for key, hint in _DEFERRED_HINTS.items():
            if reason.startswith(key):
                details["remediation_hint"] = hint
                if slot_name:
                    details["remediation_hint"] += f" (slot: {slot_name})"
                return gate_result

    if entry is None:
        details["remediation_hint"] = None
        return gate_result

    hint = entry["hint"]
    if slot_name:
        hint = f"[slot: {slot_name}] {hint}"
    details["remediation_hint"] = hint
    if "example_fix" in entry:
        details["example_fix"] = entry["example_fix"]

    # For numerical drift, surface the worst comparison numerically so
    # the LLM can size the tolerance bump.
    if reason == "numerical_drift":
        comps = details.get("comparisons", [])
        worst = _worst_comparison(comps)
        if worst is not None:
            details["worst_comparison"] = worst

    return gate_result


def _worst_comparison(comps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the single comparison with the largest ``max_abs_error``."""
    if not comps:
        return None
    try:
        worst = max(
            comps,
            key=lambda c: float(c.get("max_abs_error", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        return comps[0]
    return {
        "index": worst.get("index"),
        "max_abs_error": worst.get("max_abs_error"),
        "max_rel_error": worst.get("max_rel_error"),
    }


def known_reasons() -> tuple[str, ...]:
    """Return the sorted tuple of reason strings with first-class hints.

    Tests assert that every listed reason yields a non-None
    ``remediation_hint``.
    """
    return tuple(sorted(_HINTS))


__all__ = ["add_remediation", "known_reasons"]

"""Composite gate — chains gates with short-circuit on first rejection.

Usage::

    gate = lambda proposal, **ctx: composite_gate(
        proposal, gates=[structural_gate, differential_gate], **ctx
    )

Returns a GateResult dict whose ``details.gate_trace`` is a list of
(gate_name, status, per_gate_details) — even for accepted proposals,
so callers can audit what ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from compgen.agent.gates._remediation import add_remediation

GateFn = Callable[..., dict[str, Any]]


def composite_gate(
    proposal: dict[str, Any],
    *,
    gates: list[GateFn],
    fail_fast: bool = True,
    slot_name: str | None = None,
    **ctx: Any,
) -> dict[str, Any]:
    """Run each gate in order; short-circuit on first rejection by default.

    Args:
        proposal: The invent-slot proposal.
        gates: Ordered list of gate callables.
        fail_fast: When True (default), stop on first rejection. When
            False, run every gate and aggregate results.
        slot_name: Optional invent-slot name, propagated into the
            remediation hint for LLM consumers.

    Returns:
        {"status": "accepted"|"rejected"|"deferred",
         "details": {"gate_trace": [...], "remediation_hint"?, ...}}

    Non-accepted results always carry a ``details.reason`` (normalised)
    and ``details.remediation_hint`` (may be ``None`` for unknown
    reasons — better to say nothing than to hallucinate a fix).
    """
    trace: list[dict[str, Any]] = []
    final_status = "accepted"

    for gate in gates:
        gate_name = getattr(gate, "__name__", str(gate))
        result = gate(proposal, **ctx)
        status = result.get("status", "deferred")
        trace.append(
            {
                "gate": gate_name,
                "status": status,
                "details": result.get("details", {}),
            }
        )
        if status == "rejected":
            final_status = "rejected"
            if fail_fast:
                break
        elif status == "deferred" and final_status == "accepted":
            final_status = "deferred"

    composite_result: dict[str, Any] = {
        "status": final_status,
        "details": {"gate_trace": trace},
    }
    if final_status != "accepted":
        add_remediation(composite_result, slot_name=slot_name, ctx=ctx)
    return composite_result


__all__ = ["GateFn", "composite_gate"]

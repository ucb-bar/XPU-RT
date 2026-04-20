"""MCP transform tools: invoke_tool, propose_invent_slot,
verify_proposal, step_proposal."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from compgen.agent.gates import (
    composite_gate,
    differential_gate,
    structural_gate,
)
from compgen.mcp.session import SessionManager


def invoke_tool(
    sm: SessionManager,
    *,
    session_id: str,
    tool_name: str,
    args: dict[str, Any] | None = None,
    phase: int | None = None,
) -> dict[str, Any]:
    """Dispatch a registered LLM Tool by name through the driver."""
    session = sm.get(session_id)
    driver = session.require_driver()
    result = driver.step_tool(tool_name, args or {}, phase=phase)
    return {
        "ok": True,
        "session_id": session_id,
        **asdict(result),
    }


def propose_invent_slot(
    sm: SessionManager,
    *,
    session_id: str,
    slot_name: str,
    proposal: dict[str, Any],
    phase: int | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    """Submit a proposal to an invent-slot's gate.

    Gate result always carries ``remediation_hint`` (possibly ``None``)
    when non-accepted, so the LLM can fix the next turn.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    ctx: dict[str, Any] = {}
    if atol is not None:
        ctx["atol"] = atol
    if rtol is not None:
        ctx["rtol"] = rtol
    result = driver.step_invent(slot_name, proposal, phase=phase, gate_ctx=ctx)
    return {
        "ok": True,
        "session_id": session_id,
        **asdict(result),
    }


def verify_proposal(
    sm: SessionManager,
    *,
    session_id: str,
    proposal: dict[str, Any],
    gates: list[str] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, Any]:
    """Run gates directly on a proposal (no slot lookup).

    ``gates`` is a list of gate-name strings from the catalogue:
    ``"structural" | "differential"``. Unknown names are silently
    dropped (the driver never raises on bad input from the LLM).
    """
    _ = sm.get(session_id)  # exists-check
    gate_catalogue = {
        "structural": structural_gate,
        "differential": differential_gate,
    }
    chosen = [gate_catalogue[g] for g in (gates or ["structural"]) if g in gate_catalogue] or [structural_gate]
    ctx: dict[str, Any] = {}
    if atol is not None:
        ctx["atol"] = atol
    if rtol is not None:
        ctx["rtol"] = rtol
    res = composite_gate(proposal, gates=chosen, **ctx)
    return {"ok": True, "session_id": session_id, "gate_result": res}


def step_proposal(
    sm: SessionManager,
    *,
    session_id: str,
    action_type: str,
    target: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Translate an LLM action proposal into an env step and apply it.

    This is the direct path into :meth:`AgenticCompilationLoop._proposal_to_action`
    + :meth:`CompilerEnv.step` — no invent-slot / no extra gate, just
    the agentic-loop mapping the LLM already knows.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    result = driver.step_proposal(
        action_type,
        target=target,
        reason=reason,
    )
    return {"ok": True, "session_id": session_id, **asdict(result)}


TRANSFORM_TOOLS: list[dict[str, Any]] = [
    {
        "name": "invoke_tool",
        "description": "Invoke a registered LLM Tool by name.",
        "phase": "transform",
        "handler": invoke_tool,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "args": {"type": "object"},
                "phase": {"type": "integer"},
            },
            "required": ["session_id", "tool_name"],
        },
    },
    {
        "name": "propose_invent_slot",
        "description": "Submit a proposal to an invent-slot gate.",
        "phase": "transform",
        "handler": propose_invent_slot,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "slot_name": {"type": "string"},
                "proposal": {"type": "object"},
                "phase": {"type": "integer"},
                "atol": {"type": "number"},
                "rtol": {"type": "number"},
            },
            "required": ["session_id", "slot_name", "proposal"],
        },
    },
    {
        "name": "verify_proposal",
        "description": "Run named gates directly on a proposal.",
        "phase": "transform",
        "handler": verify_proposal,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "proposal": {"type": "object"},
                "gates": {"type": "array", "items": {"type": "string"}},
                "atol": {"type": "number"},
                "rtol": {"type": "number"},
            },
            "required": ["session_id", "proposal"],
        },
    },
    {
        "name": "step_proposal",
        "description": "Translate a typed LLM action proposal into an env step.",
        "phase": "transform",
        "handler": step_proposal,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "action_type": {"type": "string"},
                "target": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["session_id", "action_type"],
        },
    },
]


__all__ = [
    "TRANSFORM_TOOLS",
    "invoke_tool",
    "propose_invent_slot",
    "step_proposal",
    "verify_proposal",
]

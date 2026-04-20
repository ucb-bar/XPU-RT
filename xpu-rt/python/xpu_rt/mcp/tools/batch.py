"""MCP tool: ``batch_propose`` — submit multiple invent-slot proposals
in one MCP roundtrip.

Replaces N separate ``propose_invent_slot`` calls with one. Useful when
the agent already has a multi-region plan (e.g. from
``suggest_proposals``) and wants to apply all of them atomically.

Atomic mode: when ``atomic=True``, the recipe + payload are
snapshotted before the batch; on first rejection the snapshots are
rebound, leaving the session as if the batch never started. The agent
sees per-step results regardless, so it can decide what to do next
(re-batch with the failing entry removed, etc.).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from compgen.mcp.session import SessionManager


def batch_propose(
    sm: SessionManager,
    *,
    session_id: str,
    proposals: list[dict[str, Any]],
    atomic: bool = False,
    gate_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply ``proposals`` (list of ``{slot_name, proposal}``) in order.

    Args:
        proposals: One entry per intended invent-slot call. Each must
            carry ``slot_name`` and ``proposal``; may optionally carry
            ``phase`` and a per-entry ``gate_ctx`` (which overrides the
            top-level one for that entry).
        atomic: If True, on the first rejection the recipe + payload
            are rolled back to a pre-batch snapshot.
        gate_ctx: Default gate context applied to every entry that
            doesn't supply its own.

    Returns ``{ok, accepted, rejected, results: [DriverStepResult dict],
    rolled_back: bool}``.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    if not isinstance(proposals, list):
        return {
            "ok": False,
            "session_id": session_id,
            "error": "proposals must be a list",
        }
    results, rolled_back = driver.step_invent_many(
        proposals,
        atomic=atomic,
        gate_ctx=gate_ctx,
    )
    rejected = sum(1 for r in results if r.status not in {"accepted"})
    if rolled_back:
        # The whole batch was reverted to the pre-batch snapshot; even
        # the early-batch "accepted" proposals are no longer in the
        # recipe. Report them honestly so the agent isn't told things
        # landed when they didn't.
        accepted = 0
    else:
        accepted = sum(1 for r in results if r.status == "accepted")
    return {
        "ok": True,
        "session_id": session_id,
        "accepted": accepted,
        "rejected": rejected,
        "rolled_back": rolled_back,
        "results": [asdict(r) for r in results],
    }


BATCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "batch_propose",
        "description": (
            "Submit a list of invent-slot proposals in one MCP roundtrip. "
            "atomic=True rolls back recipe + payload to a pre-batch "
            "snapshot on first rejection."
        ),
        "phase": "transform",
        "handler": batch_propose,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "proposals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slot_name": {"type": "string"},
                            "proposal": {"type": "object"},
                            "phase": {"type": "integer"},
                            "gate_ctx": {"type": "object"},
                        },
                        "required": ["slot_name", "proposal"],
                    },
                },
                "atomic": {"type": "boolean", "default": False},
                "gate_ctx": {"type": "object"},
            },
            "required": ["session_id", "proposals"],
        },
    },
]


__all__ = ["BATCH_TOOLS", "batch_propose"]

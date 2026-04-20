"""MCP tool: ``suggest_proposals(session_id, slot_name, k)`` returns
ranked candidate proposals an agent can submit verbatim via
``propose_invent_slot`` (or in bulk via ``batch_propose``).

Per-slot suggesters live in :mod:`compgen.agent.suggest`. This tool
is the thin MCP-layer adapter.
"""

from __future__ import annotations

from typing import Any

from compgen.agent.suggest import (
    SUGGESTERS,
    supported_slot_names,
)
from compgen.agent.suggest import (
    suggest as _suggest,
)
from compgen.mcp.session import SessionManager


def suggest_proposals(
    sm: SessionManager,
    *,
    session_id: str,
    slot_name: str,
    k: int = 5,
) -> dict[str, Any]:
    """Return up to ``k`` ranked candidates for ``slot_name``.

    Each entry carries the ``chosen`` block ready for
    ``propose_invent_slot.proposal``, plus a one-line ``rationale`` and
    an ``expected_impact`` score in [0, 1] used to sort.

    When no suggester is registered for the slot, returns an empty
    candidate list and ``available_slots`` so the agent can pick a
    different slot to suggest against.
    """
    session = sm.get(session_id)
    driver = session.require_driver()
    compiled = session.require_compiled()

    if slot_name not in SUGGESTERS:
        return {
            "ok": True,
            "session_id": session_id,
            "slot_name": slot_name,
            "candidates": [],
            "available_slots": list(supported_slot_names()),
            "remediation_hint": (
                f"No model-aware suggester registered for {slot_name!r}. "
                f"Available: {sorted(SUGGESTERS.keys())}. Use "
                "propose_invent_slot directly with a hand-built proposal."
            ),
        }

    if driver.env.recipe is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": "Recipe IR not enabled on this session.",
        }

    candidates = _suggest(
        slot_name,
        recipe=driver.env.recipe,
        dossier=compiled.analysis_dossier,
        target=compiled.device.profile,
        k=int(k),
    )
    rendered = [
        {
            "rank": i,
            "chosen": dict(c.chosen),
            "rationale": c.rationale,
            "expected_impact": c.expected_impact,
            "target_feature_justification": c.target_feature_justification,
            "metadata": dict(c.metadata),
            # Wrapped form ready to drop into propose_invent_slot.
            "proposal": c.to_proposal(),
            # Self-describing follow-up: which MCP tool to call next +
            # the args already filled in. Makes the agent's two-turn
            # loop (suggest → submit) effectively one turn of reading.
            "next_call": c.next_call(),
            # Multiplicity surface: when a candidate represents N
            # structurally-equivalent matches, the agent sees them all
            # without paging.
            "members": list(c.members),
        }
        for i, c in enumerate(candidates)
    ]
    # Top-level batch-call hint when multiple candidates exist —
    # apply ALL ranked candidates in one batch_propose call.
    batch_next: dict[str, Any] | None = None
    if rendered:
        batch_next = {
            "tool": "batch_propose",
            "args": {
                "proposals": [{"slot_name": slot_name, "proposal": r["proposal"]} for r in rendered],
                "atomic": False,
            },
        }
    return {
        "ok": True,
        "session_id": session_id,
        "slot_name": slot_name,
        "candidate_count": len(candidates),
        "candidates": rendered,
        "next_call_apply_all": batch_next,
    }


SUGGEST_TOOLS: list[dict[str, Any]] = [
    {
        "name": "suggest_proposals",
        "description": (
            "Return ranked candidate proposals for an invent slot, built "
            "from the session's recipe + dossier + target. The agent "
            "picks one (or batches several via batch_propose) instead of "
            "constructing the proposal payload from scratch."
        ),
        "phase": "inspect",
        "handler": suggest_proposals,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "slot_name": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["session_id", "slot_name"],
        },
    },
]


__all__ = ["SUGGEST_TOOLS", "suggest_proposals"]

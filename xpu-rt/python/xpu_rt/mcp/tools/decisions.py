"""MCP tools for the agent's decision-site write path.

These tools let the LLM do four things:

* ``list_decisions`` — enumerate every decision site in the session,
  showing kind, candidates, oracle recommendation, and status.
* ``propose_decision`` — record a proposal for a site without applying
  it (the LLM drafts, then either applies itself or another turn
  finalizes). Lightweight — emits a trace event only, no side effect
  on IR.
* ``apply_decision`` — binding pick. Applied before or at enqueue
  time. Emits a ``decision(source="agent")`` trace event.
* ``override_decision`` — replace an already-resolved outcome.

The four tools operate on :class:`compgen.agent.decisions.DecisionRegistry`
stored on the MCP session. Stage plugins call the registry from the
server side (:func:`compgen.agent.decisions.get_active_registry`); these
tools are how an LLM reaches into the same registry over JSON-RPC.
"""

from __future__ import annotations

from typing import Any

from compgen.agent.decisions import DecisionRegistry
from compgen.mcp.session import McpSession, SessionManager


def _registry(session: McpSession) -> DecisionRegistry:
    return session.require_decision_registry()


def list_decisions(
    sm: SessionManager,
    *,
    session_id: str,
    status: str = "all",
    kind: str | None = None,
) -> dict[str, Any]:
    """Enumerate decision sites visible to this session.

    Args:
        status: ``"pending"`` / ``"resolved"`` / ``"overridden"`` / ``"all"``.
        kind: Filter by site kind (e.g. ``"encoding"``, ``"tile"``).
    """
    session = sm.get(session_id)
    registry = _registry(session)
    sites = registry.list_all()
    if status != "all":
        sites = [s for s in sites if s.status == status]
    if kind:
        sites = [s for s in sites if s.kind == kind]
    return {
        "ok": True,
        "session_id": session_id,
        "count": len(sites),
        "sites": [s.to_dict() for s in sites],
    }


def propose_decision(
    sm: SessionManager,
    *,
    session_id: str,
    site_id: str,
    chosen_id: str,
    rationale: str = "",
    chosen_value: Any | None = None,
    source: str = "agent",
) -> dict[str, Any]:
    """Record a non-binding proposal for a site.

    Returns the same payload ``apply_decision`` would produce but does
    NOT commit the outcome — useful for the LLM to reason across
    multiple sites before committing. No IR is mutated.
    """
    session = sm.get(session_id)
    registry = _registry(session)
    site = registry.get(site_id)
    if site is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"unknown site_id: {site_id!r}",
        }
    candidate = site.candidate_by_id(chosen_id)
    if candidate is None and not chosen_id.startswith("invent:"):
        return {
            "ok": False,
            "session_id": session_id,
            "error": (
                f"candidate {chosen_id!r} not in site; valid ids: "
                f"{[c.id for c in site.candidates]}"
            ),
        }
    from compgen.trace import DecisionPublisher, get_current_llm_turn_id

    DecisionPublisher.emit(
        decision_type=f"{site.kind}_proposal",
        site_id=site_id,
        chosen=chosen_id,
        chosen_value=chosen_value if chosen_value is not None else (
            candidate.value if candidate is not None else None
        ),
        source=source,
        rationale=rationale,
        candidates=[c.id for c in site.candidates],
        oracle_recommended_id=site.oracle_recommended_id,
        llm_turn_id=get_current_llm_turn_id(),
        phase="proposal",
    )
    return {
        "ok": True,
        "session_id": session_id,
        "site_id": site_id,
        "chosen_id": chosen_id,
        "committed": False,
    }


def apply_decision(
    sm: SessionManager,
    *,
    session_id: str,
    site_id: str,
    chosen_id: str,
    rationale: str,
    chosen_value: Any | None = None,
    source: str = "agent",
) -> dict[str, Any]:
    """Commit an agent pick. Applied before or at site enqueue time.

    If the site has not been enqueued yet, the outcome is stashed; the
    next ``enqueue`` + ``resolve`` will pull from the stash. If it has
    been enqueued but not resolved, the outcome is committed now. If
    it is already resolved, returns an error pointing at
    ``override_decision``.
    """
    session = sm.get(session_id)
    registry = _registry(session)
    from compgen.trace import get_current_llm_turn_id

    try:
        outcome = registry.apply(
            site_id,
            chosen_id=chosen_id,
            rationale=rationale,
            chosen_value=chosen_value,
            source=source,
            llm_turn_id=get_current_llm_turn_id(),
        )
    except (KeyError, RuntimeError) as exc:
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "ok": True,
        "session_id": session_id,
        "site_id": site_id,
        "committed": site_id in {s.site_id for s in registry.list_all() if s.outcome is not None},
        "outcome": outcome.to_dict(),
    }


def override_decision(
    sm: SessionManager,
    *,
    session_id: str,
    site_id: str,
    chosen_id: str,
    rationale: str,
    chosen_value: Any | None = None,
) -> dict[str, Any]:
    """Replace an already-resolved outcome for ``site_id``."""
    session = sm.get(session_id)
    registry = _registry(session)
    from compgen.trace import get_current_llm_turn_id

    try:
        outcome = registry.override(
            site_id,
            chosen_id=chosen_id,
            rationale=rationale,
            chosen_value=chosen_value,
            llm_turn_id=get_current_llm_turn_id(),
        )
    except KeyError as exc:
        return {"ok": False, "session_id": session_id, "error": str(exc)}
    return {
        "ok": True,
        "session_id": session_id,
        "site_id": site_id,
        "outcome": outcome.to_dict(),
    }


DECISION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_decisions",
        "description": (
            "Enumerate every decision site in the session. Each site "
            "declares the candidates, the oracle's non-binding "
            "recommendation, context (shapes, dtypes), and status "
            "(pending / resolved / overridden). Use this to discover "
            "what the compiler is about to choose before anything is "
            "committed to the IR."
        ),
        "phase": "inspect",
        "handler": list_decisions,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "status": {"type": "string"},
                "kind": {"type": ["string", "null"]},
            },
            "required": ["session_id"],
        },
    },
    {
        "name": "propose_decision",
        "description": (
            "Record a non-binding proposal for a site. Does NOT mutate "
            "the IR. Use to reason across multiple sites before "
            "committing via apply_decision."
        ),
        "phase": "transform",
        "handler": propose_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "site_id": {"type": "string"},
                "chosen_id": {"type": "string"},
                "rationale": {"type": "string"},
                "chosen_value": {
                    "type": ["string", "number", "boolean", "object", "array", "null"],
                    "description": "Optional novel value when chosen_id is 'invent:...'.",
                },
                "source": {"type": "string"},
            },
            "required": ["session_id", "site_id", "chosen_id"],
        },
    },
    {
        "name": "apply_decision",
        "description": (
            "Commit an agent pick at a decision site. Chosen_id must "
            "be one of the site's candidate ids, or prefixed with "
            "'invent:' to submit a novel value (verified downstream)."
        ),
        "phase": "transform",
        "handler": apply_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "site_id": {"type": "string"},
                "chosen_id": {"type": "string"},
                "rationale": {"type": "string"},
                "chosen_value": {
                    "type": ["string", "number", "boolean", "object", "array", "null"],
                    "description": "Optional novel value when chosen_id is 'invent:...'.",
                },
                "source": {"type": "string"},
            },
            "required": ["session_id", "site_id", "chosen_id", "rationale"],
        },
    },
    {
        "name": "override_decision",
        "description": (
            "Replace an already-resolved outcome for a site. Fails "
            "when the site is still pending."
        ),
        "phase": "transform",
        "handler": override_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "site_id": {"type": "string"},
                "chosen_id": {"type": "string"},
                "rationale": {"type": "string"},
                "chosen_value": {
                    "type": ["string", "number", "boolean", "object", "array", "null"],
                    "description": "Optional novel value when chosen_id is 'invent:...'.",
                },
            },
            "required": ["session_id", "site_id", "chosen_id", "rationale"],
        },
    },
]


__all__ = [
    "DECISION_TOOLS",
    "apply_decision",
    "list_decisions",
    "override_decision",
    "propose_decision",
]

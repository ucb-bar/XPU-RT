"""MCP tools — agent-facing surface for the hierarchical knowledge store.

Lets Claude Code accumulate cross-session learnings via two tools:

  * ``record_lesson``   — append a Lesson at a target's most-specific
    scope (or at an explicit scope). Persists to ``~/.compgen/knowledge/``
    so future sessions see it.
  * ``query_knowledge`` — narrowly query (scope-walked) lessons by
    target / stage / op_family / topic.
  * ``get_context_brief`` — convenience tool returning a prompt-friendly
    one-shot brief (calls ``KnowledgeStore.context_brief``).

The store under the hood is the existing ``compgen.memory.knowledge``
hierarchy; these tools are a thin MCP surface so the agent can drive
the store without going through Python imports.
"""

from __future__ import annotations

from typing import Any

from compgen.memory.knowledge import (
    KnowledgeStore,
    Lesson,
    scope_chain_for_target,
    shared_store,
)
from compgen.mcp.session import SessionManager


def _store(_sm: SessionManager) -> KnowledgeStore:
    """Use the process-wide shared store. Tests override via
    ``compgen.memory.knowledge.set_shared_store(KnowledgeStore(root=tmp))``.
    """
    return shared_store()


def record_lesson(
    sm: SessionManager,
    *,
    session_id: str,
    category: str,
    summary: str,
    target: str | None = None,
    scope: str | None = None,
    stage: str = "any",
    op_family: str = "",
    topic: str = "",
    tags: list[str] | None = None,
    applicability: str = "",
    next_action: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a lesson to the knowledge store.

    The agent provides either an explicit ``scope`` (e.g.
    ``"backends/gpu/nvidia/turing"``) or a ``target`` (e.g.
    ``"cuda-titan-rtx"``) — when ``scope`` is omitted we use the
    most-specific scope from the target's scope chain.
    """
    if not scope:
        if not target:
            return {"ok": False, "session_id": session_id,
                    "error": "either 'scope' or 'target' is required"}
        scope = scope_chain_for_target(target)[0]

    try:
        lesson = Lesson(
            scope=scope, category=category, summary=summary,
            stage=stage, op_family=op_family, topic=topic,
            tags=tuple(tags or ()), applicability=applicability,
            next_action=next_action, evidence=dict(evidence or {}),
        )
    except ValueError as exc:
        return {"ok": False, "session_id": session_id, "error": str(exc)}

    persisted = _store(sm).add(lesson)
    return {
        "ok": True, "session_id": session_id,
        "lesson_id": persisted.id,
        "scope": persisted.scope,
        "timestamp": persisted.timestamp,
    }


def query_knowledge(
    sm: SessionManager,
    *,
    session_id: str,
    target: str,
    stage: str | None = None,
    op_family: str | None = None,
    topic: str | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = 20,
) -> dict[str, Any]:
    """Walk the target's scope chain and return matching lessons."""
    chain = scope_chain_for_target(target)
    lessons = _store(sm).query(
        scope_chain=chain,
        stage=stage, op_family=op_family, topic=topic,
        categories=tuple(categories) if categories else None,
        tags=tuple(tags) if tags else None,
        limit=limit,
    )
    return {
        "ok": True, "session_id": session_id,
        "target": target,
        "scope_chain": chain,
        "lesson_count": len(lessons),
        "lessons": [
            {
                "id": l.id,
                "scope": l.scope,
                "category": l.category,
                "summary": l.summary,
                "stage": l.stage,
                "op_family": l.op_family,
                "topic": l.topic,
                "tags": list(l.tags),
                "applicability": l.applicability,
                "next_action": l.next_action,
                "timestamp": l.timestamp,
            }
            for l in lessons
        ],
    }


def get_context_brief(
    sm: SessionManager,
    *,
    session_id: str,
    target: str,
    stage: str = "kernel-gen",
    topic: str = "",
    op_family: str = "",
    max_lessons: int = 5,
) -> dict[str, Any]:
    """Return a prompt-friendly one-shot brief for the agent's context."""
    brief = _store(sm).context_brief(
        target, stage=stage, topic=topic, op_family=op_family,
        max_lessons=max_lessons,
    )
    return {
        "ok": True, "session_id": session_id,
        "target": target, "stage": stage, "topic": topic,
        "op_family": op_family, "brief": brief,
    }


KNOWLEDGE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "record_lesson",
        "description": (
            "Append a lesson to the hierarchical knowledge store. The "
            "agent uses this to durably remember a perf observation, "
            "correctness bug, hardware ceiling, design decision, or a "
            "concrete recipe that worked."
        ),
        "phase": "transform",
        "handler": record_lesson,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "category": {"type": "string",
                             "enum": ["perf", "correctness", "limit",
                                      "design", "recipe"]},
                "summary": {"type": "string"},
                "target": {"type": ["string", "null"]},
                "scope": {"type": ["string", "null"]},
                "stage": {"type": "string"},
                "op_family": {"type": "string"},
                "topic": {"type": "string"},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "applicability": {"type": "string"},
                "next_action": {"type": "string"},
                "evidence": {"type": ["object", "null"]},
            },
            "required": ["session_id", "category", "summary"],
        },
    },
    {
        "name": "query_knowledge",
        "description": (
            "Query the knowledge store for lessons applicable to a "
            "target × stage × op_family × topic. Walks the scope chain "
            "(general → arch-specific) so general principles come first."
        ),
        "phase": "inspect",
        "handler": query_knowledge,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target": {"type": "string"},
                "stage": {"type": ["string", "null"]},
                "op_family": {"type": ["string", "null"]},
                "topic": {"type": ["string", "null"]},
                "categories": {"type": ["array", "null"],
                               "items": {"type": "string"}},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "limit": {"type": ["integer", "null"]},
            },
            "required": ["session_id", "target"],
        },
    },
    {
        "name": "get_context_brief",
        "description": (
            "One-shot prompt-friendly brief of the most-relevant lessons "
            "for the given target × stage × topic × op_family — meant "
            "to be inlined into a codegen / dispatch prompt."
        ),
        "phase": "inspect",
        "handler": get_context_brief,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "target": {"type": "string"},
                "stage": {"type": "string"},
                "topic": {"type": "string"},
                "op_family": {"type": "string"},
                "max_lessons": {"type": "integer"},
            },
            "required": ["session_id", "target"],
        },
    },
]


__all__ = [
    "KNOWLEDGE_TOOLS",
    "get_context_brief",
    "query_knowledge",
    "record_lesson",
]

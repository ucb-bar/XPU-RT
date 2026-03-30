"""Error pattern learning — negative knowledge system.

Records action failures so the agent can avoid repeating the same
mistakes in future optimization runs. Stores failure patterns in
CompilerMemory with action context, enabling retrieval of relevant
warnings when similar actions are proposed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from compgen.memory.store import CompilerMemory

log = structlog.get_logger()


@dataclass(frozen=True)
class ErrorPattern:
    """A recorded failure pattern."""

    action_type: str
    context_summary: str
    failure_reason: str
    target_key: str
    occurrence_count: int = 1


def record_error_pattern(
    memory: CompilerMemory,
    action_type: str,
    region_context: str,
    failure_reason: str,
    target_key: str,
) -> str:
    """Record an action failure pattern.

    Args:
        memory: CompilerMemory instance.
        action_type: The action type that failed (e.g., "tile", "fuse").
        region_context: Summary of the region where it failed.
        failure_reason: Why it failed.
        target_key: Target profile name.

    Returns:
        Knowledge item ID.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    summary = f"FAIL {action_type} on {target_key}: {failure_reason}"
    artifact = json.dumps({
        "action_type": action_type,
        "region_context": region_context,
        "failure_reason": failure_reason,
        "target_key": target_key,
    })

    item = memory.store_knowledge(
        kind=KnowledgeKind.FAILURE_MODE,
        summary=summary,
        artifact=artifact,
        scope_kind=ScopeKind.OPERATOR_FAMILY,
        scope_key=action_type,
        source="error_pattern",
    )
    log.info(
        "error_pattern.recorded",
        action=action_type,
        reason=failure_reason[:80],
    )
    return item.knowledge_id


def retrieve_error_patterns(
    memory: CompilerMemory,
    action_type: str = "",
    target_key: str = "",
    top_k: int = 5,
) -> list[ErrorPattern]:
    """Retrieve known failure patterns for an action type.

    Args:
        memory: CompilerMemory instance.
        action_type: Filter by action type (empty = all).
        target_key: Filter by target (empty = all).
        top_k: Maximum patterns to return.

    Returns:
        List of ErrorPattern, most recent first.
    """
    from compgen.memory.schema import KnowledgeKind, ScopeKind

    items = memory.retrieve_knowledge(
        kind=KnowledgeKind.FAILURE_MODE,
        scope_kind=ScopeKind.OPERATOR_FAMILY if action_type else None,
        scope_key=action_type,
        top_k=top_k * 2,  # Fetch extra for filtering
    )

    patterns: list[ErrorPattern] = []
    for item in items:
        if item.source != "error_pattern":
            continue
        try:
            blob = memory.blobs.load(item.artifact_hash)
            data = json.loads(blob)
            if target_key and data.get("target_key", "") != target_key:
                continue
            patterns.append(ErrorPattern(
                action_type=data.get("action_type", ""),
                context_summary=data.get("region_context", ""),
                failure_reason=data.get("failure_reason", ""),
                target_key=data.get("target_key", ""),
                occurrence_count=item.uses + 1,
            ))
        except Exception:
            continue

    return patterns[:top_k]


def error_patterns_to_prompt(patterns: list[ErrorPattern]) -> list[dict]:
    """Convert error patterns to dicts suitable for prompt context.

    Args:
        patterns: List of ErrorPattern.

    Returns:
        List of dicts with action_type, failure_reason, target_key.
    """
    return [
        {
            "action_type": p.action_type,
            "failure_reason": p.failure_reason,
            "target_key": p.target_key,
            "occurrences": p.occurrence_count,
        }
        for p in patterns
    ]


__all__ = [
    "ErrorPattern",
    "error_patterns_to_prompt",
    "record_error_pattern",
    "retrieve_error_patterns",
]

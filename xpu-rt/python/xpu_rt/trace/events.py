"""Event schema for the CompGen compilation trace.

A trace is a JSONL stream of ``TraceEvent`` records, one per line. Every
event carries ``event_id``, ``parent_event_id`` (for correlation across
LLM turn → MCP tool → pass → decision chains), ``session_id`` and a
``kind`` from :class:`EventKind`.

Events are deliberately small: heavy payloads (full prompts, full IR)
live in the pre-existing per-layer recorders (:class:`LLMRecorder`,
:class:`ToolCallRecorder`, :class:`McpTranscriptRecorder`); the trace
records only a reference, a hash, or a short preview so the trace file
itself stays grep-able.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Canonical event kinds emitted onto the :class:`TraceBus`."""

    LLM_PROMPT = "llm_prompt"
    LLM_RESPONSE = "llm_response"
    MCP_CALL = "mcp_call"
    TOOL_CALL = "tool_call"
    PASS_RUN = "pass_run"
    ANALYSIS_RUN = "analysis_run"
    DECISION = "decision"
    DECISION_SITE = "decision_site"
    ORACLE_ADVISORY = "oracle_advisory"
    IR_DUMP = "ir_dump"
    STAGE_RUN = "stage_run"


class Level(str, Enum):
    """Structured log levels. Default is ``INFO``; noisy spans use ``DEBUG``."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


# Default level per event kind. Overridable per publish call.
DEFAULT_LEVEL_BY_KIND: dict[str, str] = {
    EventKind.STAGE_RUN.value: Level.INFO.value,
    EventKind.PASS_RUN.value: Level.INFO.value,
    EventKind.ANALYSIS_RUN.value: Level.INFO.value,
    EventKind.DECISION.value: Level.INFO.value,
    EventKind.DECISION_SITE.value: Level.DEBUG.value,
    EventKind.ORACLE_ADVISORY.value: Level.DEBUG.value,
    EventKind.MCP_CALL.value: Level.INFO.value,
    EventKind.TOOL_CALL.value: Level.INFO.value,
    EventKind.LLM_PROMPT.value: Level.DEBUG.value,
    EventKind.LLM_RESPONSE.value: Level.INFO.value,
    EventKind.IR_DUMP.value: Level.DEBUG.value,
}


def default_level_for(kind: str) -> str:
    return DEFAULT_LEVEL_BY_KIND.get(kind, Level.INFO.value)


def category_for(kind: str, payload: dict[str, Any] | None) -> str:
    """Return a human-friendly category tag for a trace event.

    Looks like ``pass:fold_transposes_into_dots`` or
    ``mcp_tool_call:analyze_graph`` — used by the rendered companion
    log and by log-format grep patterns such as
    ``[INFO]<agent_decision>``.
    """
    p = payload or {}
    if kind == EventKind.STAGE_RUN.value:
        return f"stage:{p.get('stage_name') or p.get('stage') or p.get('name') or 'unknown'}"
    if kind == EventKind.PASS_RUN.value:
        return f"pass:{p.get('name') or 'unknown'}"
    if kind == EventKind.ANALYSIS_RUN.value:
        return f"analysis:{p.get('analysis') or p.get('name') or 'unknown'}"
    if kind == EventKind.DECISION.value:
        return f"agent_decision:{p.get('decision_type') or 'unknown'}"
    if kind == EventKind.DECISION_SITE.value:
        return f"decision_site:{p.get('kind') or 'unknown'}"
    if kind == EventKind.ORACLE_ADVISORY.value:
        return f"oracle:{p.get('oracle') or 'unknown'}"
    if kind == EventKind.MCP_CALL.value:
        return f"mcp_tool_call:{p.get('tool') or 'unknown'}"
    if kind == EventKind.TOOL_CALL.value:
        return f"tool_call:{p.get('name') or 'unknown'}"
    if kind == EventKind.LLM_PROMPT.value:
        return f"llm_prompt:{p.get('artifact_type') or 'unknown'}"
    if kind == EventKind.LLM_RESPONSE.value:
        return f"llm_response:{p.get('model') or 'unknown'}"
    if kind == EventKind.IR_DUMP.value:
        return f"ir_dump:{p.get('phase_tag') or p.get('name') or 'unknown'}"
    return kind


class Phase(str, Enum):
    """Paired span phase marker. ``POINT`` is used for single-shot events."""

    START = "start"
    END = "end"
    POINT = "point"


@dataclass
class TraceEvent:
    """One line in ``trace.jsonl``.

    Attributes:
        event_id: Monotonic int assigned by the bus. Zero-padded 10-digit
            strings are preferred at serialization time so log tools can
            lex-sort.
        parent_event_id: The ``event_id`` of the enclosing span (from the
            bus ``ContextVar`` stack at emission time). Empty string at
            top level.
        session_id: Session identifier that partitions events across
            concurrent compiles.
        ts: ISO-8601 UTC timestamp.
        kind: See :class:`EventKind`.
        phase: ``start`` / ``end`` / ``point``.
        elapsed_ms: Elapsed ms between paired ``start`` / ``end`` events.
            Zero for ``start`` / ``point``.
        payload: Kind-specific structured dict. See module docstring for
            per-kind payload shapes.
    """

    event_id: str
    parent_event_id: str
    session_id: str
    ts: str
    kind: str
    phase: str
    level: str = Level.INFO.value
    elapsed_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise in semantic field order (no alpha sort).

        Keys appear in the order ``event_id, parent_event_id, ts,
        level, kind, phase, elapsed_ms, session_id, payload`` so a raw
        ``cat trace.jsonl`` reads left-to-right like a log line: id,
        time, level, what happened, how long, details.
        """
        return json.dumps(
            {
                "event_id": self.event_id,
                "parent_event_id": self.parent_event_id,
                "ts": self.ts,
                "level": self.level,
                "kind": self.kind,
                "phase": self.phase,
                "elapsed_ms": self.elapsed_ms,
                "session_id": self.session_id,
                "payload": self.payload,
            },
            default=str,
        )


def utc_now_iso() -> str:
    """ISO-8601 UTC with millisecond precision (``YYYY-MM-DDThh:mm:ss.sssZ``).

    Sub-second precision is required because stage / pass spans often
    complete in under a second and second-only timestamps break
    ordering in the rendered companion log.
    """
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t))}.{ms:03d}Z"


__all__ = [
    "DEFAULT_LEVEL_BY_KIND",
    "EventKind",
    "Level",
    "Phase",
    "TraceEvent",
    "category_for",
    "default_level_for",
    "utc_now_iso",
]

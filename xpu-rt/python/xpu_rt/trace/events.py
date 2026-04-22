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
    elapsed_ms: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_id": self.event_id,
                "parent_event_id": self.parent_event_id,
                "session_id": self.session_id,
                "ts": self.ts,
                "kind": self.kind,
                "phase": self.phase,
                "elapsed_ms": self.elapsed_ms,
                "payload": self.payload,
            },
            default=str,
            sort_keys=True,
        )


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["EventKind", "Phase", "TraceEvent", "utc_now_iso"]

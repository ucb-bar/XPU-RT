"""Convenience publishers for specific event kinds.

Each publisher is a thin wrapper that resolves the currently-active
:class:`TraceBus` lazily (so callers don't need to thread it through)
and exposes ``@contextmanager span(...)`` helpers that emit paired
``start`` / ``end`` events with correct elapsed timing and parent-id
nesting.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from compgen.trace.bus import get_active_bus, get_current_llm_turn_id
from compgen.trace.events import EventKind, Phase


class _BasePublisher:
    kind: str = ""

    @classmethod
    def emit(cls, **payload: Any) -> str | None:
        """Emit a single ``point`` event. Returns the event_id (or None if no bus)."""
        bus = get_active_bus()
        if bus is None:
            return None
        return bus.publish(kind=cls.kind, phase=Phase.POINT.value, payload=payload)

    @classmethod
    @contextlib.contextmanager
    def span(
        cls,
        *,
        payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> Iterator[str | None]:
        bus = get_active_bus()
        if bus is None:
            yield None
            return
        with bus.span(cls.kind, payload=payload, end_payload=end_payload) as eid:
            yield eid


class PassPublisher(_BasePublisher):
    """Span a pass invocation. Use as::

        with PassPublisher.span(payload={"name": "fold_transposes_into_dots",
                                         "config": {...}}) as span_id:
            run_pass(...)

    End payload fields (``ir_hash_before``, ``ir_hash_after``, ``stats``)
    are filled in by the caller via ``end_payload``.
    """

    kind = EventKind.PASS_RUN.value


class AnalysisPublisher(_BasePublisher):
    kind = EventKind.ANALYSIS_RUN.value


class DecisionPublisher(_BasePublisher):
    """Records an agent decision: chosen candidate + rationale.

    ``payload`` fields the recorder expects:
        decision_type: one of "pass_select" | "kernel_strategy" | ...
        chosen: free-form description of the chosen option
        candidates: list of considered options
        rationale: human-readable reason (often an LLM response excerpt)
        llm_turn_id: cross-reference to the `llm_response` event.
            Auto-populated from :func:`get_current_llm_turn_id` when
            the caller does not supply it — so every ``decision``
            emitted inside an LLM-driven step links back to that turn.
    """

    kind = EventKind.DECISION.value

    @classmethod
    def emit(cls, **payload: Any) -> str | None:  # type: ignore[override]
        bus = get_active_bus()
        if bus is None:
            return None
        if "llm_turn_id" not in payload or not payload["llm_turn_id"]:
            payload["llm_turn_id"] = get_current_llm_turn_id()
        return bus.publish(kind=cls.kind, phase=Phase.POINT.value, payload=payload)


class IRDumpPublisher(_BasePublisher):
    """Point event — a reference to a file emitted by :class:`IRDumpWriter`."""

    kind = EventKind.IR_DUMP.value


class StagePublisher(_BasePublisher):
    kind = EventKind.STAGE_RUN.value


class DecisionSitePublisher(_BasePublisher):
    """A ``decision_site`` event — emitted when a stage enqueues a choice.

    Non-binding by construction: the event records the candidates and
    the oracle's recommendation, but the actual pick is emitted later
    as a separate :class:`DecisionPublisher` event.
    """

    kind = EventKind.DECISION_SITE.value


class OraclePublisher(_BasePublisher):
    """A ``oracle_advisory`` event — a non-binding oracle output.

    Fired every time an oracle (``fusion_oracle``, ``tile_oracle``,
    ``granularity_oracle``) returns a verdict, even when nothing acts
    on that verdict. Lets reviewers see oracle activity independent of
    the resolved decisions that actually shipped.
    """

    kind = EventKind.ORACLE_ADVISORY.value


class LLMPublisher(_BasePublisher):
    kind = EventKind.LLM_RESPONSE.value


class MCPPublisher(_BasePublisher):
    kind = EventKind.MCP_CALL.value


class ToolPublisher(_BasePublisher):
    kind = EventKind.TOOL_CALL.value


__all__ = [
    "AnalysisPublisher",
    "DecisionPublisher",
    "DecisionSitePublisher",
    "IRDumpPublisher",
    "LLMPublisher",
    "MCPPublisher",
    "OraclePublisher",
    "PassPublisher",
    "StagePublisher",
    "ToolPublisher",
]

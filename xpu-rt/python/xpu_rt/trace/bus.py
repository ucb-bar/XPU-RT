"""Trace bus: single writer for every compilation-trace event.

The bus is the one place in CompGen that decides where trace lines go.
Publishers (:mod:`compgen.trace.publishers`) and recorder adapters
(:mod:`compgen.trace.adapters`) push :class:`TraceEvent` records here;
the bus assigns monotonic IDs, tracks the parent-event stack via
:class:`contextvars.ContextVar`, and serializes writes through a single
lock so the resulting ``trace.jsonl`` is always append-ordered.

Installation:

    bus = install_bus(output_dir=Path("/tmp/run"), session_id="sess_x")

After install, the canonical file is ``<output_dir>/trace/trace.jsonl``
and a mirror link lives at ``<session_dir>/trace.jsonl`` (symlink where
supported, JSON pointer file otherwise).
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

from compgen.trace.events import Phase, TraceEvent, utc_now_iso

log = structlog.get_logger()

_active_bus: ContextVar[TraceBus | None] = ContextVar("compgen_trace_bus", default=None)
_parent_stack: ContextVar[tuple[str, ...]] = ContextVar("compgen_trace_parent_stack", default=())

# Process-wide fallback so a bus installed in one task survives when a
# sibling task starts fresh (MCP stdio server dispatches each request in
# a new anyio task). ``ContextVar`` is still primary — it gives correct
# per-task parent-stack isolation — but if the ContextVar is unset we
# fall through to the last bus ``install_bus`` created in this process.
_PROCESS_BUS: TraceBus | None = None

# The event_id of the most recent ``llm_response`` event in this context.
# :class:`DecisionPublisher` reads this so every ``decision`` event has a
# back-reference to the LLM turn that produced it, closing gap #7.
_current_llm_turn: ContextVar[str] = ContextVar("compgen_trace_llm_turn", default="")


def set_current_llm_turn_id(event_id: str) -> None:
    """Set the ``llm_turn_id`` seen by subsequent :class:`DecisionPublisher` emits.

    Called by :class:`TracingLLMRecorder` whenever it publishes an
    ``llm_response`` event. Scoped to the current task via ContextVar
    so concurrent LLM calls don't clobber each other.
    """
    _current_llm_turn.set(event_id)


def get_current_llm_turn_id() -> str:
    """Read the current ``llm_turn_id`` (empty string when no LLM call is in flight)."""
    return _current_llm_turn.get()


class TraceBus:
    """Thread-safe JSONL trace writer.

    One bus per compilation. The bus owns the output file, the
    monotonic event-id counter, and a :mod:`threading` lock that
    serializes all writes.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        session_id: str,
        session_mirror: Path | None = None,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.session_id = session_id
        self.session_mirror = session_mirror
        self.enabled = enabled
        self._counter = 0
        self._lock = threading.Lock()
        self._trace_dir = self.output_dir / "trace"
        self._trace_path = self._trace_dir / "trace.jsonl"
        if self.enabled:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            self._install_session_mirror()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _install_session_mirror(self) -> None:
        if self.session_mirror is None:
            return
        mirror = Path(self.session_mirror)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mirror.exists() or mirror.is_symlink():
                mirror.unlink()
            os.symlink(self._trace_path, mirror)
        except (OSError, NotImplementedError):
            pointer = mirror.with_name(mirror.name + ".json")
            pointer.write_text(
                json.dumps(
                    {
                        "output_dir": str(self.output_dir),
                        "trace_path": str(self._trace_path),
                        "session_id": self.session_id,
                    }
                )
            )

    # ------------------------------------------------------------------
    # Core publish
    # ------------------------------------------------------------------

    @property
    def trace_path(self) -> Path:
        return self._trace_path

    def next_event_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"evt_{self._counter:010d}"

    def publish(
        self,
        *,
        kind: str,
        phase: str = Phase.POINT.value,
        payload: dict[str, Any] | None = None,
        elapsed_ms: float = 0.0,
        event_id: str | None = None,
        parent_event_id: str | None = None,
    ) -> str:
        """Append one event. Returns the emitted ``event_id``."""
        if not self.enabled:
            return event_id or ""
        eid = event_id or self.next_event_id()
        parent = parent_event_id
        if parent is None:
            stack = _parent_stack.get()
            parent = stack[-1] if stack else ""
        event = TraceEvent(
            event_id=eid,
            parent_event_id=parent,
            session_id=self.session_id,
            ts=utc_now_iso(),
            kind=kind,
            phase=phase,
            elapsed_ms=float(elapsed_ms),
            payload=payload or {},
        )
        with self._lock:
            with self._trace_path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
        return eid

    # ------------------------------------------------------------------
    # Span helpers — used by publishers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def span(
        self,
        kind: str,
        *,
        payload: dict[str, Any] | None = None,
        end_payload: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Context manager emitting a paired ``start`` / ``end`` event.

        Yields the ``event_id`` of the start event so the caller may
        attach later points to it (for example, IR dumps that link to a
        ``pass_run`` span).
        """
        start = self.publish(kind=kind, phase=Phase.START.value, payload=dict(payload or {}))
        stack = _parent_stack.get()
        token = _parent_stack.set(stack + (start,))
        import time as _time

        t0 = _time.time()
        try:
            yield start
        finally:
            elapsed_ms = (_time.time() - t0) * 1000.0
            _parent_stack.reset(token)
            combined: dict[str, Any] = {"span_id": start}
            if end_payload:
                combined.update(end_payload)
            self.publish(
                kind=kind,
                phase=Phase.END.value,
                payload=combined,
                elapsed_ms=elapsed_ms,
                parent_event_id=stack[-1] if stack else "",
            )


# ---------------------------------------------------------------------------
# Module-level accessors
# ---------------------------------------------------------------------------


def install_bus(
    output_dir: Path,
    session_id: str,
    *,
    session_mirror: Path | None = None,
    enabled: bool = True,
) -> TraceBus:
    """Install a fresh :class:`TraceBus` as the active bus.

    Replaces any previously active bus for the current context AND
    updates the process-wide fallback so sibling async tasks — in
    particular MCP ``tools/call`` handlers that spawn their own
    asyncio tasks — can still recover the bus via
    :func:`get_active_bus`.
    """
    global _PROCESS_BUS
    bus = TraceBus(
        output_dir=output_dir,
        session_id=session_id,
        session_mirror=session_mirror,
        enabled=enabled,
    )
    _active_bus.set(bus)
    _PROCESS_BUS = bus
    return bus


def get_active_bus() -> TraceBus | None:
    """Return the bus installed for the current context, or the last
    process-level bus if the ContextVar is unset for this task."""
    bus = _active_bus.get()
    if bus is not None:
        return bus
    return _PROCESS_BUS


def set_active_bus(bus: TraceBus | None) -> None:
    """Replace the active bus (used by tests and by `llm_driver`).

    Also updates the process-level fallback. Passing ``None`` clears
    both so tests start from a clean slate.
    """
    global _PROCESS_BUS
    _active_bus.set(bus)
    _PROCESS_BUS = bus


__all__ = ["TraceBus", "install_bus", "get_active_bus", "set_active_bus"]

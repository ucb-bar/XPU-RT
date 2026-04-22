"""Adapters that connect the existing recorders to the :class:`TraceBus`.

These are **composition** wrappers — none of the original recorder
classes change their public API. Each adapter's ``wrap`` classmethod is
idempotent: if the recorder has already been wrapped (sentinel attr
``_compgen_trace_bus`` set), it is returned unchanged.

Why composition: the three recorders
(:class:`compgen.llm.recorder.LLMRecorder`,
:class:`compgen.llm.recorder.ToolCallRecorder`,
:class:`compgen.mcp.transcript.McpTranscriptRecorder`) are well-tested
and already the canonical storage for their respective payloads. The
trace bus doesn't duplicate their contents — it emits short events that
point back to the per-layer log files via hashes and IDs.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from compgen.trace.bus import TraceBus, get_active_bus, set_current_llm_turn_id
from compgen.trace.events import EventKind, Phase

if TYPE_CHECKING:  # pragma: no cover - imports used only for type checking
    from compgen.llm.recorder import LLMRecorder, ToolCallRecorder
    from compgen.mcp.transcript import McpTranscriptRecorder

_SENTINEL = "_compgen_trace_bus"


def _is_bound(recorder: Any) -> bool:
    """Is this recorder already wrapped by a Tracing* adapter?

    ``True`` means a wrapper has been installed — even if the wrapper's
    bus is ``None`` (in which case the wrapper lazy-resolves via
    :func:`get_active_bus`). We intentionally do NOT treat "bus is None"
    as unbound, because that would cause repeated wrap() calls to stack.
    """
    return getattr(recorder, _SENTINEL, False) is not False


def _mark(recorder: Any, bus: TraceBus | None) -> None:
    # Store the bus or ``True`` when bus is None so ``_is_bound`` treats
    # the recorder as wrapped either way.
    setattr(recorder, _SENTINEL, bus if bus is not None else True)


# ---------------------------------------------------------------------------
# LLMRecorder
# ---------------------------------------------------------------------------


class TracingLLMRecorder:
    """Wrap an :class:`LLMRecorder` to also publish trace events.

    The wrapper exposes the same ``generate`` / ``generate_structured``
    interface as :class:`CompGenLLMProtocol`, forwards to the inner
    recorder, and publishes:

    - ``llm_prompt`` (point) with prompt hash + artifact type
    - ``llm_response`` (point) with token counts + latency + a reference
      to the JSON file :class:`LLMRecorder` just wrote

    ``bus`` is resolved lazily (per call) so the wrapper is robust to
    install ordering — wrapping the recorder before the bus is
    installed still emits events once the bus appears.
    """

    def __init__(self, inner: "LLMRecorder", bus: TraceBus | None) -> None:
        self._inner = inner
        self._bus = bus
        _mark(inner, bus)

    def _resolve_bus(self) -> TraceBus | None:
        return self._bus or get_active_bus()

    # Forwarding to CompGenLLMProtocol
    def generate(self, request: Any) -> Any:
        bus = self._resolve_bus()
        if bus is None:
            return self._inner.generate(request)
        prompt_text = _safe_prompt_text(request)
        prompt_hash = _hash_text(prompt_text)
        bus.publish(
            kind=EventKind.LLM_PROMPT.value,
            phase=Phase.POINT.value,
            payload={
                "prompt_hash": prompt_hash,
                "artifact_type": getattr(request, "artifact_type", ""),
                "prompt_preview": prompt_text[:200],
            },
        )
        response = self._inner.generate(request)
        turn_id = bus.publish(
            kind=EventKind.LLM_RESPONSE.value,
            phase=Phase.POINT.value,
            payload=_response_payload(response, self._inner, prompt_hash),
        )
        # Expose this turn's id so any ``DecisionPublisher.emit`` that
        # follows in this context can back-reference it via ``llm_turn_id``.
        set_current_llm_turn_id(turn_id)
        return response

    def generate_structured(self, request: Any, schema: dict[str, Any]) -> Any:
        bus = self._resolve_bus()
        if bus is None:
            return self._inner.generate_structured(request, schema)
        prompt_text = _safe_prompt_text(request)
        prompt_hash = _hash_text(prompt_text)
        bus.publish(
            kind=EventKind.LLM_PROMPT.value,
            phase=Phase.POINT.value,
            payload={
                "prompt_hash": prompt_hash,
                "artifact_type": getattr(request, "artifact_type", ""),
                "structured": True,
                "prompt_preview": prompt_text[:200],
            },
        )
        response = self._inner.generate_structured(request, schema)
        bus.publish(
            kind=EventKind.LLM_RESPONSE.value,
            phase=Phase.POINT.value,
            payload=_response_payload(response, self._inner, prompt_hash),
        )
        return response

    @property
    def inner(self) -> LLMRecorder:
        return self._inner

    # Allow attribute pass-through for anything we did not override.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @classmethod
    def wrap(cls, recorder: "LLMRecorder", bus: TraceBus | None = None) -> Any:
        if bus is None:
            bus = get_active_bus()
        if _is_bound(recorder):
            return recorder
        return cls(recorder, bus)


def _safe_prompt_text(request: Any) -> str:
    try:
        from compgen.llm._prompt import render_request_prompt

        return render_request_prompt(request)
    except Exception:  # noqa: BLE001
        return repr(request)[:2000]


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _response_payload(response: Any, recorder: LLMRecorder, prompt_hash: str) -> dict[str, Any]:
    return {
        "prompt_hash": prompt_hash,
        "model": getattr(response, "model_id", ""),
        "prompt_tokens": getattr(response, "prompt_tokens", 0),
        "completion_tokens": getattr(response, "completion_tokens", 0),
        "latency_ms": getattr(response, "latency_ms", 0),
        "raw_text_preview": getattr(response, "raw_text", "")[:200],
        "num_artifacts": len(getattr(response, "parsed_artifacts", []) or []),
        "call_id": recorder.total_calls,
    }


# ---------------------------------------------------------------------------
# ToolCallRecorder
# ---------------------------------------------------------------------------


class TracingToolCallRecorder:
    """Wrap a :class:`ToolCallRecorder` to also publish a ``tool_call`` event per record."""

    def __init__(self, inner: "ToolCallRecorder", bus: TraceBus | None) -> None:
        self._inner = inner
        self._bus = bus
        _mark(inner, bus)

    def _resolve_bus(self) -> TraceBus | None:
        return self._bus or get_active_bus()

    def record(self, **kwargs: Any) -> Any:
        record = self._inner.record(**kwargs)
        bus = self._resolve_bus()
        if bus is not None:
            bus.publish(
                kind=EventKind.TOOL_CALL.value,
                phase=Phase.POINT.value,
                payload={
                    "phase": record.phase,
                    "llm_turn_id": record.llm_turn_id,
                    "kind": record.kind,
                    "name": record.name,
                    "select_vs_invent": record.select_vs_invent,
                    "recipe_ir_diff": record.recipe_ir_diff,
                    "gate_result": record.gate_result,
                    "elapsed_ms": record.elapsed_ms,
                },
                elapsed_ms=float(record.elapsed_ms),
            )
        return record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def inner(self) -> "ToolCallRecorder":
        return self._inner

    @classmethod
    def wrap(cls, recorder: "ToolCallRecorder", bus: TraceBus | None = None) -> Any:
        if bus is None:
            bus = get_active_bus()
        if _is_bound(recorder):
            return recorder
        return cls(recorder, bus)


# ---------------------------------------------------------------------------
# McpTranscriptRecorder
# ---------------------------------------------------------------------------


class TracingMcpTranscriptRecorder:
    """Wrap an :class:`McpTranscriptRecorder` and emit ``mcp_call`` events."""

    def __init__(self, inner: "McpTranscriptRecorder", bus: TraceBus | None) -> None:
        self._inner = inner
        self._bus = bus
        _mark(inner, bus)

    def _resolve_bus(self) -> TraceBus | None:
        return self._bus or get_active_bus()

    def record(self, **kwargs: Any) -> Any:
        record = self._inner.record(**kwargs)
        bus = self._resolve_bus()
        if bus is not None:
            bus.publish(
                kind=EventKind.MCP_CALL.value,
                phase=Phase.POINT.value,
                payload={
                    "tool": kwargs.get("tool"),
                    "session_id": kwargs.get("session_id"),
                    "duration_ms": kwargs.get("duration_ms"),
                    "error": kwargs.get("error"),
                    "args_keys": sorted((kwargs.get("args") or {}).keys()),
                },
                elapsed_ms=float(kwargs.get("duration_ms", 0.0)),
            )
        return record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def inner(self) -> "McpTranscriptRecorder":
        return self._inner

    @classmethod
    def wrap(cls, recorder: "McpTranscriptRecorder", bus: TraceBus | None = None) -> Any:
        if bus is None:
            bus = get_active_bus()
        if _is_bound(recorder):
            return recorder
        return cls(recorder, bus)


__all__ = [
    "TracingLLMRecorder",
    "TracingMcpTranscriptRecorder",
    "TracingToolCallRecorder",
]

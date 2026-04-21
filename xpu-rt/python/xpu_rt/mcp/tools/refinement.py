"""MCP tools — agent-driven kernel refinement.

Lifts ``bench/iterate.py``'s try → bench → diagnose → refine loop to
MCP. The Python loop is still callable for non-agentic use; this
surface lets *the agent* decide when to stop refining vs keep going,
and gives every refinement attempt a durable record so the next
session can resume mid-stream.

Flow:

  1. Orchestration calls ``request_refinement`` with the current best
     kernel, the diagnosis from the last bench, and the contract.
  2. Agent reads the pending refinement via
     ``list_pending_refinements``, writes a new kernel source, and
     posts it via ``register_refinement_attempt`` along with whether
     to keep iterating (``done=False``) or accept this attempt as
     converged (``done=True``).
  3. ``lookup_refinement_history`` returns every attempt for a given
     kernel fingerprint so the agent can see what it has already
     tried (and avoid repeating itself).

Cache key = stable hash of (kernel_fingerprint). All attempts for the
same kernel cluster under one history entry.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from compgen.mcp.session import McpSession, SessionManager


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RefinementAttempt:
    """One try at a refined kernel."""

    attempt_id: str
    kernel_source: str
    perf_us: float | None = None
    correct: bool = True
    diagnosis_summary: str = ""
    rationale: str = ""
    timestamp: float = 0.0


@dataclass
class PendingRefinement:
    request_id: str
    prompt: str
    kernel_fingerprint: str
    attempt_index: int = 0
    perf_target_us: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementHistory:
    """Per-kernel ordered attempts list + a converged flag."""

    attempts: list[RefinementAttempt] = field(default_factory=list)
    converged: bool = False


@dataclass
class RefinementCache:
    pending: dict[str, PendingRefinement] = field(default_factory=dict)
    histories: dict[str, RefinementHistory] = field(default_factory=dict)


def _refinement_cache(session: McpSession) -> RefinementCache:
    cache: RefinementCache | None = getattr(session, "refinement_cache", None)
    if cache is None:
        cache = RefinementCache()
        session.refinement_cache = cache    # type: ignore[attr-defined]
    return cache


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_refinement_prompt(
    kernel_fingerprint: str,
    prior_source: str,
    diagnosis_summary: str,
    perf_target_us: float | None,
    attempt_index: int,
) -> str:
    target_line = (
        f"PERF TARGET: ≤{perf_target_us}μs"
        if perf_target_us is not None else "PERF TARGET: unspecified"
    )
    lines = [
        f"Refine kernel {kernel_fingerprint!r} (attempt #{attempt_index}).",
        "",
        target_line,
        "",
        "PRIOR SOURCE",
        "------------",
        prior_source[:4000] + ("\n# ... [trimmed]" if len(prior_source) > 4000 else ""),
        "",
        "DIAGNOSIS",
        "---------",
        diagnosis_summary or "(no diagnosis supplied)",
        "",
        "Reply by calling register_refinement_attempt with:",
        "  kernel_source:        the refined kernel",
        "  perf_us / correct:    measured numbers (None if not run yet)",
        "  done:                 true if you accept this attempt as converged",
        "  rationale:            one short sentence on what you changed and why",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def request_refinement(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_fingerprint: str,
    prior_source: str,
    diagnosis_summary: str = "",
    perf_target_us: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Queue a refinement request for the agent."""
    session = sm.get(session_id)
    cache = _refinement_cache(session)
    history = cache.histories.setdefault(kernel_fingerprint, RefinementHistory())

    if history.converged:
        # Already converged — return the last attempt's source instead of queueing.
        last = history.attempts[-1] if history.attempts else None
        return {
            "ok": True, "session_id": session_id,
            "converged": True,
            "kernel_fingerprint": kernel_fingerprint,
            "attempt_count": len(history.attempts),
            "last_kernel_source": (last.kernel_source if last else ""),
            "last_perf_us": (last.perf_us if last else None),
        }

    rid = request_id or f"refine_{uuid.uuid4().hex[:12]}"
    prompt = _render_refinement_prompt(
        kernel_fingerprint, prior_source, diagnosis_summary,
        perf_target_us, attempt_index=len(history.attempts) + 1,
    )
    cache.pending[rid] = PendingRefinement(
        request_id=rid, prompt=prompt,
        kernel_fingerprint=kernel_fingerprint,
        attempt_index=len(history.attempts) + 1,
        perf_target_us=perf_target_us,
    )
    return {
        "ok": True, "session_id": session_id,
        "converged": False,
        "request_id": rid,
        "kernel_fingerprint": kernel_fingerprint,
        "attempt_index": len(history.attempts) + 1,
        "prompt": prompt,
        "next_call": {
            "tool": "register_refinement_attempt",
            "args": {"session_id": session_id, "request_id": rid,
                     "kernel_source": "<refined source>",
                     "perf_us": "<float|null>", "correct": "<bool>",
                     "done": "<bool>", "rationale": "<string>"},
        },
    }


def register_refinement_attempt(
    sm: SessionManager,
    *,
    session_id: str,
    request_id: str,
    kernel_source: str,
    perf_us: float | None = None,
    correct: bool = True,
    done: bool = False,
    rationale: str = "",
) -> dict[str, Any]:
    """Fulfill a refinement request. ``done=True`` marks the kernel
    converged; further refinement requests for the same fingerprint
    short-circuit until the cache is reset (e.g. perf regresses)."""
    session = sm.get(session_id)
    cache = _refinement_cache(session)
    pending = cache.pending.pop(request_id, None)
    if pending is None:
        return {"ok": False, "session_id": session_id,
                "error": f"unknown or already-fulfilled request_id {request_id!r}"}
    if not kernel_source.strip():
        cache.pending[request_id] = pending
        return {"ok": False, "session_id": session_id,
                "error": "kernel_source is empty; re-queued"}
    history = cache.histories.setdefault(
        pending.kernel_fingerprint, RefinementHistory(),
    )
    attempt = RefinementAttempt(
        attempt_id=f"a{pending.attempt_index:03d}",
        kernel_source=kernel_source,
        perf_us=perf_us, correct=bool(correct),
        diagnosis_summary="",
        rationale=rationale,
        timestamp=time.time(),
    )
    history.attempts.append(attempt)
    if done:
        history.converged = True
    return {
        "ok": True, "session_id": session_id,
        "kernel_fingerprint": pending.kernel_fingerprint,
        "attempt_id": attempt.attempt_id,
        "attempt_count": len(history.attempts),
        "converged": history.converged,
    }


def lookup_refinement_history(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_fingerprint: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _refinement_cache(session)
    history = cache.histories.get(kernel_fingerprint)
    if history is None:
        return {"ok": True, "session_id": session_id,
                "kernel_fingerprint": kernel_fingerprint,
                "attempt_count": 0, "converged": False, "attempts": []}
    return {
        "ok": True, "session_id": session_id,
        "kernel_fingerprint": kernel_fingerprint,
        "attempt_count": len(history.attempts),
        "converged": history.converged,
        "attempts": [
            {
                "attempt_id": a.attempt_id,
                "perf_us": a.perf_us,
                "correct": a.correct,
                "rationale": a.rationale,
                "kernel_source": a.kernel_source,
                "timestamp": a.timestamp,
            }
            for a in history.attempts
        ],
    }


def list_pending_refinements(
    sm: SessionManager, *, session_id: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _refinement_cache(session)
    return {
        "ok": True, "session_id": session_id,
        "pending_count": len(cache.pending),
        "requests": [
            {
                "request_id": rid,
                "kernel_fingerprint": req.kernel_fingerprint,
                "attempt_index": req.attempt_index,
                "perf_target_us": req.perf_target_us,
                "prompt": req.prompt,
            }
            for rid, req in cache.pending.items()
        ],
    }


REFINEMENT_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_refinement",
        "description": (
            "Queue a refinement request for the agent — supplies prior "
            "kernel source + diagnosis. Short-circuits if this kernel "
            "fingerprint has already converged."
        ),
        "phase": "transform",
        "handler": request_refinement,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_fingerprint": {"type": "string"},
                "prior_source": {"type": "string"},
                "diagnosis_summary": {"type": "string"},
                "perf_target_us": {"type": ["number", "null"]},
                "request_id": {"type": ["string", "null"]},
            },
            "required": ["session_id", "kernel_fingerprint", "prior_source"],
        },
    },
    {
        "name": "register_refinement_attempt",
        "description": (
            "Fulfill a pending refinement with a new kernel source. "
            "Setting done=true marks the kernel converged."
        ),
        "phase": "transform",
        "handler": register_refinement_attempt,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "request_id": {"type": "string"},
                "kernel_source": {"type": "string"},
                "perf_us": {"type": ["number", "null"]},
                "correct": {"type": "boolean"},
                "done": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["session_id", "request_id", "kernel_source"],
        },
    },
    {
        "name": "lookup_refinement_history",
        "description": "Return every refinement attempt for a kernel fingerprint.",
        "phase": "inspect",
        "handler": lookup_refinement_history,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_fingerprint": {"type": "string"},
            },
            "required": ["session_id", "kernel_fingerprint"],
        },
    },
    {
        "name": "list_pending_refinements",
        "description": "List outstanding refinement requests for the agent.",
        "phase": "inspect",
        "handler": list_pending_refinements,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "REFINEMENT_TOOLS",
    "PendingRefinement",
    "RefinementAttempt",
    "RefinementCache",
    "RefinementHistory",
    "list_pending_refinements",
    "lookup_refinement_history",
    "register_refinement_attempt",
    "request_refinement",
]

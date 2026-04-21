"""MCP tools — agent-driven kernel bench.

Mirrors the request/register/lookup/list pattern of
``compgen.mcp.tools.kernel``. Lets the agent (Claude Code) be the entity
that runs (or routes) bench measurements:

  1. ``request_kernel_bench`` — orchestration queues a bench request
     for a kernel + a sample-shape signature.
  2. The agent reads pending requests via ``list_pending_bench_requests``,
     either runs the bench locally (if it has CUDA + the kernel source)
     or delegates to a remote runner, and posts the verdict via
     ``register_bench_result``.
  3. ``McpBenchFn`` (a callable that satisfies the optimizer's
     ``BenchFn`` slot) round-trips through these tools — cache hit
     returns the recorded perf; cache miss queues a request and
     returns a placeholder ``BenchResult(perf_us=None, correct=True)``
     so the optimization pass can keep moving and pick up the verdict
     on the next loop iteration.

Bench fingerprint = stable hash of (kernel_fingerprint, shape_signature,
dtype_signature). Two bench requests with the same signatures share one
result across compile passes and across sessions (rehydrated from the
KernelDB on first access if the kernel store wrote a perf record).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from compgen.mcp.session import McpSession, SessionManager
from compgen.memory.kernel_db import KernelDB, KernelPerfRecord, shared_db


# ---------------------------------------------------------------------------
# Cache primitives
# ---------------------------------------------------------------------------


@dataclass
class PendingBenchRequest:
    request_id: str
    prompt: str
    fingerprint: str
    kernel_fingerprint: str
    target: str = ""
    op_family: str = ""
    shape_signature: str = ""
    dtype_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchResultEntry:
    perf_us: float | None
    correct: bool
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchCache:
    pending: dict[str, PendingBenchRequest] = field(default_factory=dict)
    entries: dict[str, BenchResultEntry] = field(default_factory=dict)


def _bench_cache(session: McpSession) -> BenchCache:
    cache: BenchCache | None = getattr(session, "bench_cache", None)
    if cache is None:
        cache = BenchCache()
        session.bench_cache = cache    # type: ignore[attr-defined]
    return cache


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def bench_fingerprint(
    kernel_fingerprint: str,
    shape_signature: str,
    dtype_signature: str,
) -> str:
    payload = json.dumps(
        {
            "kernel_fp": kernel_fingerprint,
            "shapes": shape_signature,
            "dtypes": dtype_signature,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_bench_prompt(
    kernel_fingerprint: str,
    target: str,
    op_family: str,
    shape_signature: str,
    dtype_signature: str,
    perf_target_us: float | None,
) -> str:
    lines = [
        f"Bench kernel {kernel_fingerprint!r} on target {target!r}.",
        "",
        f"OP FAMILY:       {op_family or '?'}",
        f"SHAPE SIGNATURE: {shape_signature or '?'}",
        f"DTYPE SIGNATURE: {dtype_signature or '?'}",
    ]
    if perf_target_us is not None:
        lines.append(f"PERF TARGET:     ≤{perf_target_us}μs")
    lines += [
        "",
        "Reply by calling register_bench_result with:",
        "  perf_us:  measured median latency in microseconds (None on failure)",
        "  correct:  true/false — whether the kernel matches the eager reference",
        "  notes:    short observation (atol, max_rel_err, anything load-bearing)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def request_kernel_bench(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_fingerprint: str,
    shape_signature: str = "",
    dtype_signature: str = "",
    target: str = "",
    op_family: str = "",
    perf_target_us: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Queue a bench request for the agent to fulfil.

    On cache hit (same signature already measured this session OR
    rehydrated from KernelDB), returns the recorded ``perf_us`` +
    ``correct`` directly with ``found_in_cache=True``.
    """
    session = sm.get(session_id)
    cache = _bench_cache(session)
    fp = bench_fingerprint(kernel_fingerprint, shape_signature, dtype_signature)

    cached = cache.entries.get(fp)
    if cached is None and target and op_family:
        # Rehydrate from the persistent KernelDB on first lookup.
        rec = shared_db().best_kernel_perf(target, op_family, kernel_fingerprint)
        if rec is not None:
            cached = BenchResultEntry(
                perf_us=rec.perf_us, correct=rec.correctness_passed,
                notes="loaded from kernel_db",
                metadata={"loaded_from_disk": True, "measured_at": rec.measured_at},
            )
            cache.entries[fp] = cached
    if cached is not None:
        return {
            "ok": True,
            "session_id": session_id,
            "found_in_cache": True,
            "fingerprint": fp,
            "perf_us": cached.perf_us,
            "correct": cached.correct,
            "notes": cached.notes,
        }

    rid = request_id or f"bench_{uuid.uuid4().hex[:12]}"
    prompt = _render_bench_prompt(
        kernel_fingerprint, target, op_family,
        shape_signature, dtype_signature, perf_target_us,
    )
    cache.pending[rid] = PendingBenchRequest(
        request_id=rid, prompt=prompt, fingerprint=fp,
        kernel_fingerprint=kernel_fingerprint,
        target=target, op_family=op_family,
        shape_signature=shape_signature, dtype_signature=dtype_signature,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "found_in_cache": False,
        "request_id": rid,
        "fingerprint": fp,
        "prompt": prompt,
        "next_call": {
            "tool": "register_bench_result",
            "args": {"session_id": session_id, "request_id": rid,
                     "perf_us": "<float|null>", "correct": "<bool>",
                     "notes": "<string>"},
        },
    }


def register_bench_result(
    sm: SessionManager,
    *,
    session_id: str,
    request_id: str,
    perf_us: float | None,
    correct: bool,
    notes: str = "",
) -> dict[str, Any]:
    """Fulfill a pending bench request. Caches in the session AND
    appends a record to the persistent KernelDB so future sessions
    skip the bench."""
    session = sm.get(session_id)
    cache = _bench_cache(session)
    pending = cache.pending.pop(request_id, None)
    if pending is None:
        return {
            "ok": False, "session_id": session_id,
            "error": f"unknown or already-fulfilled request_id {request_id!r}",
        }
    cache.entries[pending.fingerprint] = BenchResultEntry(
        perf_us=perf_us, correct=bool(correct), notes=notes,
    )
    # Persist to KernelDB so future processes/sessions hit cache.
    if pending.target and pending.op_family:
        shared_db().record_kernel_perf(KernelPerfRecord(
            target=pending.target, op_family=pending.op_family,
            fingerprint=pending.kernel_fingerprint,
            perf_us=float(perf_us or 0.0),
            correctness_passed=bool(correct),
            measured_at=time.time(),
            notes=f"agent: {notes}" if notes else "agent",
        ))
    return {
        "ok": True, "session_id": session_id,
        "fingerprint": pending.fingerprint,
        "cached_results": len(cache.entries),
    }


def lookup_bench_result(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_fingerprint: str,
    shape_signature: str = "",
    dtype_signature: str = "",
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _bench_cache(session)
    fp = bench_fingerprint(kernel_fingerprint, shape_signature, dtype_signature)
    entry = cache.entries.get(fp)
    if entry is None:
        return {"ok": True, "session_id": session_id,
                "found": False, "fingerprint": fp}
    return {
        "ok": True, "session_id": session_id,
        "found": True, "fingerprint": fp,
        "perf_us": entry.perf_us, "correct": entry.correct,
        "notes": entry.notes,
    }


def list_pending_bench_requests(
    sm: SessionManager, *, session_id: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _bench_cache(session)
    return {
        "ok": True, "session_id": session_id,
        "pending_count": len(cache.pending),
        "requests": [
            {
                "request_id": rid,
                "prompt": req.prompt,
                "fingerprint": req.fingerprint,
                "kernel_fingerprint": req.kernel_fingerprint,
                "target": req.target,
                "op_family": req.op_family,
                "shape_signature": req.shape_signature,
            }
            for rid, req in cache.pending.items()
        ],
    }


# ---------------------------------------------------------------------------
# BenchFn adapter for compgen.agent.kernel_optimizer
# ---------------------------------------------------------------------------


@dataclass
class McpBenchFn:
    """Routes the optimizer's per-region bench through MCP.

    Cache hit → returns the recorded perf. Cache miss → queues a
    pending request for the agent and returns a placeholder
    ``BenchResult(perf_us=None, correct=True)`` so the optimizer can
    proceed; the next pass will hit the cache.
    """

    sm: SessionManager
    session_id: str

    def __call__(self, contract, codegen_result):  # noqa: ANN001
        from compgen.agent.kernel_optimizer import (
            BenchResult,
            fingerprint_for,
        )

        kernel_fp = fingerprint_for(contract)
        target = contract.orchestration.execution.hardware.target_name
        op_family = contract.archetype.value
        shape_sig = ";".join(
            ",".join(str(d) for d in t.shape.dims)
            for t in contract.io.inputs
        )
        dtype_sig = ";".join(
            "/".join(t.dtype_class) for t in contract.io.inputs
        )
        out = request_kernel_bench(
            self.sm, session_id=self.session_id,
            kernel_fingerprint=kernel_fp,
            shape_signature=shape_sig, dtype_signature=dtype_sig,
            target=target, op_family=op_family,
        )
        if out.get("found_in_cache"):
            return BenchResult(
                perf_us=out.get("perf_us"),
                correct=bool(out.get("correct", True)),
                notes=out.get("notes", ""),
            )
        return BenchResult(perf_us=None, correct=True,
                           notes="bench queued via MCP")


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


BENCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_kernel_bench",
        "description": (
            "Queue a kernel-bench request for the agent. On cache hit "
            "(same kernel × shape × dtype already measured), returns "
            "the recorded perf and correctness directly."
        ),
        "phase": "transform",
        "handler": request_kernel_bench,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_fingerprint": {"type": "string"},
                "shape_signature": {"type": "string"},
                "dtype_signature": {"type": "string"},
                "target": {"type": "string"},
                "op_family": {"type": "string"},
                "perf_target_us": {"type": ["number", "null"]},
                "request_id": {"type": ["string", "null"]},
            },
            "required": ["session_id", "kernel_fingerprint"],
        },
    },
    {
        "name": "register_bench_result",
        "description": (
            "Fulfill a pending bench request with the measured perf "
            "and correctness. Persists to the on-disk KernelDB."
        ),
        "phase": "transform",
        "handler": register_bench_result,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "request_id": {"type": "string"},
                "perf_us": {"type": ["number", "null"]},
                "correct": {"type": "boolean"},
                "notes": {"type": "string"},
            },
            "required": ["session_id", "request_id", "perf_us", "correct"],
        },
    },
    {
        "name": "lookup_bench_result",
        "description": "Cache lookup by (kernel × shape × dtype) fingerprint.",
        "phase": "inspect",
        "handler": lookup_bench_result,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_fingerprint": {"type": "string"},
                "shape_signature": {"type": "string"},
                "dtype_signature": {"type": "string"},
            },
            "required": ["session_id", "kernel_fingerprint"],
        },
    },
    {
        "name": "list_pending_bench_requests",
        "description": "List outstanding bench requests for the agent to fulfil.",
        "phase": "inspect",
        "handler": list_pending_bench_requests,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "BENCH_TOOLS",
    "BenchCache",
    "BenchResultEntry",
    "McpBenchFn",
    "PendingBenchRequest",
    "bench_fingerprint",
    "list_pending_bench_requests",
    "lookup_bench_result",
    "register_bench_result",
    "request_kernel_bench",
]

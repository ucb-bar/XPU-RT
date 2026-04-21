"""MCP tools — in-session kernel codegen + cache.

The single-process MCP transport processes tool calls sequentially, so we
can't have a provider *block* on a callback from inside a tool handler.
Instead the in-session codegen flow is two-phase:

  1. The orchestration layer calls ``request_kernel_codegen`` for each
     KernelContractV3 it needs. Each request lands in the session's
     pending queue with a structured prompt.
  2. The agent (Claude Code) reads the pending requests via
     ``list_pending_kernel_requests``, generates each kernel from the
     prompt's KernelFacingView, and calls ``register_kernel_result`` to
     fulfill them. Results land in the in-session cache.
  3. ``ClaudeCodeKernelProvider``'s ``InSessionCodegen`` callable
     consults the cache; cache hit = zero extra API cost (everything
     happens inside the calling Claude Code session).

Cache misses make the provider return ``found=False``, which the
escalation router falls through to autocomp on.

Cache fingerprint = stable hash of (op_name, archetype, granularity, IO
shapes / dtypes / layouts, attributes, target_name). Two regions with
the same fingerprint share one kernel even across compile passes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from compgen.kernels.store import KernelStore, shared_store
from compgen.mcp.session import McpSession, SessionManager


# ---------------------------------------------------------------------------
# In-session cache structures
# ---------------------------------------------------------------------------


@dataclass
class PendingRequest:
    request_id: str
    prompt: str
    fingerprint: str
    perf_target_us: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Salient contract fields kept for the disk-store write-through.
    op_name: str = ""
    archetype: str = ""
    granularity: str = ""
    target: str = ""


@dataclass
class KernelCacheEntry:
    kernel_code: str
    language: str = "unknown"
    perf_us: float | None = None
    correctness_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelCache:
    """Per-session pending-requests + fulfilled-results store."""

    pending: dict[str, PendingRequest] = field(default_factory=dict)
    entries: dict[str, KernelCacheEntry] = field(default_factory=dict)


def _kernel_cache(session: McpSession) -> KernelCache:
    """Lazy-initialise the per-session kernel cache.

    On first access, rehydrates entries from the on-disk
    :class:`KernelStore` so kernels generated in a prior session land
    pre-warmed in this one. Disk persistence makes the kernel cache
    cross-session: write once, every future ``compile_with_llm`` /
    Claude Code session sees the kernel.
    """
    cache = session.kernel_cache
    if cache is None:
        cache = KernelCache()
        session.kernel_cache = cache
        store = shared_store()
        for stored in store.list_all():
            payload = store.get(stored.fingerprint)
            if payload is None:
                continue
            entry, source = payload
            cache.entries[stored.fingerprint] = KernelCacheEntry(
                kernel_code=source,
                language=entry.language,
                perf_us=entry.perf_us,
                correctness_passed=entry.correctness_passed,
                metadata={"loaded_from_disk": True, "path": entry.path},
            )
    return cache


# ---------------------------------------------------------------------------
# Fingerprinting + prompt rendering
# ---------------------------------------------------------------------------


def contract_fingerprint(contract_v3: dict) -> str:
    """Stable fingerprint over the kernel-relevant subset of a v3 contract.

    Two contracts that the kernel implementation would treat identically
    must share a fingerprint, so the cache hits across compile passes
    and across regions.
    """
    relevant = {
        "op_name": contract_v3.get("op_name"),
        "archetype": contract_v3.get("archetype"),
        "granularity": contract_v3.get("granularity"),
        "io": contract_v3.get("io"),
        # Target name lives under orchestration.execution.hardware
        "target": (
            ((contract_v3.get("orchestration") or {}).get("execution") or {})
            .get("hardware", {}).get("target_name")
        ),
    }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _render_prompt(contract_v3: dict, perf_target_us: float | None) -> str:
    """Render a v3 contract into the prompt the agent will see.

    Keeps it terse — the agent already has the v3 schema in its context;
    we just surface the *concrete* values for this contract.
    """
    archetype = contract_v3.get("archetype", "?")
    op_name = contract_v3.get("op_name", "?")
    target = (
        ((contract_v3.get("orchestration") or {}).get("execution") or {})
        .get("hardware", {}).get("target_name", "?")
    )
    io = contract_v3.get("io", {})
    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    attrs = io.get("attributes", [])
    numerics = io.get("numerics", {})

    lines = [
        f"Generate a {archetype} kernel for {op_name!r} on target {target!r}.",
        "",
        "INPUTS:",
        *[f"  - {t.get('name')}: shape={t.get('shape', {}).get('dims')} "
          f"dtype_class={t.get('dtype_class')} layout={t.get('layout')} "
          f"alignment={t.get('alignment_bytes')}B"
          for t in inputs],
        "",
        "OUTPUTS:",
        *[f"  - {t.get('name')}: shape={t.get('shape', {}).get('dims')} "
          f"dtype_class={t.get('dtype_class')} layout={t.get('layout')}"
          for t in outputs],
        "",
        f"ATTRIBUTES: {attrs}",
        f"NUMERICS:   {numerics}",
    ]
    if perf_target_us is not None:
        lines.append(f"PERF TARGET: ≤{perf_target_us}μs")
    lines.append("")
    lines.append(
        "Respond by calling register_kernel_result with the kernel source "
        "(no markdown fences, no explanation)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def request_kernel_codegen(
    sm: SessionManager,
    *,
    session_id: str,
    contract_v3: dict,
    perf_target_us: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Register a kernel-codegen request in the session's pending queue.

    Returns the ``request_id`` and the rendered prompt the agent should
    respond to via ``register_kernel_result``. If the contract's
    fingerprint already has a cache entry, returns the cached kernel
    immediately (``found=True``) and skips queueing.
    """
    session = sm.get(session_id)
    cache = _kernel_cache(session)
    fingerprint = contract_fingerprint(contract_v3)

    cached = cache.entries.get(fingerprint)
    if cached is not None:
        return {
            "ok": True,
            "session_id": session_id,
            "found_in_cache": True,
            "fingerprint": fingerprint,
            "kernel_code": cached.kernel_code,
            "language": cached.language,
        }

    rid = request_id or f"req_{uuid.uuid4().hex[:12]}"
    prompt = _render_prompt(contract_v3, perf_target_us)
    target_name = (
        ((contract_v3.get("orchestration") or {}).get("execution") or {})
        .get("hardware", {}).get("target_name", "")
    )
    cache.pending[rid] = PendingRequest(
        request_id=rid,
        prompt=prompt,
        fingerprint=fingerprint,
        perf_target_us=perf_target_us,
        op_name=str(contract_v3.get("op_name", "")),
        archetype=str(contract_v3.get("archetype", "")),
        granularity=str(contract_v3.get("granularity", "")),
        target=target_name,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "found_in_cache": False,
        "request_id": rid,
        "fingerprint": fingerprint,
        "prompt": prompt,
        "next_call": {
            "tool": "register_kernel_result",
            "args": {"session_id": session_id, "request_id": rid,
                     "kernel_code": "<your kernel source>",
                     "language": "<triton|cuda|c|python|...>"},
        },
    }


def register_kernel_result(
    sm: SessionManager,
    *,
    session_id: str,
    request_id: str,
    kernel_code: str,
    language: str = "unknown",
    perf_us: float | None = None,
    correctness_passed: bool = False,
) -> dict[str, Any]:
    """Fulfill a pending codegen request. Stores in cache by fingerprint."""
    session = sm.get(session_id)
    cache = _kernel_cache(session)
    pending = cache.pending.pop(request_id, None)
    if pending is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"unknown or already-fulfilled request_id {request_id!r}",
        }
    if not kernel_code.strip():
        # Re-queue so the agent can retry; empty body would poison the cache.
        cache.pending[request_id] = pending
        return {
            "ok": False,
            "session_id": session_id,
            "error": "kernel_code is empty; re-queued request",
        }
    cache.entries[pending.fingerprint] = KernelCacheEntry(
        kernel_code=kernel_code,
        language=language,
        perf_us=perf_us,
        correctness_passed=correctness_passed,
    )
    # Write through to the on-disk store so the kernel survives this
    # session — future sessions / processes pick it up via the
    # rehydrate-on-open path in ``_kernel_cache``.
    stored = shared_store().put(
        pending.fingerprint,
        kernel_code,
        target=pending.target,
        language=language,
        op_name=pending.op_name,
        archetype=pending.archetype,
        granularity=pending.granularity,
        perf_us=perf_us,
        correctness_passed=correctness_passed,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "fingerprint": pending.fingerprint,
        "cached_kernels": len(cache.entries),
        "persisted_path": stored.path,
    }


def lookup_cached_kernel(
    sm: SessionManager,
    *,
    session_id: str,
    contract_v3: dict,
) -> dict[str, Any]:
    """Cache hit-or-miss check by v3-contract fingerprint."""
    session = sm.get(session_id)
    cache = _kernel_cache(session)
    fp = contract_fingerprint(contract_v3)
    entry = cache.entries.get(fp)
    if entry is None:
        return {
            "ok": True,
            "session_id": session_id,
            "found": False,
            "fingerprint": fp,
        }
    return {
        "ok": True,
        "session_id": session_id,
        "found": True,
        "fingerprint": fp,
        "kernel_code": entry.kernel_code,
        "language": entry.language,
        "perf_us": entry.perf_us,
        "correctness_passed": entry.correctness_passed,
    }


def list_pending_kernel_requests(
    sm: SessionManager,
    *,
    session_id: str,
) -> dict[str, Any]:
    """Snapshot of outstanding requests so the agent can fulfill them in
    a batch (one ``register_kernel_result`` call per request)."""
    session = sm.get(session_id)
    cache = _kernel_cache(session)
    return {
        "ok": True,
        "session_id": session_id,
        "pending_count": len(cache.pending),
        "requests": [
            {
                "request_id": rid,
                "prompt": req.prompt,
                "fingerprint": req.fingerprint,
                "perf_target_us": req.perf_target_us,
            }
            for rid, req in cache.pending.items()
        ],
    }


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


KERNEL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_kernel_codegen",
        "description": (
            "Register a kernel-codegen request. The agent fulfills it by "
            "calling register_kernel_result with the generated source. "
            "On cache hit (same KernelContractV3 fingerprint already "
            "fulfilled), returns the cached kernel directly."
        ),
        "phase": "transform",
        "handler": request_kernel_codegen,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "contract_v3": {"type": "object"},
                "perf_target_us": {"type": ["number", "null"]},
                "request_id": {"type": ["string", "null"]},
            },
            "required": ["session_id", "contract_v3"],
        },
    },
    {
        "name": "register_kernel_result",
        "description": (
            "Fulfill a pending codegen request with kernel source. "
            "Stores in the session's cache by contract-fingerprint so "
            "subsequent identical requests hit immediately."
        ),
        "phase": "transform",
        "handler": register_kernel_result,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "request_id": {"type": "string"},
                "kernel_code": {"type": "string"},
                "language": {"type": "string"},
                "perf_us": {"type": ["number", "null"]},
                "correctness_passed": {"type": "boolean"},
            },
            "required": ["session_id", "request_id", "kernel_code"],
        },
    },
    {
        "name": "lookup_cached_kernel",
        "description": (
            "Check whether a kernel for this v3 contract is already cached."
        ),
        "phase": "inspect",
        "handler": lookup_cached_kernel,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "contract_v3": {"type": "object"},
            },
            "required": ["session_id", "contract_v3"],
        },
    },
    {
        "name": "list_pending_kernel_requests",
        "description": (
            "List all outstanding codegen requests in the session so the "
            "agent can fulfill them in a batch."
        ),
        "phase": "inspect",
        "handler": list_pending_kernel_requests,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "KERNEL_TOOLS",
    "KernelCache",
    "KernelCacheEntry",
    "PendingRequest",
    "contract_fingerprint",
    "list_pending_kernel_requests",
    "lookup_cached_kernel",
    "register_kernel_result",
    "request_kernel_codegen",
]

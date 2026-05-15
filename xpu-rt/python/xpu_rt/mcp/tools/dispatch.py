"""MCP tools — in-session HW-aware dispatch decisions.

Mirror of ``compgen.mcp.tools.kernel`` but for the dispatch decision
flow (W6.4): the orchestration layer asks the agent (Claude Code) to
read a region's hardware envelopes and decide which target +
granularity is best.

Two-phase, like the kernel-codegen flow:

  1. Orchestration calls ``request_dispatch_decision`` for a region
     with one-or-more candidate ``HardwareEnvelope`` summaries. The
     request lands in the session's pending queue with a structured
     prompt.
  2. The agent reads pending requests via
     ``list_pending_dispatch_decisions``, analyses each region/spec,
     and calls ``register_dispatch_decision`` with a JSON verdict
     of the same shape ``hw_aware_dispatch._parse_llm_decision``
     consumes.
  3. ``McpDispatchLLM`` (an LLM-protocol adapter that round-trips
     through these tools) lets ``decide_dispatch`` reuse the agent
     loop transparently.

The tools never call any external API — every "LLM call" round-trips
through Claude Code in the same session.

Cache fingerprint = stable hash of (region op_names, target names,
perf budget, objective). Identical region × identical target list
shares one decision across compile passes.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from compgen.llm.base import (
    GenerationRequest,
    GenerationResponse,
    Objective,
)
from compgen.mcp.session import McpSession, SessionManager

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PendingDispatchRequest:
    request_id: str
    prompt: str
    fingerprint: str
    region_summary: str = ""
    target_names: list[str] = field(default_factory=list)
    perf_budget_us: float | None = None
    objective: str = "latency"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchDecisionEntry:
    """Cached agent verdict — the JSON the agent submitted."""

    decision_json: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchCache:
    """Per-session pending requests + fulfilled decisions."""

    pending: dict[str, PendingDispatchRequest] = field(default_factory=dict)
    entries: dict[str, DispatchDecisionEntry] = field(default_factory=dict)


def _dispatch_cache(session: McpSession) -> DispatchCache:
    """Lazy-initialise the per-session dispatch cache.

    Stored under ``session.dispatch_cache`` (added on first access — the
    session object accepts ad-hoc attributes since it's a plain
    dataclass)."""
    cache: DispatchCache | None = getattr(session, "dispatch_cache", None)
    if cache is None:
        cache = DispatchCache()
        session.dispatch_cache = cache  # type: ignore[attr-defined]
    return cache


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def dispatch_fingerprint(
    region_op_names: list[str],
    envelope_targets: list[str],
    perf_budget_us: float | None,
    objective: str,
) -> str:
    payload = json.dumps(
        {
            "region": region_op_names,
            "targets": envelope_targets,
            "budget": perf_budget_us,
            "objective": objective,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Prompt rendering — what the agent sees
# ---------------------------------------------------------------------------


def _render_dispatch_prompt(
    region_summary: str,
    envelope_summaries: list[str],
    priors: list[dict[str, Any]],
    perf_budget_us: float | None,
    objective: str,
) -> str:
    lines = [
        "You are the in-session dispatch oracle. Decide for each candidate",
        "target which kernel granularity (MICRO ukernel / NORMAL kernel /",
        "MEGA persistent kernel) the region should be dispatched as, AND",
        "pick the BEST target overall under the stated optimisation budget.",
        "",
        "REGION",
        "------",
        region_summary,
        "",
        "CANDIDATE TARGETS",
        "-----------------",
        *[f"  - {s}" for s in envelope_summaries],
        "",
        "DETERMINISTIC PRIORS",
        "--------------------",
        *[
            f"  - {p['target']}: granularity={p['granularity']} "
            f"(confidence={p['confidence']:.2f}); reason: {p['reason']}"
            for p in priors
        ],
        "",
        f"PERF BUDGET: {perf_budget_us}us" if perf_budget_us else "PERF BUDGET: unspecified",
        f"OBJECTIVE:   {objective}",
        "",
        "Reply by calling register_dispatch_decision with this JSON shape:",
        "",
        '{ "per_target": { "<target>": { "granularity": "micro|normal|mega",',
        '                                 "rationale": "<one sentence>" } },',
        '  "best_target": "<target>",',
        '  "best_rationale": "<one sentence>" }',
    ]
    return "\n".join(lines)


def _envelope_summary(env: dict[str, Any]) -> str:
    parts = [
        f"target={env.get('target_name', '?')}",
        f"vector_lanes={env.get('vector_lanes', '?')}",
        f"scratchpad_bytes={env.get('scratchpad_bytes', '?')}",
        f"register_bytes={env.get('register_bytes', '?')}",
    ]
    if env.get("peak_bandwidth_gbps"):
        parts.append(f"bw={env['peak_bandwidth_gbps']:.0f}GB/s")
    if env.get("native_dtypes"):
        parts.append(f"dtypes={','.join(env['native_dtypes'])}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# MCP tool handlers
# ---------------------------------------------------------------------------


def request_dispatch_decision(
    sm: SessionManager,
    *,
    session_id: str,
    region_summary: str,
    region_op_names: list[str],
    envelopes: list[dict[str, Any]],
    priors: list[dict[str, Any]] | None = None,
    perf_budget_us: float | None = None,
    objective: str = "latency",
    request_id: str | None = None,
) -> dict[str, Any]:
    """Queue a HW-aware dispatch decision for the agent to fulfil.

    On cache hit (same fingerprint already decided this session OR a
    prior session via memory load), returns the cached decision JSON
    directly with ``found_in_cache=True``. Otherwise queues a pending
    request and returns the prompt the agent should answer via
    ``register_dispatch_decision``.

    Args:
        region_summary: One-line description of the region.
        region_op_names: Op names in the region — used for fingerprint.
        envelopes: List of HardwareEnvelope-shaped dicts.
        priors: Optional list of deterministic-oracle priors per
            target (each dict: ``target``, ``granularity``, ``confidence``,
            ``reason``). Surfaced in the prompt so the agent has a
            starting point.
        perf_budget_us: Optional budget.
        objective: ``latency``/``throughput``/``memory``/``energy``.
    """
    session = sm.get(session_id)
    cache = _dispatch_cache(session)
    target_names = [str(e.get("target_name", "")) for e in envelopes]
    fp = dispatch_fingerprint(region_op_names, target_names, perf_budget_us, objective)

    cached = cache.entries.get(fp)
    if cached is not None:
        return {
            "ok": True,
            "session_id": session_id,
            "found_in_cache": True,
            "fingerprint": fp,
            "decision_json": cached.decision_json,
        }

    rid = request_id or f"disp_{uuid.uuid4().hex[:12]}"
    prompt = _render_dispatch_prompt(
        region_summary,
        [_envelope_summary(e) for e in envelopes],
        priors or [],
        perf_budget_us,
        objective,
    )
    cache.pending[rid] = PendingDispatchRequest(
        request_id=rid,
        prompt=prompt,
        fingerprint=fp,
        region_summary=region_summary,
        target_names=list(target_names),
        perf_budget_us=perf_budget_us,
        objective=objective,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "found_in_cache": False,
        "request_id": rid,
        "fingerprint": fp,
        "prompt": prompt,
        "next_call": {
            "tool": "register_dispatch_decision",
            "args": {
                "session_id": session_id,
                "request_id": rid,
                "decision_json": "<JSON object — see prompt>",
            },
        },
    }


def register_dispatch_decision(
    sm: SessionManager,
    *,
    session_id: str,
    request_id: str,
    decision_json: str,
) -> dict[str, Any]:
    """Fulfill a pending dispatch decision."""
    session = sm.get(session_id)
    cache = _dispatch_cache(session)
    pending = cache.pending.pop(request_id, None)
    if pending is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"unknown or already-fulfilled request_id {request_id!r}",
        }
    body = decision_json.strip()
    if not body:
        cache.pending[request_id] = pending
        return {
            "ok": False,
            "session_id": session_id,
            "error": "decision_json is empty; re-queued request",
        }
    # Validate JSON shape minimally — must parse + carry per_target
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        cache.pending[request_id] = pending
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"decision_json is not valid JSON: {exc}",
        }
    if not isinstance(parsed, dict) or "per_target" not in parsed:
        cache.pending[request_id] = pending
        return {
            "ok": False,
            "session_id": session_id,
            "error": "decision_json must be an object with a 'per_target' key",
        }
    cache.entries[pending.fingerprint] = DispatchDecisionEntry(decision_json=body)
    return {
        "ok": True,
        "session_id": session_id,
        "fingerprint": pending.fingerprint,
        "cached_decisions": len(cache.entries),
    }


def lookup_dispatch_decision(
    sm: SessionManager,
    *,
    session_id: str,
    region_op_names: list[str],
    envelope_targets: list[str],
    perf_budget_us: float | None = None,
    objective: str = "latency",
) -> dict[str, Any]:
    """Cache hit-or-miss check by fingerprint."""
    session = sm.get(session_id)
    cache = _dispatch_cache(session)
    fp = dispatch_fingerprint(region_op_names, envelope_targets, perf_budget_us, objective)
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
        "decision_json": entry.decision_json,
    }


def list_pending_dispatch_decisions(
    sm: SessionManager,
    *,
    session_id: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _dispatch_cache(session)
    return {
        "ok": True,
        "session_id": session_id,
        "pending_count": len(cache.pending),
        "requests": [
            {
                "request_id": rid,
                "prompt": req.prompt,
                "fingerprint": req.fingerprint,
                "region_summary": req.region_summary,
                "target_names": req.target_names,
                "perf_budget_us": req.perf_budget_us,
                "objective": req.objective,
            }
            for rid, req in cache.pending.items()
        ],
    }


# ---------------------------------------------------------------------------
# CompGenLLMProtocol adapter — lets the optimizer route through MCP
# ---------------------------------------------------------------------------


@dataclass
class McpDispatchLLM:
    """Adapter that lets ``decide_dispatch`` route the LLM call through
    the in-session MCP dispatch-decision flow.

    Usage::

        llm = McpDispatchLLM(sm=session_manager, session_id="default")
        verdict = decide_dispatch(region, envelopes=[env], llm=llm)

    Behaviour:
      * On cache hit, returns the cached decision JSON immediately
        (one MCP round-trip).
      * On miss, queues the request and *blocks waiting* is not an
        option in single-process MCP. Instead we return an empty
        response — ``decide_dispatch``'s parser then fails to parse
        and falls back to the deterministic oracle. The caller is
        expected to drive the agent through ``list_pending_*`` +
        ``register_dispatch_decision`` between optimisation passes.

    This mirrors how ``ClaudeCodeKernelProvider`` works for kernel
    codegen: cache-hit-fast-path + drop-through-on-miss.
    """

    sm: SessionManager
    session_id: str
    perf_budget_us: float | None = None
    objective: Objective = Objective.LATENCY

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Reconstruct the dispatch context from the request — we encoded
        # the region summary + envelope summaries into context fields.
        ctx = request.context
        region_summary = ctx.model_ir_summary or ""
        envelope_summaries = (ctx.target_profile_summary or "").splitlines()
        # Try the cache first (no harm if miss).
        target_names = [
            line.split("target=")[1].split(" ")[0].strip("|").strip()
            for line in envelope_summaries
            if "target=" in line
        ]
        op_names = [region_summary]
        objective = ctx.objective.value if ctx.objective else self.objective.value
        cached = lookup_dispatch_decision(
            self.sm,
            session_id=self.session_id,
            region_op_names=op_names,
            envelope_targets=target_names,
            perf_budget_us=self.perf_budget_us,
            objective=objective,
        )
        if cached.get("found"):
            return GenerationResponse(
                raw_text=cached["decision_json"],
                parsed_artifacts=[],
                model_id="mcp-dispatch-cache",
            )
        # Miss → queue it for the agent. Return empty so the caller's
        # parser falls back to the deterministic oracle this pass.
        envelopes_dicts: list[dict[str, Any]] = []
        for line in envelope_summaries:
            d: dict[str, Any] = {}
            for kv in line.split("|"):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                d[k.strip()] = v.strip()
            if d:
                envelopes_dicts.append(d)
        request_dispatch_decision(
            self.sm,
            session_id=self.session_id,
            region_summary=region_summary,
            region_op_names=op_names,
            envelopes=envelopes_dicts,
            perf_budget_us=self.perf_budget_us,
            objective=objective,
        )
        return GenerationResponse(
            raw_text="",
            parsed_artifacts=[],
            model_id="mcp-dispatch-pending",
        )

    def generate_structured(self, request, schema):  # noqa: ANN001
        return self.generate(request)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


DISPATCH_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_dispatch_decision",
        "description": (
            "Queue a HW-aware dispatch decision for the agent to fulfil. "
            "On cache hit, returns the previously-decided JSON directly."
        ),
        "phase": "transform",
        "handler": request_dispatch_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "region_summary": {"type": "string"},
                "region_op_names": {"type": "array", "items": {"type": "string"}},
                "envelopes": {"type": "array", "items": {"type": "object"}},
                "priors": {"type": ["array", "null"], "items": {"type": "object"}},
                "perf_budget_us": {"type": ["number", "null"]},
                "objective": {"type": "string"},
                "request_id": {"type": ["string", "null"]},
            },
            "required": ["session_id", "region_summary", "region_op_names", "envelopes"],
        },
    },
    {
        "name": "register_dispatch_decision",
        "description": (
            "Fulfill a pending dispatch decision with a JSON verdict. "
            "Validates the JSON shape and stores it in the session's cache."
        ),
        "phase": "transform",
        "handler": register_dispatch_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "request_id": {"type": "string"},
                "decision_json": {"type": "string"},
            },
            "required": ["session_id", "request_id", "decision_json"],
        },
    },
    {
        "name": "lookup_dispatch_decision",
        "description": (
            "Check whether a dispatch decision for this region × target list is already cached in the session."
        ),
        "phase": "inspect",
        "handler": lookup_dispatch_decision,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "region_op_names": {"type": "array", "items": {"type": "string"}},
                "envelope_targets": {"type": "array", "items": {"type": "string"}},
                "perf_budget_us": {"type": ["number", "null"]},
                "objective": {"type": "string"},
            },
            "required": ["session_id", "region_op_names", "envelope_targets"],
        },
    },
    {
        "name": "list_pending_dispatch_decisions",
        "description": "List outstanding dispatch decisions for the agent to fulfil.",
        "phase": "inspect",
        "handler": list_pending_dispatch_decisions,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "DISPATCH_TOOLS",
    "DispatchCache",
    "DispatchDecisionEntry",
    "McpDispatchLLM",
    "PendingDispatchRequest",
    "dispatch_fingerprint",
    "list_pending_dispatch_decisions",
    "lookup_dispatch_decision",
    "register_dispatch_decision",
    "request_dispatch_decision",
]

"""MCP tools — agent-driven Triton autotune trials.

Per-shape autotune today is a Python-only loop: ``@triton.autotune``
sweeps the configs at first call, records the winner, persists via
``compgen.bench.autotune_cache``. With these tools the agent can:

  * gate per-shape trials (skip when a winner is already cached)
  * propose a *narrowed* config grid for the next trial based on
    its knowledge brief / lessons
  * record the picked config so the on-disk
    ``~/.compgen/autotune/`` cache surfaces it for every future
    process / session

Cache key = ``(kernel_qualname, key_tuple_repr)`` mirroring Triton's
internal autotune key. Two trials at the same key share one pick.
"""

from __future__ import annotations

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
class AutotunePick:
    """Best-known config for a (kernel, key) pair."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    maxnreg: int | None = None
    perf_us: float | None = None
    notes: str = ""
    timestamp: float = 0.0


@dataclass
class PendingAutotuneTrial:
    request_id: str
    prompt: str
    kernel_qualname: str
    key_repr: str
    candidate_configs: list[dict[str, Any]] = field(default_factory=list)
    perf_target_us: float | None = None


@dataclass
class AutotuneCache:
    pending: dict[str, PendingAutotuneTrial] = field(default_factory=dict)
    picks: dict[str, AutotunePick] = field(default_factory=dict)


def _autotune_cache(session: McpSession) -> AutotuneCache:
    cache: AutotuneCache | None = getattr(session, "autotune_cache", None)
    if cache is None:
        cache = AutotuneCache()
        session.autotune_cache = cache  # type: ignore[attr-defined]
    return cache


def _autotune_key(kernel_qualname: str, key_repr: str) -> str:
    return f"{kernel_qualname}::{key_repr}"


# ---------------------------------------------------------------------------
# On-disk persistence (delegates to compgen.bench.autotune_cache layout)
# ---------------------------------------------------------------------------


def _disk_path(kernel_qualname: str):
    """Path to the JSON file ``compgen.bench.autotune_cache`` writes."""
    from compgen.bench.autotune_cache import default_cache_root

    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in kernel_qualname)
    return default_cache_root() / f"{safe}.json"


def _load_disk_picks(kernel_qualname: str) -> dict[str, AutotunePick]:
    path = _disk_path(kernel_qualname)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, AutotunePick] = {}
    for k, v in raw.items():
        out[_autotune_key(kernel_qualname, k)] = AutotunePick(
            kwargs=dict(v.get("kwargs", {})),
            num_warps=int(v.get("num_warps", 4)),
            num_stages=int(v.get("num_stages", 2)),
            num_ctas=int(v.get("num_ctas", 1)),
            maxnreg=v.get("maxnreg"),
            perf_us=v.get("perf_us"),
            notes=str(v.get("notes", "")),
            timestamp=float(v.get("timestamp", 0.0)),
        )
    return out


def _persist_disk_pick(kernel_qualname: str, key_repr: str, pick: AutotunePick) -> None:
    path = _disk_path(kernel_qualname)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    data[key_repr] = {
        "kwargs": pick.kwargs,
        "num_warps": pick.num_warps,
        "num_stages": pick.num_stages,
        "num_ctas": pick.num_ctas,
        "maxnreg": pick.maxnreg,
        "perf_us": pick.perf_us,
        "notes": pick.notes,
        "timestamp": pick.timestamp,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _render_autotune_prompt(
    kernel_qualname: str,
    key_repr: str,
    candidate_configs: list[dict[str, Any]],
    perf_target_us: float | None,
) -> str:
    cfgs_text = (
        "\n".join(f"  - {json.dumps(c, sort_keys=True)}" for c in candidate_configs)
        or "  (no shortlist supplied — agent should pick from the standard grid)"
    )
    target = f"PERF TARGET: ≤{perf_target_us}μs" if perf_target_us else "PERF TARGET: unspecified"
    return "\n".join(
        [
            f"Pick the best autotune config for kernel {kernel_qualname!r}.",
            "",
            f"KEY:    {key_repr}",
            target,
            "",
            "CANDIDATE CONFIGS:",
            cfgs_text,
            "",
            "Reply by calling register_autotune_pick with the winning config:",
            "  num_warps, num_stages, num_ctas, maxnreg, kwargs (BLOCK_M etc.)",
            "  perf_us:    measured median latency",
            "  notes:      one short sentence",
        ]
    )


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def request_autotune_trial(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_qualname: str,
    key_repr: str,
    candidate_configs: list[dict[str, Any]] | None = None,
    perf_target_us: float | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Request an autotune pick from the agent.

    On cache hit (in-session OR rehydrated from
    ``~/.compgen/autotune/``), returns the recorded pick directly.
    """
    session = sm.get(session_id)
    cache = _autotune_cache(session)
    full_key = _autotune_key(kernel_qualname, key_repr)

    if full_key not in cache.picks:
        # Try disk rehydration on first reference.
        for k, p in _load_disk_picks(kernel_qualname).items():
            cache.picks.setdefault(k, p)

    cached = cache.picks.get(full_key)
    if cached is not None:
        return {
            "ok": True,
            "session_id": session_id,
            "found_in_cache": True,
            "kernel_qualname": kernel_qualname,
            "key_repr": key_repr,
            "pick": {
                "kwargs": cached.kwargs,
                "num_warps": cached.num_warps,
                "num_stages": cached.num_stages,
                "num_ctas": cached.num_ctas,
                "maxnreg": cached.maxnreg,
                "perf_us": cached.perf_us,
                "notes": cached.notes,
            },
        }

    rid = request_id or f"tune_{uuid.uuid4().hex[:12]}"
    prompt = _render_autotune_prompt(
        kernel_qualname,
        key_repr,
        candidate_configs or [],
        perf_target_us,
    )
    cache.pending[rid] = PendingAutotuneTrial(
        request_id=rid,
        prompt=prompt,
        kernel_qualname=kernel_qualname,
        key_repr=key_repr,
        candidate_configs=list(candidate_configs or []),
        perf_target_us=perf_target_us,
    )
    return {
        "ok": True,
        "session_id": session_id,
        "found_in_cache": False,
        "request_id": rid,
        "prompt": prompt,
        "next_call": {
            "tool": "register_autotune_pick",
            "args": {"session_id": session_id, "request_id": rid, "kwargs": "<dict>", "num_warps": 4, "num_stages": 2},
        },
    }


def register_autotune_pick(
    sm: SessionManager,
    *,
    session_id: str,
    request_id: str,
    kwargs: dict[str, Any] | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    num_ctas: int = 1,
    maxnreg: int | None = None,
    perf_us: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Fulfill a pending autotune trial. Persists to disk."""
    session = sm.get(session_id)
    cache = _autotune_cache(session)
    pending = cache.pending.pop(request_id, None)
    if pending is None:
        return {
            "ok": False,
            "session_id": session_id,
            "error": f"unknown or already-fulfilled request_id {request_id!r}",
        }
    pick = AutotunePick(
        kwargs=dict(kwargs or {}),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        num_ctas=int(num_ctas),
        maxnreg=maxnreg,
        perf_us=perf_us,
        notes=notes,
        timestamp=time.time(),
    )
    full_key = _autotune_key(pending.kernel_qualname, pending.key_repr)
    cache.picks[full_key] = pick
    try:
        _persist_disk_pick(pending.kernel_qualname, pending.key_repr, pick)
    except OSError as exc:
        return {
            "ok": True,
            "session_id": session_id,
            "warning": f"in-memory cached but disk persist failed: {exc}",
            "kernel_qualname": pending.kernel_qualname,
            "key_repr": pending.key_repr,
        }
    return {
        "ok": True,
        "session_id": session_id,
        "kernel_qualname": pending.kernel_qualname,
        "key_repr": pending.key_repr,
        "cached_picks": len(cache.picks),
    }


def lookup_autotune_pick(
    sm: SessionManager,
    *,
    session_id: str,
    kernel_qualname: str,
    key_repr: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _autotune_cache(session)
    full_key = _autotune_key(kernel_qualname, key_repr)
    if full_key not in cache.picks:
        for k, p in _load_disk_picks(kernel_qualname).items():
            cache.picks.setdefault(k, p)
    pick = cache.picks.get(full_key)
    if pick is None:
        return {
            "ok": True,
            "session_id": session_id,
            "found": False,
            "kernel_qualname": kernel_qualname,
            "key_repr": key_repr,
        }
    return {
        "ok": True,
        "session_id": session_id,
        "found": True,
        "kernel_qualname": kernel_qualname,
        "key_repr": key_repr,
        "pick": {
            "kwargs": pick.kwargs,
            "num_warps": pick.num_warps,
            "num_stages": pick.num_stages,
            "num_ctas": pick.num_ctas,
            "maxnreg": pick.maxnreg,
            "perf_us": pick.perf_us,
            "notes": pick.notes,
        },
    }


def list_pending_autotune_trials(
    sm: SessionManager,
    *,
    session_id: str,
) -> dict[str, Any]:
    session = sm.get(session_id)
    cache = _autotune_cache(session)
    return {
        "ok": True,
        "session_id": session_id,
        "pending_count": len(cache.pending),
        "requests": [
            {
                "request_id": rid,
                "kernel_qualname": req.kernel_qualname,
                "key_repr": req.key_repr,
                "candidate_configs": req.candidate_configs,
                "perf_target_us": req.perf_target_us,
                "prompt": req.prompt,
            }
            for rid, req in cache.pending.items()
        ],
    }


AUTOTUNE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "request_autotune_trial",
        "description": (
            "Request an agent-driven autotune pick for a (kernel, key) "
            "pair. Returns a cached pick on hit (in-session OR "
            "rehydrated from ~/.compgen/autotune/)."
        ),
        "phase": "transform",
        "handler": request_autotune_trial,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_qualname": {"type": "string"},
                "key_repr": {"type": "string"},
                "candidate_configs": {"type": ["array", "null"], "items": {"type": "object"}},
                "perf_target_us": {"type": ["number", "null"]},
                "request_id": {"type": ["string", "null"]},
            },
            "required": ["session_id", "kernel_qualname", "key_repr"],
        },
    },
    {
        "name": "register_autotune_pick",
        "description": (
            "Fulfill a pending autotune trial with the agent's pick. Persists to ~/.compgen/autotune/<kernel>.json."
        ),
        "phase": "transform",
        "handler": register_autotune_pick,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "request_id": {"type": "string"},
                "kwargs": {"type": ["object", "null"]},
                "num_warps": {"type": "integer"},
                "num_stages": {"type": "integer"},
                "num_ctas": {"type": "integer"},
                "maxnreg": {"type": ["integer", "null"]},
                "perf_us": {"type": ["number", "null"]},
                "notes": {"type": "string"},
            },
            "required": ["session_id", "request_id"],
        },
    },
    {
        "name": "lookup_autotune_pick",
        "description": "Cache lookup for a (kernel, key) autotune pick.",
        "phase": "inspect",
        "handler": lookup_autotune_pick,
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "kernel_qualname": {"type": "string"},
                "key_repr": {"type": "string"},
            },
            "required": ["session_id", "kernel_qualname", "key_repr"],
        },
    },
    {
        "name": "list_pending_autotune_trials",
        "description": "List outstanding autotune trial requests for the agent.",
        "phase": "inspect",
        "handler": list_pending_autotune_trials,
        "input_schema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
]


__all__ = [
    "AUTOTUNE_TOOLS",
    "AutotuneCache",
    "AutotunePick",
    "PendingAutotuneTrial",
    "list_pending_autotune_trials",
    "lookup_autotune_pick",
    "register_autotune_pick",
    "request_autotune_trial",
]

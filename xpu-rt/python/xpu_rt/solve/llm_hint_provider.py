"""Solver-hint provider that can read LLM-supplied hints.

Three modes:

* ``rule_based`` (default) — the deterministic
  :func:`compgen.solve.solver_hints.rule_based_memory_hints`
  always runs. No LLM call.

* ``llm_file`` — the operator (or an upstream agent-decision
  request handler) has already written hint JSON to a path on
  disk. We load and validate it; if anything is malformed we fall
  back to ``rule_based`` honestly (never a fake hint).

* ``merged`` — combine ``rule_based`` and ``llm_file``,
  preferring higher-confidence entries. This is the recommended
  production-path mode: the rule-based heuristic provides
  baseline coverage; the LLM augments with insights the rules
  miss.

The LLM call itself is **not in scope of this module** — the
existing CompGen agent-decision flow handles that. A skill /
operator invokes Claude Code (or another LLM) with the
``LLMHintRequest`` payload, the LLM emits a JSON document at the
declared path, and this module reads it.

Hint payload schema (what the LLM is asked to produce):

::

    {
        "schema_version": "memory_hints_v1",
        "source": "llm",
        "tier_hints": [
            {"buffer_id": "...", "tier_id": "...",
             "confidence": 0.0-1.0, "reason": "..."},
            ...
        ],
        "offset_warm_start": [
            {"buffer_id": "...", "offset_bytes": 1234, "reason": "..."},
            ...
        ],
        "stage_partition": [
            {"stage_id": "...", "buffer_ids": ["..."], "reason": "..."},
            ...
        ],
        "symmetry_classes": [
            {"class_id": "...", "buffer_ids": ["..."], "reason": "..."},
            ...
        ],
        "confidence_summary": {"tier_hints_fraction": 0.0-1.0, ...}
    }
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from compgen.solve.solver_hints import (
    MemoryHints,
    merge_hints,
    rule_based_memory_hints,
)

__all__ = [
    "read_llm_hints_from_file",
    "get_memory_hints",
    "write_llm_hint_request",
]


def read_llm_hints_from_file(path: str | Path) -> MemoryHints | None:
    """Load an LLM hint document from disk.

    Returns ``None`` on parse/validation errors — callers fall back
    to the rule-based heuristic honestly rather than failing the
    whole solve.
    """

    p = Path(path)
    if not p.is_file():
        return None
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(body, dict):
        return None
    try:
        return MemoryHints.from_dict(body)
    except (KeyError, TypeError, ValueError):
        return None


def get_memory_hints(
    plan_input: Any,
    *,
    mode: str = "rule_based",
    llm_hint_path: str | Path | None = None,
) -> MemoryHints:
    """Return a hint bundle per the requested mode.

    Args:
        plan_input: ``MemoryPlanInput`` from the planner.
        mode: ``rule_based`` | ``llm_file`` | ``merged``.
            ``rule_based`` always runs the deterministic heuristic.
            ``llm_file`` loads the LLM document; returns rule_based
              honestly if the file is missing or malformed.
            ``merged`` is the union of both.
        llm_hint_path: required when mode != ``rule_based``.
    """

    if mode == "rule_based":
        return rule_based_memory_hints(plan_input)

    if mode in ("llm_file", "merged"):
        if llm_hint_path is None:
            llm_hint_path = os.environ.get("COMPGEN_LLM_HINT_PATH")
        if llm_hint_path is None:
            # No file configured → degrade to rule-based honestly.
            return rule_based_memory_hints(plan_input)
        llm = read_llm_hints_from_file(llm_hint_path)
        if llm is None:
            return rule_based_memory_hints(plan_input)
        if mode == "llm_file":
            return llm
        rule = rule_based_memory_hints(plan_input)
        return merge_hints(rule, llm)

    raise ValueError(f"unknown hint mode: {mode!r}")


def write_llm_hint_request(plan_input: Any, *, out_path: str | Path) -> Path:
    """Write a typed *request* document the LLM (or operator) reads
    to produce a hint document.

    The request is intentionally simple: the LLM sees the buffer
    list and tier list, plus a short prompt explaining what kinds
    of hints it can return. It writes the response back to a path
    declared in ``request["expected_response_path"]``.

    This module does NOT call the LLM — that's the role of the
    agent-decision-request handler upstream. We only serialize the
    typed request so the operator/agent can fulfill it.
    """

    from compgen.solve.memory_planner import MemoryPlanInput

    if not isinstance(plan_input, MemoryPlanInput):
        raise TypeError(
            "write_llm_hint_request requires a MemoryPlanInput"
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "schema_version": "memory_hint_request_v1",
        "buffers": [
            {
                "buffer_id": b.buffer_id,
                "size_bytes": b.size_bytes,
                "lifetime_start": b.lifetime_start,
                "lifetime_end": b.lifetime_end,
                "allowed_tiers": list(b.allowed_tiers),
            }
            for b in plan_input.buffers
        ],
        "tiers": [
            {"tier_id": t.tier_id, "capacity_bytes": t.capacity_bytes, "weight": t.weight}
            for t in plan_input.tier_capacities
        ],
        "alias_candidates": [
            {"buffer_a": a.buffer_a, "buffer_b": a.buffer_b}
            for a in plan_input.alias_candidates
        ],
        "prompt": (
            "Produce a JSON document matching MemoryHints "
            "(schema_version=memory_hints_v1). Best-effort only: "
            "confidence < 0.9 is a warm-start hint, >= 0.9 is "
            "fixed. You don't need to cover every buffer — even "
            "65-75% coverage exponentially shrinks the MILP "
            "search space."
        ),
    }
    out.write_text(json.dumps(request, sort_keys=True, indent=2))
    return out

"""H1 — Typed ``ToolDelta`` envelope for every MCP dispatch.

Section 11 Dream 2 in the plan: every tool call that touches the
session emits a typed, frozen ``ToolDelta`` envelope summarising
*everything that changed* — recipe edits, payload edits, decision
resolutions, artifact writes, knowledge writes, bench/LLM/verifier
call IDs, cost (wall-ms, llm-tokens, verifier-seconds). The envelope
is *additive*: callers still get the raw `return_value` field, but
the trace bus now carries a structured record.

The envelope is intentionally light:

* large IR state (``recipe_module`` / ``payload_module``) is diffed
  by **content hash** — we record before/after hashes, not the full IR;
* small state (``decision_registry`` keys, ``kernel_cache`` keys,
  ``bench_cache`` keys) is diffed by keyset;
* read-only tools produce a ToolDelta with empty ``state_changes``.

Backward compat: callers reading ``dispatch_tool`` results still get
the same dict; the ToolDelta envelope flows into the recorder (and
optionally back to the caller via the ``--envelope`` MCP option,
which is off by default).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Sub-typed envelope pieces (frozen, plan-verbatim shapes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """Resource cost charged by one tool invocation."""

    wall_ms: float = 0.0
    llm_tokens: int = 0
    verifier_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_ms": self.wall_ms,
            "llm_tokens": self.llm_tokens,
            "verifier_seconds": self.verifier_seconds,
        }


@dataclass(frozen=True)
class RecipeOpEdit:
    """One mutation to the Recipe-IR module."""

    op_id: str
    kind: str  # appended | mutated | removed
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "kind": self.kind, "summary": self.summary}


@dataclass(frozen=True)
class PayloadEdit:
    """One mutation to the Payload-IR module."""

    region_id: str
    kind: str
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "kind": self.kind,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class DecisionResolved:
    """One decision-registry key resolved."""

    decision_key: str
    resolution: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"decision_key": self.decision_key, "resolution": self.resolution}


@dataclass(frozen=True)
class ArtifactWritten:
    """One artifact path emitted by the tool."""

    path: str
    kind: str = "artifact"

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind}


@dataclass(frozen=True)
class KnowledgeWritten:
    """One entry written to the knowledge / lesson store."""

    key: str
    kind: str = "lesson"

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "kind": self.kind}


@dataclass(frozen=True)
class BenchID:
    """Identifier of a bench result the tool produced or consumed."""

    bench_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"bench_id": self.bench_id}


@dataclass(frozen=True)
class LLMCallID:
    """Identifier of one LLM call attributed to the tool."""

    call_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"call_id": self.call_id}


@dataclass(frozen=True)
class VerifierRunID:
    """Identifier of one verifier run attributed to the tool."""

    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id}


# ---------------------------------------------------------------------------
# State-change record + envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateChanges:
    """The diff between pre- and post-call session state."""

    recipe_ops: tuple[RecipeOpEdit, ...] = ()
    payload_edits: tuple[PayloadEdit, ...] = ()
    decisions: tuple[DecisionResolved, ...] = ()
    recipe_hash_before: str = ""
    recipe_hash_after: str = ""
    payload_hash_before: str = ""
    payload_hash_after: str = ""

    @property
    def is_empty(self) -> bool:
        """True iff this ToolDelta represents a no-op (read-only call)."""

        return (
            not self.recipe_ops
            and not self.payload_edits
            and not self.decisions
            and self.recipe_hash_before == self.recipe_hash_after
            and self.payload_hash_before == self.payload_hash_after
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipe_ops": [r.to_dict() for r in self.recipe_ops],
            "payload_edits": [p.to_dict() for p in self.payload_edits],
            "decisions": [d.to_dict() for d in self.decisions],
            "recipe_hash_before": self.recipe_hash_before,
            "recipe_hash_after": self.recipe_hash_after,
            "payload_hash_before": self.payload_hash_before,
            "payload_hash_after": self.payload_hash_after,
        }


@dataclass(frozen=True)
class SideEffects:
    """External-world effects (artifacts, knowledge, bench/LLM IDs)."""

    artifacts: tuple[ArtifactWritten, ...] = ()
    knowledge: tuple[KnowledgeWritten, ...] = ()
    benches: tuple[BenchID, ...] = ()
    llm_calls: tuple[LLMCallID, ...] = ()
    verifier_runs: tuple[VerifierRunID, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": [a.to_dict() for a in self.artifacts],
            "knowledge": [k.to_dict() for k in self.knowledge],
            "benches": [b.to_dict() for b in self.benches],
            "llm_calls": [c.to_dict() for c in self.llm_calls],
            "verifier_runs": [v.to_dict() for v in self.verifier_runs],
        }


@dataclass(frozen=True)
class ToolDelta:
    """The canonical Section-11 Dream 2 envelope.

    Emitted once per dispatched MCP tool call. The recorder consumes
    it; the caller optionally receives it when the ``--envelope`` flag
    is set on the session.
    """

    tool: str
    args_hash: str
    timestamp: float
    state_changes: StateChanges = field(default_factory=StateChanges)
    side_effects: SideEffects = field(default_factory=SideEffects)
    return_value: Any | None = None
    cost: Cost = field(default_factory=Cost)
    status: str = "ok"  # ok | blocked | error
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "compgen_tool_delta_v1",
            "tool": self.tool,
            "args_hash": self.args_hash,
            "timestamp": self.timestamp,
            "state_changes": self.state_changes.to_dict(),
            "side_effects": self.side_effects.to_dict(),
            "return_value": self.return_value,
            "cost": self.cost.to_dict(),
            "status": self.status,
            "blocked_reason": self.blocked_reason,
        }


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def canonical_args_hash(args: dict[str, Any]) -> str:
    """SHA-256 of canonical-JSON-serialised args (first 16 hex chars).

    Mirrors the ToolCard input-hash convention so a single args hash is
    comparable across the ToolCard / dispatch / recorder surfaces.
    """

    try:
        blob = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    except Exception:  # noqa: BLE001
        blob = repr(sorted(args.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _module_hash(mod: Any | None) -> str:
    """Hash of an xDSL/MLIR module's textual form.

    Returns an empty string for missing modules so a tool that doesn't
    touch the IR produces ``recipe_hash_before == recipe_hash_after``.
    """

    if mod is None:
        return ""
    try:
        text = str(mod)
    except Exception:  # noqa: BLE001
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _snapshot_decision_keys(sm: Any) -> set[str]:
    """Best-effort snapshot of the live decision-registry keys.

    Returns an empty set if the session has no decision registry; the
    diff comes out empty as a result.
    """

    try:
        reg = getattr(sm, "decision_registry", None)
        if reg is None:
            return set()
        return set(getattr(reg, "_decisions", {}).keys())
    except Exception:  # noqa: BLE001
        return set()


def _snapshot_modules(sm: Any) -> tuple[str, str]:
    """Pre/post hashes of the live Recipe and Payload IR modules."""

    driver = getattr(sm, "driver", None)
    env = getattr(driver, "env", None) if driver is not None else None
    recipe = getattr(env, "recipe", None) if env is not None else None
    payload = getattr(env, "_payload_module", None) if env is not None else None
    if payload is None and env is not None:
        payload = getattr(env, "payload_module", None)
        if callable(payload):
            try:
                payload = payload()
            except Exception:  # noqa: BLE001
                payload = None
    return _module_hash(recipe), _module_hash(payload)


def build_state_changes(
    *,
    sm: Any,
    pre_recipe: str,
    pre_payload: str,
    pre_decisions: set[str],
) -> StateChanges:
    """Compare a pre-call snapshot against the live session state."""

    post_recipe, post_payload = _snapshot_modules(sm)
    post_decisions = _snapshot_decision_keys(sm)
    new_decisions = sorted(post_decisions - pre_decisions)
    return StateChanges(
        recipe_ops=(),
        payload_edits=(),
        decisions=tuple(DecisionResolved(decision_key=k) for k in new_decisions),
        recipe_hash_before=pre_recipe,
        recipe_hash_after=post_recipe,
        payload_hash_before=pre_payload,
        payload_hash_after=post_payload,
    )


def now_timestamp() -> float:
    """Monotonic-ish wall clock for envelope timestamping."""

    return time.time()


__all__ = [
    "ArtifactWritten",
    "BenchID",
    "Cost",
    "DecisionResolved",
    "KnowledgeWritten",
    "LLMCallID",
    "PayloadEdit",
    "RecipeOpEdit",
    "SideEffects",
    "StateChanges",
    "ToolDelta",
    "VerifierRunID",
    "build_state_changes",
    "canonical_args_hash",
    "now_timestamp",
]

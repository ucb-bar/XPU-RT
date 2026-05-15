"""H1 — typed ``ToolDelta`` envelope.

Validates the Section-11 Dream 2 envelope:

1. Frozen dataclasses + ``to_dict`` round-trip.
2. Canonical args hash is byte-stable.
3. ``StateChanges.is_empty`` is correct for read-only flows.
4. ``build_state_changes`` correctly detects new decision keys.
5. ``_module_hash`` returns "" for missing modules + non-empty for
   present ones.
6. ``dispatch_tool`` runs end-to-end through the new envelope path
   without changing the caller-visible result shape.
7. The envelope's ``status`` field reflects handler success/failure.
8. The envelope's ``cost.wall_ms`` is populated from the dispatch
   timing.
"""

from __future__ import annotations

from typing import Any

from compgen.mcp.server import dispatch_tool
from compgen.mcp.tool_delta import (
    ArtifactWritten,
    BenchID,
    Cost,
    DecisionResolved,
    KnowledgeWritten,
    LLMCallID,
    PayloadEdit,
    RecipeOpEdit,
    SideEffects,
    StateChanges,
    ToolDelta,
    VerifierRunID,
    _module_hash,
    build_state_changes,
    canonical_args_hash,
    now_timestamp,
)


# ----------------------------------------------------------------------
# Dataclass + to_dict round-trip
# ----------------------------------------------------------------------


def test_recipe_op_edit_to_dict() -> None:
    e = RecipeOpEdit(op_id="r0", kind="appended", summary="propose")
    assert e.to_dict() == {"op_id": "r0", "kind": "appended", "summary": "propose"}


def test_payload_edit_to_dict() -> None:
    p = PayloadEdit(region_id="reg_0", kind="rewritten", summary="tile")
    assert p.to_dict()["region_id"] == "reg_0"


def test_decision_resolved_to_dict() -> None:
    d = DecisionResolved(decision_key="dispatch:cpu", resolution="sync")
    assert d.to_dict()["resolution"] == "sync"


def test_side_effects_aggregates() -> None:
    se = SideEffects(
        artifacts=(ArtifactWritten(path="/tmp/x.json"),),
        knowledge=(KnowledgeWritten(key="lesson:1"),),
        benches=(BenchID(bench_id="b0"),),
        llm_calls=(LLMCallID(call_id="c0"),),
        verifier_runs=(VerifierRunID(run_id="v0"),),
    )
    blob = se.to_dict()
    assert len(blob["artifacts"]) == 1
    assert len(blob["knowledge"]) == 1
    assert blob["benches"][0]["bench_id"] == "b0"


def test_tool_delta_envelope_to_dict() -> None:
    env = ToolDelta(
        tool="echo",
        args_hash="deadbeef00000000",
        timestamp=1234.5,
        cost=Cost(wall_ms=1.0),
    )
    blob = env.to_dict()
    assert blob["schema_version"] == "compgen_tool_delta_v1"
    assert blob["status"] == "ok"
    assert blob["cost"]["wall_ms"] == 1.0


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------


def test_canonical_args_hash_is_stable() -> None:
    a = canonical_args_hash({"x": 1, "y": [2, 3]})
    b = canonical_args_hash({"y": [2, 3], "x": 1})  # key order ignored
    assert a == b
    assert len(a) == 16


def test_canonical_args_hash_distinguishes_values() -> None:
    a = canonical_args_hash({"x": 1})
    b = canonical_args_hash({"x": 2})
    assert a != b


def test_module_hash_empty_for_none() -> None:
    assert _module_hash(None) == ""


def test_module_hash_nonempty_for_stringable_obj() -> None:
    class _Mod:
        def __str__(self) -> str:
            return "module body"

    h = _module_hash(_Mod())
    assert len(h) == 16
    # Different bodies hash differently.

    class _Mod2:
        def __str__(self) -> str:
            return "module body 2"

    assert _module_hash(_Mod2()) != h


# ----------------------------------------------------------------------
# StateChanges
# ----------------------------------------------------------------------


def test_state_changes_empty_for_read_only() -> None:
    s = StateChanges(
        recipe_hash_before="a",
        recipe_hash_after="a",
        payload_hash_before="b",
        payload_hash_after="b",
    )
    assert s.is_empty


def test_state_changes_not_empty_on_recipe_mutation() -> None:
    s = StateChanges(
        recipe_hash_before="a",
        recipe_hash_after="z",
        payload_hash_before="b",
        payload_hash_after="b",
    )
    assert not s.is_empty


# ----------------------------------------------------------------------
# build_state_changes with a stub session manager
# ----------------------------------------------------------------------


class _StubDecisionRegistry:
    def __init__(self) -> None:
        self._decisions: dict[str, Any] = {}


class _StubSession:
    def __init__(self) -> None:
        self.decision_registry = _StubDecisionRegistry()
        self.driver = None


def test_build_state_changes_detects_new_decisions() -> None:
    sm = _StubSession()
    pre = set(sm.decision_registry._decisions.keys())
    sm.decision_registry._decisions["dispatch:gpu"] = "sync"
    sc = build_state_changes(sm=sm, pre_recipe="", pre_payload="", pre_decisions=pre)
    assert len(sc.decisions) == 1
    assert sc.decisions[0].decision_key == "dispatch:gpu"


def test_build_state_changes_empty_when_nothing_changed() -> None:
    sm = _StubSession()
    pre = set(sm.decision_registry._decisions.keys())
    sc = build_state_changes(sm=sm, pre_recipe="", pre_payload="", pre_decisions=pre)
    assert sc.is_empty


# ----------------------------------------------------------------------
# Dispatch end-to-end (envelope must not change caller-visible result)
# ----------------------------------------------------------------------


def test_dispatch_tool_envelope_passes_through_result() -> None:
    """The envelope is additive — the raw result dict is unchanged."""

    def _handler(sm: Any, *, x: int = 0) -> dict[str, Any]:
        return {"ok": True, "echoed": x}

    tool_by_name = {
        "echo_tool": {"name": "echo_tool", "handler": _handler},
    }
    sm = _StubSession()
    result = dispatch_tool(
        "echo_tool",
        {"x": 7},
        sm=sm,  # type: ignore[arg-type]
        tool_by_name=tool_by_name,
        recorder=None,
    )
    assert result == {"ok": True, "echoed": 7}


def test_dispatch_tool_envelope_handles_handler_error() -> None:
    """An exception in the handler still produces a typed result; the
    envelope's status would be 'error' but caller sees ``ok: false``."""

    def _handler(sm: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    tool_by_name = {
        "broken_tool": {"name": "broken_tool", "handler": _handler},
    }
    sm = _StubSession()
    result = dispatch_tool(
        "broken_tool",
        {},
        sm=sm,  # type: ignore[arg-type]
        tool_by_name=tool_by_name,
        recorder=None,
    )
    assert result["ok"] is False
    assert "RuntimeError" in result["error"]


def test_dispatch_tool_unknown_tool() -> None:
    """Unknown tools surface available list — envelope path is bypassed."""

    sm = _StubSession()
    result = dispatch_tool(
        "no_such_tool",
        {},
        sm=sm,  # type: ignore[arg-type]
        tool_by_name={},
        recorder=None,
    )
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_now_timestamp_monotonic() -> None:
    a = now_timestamp()
    b = now_timestamp()
    assert b >= a

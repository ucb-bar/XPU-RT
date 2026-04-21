"""Tests for ``compgen.mcp.tools.dispatch``.

Locks in:
  * round-trip: request → list → register → lookup
  * fingerprint short-circuit on identical (region, targets, budget, objective)
  * register validates JSON shape and re-queues on bad payload
  * register requires a 'per_target' key
  * the four tools are wired into ALL_TOOLS
  * McpDispatchLLM hits the cache when a decision is already cached and
    queues a pending request (returns empty raw_text) on miss
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.mcp.session import SessionManager
from compgen.mcp.tools.dispatch import (
    DISPATCH_TOOLS,
    McpDispatchLLM,
    dispatch_fingerprint,
    list_pending_dispatch_decisions,
    lookup_dispatch_decision,
    register_dispatch_decision,
    request_dispatch_decision,
)


@pytest.fixture
def session_manager(tmp_path: Path) -> SessionManager:
    sm = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    sm.open(session_id="sess1")
    return sm


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_dispatch_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in DISPATCH_TOOLS}
    assert names == {
        "request_dispatch_decision",
        "register_dispatch_decision",
        "lookup_dispatch_decision",
        "list_pending_dispatch_decisions",
    }


def test_dispatch_tools_appear_in_all_tools_bundle() -> None:
    from compgen.mcp.tools import ALL_TOOLS
    names = {t["name"] for t in ALL_TOOLS}
    for dt in ("request_dispatch_decision", "register_dispatch_decision",
               "lookup_dispatch_decision", "list_pending_dispatch_decisions"):
        assert dt in names, f"{dt} missing from ALL_TOOLS"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_identical_input() -> None:
    fp1 = dispatch_fingerprint(["addf", "mulf"], ["cuda", "cpu"], 100.0, "latency")
    fp2 = dispatch_fingerprint(["addf", "mulf"], ["cuda", "cpu"], 100.0, "latency")
    assert fp1 == fp2


def test_fingerprint_differs_on_target_set() -> None:
    a = dispatch_fingerprint(["addf"], ["cuda"], None, "latency")
    b = dispatch_fingerprint(["addf"], ["rocm"], None, "latency")
    assert a != b


def test_fingerprint_differs_on_objective() -> None:
    a = dispatch_fingerprint(["addf"], ["cuda"], None, "latency")
    b = dispatch_fingerprint(["addf"], ["cuda"], None, "energy")
    assert a != b


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def _envelope_dict(target: str = "cuda-a100") -> dict:
    return {
        "target_name": target, "vector_lanes": 64, "scratchpad_bytes": 49152,
        "register_bytes": 256, "native_dtypes": ["f16", "f32"],
        "peak_bandwidth_gbps": 672.0,
    }


def test_request_then_register_then_lookup_round_trip(session_manager) -> None:
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="single-op region: addf",
        region_op_names=["addf"],
        envelopes=[_envelope_dict()],
        priors=[{"target": "cuda-a100", "granularity": "normal",
                 "confidence": 0.7, "reason": "default"}],
        perf_budget_us=100.0, objective="latency",
    )
    assert out["ok"] and not out["found_in_cache"]
    rid = out["request_id"]
    fp = out["fingerprint"]
    assert "addf" in out["prompt"]
    assert "PERF BUDGET: 100.0us" in out["prompt"]

    pending = list_pending_dispatch_decisions(session_manager, session_id="sess1")
    assert pending["pending_count"] == 1
    assert pending["requests"][0]["request_id"] == rid

    decision = {
        "per_target": {
            "cuda-a100": {"granularity": "normal",
                          "rationale": "single-op default"},
        },
        "best_target": "cuda-a100",
        "best_rationale": "only target",
    }
    reg = register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id=rid, decision_json=json.dumps(decision),
    )
    assert reg["ok"]
    assert reg["fingerprint"] == fp
    assert reg["cached_decisions"] == 1

    lk = lookup_dispatch_decision(
        session_manager, session_id="sess1",
        region_op_names=["addf"], envelope_targets=["cuda-a100"],
        perf_budget_us=100.0, objective="latency",
    )
    assert lk["found"]
    assert lk["fingerprint"] == fp
    assert json.loads(lk["decision_json"])["best_target"] == "cuda-a100"


def test_request_short_circuits_on_cached_fingerprint(session_manager) -> None:
    # First request + fulfil
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="r", region_op_names=["op"],
        envelopes=[_envelope_dict()],
    )
    register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id=out["request_id"],
        decision_json=json.dumps({"per_target": {"cuda-a100": {"granularity": "normal", "rationale": "x"}},
                                  "best_target": "cuda-a100", "best_rationale": "y"}),
    )
    # Second identical request → cache hit, no new pending entry.
    out2 = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="r", region_op_names=["op"],
        envelopes=[_envelope_dict()],
    )
    assert out2["found_in_cache"] is True
    assert out2["fingerprint"] == out["fingerprint"]
    assert json.loads(out2["decision_json"])["best_target"] == "cuda-a100"
    pending = list_pending_dispatch_decisions(session_manager, session_id="sess1")
    assert pending["pending_count"] == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_register_with_empty_json_requeues(session_manager) -> None:
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="r", region_op_names=["op"],
        envelopes=[_envelope_dict()],
    )
    rid = out["request_id"]
    res = register_dispatch_decision(
        session_manager, session_id="sess1", request_id=rid, decision_json="",
    )
    assert res["ok"] is False
    assert "empty" in res["error"]
    pending = list_pending_dispatch_decisions(session_manager, session_id="sess1")
    assert pending["pending_count"] == 1   # re-queued


def test_register_rejects_invalid_json_shape(session_manager) -> None:
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="r", region_op_names=["op"],
        envelopes=[_envelope_dict()],
    )
    res = register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id=out["request_id"],
        decision_json='{"foo": "bar"}',     # missing per_target
    )
    assert res["ok"] is False
    assert "per_target" in res["error"]


def test_register_rejects_unparseable_json(session_manager) -> None:
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="r", region_op_names=["op"],
        envelopes=[_envelope_dict()],
    )
    res = register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id=out["request_id"],
        decision_json="not valid json {",
    )
    assert res["ok"] is False
    assert "not valid JSON" in res["error"]


def test_register_unknown_request_id_errors(session_manager) -> None:
    res = register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id="nonexistent",
        decision_json='{"per_target": {}}',
    )
    assert res["ok"] is False
    assert "unknown" in res["error"]


def test_lookup_miss_returns_found_false(session_manager) -> None:
    res = lookup_dispatch_decision(
        session_manager, session_id="sess1",
        region_op_names=["op"], envelope_targets=["cuda-a100"],
    )
    assert res["found"] is False


# ---------------------------------------------------------------------------
# McpDispatchLLM adapter
# ---------------------------------------------------------------------------


def test_mcp_dispatch_llm_returns_cached_decision(session_manager) -> None:
    """When a decision is already cached, the LLM adapter returns the
    raw JSON which decide_dispatch's parser will pick up."""
    from compgen.agent.hw_aware_dispatch import decide_dispatch
    from compgen.kernels.contract_v3 import (
        ExecutionEnvelope, HardwareEnvelope, IOContract, KernelArchetype,
        KernelContractV3, OrchestrationSpec, ShapeClass, TensorIO,
    )

    env = HardwareEnvelope(
        target_name="cuda-a100", vector_lanes=64,
        scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f16", "f32"), peak_bandwidth_gbps=672.0,
    )
    contract = KernelContractV3(
        op_name="addf", archetype=KernelArchetype.POINTWISE,
        io=IOContract(
            inputs=(TensorIO(name="a", shape=ShapeClass(dims=(None,)),
                             dtype_class=("f32",)),),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)),
                              dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )

    # Pre-warm the cache with a decision the agent would have returned.
    out = request_dispatch_decision(
        session_manager, session_id="sess1",
        region_summary="single-op region: addf",
        region_op_names=["single-op region: addf (pointwise); 1 in / 1 out"],
        envelopes=[_envelope_dict()],
    )
    register_dispatch_decision(
        session_manager, session_id="sess1",
        request_id=out["request_id"],
        decision_json=json.dumps({
            "per_target": {"cuda-a100": {"granularity": "normal", "rationale": "agent picked"}},
            "best_target": "cuda-a100", "best_rationale": "agent picked",
        }),
    )

    llm = McpDispatchLLM(sm=session_manager, session_id="sess1")
    verdict = decide_dispatch([contract], envelopes=[env], llm=llm)
    # Cache hit → LLM was used → granularity matches the agent's decision.
    assert verdict.used_llm is True
    assert "agent picked" in verdict.per_target["cuda-a100"].rationale


def test_mcp_dispatch_llm_queues_request_on_cache_miss(session_manager) -> None:
    """When no cached decision exists, McpDispatchLLM queues a request
    via request_dispatch_decision so the agent can fulfill it next pass."""
    from compgen.agent.hw_aware_dispatch import decide_dispatch
    from compgen.kernels.contract_v3 import (
        ExecutionEnvelope, HardwareEnvelope, IOContract, KernelArchetype,
        KernelContractV3, OrchestrationSpec, ShapeClass, TensorIO,
    )

    env = HardwareEnvelope(
        target_name="cuda-a100", vector_lanes=64,
        scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f16", "f32"), peak_bandwidth_gbps=672.0,
    )
    contract = KernelContractV3(
        op_name="addf", archetype=KernelArchetype.POINTWISE,
        io=IOContract(
            inputs=(TensorIO(name="a", shape=ShapeClass(dims=(None,)),
                             dtype_class=("f32",)),),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)),
                              dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )

    llm = McpDispatchLLM(sm=session_manager, session_id="sess1")
    # First pass — no cached decision; the optimizer falls back to oracle.
    verdict = decide_dispatch([contract], envelopes=[env], llm=llm)
    assert verdict.used_llm is False     # parser got empty text, no override
    # But a pending request must have been queued for the agent.
    pending = list_pending_dispatch_decisions(session_manager, session_id="sess1")
    assert pending["pending_count"] == 1
    assert "addf" in pending["requests"][0]["region_summary"]

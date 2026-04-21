"""End-to-end test: ``optimize_via_mcp`` drives the W6 loop entirely
through MCP-backed callbacks.

Locks in:
  * McpCodegenFn cache miss queues a request via request_kernel_codegen
  * McpCodegenFn cache hit returns the agent-supplied source compiled
    into a Python callable
  * optimize_via_mcp produces an OptimizedModel whose decisions reference
    "mcp_pending" (first pass) or "mcp_cache:<lang>" (after agent fulfils)
  * The flow is fully MCP-driven: no codegen_fn / bench_fn / llm slot
    is left to the caller — every callback resolves through the session.
  * request_model_optimization surfaces the pending-queue counts so the
    agent knows what to drain next.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.agent.mcp_optimizer import (
    McpCodegenFn,
    OPTIMIZE_TOOLS,
    optimize_via_mcp,
    register_optimization_progress,
    request_model_optimization,
)
from compgen.agent.kernel_optimizer import fingerprint_for
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope, HardwareEnvelope, IOContract, KernelArchetype,
    KernelContractV3, OrchestrationSpec, ShapeClass, TensorIO,
)
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.bench import (
    list_pending_bench_requests, register_bench_result,
)
from compgen.mcp.tools.dispatch import (
    list_pending_dispatch_decisions, register_dispatch_decision,
)
from compgen.mcp.tools.kernel import (
    list_pending_kernel_requests, register_kernel_result,
)
from compgen.memory.kernel_db import KernelDB, set_shared_db
from compgen.kernels.store import KernelStore, set_shared_store


@pytest.fixture
def isolated_db(tmp_path: Path):
    db = KernelDB(path=tmp_path / "kernel_db.sqlite")
    set_shared_db(db)
    yield db
    set_shared_db(None)


@pytest.fixture(autouse=True)
def isolated_kernel_store(tmp_path: Path):
    set_shared_store(KernelStore(root=tmp_path / "kernel_store"))
    yield
    set_shared_store(None)


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


def _matmul(target: str = "cuda-a100") -> KernelContractV3:
    env = HardwareEnvelope(
        target_name=target, vector_lanes=64,
        scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f16",), peak_bandwidth_gbps=672.0,
    )
    return KernelContractV3(
        op_name="matmul", archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="lhs", shape=ShapeClass(dims=(64, 64)),
                         dtype_class=("f16",)),
                TensorIO(name="rhs", shape=ShapeClass(dims=(64, 64)),
                         dtype_class=("f16",)),
            ),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(64, 64)),
                              dtype_class=("f16",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


# ---------------------------------------------------------------------------
# McpCodegenFn behaviour
# ---------------------------------------------------------------------------


def test_mcp_codegen_fn_queues_request_on_cache_miss(sm, isolated_db) -> None:
    cg = McpCodegenFn(sm=sm, session_id="sess1")
    # Build a TargetDispatchDecision-shaped placeholder.
    from compgen.agent.hw_aware_dispatch import TargetDispatchDecision
    from compgen.kernels.granularity_oracle import GranularityVerdict
    from compgen.kernels.contract_v3 import Granularity
    decision = TargetDispatchDecision(
        target="cuda-a100", granularity=Granularity.NORMAL,
        adapter_name="cuda", rationale="x", confidence=0.7,
        deterministic_prior=GranularityVerdict(
            granularity=Granularity.NORMAL, reason="x", confidence=0.7,
        ),
    )
    out = cg(_matmul(), decision)
    assert out.provider_name == "mcp_pending"
    pending = list_pending_kernel_requests(sm, session_id="sess1")
    assert pending["pending_count"] == 1


def test_mcp_codegen_fn_returns_agent_source_on_cache_hit(sm, isolated_db) -> None:
    cg = McpCodegenFn(sm=sm, session_id="sess1")
    contract = _matmul()
    from compgen.agent.hw_aware_dispatch import TargetDispatchDecision
    from compgen.kernels.granularity_oracle import GranularityVerdict
    from compgen.kernels.contract_v3 import Granularity
    decision = TargetDispatchDecision(
        target="cuda-a100", granularity=Granularity.NORMAL,
        adapter_name="cuda", rationale="x", confidence=0.7,
        deterministic_prior=GranularityVerdict(
            granularity=Granularity.NORMAL, reason="x", confidence=0.7,
        ),
    )
    # First call queues; agent fulfils; second call hits cache.
    cg(contract, decision)
    pending = list_pending_kernel_requests(sm, session_id="sess1")
    rid = pending["requests"][0]["request_id"]
    register_kernel_result(
        sm, session_id="sess1", request_id=rid,
        kernel_code="def kernel(x):\n    return x * 2\n",
        language="python",
    )
    second = cg(contract, decision)
    assert second.provider_name == "mcp_cache:python"
    assert "def kernel" in second.source
    assert second.callable_kernel(7) == 14


# ---------------------------------------------------------------------------
# Full optimize_via_mcp loop
# ---------------------------------------------------------------------------


def test_optimize_via_mcp_runs_and_queues_pending_work(sm, isolated_db) -> None:
    """First pass: every region's codegen and bench should miss the
    cache, queueing one entry each. The optimization summary still
    completes (with placeholder kernels)."""
    contracts = [_matmul()]
    optim = optimize_via_mcp(
        model_fn=None, target="cuda-a100",
        contracts=contracts, sm=sm, session_id="sess1",
    )
    assert len(optim.decisions) == 1
    d = optim.decisions[0]
    assert d.provider_name in ("mcp_pending", "mcp_cache:python")

    # Pending queues populated by the loop.
    pending_codegen = list_pending_kernel_requests(sm, session_id="sess1")
    pending_bench = list_pending_bench_requests(sm, session_id="sess1")
    assert pending_codegen["pending_count"] >= 1
    assert pending_bench["pending_count"] >= 1


def test_optimize_via_mcp_second_pass_picks_up_agent_results(sm, isolated_db) -> None:
    """Pre-fulfil the dispatch + codegen + bench requests as if Claude
    Code already responded. A second optimisation pass should hit
    every cache and report cached=True / mcp_cache:python."""
    contracts = [_matmul()]

    # Pass 1: queue the requests.
    optimize_via_mcp(
        model_fn=None, target="cuda-a100",
        contracts=contracts, sm=sm, session_id="sess1",
    )

    # Agent drains every queue.
    for rid_meta in list_pending_dispatch_decisions(sm, session_id="sess1")["requests"]:
        register_dispatch_decision(
            sm, session_id="sess1", request_id=rid_meta["request_id"],
            decision_json=json.dumps({
                "per_target": {"cuda-a100": {"granularity": "normal",
                                              "rationale": "agent picked"}},
                "best_target": "cuda-a100", "best_rationale": "agent",
            }),
        )
    for rid_meta in list_pending_kernel_requests(sm, session_id="sess1")["requests"]:
        register_kernel_result(
            sm, session_id="sess1", request_id=rid_meta["request_id"],
            kernel_code="def kernel(*args, **kw):\n    return 0\n",
            language="python", correctness_passed=True, perf_us=8.0,
        )
    for rid_meta in list_pending_bench_requests(sm, session_id="sess1")["requests"]:
        register_bench_result(
            sm, session_id="sess1", request_id=rid_meta["request_id"],
            perf_us=8.0, correct=True, notes="agent-bench",
        )

    # Pass 2 — every primitive should hit cache.
    optim2 = optimize_via_mcp(
        model_fn=None, target="cuda-a100",
        contracts=contracts, sm=sm, session_id="sess1",
    )
    d2 = optim2.decisions[0]
    # Either cached==True (kernel_db cache) OR provider mcp_cache (agent path).
    assert d2.cached or d2.provider_name.startswith("mcp_cache")


# ---------------------------------------------------------------------------
# Optimization-tracking MCP tools
# ---------------------------------------------------------------------------


def test_request_model_optimization_surfaces_pending_counts(sm, isolated_db) -> None:
    """After an optimize_via_mcp pass, request_model_optimization
    should report the queue depths so the agent knows what to drain."""
    contracts = [_matmul()]
    optim = optimize_via_mcp(
        model_fn=None, target="cuda-a100",
        contracts=contracts, sm=sm, session_id="sess1",
    )
    fp = optim.decisions[0].fingerprint
    out = request_model_optimization(
        sm, session_id="sess1", target="cuda-a100",
        contract_fingerprints=[fp],
        perf_budget_us=100.0, objective="latency",
    )
    assert out["ok"]
    assert out["contracts_tracked"] == 1
    assert "pending" in out
    assert all(k in out["pending"] for k in ("codegen", "dispatch", "bench"))


def test_register_optimization_progress_tracks_passes(sm) -> None:
    out = register_optimization_progress(
        sm, session_id="sess1",
        summary="pass 1: queued 3 codegen + 3 bench",
        completed_passes=1,
    )
    assert out["ok"] and out["completed_passes"] == 1
    out2 = register_optimization_progress(
        sm, session_id="sess1",
        summary="pass 2: 3 cache hits", completed_passes=2,
    )
    assert out2["completed_passes"] == 2
    assert "cache hits" in out2["last_summary"]


# ---------------------------------------------------------------------------
# Optimize-tools registered in ALL_TOOLS
# ---------------------------------------------------------------------------


def test_optimize_tools_in_all_tools_bundle() -> None:
    from compgen.mcp.tools import ALL_TOOLS
    names = {t["name"] for t in ALL_TOOLS}
    for n in ("request_model_optimization", "register_optimization_progress"):
        assert n in names


def test_optimize_tools_have_unique_names() -> None:
    names = [t["name"] for t in OPTIMIZE_TOOLS]
    assert len(names) == len(set(names))

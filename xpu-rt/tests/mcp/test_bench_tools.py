"""Tests for ``compgen.mcp.tools.bench``.

Locks in:
  * round-trip: request → list → register → lookup
  * fingerprint short-circuit on identical (kernel × shape × dtype)
  * register persists to KernelDB so future sessions hit the cache
  * KernelDB rehydration: a perf already on disk surfaces on first lookup
  * McpBenchFn cache hit returns the recorded perf; cache miss returns
    a placeholder (perf_us=None, correct=True) and queues the request
"""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.kernels.contract_v3 import (
    ExecutionEnvelope,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    OrchestrationSpec,
    ShapeClass,
    TensorIO,
)
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.bench import (
    BENCH_TOOLS,
    McpBenchFn,
    bench_fingerprint,
    list_pending_bench_requests,
    lookup_bench_result,
    register_bench_result,
    request_kernel_bench,
)
from compgen.memory.kernel_db import (
    KernelDB,
    KernelPerfRecord,
    set_shared_db,
)


@pytest.fixture
def isolated_db(tmp_path: Path):
    db = KernelDB(path=tmp_path / "kernel_db.sqlite")
    set_shared_db(db)
    yield db
    set_shared_db(None)


@pytest.fixture
def sm(tmp_path: Path) -> SessionManager:
    s = SessionManager(scratch_root=tmp_path / "compgen_mcp")
    s.open(session_id="sess1")
    return s


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_bench_tools_registered_with_expected_names() -> None:
    names = {t["name"] for t in BENCH_TOOLS}
    assert names == {
        "request_kernel_bench",
        "register_bench_result",
        "lookup_bench_result",
        "list_pending_bench_requests",
    }


def test_bench_tools_in_all_tools_bundle() -> None:
    from compgen.mcp.tools import ALL_TOOLS

    names = {t["name"] for t in ALL_TOOLS}
    for n in ("request_kernel_bench", "register_bench_result", "lookup_bench_result", "list_pending_bench_requests"):
        assert n in names


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_for_same_input() -> None:
    a = bench_fingerprint("kfp1", "32x32", "f16")
    b = bench_fingerprint("kfp1", "32x32", "f16")
    assert a == b


def test_fingerprint_differs_on_shape() -> None:
    a = bench_fingerprint("kfp1", "32x32", "f16")
    b = bench_fingerprint("kfp1", "64x64", "f16")
    assert a != b


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_request_then_register_then_lookup(sm, isolated_db) -> None:
    out = request_kernel_bench(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp1",
        shape_signature="64x64",
        dtype_signature="f16",
        target="cuda-a100",
        op_family="compute_tiled",
        perf_target_us=100.0,
    )
    assert out["ok"] and not out["found_in_cache"]
    rid = out["request_id"]
    fp = out["fingerprint"]
    assert "PERF TARGET" in out["prompt"]

    pending = list_pending_bench_requests(sm, session_id="sess1")
    assert pending["pending_count"] == 1
    assert pending["requests"][0]["request_id"] == rid

    reg = register_bench_result(
        sm,
        session_id="sess1",
        request_id=rid,
        perf_us=42.0,
        correct=True,
        notes="ok",
    )
    assert reg["ok"] and reg["fingerprint"] == fp

    lk = lookup_bench_result(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp1",
        shape_signature="64x64",
        dtype_signature="f16",
    )
    assert lk["found"] and lk["perf_us"] == 42.0 and lk["correct"]


def test_request_short_circuits_on_cached_fingerprint(sm, isolated_db) -> None:
    out = request_kernel_bench(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp2",
        shape_signature="32",
        target="cuda-a100",
        op_family="pointwise",
    )
    register_bench_result(
        sm,
        session_id="sess1",
        request_id=out["request_id"],
        perf_us=10.0,
        correct=True,
    )
    out2 = request_kernel_bench(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp2",
        shape_signature="32",
        target="cuda-a100",
        op_family="pointwise",
    )
    assert out2["found_in_cache"] is True
    assert out2["perf_us"] == 10.0


# ---------------------------------------------------------------------------
# KernelDB persistence
# ---------------------------------------------------------------------------


def test_register_persists_to_kernel_db(sm, isolated_db) -> None:
    out = request_kernel_bench(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp3",
        shape_signature="x",
        target="cuda-a100",
        op_family="reduce",
    )
    register_bench_result(
        sm,
        session_id="sess1",
        request_id=out["request_id"],
        perf_us=7.5,
        correct=True,
        notes="great",
    )
    rec = isolated_db.best_kernel_perf("cuda-a100", "reduce", "kfp3")
    assert rec is not None
    assert rec.perf_us == 7.5
    assert rec.correctness_passed


def test_request_rehydrates_from_kernel_db_on_first_access(sm, isolated_db) -> None:
    # Pre-seed the KernelDB with a perf record.
    isolated_db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="memory",
            fingerprint="kfp4",
            perf_us=55.0,
            correctness_passed=True,
            measured_at=12345.0,
            notes="seeded",
        )
    )
    # Fresh request must surface the cached result without queueing.
    out = request_kernel_bench(
        sm,
        session_id="sess1",
        kernel_fingerprint="kfp4",
        shape_signature="",
        target="cuda-a100",
        op_family="memory",
    )
    assert out["found_in_cache"] is True
    assert out["perf_us"] == 55.0


# ---------------------------------------------------------------------------
# McpBenchFn adapter
# ---------------------------------------------------------------------------


def _matmul_contract(target: str = "cuda-a100") -> KernelContractV3:
    env = HardwareEnvelope(
        target_name=target,
        vector_lanes=64,
        scratchpad_bytes=49152,
        register_bytes=256,
        native_dtypes=("f16",),
        peak_bandwidth_gbps=672.0,
    )
    return KernelContractV3(
        op_name="matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="lhs", shape=ShapeClass(dims=(128, 256)), dtype_class=("f16",)),
                TensorIO(name="rhs", shape=ShapeClass(dims=(256, 128)), dtype_class=("f16",)),
            ),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(128, 128)), dtype_class=("f16",)),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


def test_mcp_bench_fn_returns_placeholder_on_miss_and_queues(sm, isolated_db) -> None:
    bench = McpBenchFn(sm=sm, session_id="sess1")
    contract = _matmul_contract()
    result = bench(contract, codegen_result=None)
    assert result.perf_us is None
    assert result.correct is True
    assert "queued" in result.notes
    pending = list_pending_bench_requests(sm, session_id="sess1")
    assert pending["pending_count"] == 1


def test_mcp_bench_fn_returns_recorded_perf_on_hit(sm, isolated_db) -> None:
    bench = McpBenchFn(sm=sm, session_id="sess1")
    contract = _matmul_contract()
    # First call queues — agent fulfils it.
    bench(contract, codegen_result=None)
    pending = list_pending_bench_requests(sm, session_id="sess1")
    rid = pending["requests"][0]["request_id"]
    register_bench_result(
        sm,
        session_id="sess1",
        request_id=rid,
        perf_us=15.0,
        correct=True,
        notes="agent measured",
    )
    # Second call hits the cache.
    second = bench(contract, codegen_result=None)
    assert second.perf_us == 15.0
    assert second.correct is True

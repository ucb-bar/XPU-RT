"""Tests for ``compgen.memory.kernel_db``.

Round-trips the two new tables (kernel_perf, fusion_decisions) and
verifies the calibration-helper (average_observed_speedup) returns
sane values that the fusion oracle can use to weight predictions.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from compgen.memory.kernel_db import (
    FusionDecisionRecord,
    KernelDB,
    KernelPerfRecord,
)


@pytest.fixture
def db(tmp_path: Path) -> KernelDB:
    d = KernelDB(path=tmp_path / "test_kernel.sqlite")
    yield d
    d.close()


# ---------------------------------------------------------------------------
# kernel_perf round-trip + best-of selector
# ---------------------------------------------------------------------------


def test_record_then_best_returns_lowest_perf_us(db: KernelDB) -> None:
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="abc",
            perf_us=100.0,
            correctness_passed=True,
        )
    )
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="abc",
            perf_us=80.0,
            correctness_passed=True,
        )
    )
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="abc",
            perf_us=120.0,
            correctness_passed=True,
        )
    )
    best = db.best_kernel_perf("cuda-a100", "matmul", "abc")
    assert best is not None
    assert best.perf_us == 80.0


def test_best_ignores_correctness_failures(db: KernelDB) -> None:
    """A 30μs kernel that didn't pass correctness shouldn't beat a 60μs one
    that did."""
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="abc",
            perf_us=30.0,
            correctness_passed=False,
        )
    )
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="abc",
            perf_us=60.0,
            correctness_passed=True,
        )
    )
    best = db.best_kernel_perf("cuda-a100", "matmul", "abc")
    assert best is not None
    assert best.perf_us == 60.0


def test_best_returns_none_when_no_record(db: KernelDB) -> None:
    assert db.best_kernel_perf("cuda-a100", "matmul", "missing") is None


def test_list_kernel_perf_filters_by_target_and_op(db: KernelDB) -> None:
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="matmul",
            fingerprint="a",
            perf_us=10.0,
            correctness_passed=True,
        )
    )
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-a100",
            op_family="softmax",
            fingerprint="b",
            perf_us=20.0,
            correctness_passed=True,
        )
    )
    db.record_kernel_perf(
        KernelPerfRecord(
            target="cuda-h100",
            op_family="matmul",
            fingerprint="c",
            perf_us=5.0,
            correctness_passed=True,
        )
    )

    a100_only = db.list_kernel_perf(target="cuda-a100")
    assert {r.fingerprint for r in a100_only} == {"a", "b"}

    matmul_only = db.list_kernel_perf(op_family="matmul")
    assert {r.fingerprint for r in matmul_only} == {"a", "c"}


# ---------------------------------------------------------------------------
# fusion_decisions round-trip + calibration helper
# ---------------------------------------------------------------------------


def test_fusion_history_returns_records_sorted_by_time(db: KernelDB) -> None:
    t0 = time.time()
    db.record_fusion_decision(
        FusionDecisionRecord(
            target="cuda-a100",
            producer_role="matmul",
            consumer_role="silu",
            decision="fuse",
            predicted_speedup=1.5,
            observed_speedup=1.2,
            measured_at=t0,
        )
    )
    db.record_fusion_decision(
        FusionDecisionRecord(
            target="cuda-a100",
            producer_role="matmul",
            consumer_role="silu",
            decision="fuse",
            predicted_speedup=1.5,
            observed_speedup=0.9,
            measured_at=t0 + 100,
        )
    )
    h = db.fusion_history("cuda-a100", "matmul", "silu")
    assert len(h) == 2
    # Sorted DESC by measured_at — newer first
    assert h[0].measured_at > h[1].measured_at


def test_average_observed_speedup_calibrates_predictions(db: KernelDB) -> None:
    """Mean of observed values lets the cost model learn over time."""
    for obs in (1.0, 1.2, 0.9, 1.1, 0.8):
        db.record_fusion_decision(
            FusionDecisionRecord(
                target="cuda-a100",
                producer_role="matmul",
                consumer_role="silu",
                decision="fuse",
                predicted_speedup=1.5,
                observed_speedup=obs,
            )
        )
    avg = db.average_observed_speedup("cuda-a100", "matmul", "silu")
    assert avg is not None
    assert 0.99 < avg < 1.01  # mean of (1.0, 1.2, 0.9, 1.1, 0.8) = 1.0


def test_average_observed_speedup_is_none_when_only_null_observations(db: KernelDB) -> None:
    """A predicted-but-not-yet-measured fusion records observed=None."""
    db.record_fusion_decision(
        FusionDecisionRecord(
            target="cuda-a100",
            producer_role="matmul",
            consumer_role="silu",
            decision="fuse",
            predicted_speedup=1.5,
            observed_speedup=None,
        )
    )
    avg = db.average_observed_speedup("cuda-a100", "matmul", "silu")
    assert avg is None


def test_average_speedup_isolates_by_target_and_pair(db: KernelDB) -> None:
    db.record_fusion_decision(
        FusionDecisionRecord(
            target="cuda-a100",
            producer_role="matmul",
            consumer_role="silu",
            decision="fuse",
            predicted_speedup=1.5,
            observed_speedup=2.0,
        )
    )
    db.record_fusion_decision(
        FusionDecisionRecord(
            target="cuda-h100",
            producer_role="matmul",
            consumer_role="silu",
            decision="fuse",
            predicted_speedup=1.5,
            observed_speedup=1.2,
        )
    )
    a = db.average_observed_speedup("cuda-a100", "matmul", "silu")
    h = db.average_observed_speedup("cuda-h100", "matmul", "silu")
    assert a == pytest.approx(2.0)
    assert h == pytest.approx(1.2)

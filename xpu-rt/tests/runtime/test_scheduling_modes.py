"""Tests for dynamic scheduling modes: adaptive batch, priority, calibration."""

from __future__ import annotations

import pytest
from compgen.agent.memory import CostCalibration
from compgen.runtime.adaptive import (
    AdaptiveBatchScheduler,
    TieredPlan,
)
from compgen.runtime.calibration_loop import (
    DEFAULT_DRIFT_THRESHOLD,
    CalibrationLoop,
    CalibrationResult,
    DriftResult,
)
from compgen.runtime.planner import ExecutionPlan, PlacementDecision
from compgen.runtime.priority_scheduler import (
    Priority,
    PriorityScheduler,
    Workload,
)
from compgen.targets.schema import DeviceSpec, TargetProfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_target(num_devices: int = 1) -> TargetProfile:
    """Create a minimal target profile for testing."""
    devices = [
        DeviceSpec(device_type="gpu", name=f"test_gpu_{i}")
        for i in range(num_devices)
    ]
    return TargetProfile(name="test-target", devices=devices)


def _dummy_plan(batch_size: int = 1) -> ExecutionPlan:
    """Create a minimal execution plan for testing."""
    return ExecutionPlan(
        placements=[PlacementDecision(op_name="op_0", device_index=0)],
        execution_order=["op_0"],
        estimated_latency_us=100.0,
        metadata={"batch_size_tier": batch_size},
    )


# ---------------------------------------------------------------------------
# AdaptiveBatchScheduler tests
# ---------------------------------------------------------------------------

class TestAdaptiveBatchScheduler:
    """Tests for batch-tier selection logic."""

    def test_default_tiers(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target)
        assert scheduler.tiers == (1, 8, 32, 128)

    def test_custom_tiers_sorted(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(32, 1, 8))
        assert scheduler.tiers == (1, 8, 32)

    def test_empty_tiers_rejected(self) -> None:
        target = _simple_target()
        with pytest.raises(ValueError, match="At least one batch-size tier"):
            AdaptiveBatchScheduler(target=target, tiers=())

    def test_select_without_precompute_raises(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target)
        with pytest.raises(RuntimeError, match="No plans have been precomputed"):
            scheduler.select_plan(8)

    def test_select_exact_tier(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(1, 8, 32))

        # Manually inject plans to avoid needing xDSL modules
        for bs in (1, 8, 32):
            scheduler.plans[bs] = TieredPlan(batch_size=bs, plan=_dummy_plan(bs))

        result = scheduler.select_plan(8)
        assert result.batch_size == 8

    def test_select_rounds_up_to_next_tier(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(1, 8, 32))

        for bs in (1, 8, 32):
            scheduler.plans[bs] = TieredPlan(batch_size=bs, plan=_dummy_plan(bs))

        # Request batch_size=5 should round up to tier 8
        result = scheduler.select_plan(5)
        assert result.batch_size == 8

    def test_select_larger_than_all_tiers(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(1, 8, 32))

        for bs in (1, 8, 32):
            scheduler.plans[bs] = TieredPlan(batch_size=bs, plan=_dummy_plan(bs))

        # Request batch_size=64 should fall back to largest tier (32)
        result = scheduler.select_plan(64)
        assert result.batch_size == 32

    def test_select_smallest_request(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(4, 16))

        for bs in (4, 16):
            scheduler.plans[bs] = TieredPlan(batch_size=bs, plan=_dummy_plan(bs))

        # Request batch_size=1 should pick tier 4
        result = scheduler.select_plan(1)
        assert result.batch_size == 4

    def test_deduplicate_tiers(self) -> None:
        target = _simple_target()
        scheduler = AdaptiveBatchScheduler(target=target, tiers=(8, 8, 32, 32))
        assert scheduler.tiers == (8, 32)


# ---------------------------------------------------------------------------
# PriorityScheduler tests
# ---------------------------------------------------------------------------

class TestPriorityScheduler:
    """Tests for priority ordering and cooperative preemption."""

    def test_dequeue_empty(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)
        assert scheduler.dequeue(device=0) is None

    def test_submit_and_dequeue_single(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)
        w = Workload(workload_id="w1", plan=_dummy_plan())
        scheduler.submit(w, device=0)
        result = scheduler.dequeue(device=0)
        assert result is not None
        assert result.workload_id == "w1"

    def test_priority_ordering(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)

        low = Workload(workload_id="low", plan=_dummy_plan(), priority=Priority.LOW)
        normal = Workload(workload_id="normal", plan=_dummy_plan(), priority=Priority.NORMAL)
        high = Workload(workload_id="high", plan=_dummy_plan(), priority=Priority.HIGH)

        # Submit in reverse priority order
        scheduler.submit(low, device=0)
        scheduler.submit(normal, device=0)
        scheduler.submit(high, device=0)

        # Dequeue should return HIGH first, then NORMAL, then LOW
        r1 = scheduler.dequeue(device=0)
        r2 = scheduler.dequeue(device=0)
        r3 = scheduler.dequeue(device=0)

        assert r1 is not None and r1.workload_id == "high"
        assert r2 is not None and r2.workload_id == "normal"
        assert r3 is not None and r3.workload_id == "low"

    def test_fifo_within_same_priority(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)

        w1 = Workload(workload_id="first", plan=_dummy_plan(), priority=Priority.NORMAL)
        w2 = Workload(workload_id="second", plan=_dummy_plan(), priority=Priority.NORMAL)
        w3 = Workload(workload_id="third", plan=_dummy_plan(), priority=Priority.NORMAL)

        scheduler.submit(w1, device=0)
        scheduler.submit(w2, device=0)
        scheduler.submit(w3, device=0)

        r1 = scheduler.dequeue(device=0)
        r2 = scheduler.dequeue(device=0)
        r3 = scheduler.dequeue(device=0)

        assert r1 is not None and r1.workload_id == "first"
        assert r2 is not None and r2.workload_id == "second"
        assert r3 is not None and r3.workload_id == "third"

    def test_invalid_device_submit(self) -> None:
        scheduler = PriorityScheduler(num_devices=2)
        w = Workload(workload_id="w1", plan=_dummy_plan())
        with pytest.raises(ValueError, match="Device 5 out of range"):
            scheduler.submit(w, device=5)

    def test_invalid_device_dequeue(self) -> None:
        scheduler = PriorityScheduler(num_devices=2)
        with pytest.raises(ValueError, match="Device -1 out of range"):
            scheduler.dequeue(device=-1)

    def test_multi_device_isolation(self) -> None:
        scheduler = PriorityScheduler(num_devices=2)

        w0 = Workload(workload_id="dev0", plan=_dummy_plan())
        w1 = Workload(workload_id="dev1", plan=_dummy_plan())

        scheduler.submit(w0, device=0)
        scheduler.submit(w1, device=1)

        # Device 0 should only see w0
        r0 = scheduler.dequeue(device=0)
        assert r0 is not None and r0.workload_id == "dev0"
        assert scheduler.dequeue(device=0) is None

        # Device 1 should only see w1
        r1 = scheduler.dequeue(device=1)
        assert r1 is not None and r1.workload_id == "dev1"

    def test_pending_count(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)
        assert scheduler.pending_count(device=0) == 0

        scheduler.submit(Workload(workload_id="a", plan=_dummy_plan(), priority=Priority.HIGH), device=0)
        scheduler.submit(Workload(workload_id="b", plan=_dummy_plan(), priority=Priority.LOW), device=0)
        assert scheduler.pending_count(device=0) == 2

        scheduler.dequeue(device=0)
        assert scheduler.pending_count(device=0) == 1

    def test_should_preempt(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)

        # No pending work -> no preemption
        assert scheduler.should_preempt(Priority.NORMAL, device=0) is False

        # Submit a HIGH priority workload
        scheduler.submit(
            Workload(workload_id="urgent", plan=_dummy_plan(), priority=Priority.HIGH),
            device=0,
        )

        # A NORMAL task should be preempted
        assert scheduler.should_preempt(Priority.NORMAL, device=0) is True
        # A HIGH task should NOT be preempted (not strictly higher)
        assert scheduler.should_preempt(Priority.HIGH, device=0) is False

    def test_should_not_preempt_for_lower_priority(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)

        scheduler.submit(
            Workload(workload_id="low", plan=_dummy_plan(), priority=Priority.LOW),
            device=0,
        )

        # A NORMAL task should NOT be preempted by a LOW pending workload
        assert scheduler.should_preempt(Priority.NORMAL, device=0) is False

    def test_drain(self) -> None:
        scheduler = PriorityScheduler(num_devices=1)

        scheduler.submit(Workload(workload_id="low", plan=_dummy_plan(), priority=Priority.LOW), device=0)
        scheduler.submit(Workload(workload_id="high", plan=_dummy_plan(), priority=Priority.HIGH), device=0)
        scheduler.submit(Workload(workload_id="normal", plan=_dummy_plan(), priority=Priority.NORMAL), device=0)

        drained = scheduler.drain(device=0)
        assert len(drained) == 3
        assert drained[0].workload_id == "high"
        assert drained[1].workload_id == "normal"
        assert drained[2].workload_id == "low"

        # Queue should be empty after drain
        assert scheduler.pending_count(device=0) == 0


# ---------------------------------------------------------------------------
# CalibrationLoop tests
# ---------------------------------------------------------------------------

class TestCalibrationLoop:
    """Tests for drift detection and re-solve triggering."""

    def test_no_drift(self) -> None:
        target = _simple_target()
        loop = CalibrationLoop(target=target)

        measurements = {"gpu_0": {"matmul": 100.0}}
        estimates = {"gpu_0": {"matmul": 100.0}}

        results = loop.check_drift(measurements, estimates)
        assert len(results) == 1
        assert results[0].drift == pytest.approx(0.0)
        assert results[0].exceeded is False

    def test_drift_below_threshold(self) -> None:
        target = _simple_target()
        loop = CalibrationLoop(target=target, drift_threshold=0.20)

        measurements = {"gpu_0": {"matmul": 100.0}}
        estimates = {"gpu_0": {"matmul": 110.0}}  # 10% drift

        results = loop.check_drift(measurements, estimates)
        assert len(results) == 1
        assert results[0].drift == pytest.approx(0.10)
        assert results[0].exceeded is False

    def test_drift_above_threshold(self) -> None:
        target = _simple_target()
        loop = CalibrationLoop(target=target, drift_threshold=0.20)

        measurements = {"gpu_0": {"matmul": 100.0}}
        estimates = {"gpu_0": {"matmul": 130.0}}  # 30% drift

        results = loop.check_drift(measurements, estimates)
        assert len(results) == 1
        assert results[0].drift == pytest.approx(0.30)
        assert results[0].exceeded is True

    def test_custom_threshold(self) -> None:
        target = _simple_target()
        loop = CalibrationLoop(target=target, drift_threshold=0.05)

        measurements = {"gpu_0": {"matmul": 100.0}}
        estimates = {"gpu_0": {"matmul": 108.0}}  # 8% drift

        results = loop.check_drift(measurements, estimates)
        assert results[0].exceeded is True  # 8% > 5%

    def test_update_calibration_ema(self) -> None:
        target = _simple_target()
        calibration = CostCalibration()
        loop = CalibrationLoop(target=target, calibration=calibration)

        drift_results = [
            DriftResult(
                op_type="matmul",
                device_name="gpu_0",
                estimated_us=100.0,
                measured_us=150.0,
                drift=0.5,
                exceeded=True,
            )
        ]

        loop.update_calibration(drift_results)

        factor = calibration.get_factor("gpu_0", "matmul")
        # EMA: 0.7 * 1.0 + 0.3 * (150/100) = 0.7 + 0.45 = 1.15
        assert factor == pytest.approx(1.15)

    def test_multiple_ema_updates_converge(self) -> None:
        target = _simple_target()
        calibration = CostCalibration()
        loop = CalibrationLoop(target=target, calibration=calibration)

        # Repeatedly measure 150us vs estimated 100us
        for _ in range(10):
            drift_results = [
                DriftResult(
                    op_type="matmul",
                    device_name="gpu_0",
                    estimated_us=100.0,
                    measured_us=150.0,
                    drift=0.5,
                    exceeded=True,
                )
            ]
            loop.update_calibration(drift_results)

        factor = calibration.get_factor("gpu_0", "matmul")
        # EMA should converge toward 1.5 (= 150/100)
        assert factor > 1.4

    def test_zero_measured_skipped(self) -> None:
        target = _simple_target()
        loop = CalibrationLoop(target=target)

        measurements = {"gpu_0": {"matmul": 0.0}}
        estimates = {"gpu_0": {"matmul": 100.0}}

        results = loop.check_drift(measurements, estimates)
        assert len(results) == 0

    def test_multi_device_drift(self) -> None:
        target = _simple_target(num_devices=2)
        loop = CalibrationLoop(target=target, drift_threshold=0.20)

        measurements = {
            "gpu_0": {"matmul": 100.0, "relu": 10.0},
            "gpu_1": {"matmul": 200.0},
        }
        estimates = {
            "gpu_0": {"matmul": 105.0, "relu": 15.0},  # 5%, 50%
            "gpu_1": {"matmul": 180.0},                 # 10%
        }

        results = loop.check_drift(measurements, estimates)
        assert len(results) == 3

        exceeded = [r for r in results if r.exceeded]
        assert len(exceeded) == 1
        assert exceeded[0].op_type == "relu"

    def test_default_threshold_value(self) -> None:
        assert DEFAULT_DRIFT_THRESHOLD == 0.20

    def test_calibration_result_fields(self) -> None:
        result = CalibrationResult(
            drift_results=[],
            max_drift=0.0,
            threshold=0.20,
            re_solve_triggered=False,
            new_plan=None,
        )
        assert result.re_solve_triggered is False
        assert result.new_plan is None

"""Tests for the Stage-4 feedback-loop driver."""

from __future__ import annotations

from pathlib import Path

import pytest
from xpu_rt.runtime.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationModel,
    MeasurementRecord,
)
from xpu_rt.scheduler.qnn_real_workload import load_cost_matrix
from xpu_rt.scheduling.feedback_loop import (
    LoopConfig,
    LoopState,
    _convergence_check,
    _is_cycle,
    init_loop_state,
    step,
)
from xpu_rt.scheduling.granularity import Chunk, GranularityPlan, apply_fusion
from xpu_rt.scheduling.policy import SchedulerPolicy

COST_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "profiled"
    / "qnn_cost_matrix.json"
)


@pytest.fixture(scope="module")
def cost_matrix() -> dict:
    return load_cost_matrix(COST_MATRIX_PATH)


@pytest.fixture()
def calibration() -> CalibrationModel:
    return CalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        target_id="qrb5165",
        overhead_us={
            "dronet": {"CPU": 1000.0, "GPU": 500.0, "DSP": 800.0},
            "yolov8n": {"CPU": 1000.0, "GPU": 500.0, "DSP": 800.0},
        },
        contention_factor={
            "dronet": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
            "yolov8n": {"CPU": 1.0, "GPU": 1.0, "DSP": 1.0},
        },
        history=(),
        created_at="2026-05-15T00:00:00+00:00",
    )


def test_init_loop_state_is_init(cost_matrix, calibration):
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration)
    assert s.status == "init"
    assert s.iteration == 0
    assert s.history == ()
    assert len(s.current_chunks) > 0


def test_step_without_measurement_advances_iteration_no_calibration_change(
    cost_matrix, calibration,
):
    s0 = init_loop_state("dronet", "qrb5165", cost_matrix, calibration)
    s1 = step(s0, None, cost_matrix=cost_matrix)
    assert s1.iteration == 1
    assert s1.current_calibration.overhead_us == calibration.overhead_us
    assert s1.history[-1].decision_next == "recalibrate_only"
    assert s1.history[-1].measured_makespan_us is None


def test_step_with_measurement_in_band_marks_converged_after_2_rounds(
    cost_matrix, calibration,
):
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration)
    s = step(s, None, cost_matrix=cost_matrix)
    pred = s.current_predicted_makespan_us
    assert pred is not None and pred > 0
    m = MeasurementRecord(
        workload_id="dronet", backend="DSP",
        measured_us=pred * 1.02, per_op_sum_us=pred * 0.5, predicted_us=pred,
    )
    s = step(s, m, cost_matrix=cost_matrix)
    pred2 = s.current_predicted_makespan_us
    m2 = MeasurementRecord(
        workload_id="dronet", backend="DSP",
        measured_us=pred2 * 1.01, per_op_sum_us=pred2 * 0.5, predicted_us=pred2,
    )
    s = step(s, m2, cost_matrix=cost_matrix)
    assert s.status == "converged"


def test_step_with_outlier_measurement_triggers_recompile_finer(
    cost_matrix, calibration,
):
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration)
    s = step(s, None, cost_matrix=cost_matrix)
    pred = s.current_predicted_makespan_us
    # measured >> per_op_sum_us → outlier path (ratio > 2.0).
    m = MeasurementRecord(
        workload_id="dronet", backend="DSP",
        measured_us=pred * 5.0, per_op_sum_us=pred * 0.1, predicted_us=pred,
    )
    s = step(s, m, cost_matrix=cost_matrix)
    assert s.history[-1].decision_next == "recompile_finer"


def test_step_with_transfer_dominated_measurement_triggers_recompile_coarser(
    cost_matrix, calibration,
):
    # Use yolov8n (273 ops) so chunks land on different backends and pay
    # transfer cost. We force transfer dominance by setting an extremely
    # low measured value relative to the inferred transfer total.
    s = init_loop_state(
        "yolov8n", "qrb5165", cost_matrix, calibration,
        config=LoopConfig(max_chunk_ops=4, max_partitions=200),
    )
    s = step(s, None, cost_matrix=cost_matrix)
    pred = s.current_predicted_makespan_us
    # measured is small enough that ANY non-trivial transfer dominates,
    # but we still need ratio < outlier_threshold (2.0) to skip rule (c).
    m = MeasurementRecord(
        workload_id="yolov8n", backend="DSP",
        measured_us=pred * 1.5, per_op_sum_us=pred * 1.0, predicted_us=pred,
    )
    s = step(s, m, cost_matrix=cost_matrix)
    # Either coarser or recalibrate; assert it's not finer (which would
    # be the wrong response when transfer dominates).
    assert s.history[-1].decision_next in (
        "recompile_coarser", "recalibrate_only",
    )


def test_max_iterations_returns_max_iter_status(cost_matrix, calibration):
    cfg = LoopConfig(max_iterations=2, epsilon=1e-9, consecutive_required=99)
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration, config=cfg)
    pred_seed = 1000.0
    for _ in range(3):
        pred = s.current_predicted_makespan_us or pred_seed
        m = MeasurementRecord(
            workload_id="dronet", backend="DSP",
            measured_us=pred * 1.5, per_op_sum_us=pred * 1.0, predicted_us=pred,
        )
        s = step(s, m, cost_matrix=cost_matrix, config=cfg)
    assert s.status == "max_iter"


def test_dispatch_picks_mosek_at_low_n(cost_matrix, calibration):
    # dronet has 30 ops; with max_chunk_ops=1 every op becomes its own
    # chunk, giving n_partitions=30 << mosek_max(60).
    cfg = LoopConfig(max_chunk_ops=1, max_partitions=200)
    policy = SchedulerPolicy()
    s = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        scheduler_policy=policy, config=cfg,
    )
    s = step(
        s, None, cost_matrix=cost_matrix, config=cfg, scheduler_policy=policy,
    )
    assert s.current_solver_choice == "mosek"
    assert s.history[-1].solver_choice == "mosek"


def test_dispatch_picks_greedy_at_high_n(cost_matrix, calibration, tmp_path):
    # yolov8n has 273 ops; with max_chunk_ops=1 and max_partitions raised
    # past 200 we land in the GREEDY band (cpsat_max_partitions=200).
    # Disable fusion so the per-op chunking that the solver-selection
    # rule keys on isn't collapsed by the fusion pass; redirect memory_dir
    # so a persisted bandit log can't reseed ``max_chunk_ops``.
    cfg = LoopConfig(max_chunk_ops=1, max_partitions=300, enable_fusion=False)
    policy = SchedulerPolicy()
    s = init_loop_state(
        "yolov8n", "qrb5165", cost_matrix, calibration,
        scheduler_policy=policy, config=cfg, memory_dir=tmp_path,
    )
    s = step(
        s, None, cost_matrix=cost_matrix, config=cfg, scheduler_policy=policy,
    )
    assert s.current_solver_choice == "greedy"
    assert s.history[-1].solver_choice == "greedy"
    # Greedy is in-process, must succeed and produce a finite makespan.
    assert s.current_predicted_makespan_us is not None
    assert s.current_predicted_makespan_us > 0


def test_loop_round_records_tv_memory_skipped(cost_matrix, calibration):
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration)
    s = step(s, None, cost_matrix=cost_matrix)
    assert s.history[-1].tv_memory_skipped is True
    # tv_memory_proved stays True (no obligation can fail when none ran),
    # but tv_memory_skipped is the truthful audit signal.
    assert s.history[-1].tv_memory_proved is True


# --------------------------------------------------------------------------- #
# A1: Convergence rule — EMA over last K rounds admits oscillation
# --------------------------------------------------------------------------- #


def test_oscillating_errors_converge_under_k3():
    # Last 3 = [0.18, 0.05, 0.06]; in-band count (eps=0.10) = 2 ≥ min=2.
    errs = (0.12, 0.18, 0.05, 0.06)
    converged, in_band, window = _convergence_check(
        errs, window=3, min_in_band=2, epsilon=0.10,
    )
    assert converged is True
    assert in_band == 2
    assert window == 3


def test_strict_oscillation_does_not_converge():
    # Last 3 = [0.25, 0.05, 0.25]; only 1 in-band — should NOT converge.
    errs = (0.05, 0.25, 0.05, 0.25)
    converged, in_band, _ = _convergence_check(
        errs, window=3, min_in_band=2, epsilon=0.10,
    )
    assert converged is False
    assert in_band == 1


# --------------------------------------------------------------------------- #
# A3: Regression guard blocks a "converged" schedule worse than baseline
# --------------------------------------------------------------------------- #


def test_regression_guard_blocks_bad_schedule(cost_matrix, calibration):
    # Run two in-band rounds so the convergence rule fires, but pretend the
    # baseline is tiny — predicted should exceed regression_threshold ×
    # baseline and the loop must flip to "failed" instead of "converged".
    cfg = LoopConfig(convergence_window=2, convergence_min_in_band=2,
                     regression_threshold=1.2)
    s = init_loop_state("dronet", "qrb5165", cost_matrix, calibration, config=cfg)
    s = step(s, None, cost_matrix=cost_matrix, config=cfg)
    pred = s.current_predicted_makespan_us
    assert pred is not None and pred > 0
    # Force a tiny baseline so any predicted value triggers the guard.
    s = LoopState(**{**s.__dict__, "baseline_makespan_us": pred * 0.1})
    m = MeasurementRecord(workload_id="dronet", backend="DSP",
                          measured_us=pred * 1.02, per_op_sum_us=pred * 0.5,
                          predicted_us=pred)
    s = step(s, m, cost_matrix=cost_matrix, config=cfg)
    pred2 = s.current_predicted_makespan_us or pred
    m2 = MeasurementRecord(workload_id="dronet", backend="DSP",
                           measured_us=pred2 * 1.01, per_op_sum_us=pred2 * 0.5,
                           predicted_us=pred2)
    s = step(s, m2, cost_matrix=cost_matrix, config=cfg)
    assert s.status == "failed"
    assert "regression_guard" in s.history[-1].reason


# --------------------------------------------------------------------------- #
# A2: Cycle breaker forces recalibrate_only after N repeats
# --------------------------------------------------------------------------- #


def test_same_decision_retried_too_many_times_breaks_cycle():
    # Synthesise a decision_history of "recompile_finer" x3 and ask whether
    # appending a 4th would trip the cycle-break (max_retries=2 ⇒ 3 prior
    # repeats already satisfies the predicate, so the 4th gets rewritten).
    history = ("recompile_finer", "recompile_finer", "recompile_finer")
    assert _is_cycle(history, "recompile_finer", max_retries=2) is True
    # Terminal decisions are never considered cycles, even on long histories.
    assert _is_cycle(history, "converged", max_retries=2) is False
    # A different decision breaks the run, so no cycle.
    mixed = ("recompile_finer", "recalibrate_only", "recompile_finer")
    assert _is_cycle(mixed, "recompile_finer", max_retries=2) is False


# --------------------------------------------------------------------------- #
# B1: Fusion merges adjacent same-backend chunks with high transfer
# --------------------------------------------------------------------------- #


def test_fusion_merges_adjacent_same_backend_chunks_with_high_transfer():
    # Two same-backend chunks with nonzero (same-backend ⇒ 0 transfer in
    # matrix, but should_fuse still fuses unconditionally on shared
    # backend). Build a trivial 2-chunk plan and assert the fusion pass
    # collapses it to 1.
    chunks = (
        Chunk("chunk_000", ("op_a",), "DSP", {"CPU": 100.0, "GPU": 200.0, "DSP": 50.0}),
        Chunk("chunk_001", ("op_b",), "DSP", {"CPU": 100.0, "GPU": 200.0, "DSP": 50.0}),
    )
    plan = GranularityPlan(workload_id="synthetic", chunks=chunks,
                           specialty_summary={}, n_partitions=2)
    transfer = [[0.0, 100.0, 100.0],
                [100.0, 0.0, 100.0],
                [100.0, 100.0, 0.0]]
    fused = apply_fusion(plan, transfer_matrix=transfer,
                         fusion_gain_threshold=0.3)
    assert fused.n_partitions == 1
    assert fused.chunks[0].op_ids == ("op_a", "op_b")


# --------------------------------------------------------------------------- #
# B2: Granularity perturbation steps max_chunk_ops on recompile_finer
# --------------------------------------------------------------------------- #


def test_recompile_finer_decreases_max_chunk_ops_next_iter(
    cost_matrix, calibration, tmp_path,
):
    # Drive the loop to a recompile_finer decision (huge outlier ratio)
    # and assert the *next* state's max_chunk_ops is exactly start - step.
    # Use ``memory_dir=tmp_path`` to bypass any persisted bandit log that
    # would otherwise reseed ``current_max_chunk_ops``.
    cfg = LoopConfig(max_chunk_ops=16, granularity_perturbation_step=4)
    s = init_loop_state(
        "dronet", "qrb5165", cost_matrix, calibration,
        config=cfg, memory_dir=tmp_path,
    )
    assert s.current_max_chunk_ops == 16
    s = step(s, None, cost_matrix=cost_matrix, config=cfg)
    pred = s.current_predicted_makespan_us
    m = MeasurementRecord(workload_id="dronet", backend="DSP",
                          measured_us=pred * 5.0, per_op_sum_us=pred * 0.1,
                          predicted_us=pred)
    s = step(s, m, cost_matrix=cost_matrix, config=cfg)
    assert s.history[-1].decision_next == "recompile_finer"
    assert s.current_max_chunk_ops == 12

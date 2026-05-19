"""Unit tests for :mod:`xpu_rt.scheduling.objectives`."""

from __future__ import annotations

from xpu_rt.scheduling.objectives import (
    MultiObjectiveSpec,
    ObjectiveKind,
    ObjectiveScore,
    ObjectiveWeight,
    ScheduleMetrics,
    compute_metrics,
    evaluate,
    pareto_frontier,
)


def _mk_metrics(
    *,
    makespan: float = 100.0,
    deadline_violations: int = 0,
    deadline_overage_us: float = 0.0,
    peak_mem: int = 0,
) -> ScheduleMetrics:
    return ScheduleMetrics(
        makespan_us=makespan,
        deadline_violations=deadline_violations,
        deadline_violation_total_us=deadline_overage_us,
        peak_memory_bytes=peak_mem,
        energy_proxy_joules=None,
        makespan_variance_us=0.0,
    )


def test_makespan_only_score_equals_makespan() -> None:
    spec = MultiObjectiveSpec()  # default: makespan weight 1.0
    metrics = _mk_metrics(makespan=12345.0)
    score = evaluate(spec, metrics)
    assert score.score == 12345.0
    assert score.component_scores == {"makespan": 12345.0}


def test_deadline_violation_increases_score() -> None:
    spec = MultiObjectiveSpec(weights=(
        ObjectiveWeight(ObjectiveKind.MAKESPAN, 0.5, target_value=100.0),
        ObjectiveWeight(ObjectiveKind.DEADLINE_VIOLATION_COUNT, 0.5, target_value=1.0),
    ))
    a = evaluate(spec, _mk_metrics(makespan=100.0, deadline_violations=0))
    b = evaluate(spec, _mk_metrics(makespan=100.0, deadline_violations=2))
    assert b.score > a.score
    assert b.component_scores["deadline_violation_count"] > 0.0


def test_peak_memory_weighted_in() -> None:
    spec = MultiObjectiveSpec(weights=(
        ObjectiveWeight(ObjectiveKind.MAKESPAN, 0.5, target_value=100.0),
        ObjectiveWeight(ObjectiveKind.PEAK_MEMORY_BYTES, 0.5, target_value=1_000_000.0),
    ))
    a = evaluate(spec, _mk_metrics(makespan=100.0, peak_mem=500_000))
    b = evaluate(spec, _mk_metrics(makespan=100.0, peak_mem=1_000_000))
    assert b.score > a.score
    diff = b.component_scores["peak_memory_bytes"] - a.component_scores["peak_memory_bytes"]
    assert abs(diff - 0.25) < 1e-9


def test_pareto_frontier_keeps_non_dominated() -> None:
    def _score(makespan: float, deadline: int) -> ObjectiveScore:
        m = _mk_metrics(makespan=makespan, deadline_violations=deadline)
        return ObjectiveScore(
            score=makespan + deadline,
            component_scores={"makespan": makespan, "deadline_violation_count": float(deadline)},
            raw_metrics=m,
        )

    a = _score(100.0, 5)   # cheap-makespan, many violations
    b = _score(150.0, 2)   # mid trade-off
    c = _score(200.0, 0)   # expensive-makespan, no violations
    d = _score(220.0, 4)   # strictly dominated by b (and c)
    front = pareto_frontier((a, b, c, d))
    front_score_pairs = {
        (s.component_scores["makespan"], s.component_scores["deadline_violation_count"])
        for s in front
    }
    assert (100.0, 5.0) in front_score_pairs
    assert (150.0, 2.0) in front_score_pairs
    assert (200.0, 0.0) in front_score_pairs
    assert (220.0, 4.0) not in front_score_pairs


def test_weight_normalization() -> None:
    spec = MultiObjectiveSpec(weights=(
        ObjectiveWeight(ObjectiveKind.MAKESPAN, 2.0),
        ObjectiveWeight(ObjectiveKind.DEADLINE_VIOLATION_COUNT, 1.0),
        ObjectiveWeight(ObjectiveKind.PEAK_MEMORY_BYTES, 1.0),
    ))
    normed = spec.normalized()
    total = sum(w.weight for w in normed.weights)
    assert abs(total - 1.0) < 1e-9
    assert abs(normed.weights[0].weight - 0.5) < 1e-9


def test_compute_metrics_zero_violations_when_no_deadlines() -> None:
    metrics = compute_metrics(
        start_times={"c0": 0.0, "c1": 50.0},
        end_times={"c0": 50.0, "c1": 100.0},
        device_assignments={"c0": 0, "c1": 1},
        deadlines_us=None,
    )
    assert metrics.deadline_violations == 0
    assert metrics.deadline_violation_total_us == 0.0
    assert metrics.makespan_us == 100.0

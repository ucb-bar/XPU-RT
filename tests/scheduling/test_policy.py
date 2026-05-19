"""Unit tests for the scheduler / memory-planner solver policy."""

from __future__ import annotations

from xpu_rt.scheduling.policy import (
    MemoryPlannerChoice,
    MemoryPlannerPolicy,
    SchedulerPolicy,
    SolverChoice,
)


# ---------------------------------------------------------------------------
# SchedulerPolicy
# ---------------------------------------------------------------------------


def test_scheduler_small_chooses_mosek() -> None:
    assert SchedulerPolicy().choose(1) is SolverChoice.MOSEK


def test_scheduler_mosek_upper_boundary() -> None:
    assert SchedulerPolicy().choose(60) is SolverChoice.MOSEK


def test_scheduler_just_above_mosek_chooses_cpsat() -> None:
    assert SchedulerPolicy().choose(61) is SolverChoice.CPSAT


def test_scheduler_cpsat_upper_boundary() -> None:
    assert SchedulerPolicy().choose(200) is SolverChoice.CPSAT


def test_scheduler_just_above_cpsat_chooses_greedy() -> None:
    assert SchedulerPolicy().choose(201) is SolverChoice.GREEDY


def test_scheduler_very_large_chooses_greedy() -> None:
    assert SchedulerPolicy().choose(5000) is SolverChoice.GREEDY


def test_scheduler_custom_thresholds() -> None:
    policy = SchedulerPolicy(mosek_max_partitions=30, cpsat_max_partitions=100)
    assert policy.choose(50) is SolverChoice.CPSAT
    assert policy.choose(30) is SolverChoice.MOSEK
    assert policy.choose(101) is SolverChoice.GREEDY


def test_scheduler_reason_mentions_thresholds() -> None:
    text = SchedulerPolicy().reason(120)
    assert text
    assert "60" in text
    assert "200" in text


# ---------------------------------------------------------------------------
# MemoryPlannerPolicy
# ---------------------------------------------------------------------------


def test_memory_tight_small_few_chooses_milp() -> None:
    policy = MemoryPlannerPolicy()
    choice = policy.choose(
        n_buffers=50,
        projected_tier_usage_ratio=0.95,
        projected_total_bytes=16 * 2**20,
    )
    assert choice is MemoryPlannerChoice.MILP


def test_memory_loose_tier_chooses_greedy() -> None:
    policy = MemoryPlannerPolicy()
    choice = policy.choose(
        n_buffers=50,
        projected_tier_usage_ratio=0.5,
        projected_total_bytes=16 * 2**20,
    )
    assert choice is MemoryPlannerChoice.GREEDY


def test_memory_tight_but_huge_bytes_chooses_greedy() -> None:
    policy = MemoryPlannerPolicy()
    choice = policy.choose(
        n_buffers=50,
        projected_tier_usage_ratio=0.95,
        projected_total_bytes=128 * 2**20,
    )
    assert choice is MemoryPlannerChoice.GREEDY


def test_memory_tight_but_many_buffers_chooses_greedy() -> None:
    policy = MemoryPlannerPolicy()
    choice = policy.choose(
        n_buffers=300,
        projected_tier_usage_ratio=0.95,
        projected_total_bytes=16 * 2**20,
    )
    assert choice is MemoryPlannerChoice.GREEDY


def test_memory_boundary_tightness_threshold_chooses_milp() -> None:
    policy = MemoryPlannerPolicy()
    choice = policy.choose(
        n_buffers=10,
        projected_tier_usage_ratio=0.85,
        projected_total_bytes=1 * 2**20,
    )
    assert choice is MemoryPlannerChoice.MILP


def test_memory_reason_non_empty_for_each_branch() -> None:
    policy = MemoryPlannerPolicy()
    a = policy.reason(50, 0.95, 16 * 2**20)
    b = policy.reason(50, 0.5, 16 * 2**20)
    c = policy.reason(50, 0.95, 128 * 2**20)
    d = policy.reason(300, 0.95, 16 * 2**20)
    for text in (a, b, c, d):
        assert text
        assert isinstance(text, str)

"""Tests for the per-SM CP-SAT queue solver (Algorithm 1, ETC paper)."""

from __future__ import annotations

import pytest

from compgen.solve.per_sm_queue import (
    EventEdge,
    PerSMSchedule,
    TileTask,
    solve_per_sm_queue,
)


# ---------------------------------------------------------------------------
# Trivial / boundary cases
# ---------------------------------------------------------------------------


def test_empty_task_list_is_feasible() -> None:
    sched = solve_per_sm_queue(tasks=[], edges=[], sm_count=4)
    assert sched.feasible
    assert sched.makespan_us == 0.0
    assert sched.assignment == {}
    assert sched.per_sm_order == {}


def test_zero_sm_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="sm_count"):
        solve_per_sm_queue(tasks=[TileTask("t", "f")], edges=[], sm_count=0)


def test_single_task_assigned_to_some_sm() -> None:
    sched = solve_per_sm_queue(
        tasks=[TileTask("t0", "matmul", duration_us=2.0)],
        edges=[],
        sm_count=4,
    )
    assert sched.feasible
    assert sched.makespan_us == pytest.approx(2.0, abs=0.01)
    assert "t0" in sched.assignment
    assert 0 <= sched.assignment["t0"] < 4


# ---------------------------------------------------------------------------
# Per-SM no-overlap and dependency edges
# ---------------------------------------------------------------------------


def test_two_independent_tasks_run_in_parallel_on_distinct_sms() -> None:
    """With 2 SMs and 2 independent tasks of equal length, optimal makespan
    equals one task duration -- the solver must place them on different SMs."""
    sched = solve_per_sm_queue(
        tasks=[
            TileTask("a", "matmul", duration_us=3.0),
            TileTask("b", "matmul", duration_us=3.0),
        ],
        edges=[],
        sm_count=2,
    )
    assert sched.feasible
    assert sched.makespan_us == pytest.approx(3.0, abs=0.01)
    assert sched.assignment["a"] != sched.assignment["b"]


def test_dependency_edge_serialises_two_tasks() -> None:
    """A dependency edge prevents parallel execution; makespan must be the sum."""
    sched = solve_per_sm_queue(
        tasks=[
            TileTask("producer", "matmul", duration_us=2.0),
            TileTask("consumer", "reduce_scatter", duration_us=2.0),
        ],
        edges=[EventEdge("producer", "consumer", event="E")],
        sm_count=4,
    )
    assert sched.feasible
    assert sched.makespan_us == pytest.approx(4.0, abs=0.01)
    assert sched.start_times["consumer"] >= sched.start_times["producer"] + 2.0 - 1e-3


def test_diamond_dag_respects_all_edges() -> None:
    r"""    A
       / \
      B   C
       \ /
        D
    With 2 SMs, optimal: A on SM0 (0..1), B on SM0 (1..3), C on SM1 (1..3),
    D on either (3..4). Makespan = 4.
    """
    sched = solve_per_sm_queue(
        tasks=[
            TileTask("A", "f", duration_us=1.0),
            TileTask("B", "f", duration_us=2.0),
            TileTask("C", "f", duration_us=2.0),
            TileTask("D", "f", duration_us=1.0),
        ],
        edges=[
            EventEdge("A", "B"),
            EventEdge("A", "C"),
            EventEdge("B", "D"),
            EventEdge("C", "D"),
        ],
        sm_count=2,
    )
    assert sched.feasible
    assert sched.makespan_us == pytest.approx(4.0, abs=0.01)


def test_per_sm_order_is_sorted_by_start_time() -> None:
    sched = solve_per_sm_queue(
        tasks=[
            TileTask("t0", "f", duration_us=1.0),
            TileTask("t1", "f", duration_us=1.0),
            TileTask("t2", "f", duration_us=1.0),
        ],
        edges=[EventEdge("t0", "t1"), EventEdge("t1", "t2")],
        sm_count=1,
    )
    assert sched.feasible
    queue = sched.per_sm_order[0]
    assert queue == ["t0", "t1", "t2"]


# ---------------------------------------------------------------------------
# Affinity hints (used to honour LLM placement intents)
# ---------------------------------------------------------------------------


def test_affinity_pins_task_to_requested_sm() -> None:
    sched = solve_per_sm_queue(
        tasks=[
            TileTask("p", "f", duration_us=1.0, affinity_sm=2),
            TileTask("q", "f", duration_us=1.0),
        ],
        edges=[],
        sm_count=4,
    )
    assert sched.feasible
    assert sched.assignment["p"] == 2


def test_out_of_range_affinity_is_rejected() -> None:
    with pytest.raises(ValueError, match="affinity_sm"):
        solve_per_sm_queue(
            tasks=[TileTask("p", "f", duration_us=1.0, affinity_sm=42)],
            edges=[],
            sm_count=4,
        )


# ---------------------------------------------------------------------------
# Output type sanity
# ---------------------------------------------------------------------------


def test_returns_per_sm_schedule_instance() -> None:
    sched = solve_per_sm_queue(
        tasks=[TileTask("t", "f")], edges=[], sm_count=1
    )
    assert isinstance(sched, PerSMSchedule)
    assert sched.solve_time_ms >= 0.0

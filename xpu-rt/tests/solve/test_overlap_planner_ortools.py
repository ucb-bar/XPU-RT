"""CP-SAT overlap planner end-to-end tests."""

from __future__ import annotations

import pytest

from compgen.solve.overlap_planner import (
    Dependency,
    Operation,
    OverlapPlanInput,
    Resource,
    plan_overlap,
)
from compgen.solve.solver_types import SolverBackendName, SolverStatus


def test_dependency_orders_consumer_after_producer():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("A", duration=5, resource_id="q0"),
            Operation("B", duration=3, resource_id="q1"),
        ),
        dependencies=(Dependency("A", "B"),),
        resources=(Resource("q0"), Resource("q1")),
    )
    response, sched = plan_overlap(plan_input)
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert sched is not None
    by_id = {s.op_id: s for s in sched.schedule}
    assert by_id["A"].end_tick <= by_id["B"].start_tick
    assert sched.makespan == 8


def test_independent_ops_overlap_on_different_resources():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("A", duration=5, resource_id="q0"),
            Operation("B", duration=5, resource_id="q1"),
        ),
        resources=(Resource("q0"), Resource("q1")),
    )
    response, sched = plan_overlap(plan_input)
    assert sched is not None
    assert sched.makespan == 5  # overlap, not serialized


def test_same_resource_no_overlap():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("A", duration=5, resource_id="q0"),
            Operation("B", duration=5, resource_id="q0"),
        ),
        resources=(Resource("q0"),),
    )
    response, sched = plan_overlap(plan_input)
    assert sched is not None
    by_id = {s.op_id: s for s in sched.schedule}
    # Ranges disjoint on the same resource
    a_lo, a_hi = by_id["A"].start_tick, by_id["A"].end_tick
    b_lo, b_hi = by_id["B"].start_tick, by_id["B"].end_tick
    assert a_hi <= b_lo or b_hi <= a_lo
    assert sched.makespan == 10


def test_copy_finishes_before_consumer():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("copy_H2D", duration=4, resource_id="dma", kind="copy"),
            Operation("gpu_kernel", duration=6, resource_id="gpu_queue"),
        ),
        dependencies=(Dependency("copy_H2D", "gpu_kernel"),),
        resources=(Resource("dma", kind="dma"), Resource("gpu_queue")),
    )
    response, sched = plan_overlap(plan_input)
    assert sched is not None
    by_id = {s.op_id: s for s in sched.schedule}
    assert by_id["copy_H2D"].end_tick <= by_id["gpu_kernel"].start_tick


def test_impossible_deadline_returns_infeasible():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("A", duration=10, resource_id="q0"),
            Operation("B", duration=10, resource_id="q0"),
        ),
        resources=(Resource("q0"),),
        deadline=15,  # need 20 ticks, only 15 allowed
    )
    response, sched = plan_overlap(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert sched is None


def test_solved_schedule_round_trips_through_dict():
    plan_input = OverlapPlanInput(
        operations=(Operation("A", duration=2, resource_id="q0"),),
        resources=(Resource("q0"),),
    )
    response, sched = plan_overlap(plan_input)
    assert sched is not None
    body = sched.to_dict()
    assert body["schema_version"] == "overlap_schedule_solver_v1"
    assert body["solver_backend"] == "ortools_cp_sat"
    assert body["makespan"] == 2
    assert response.formulation_hash == body["formulation_hash"]


def test_invalid_dependency_reference_returns_infeasible():
    plan_input = OverlapPlanInput(
        operations=(Operation("A", duration=2, resource_id="q0"),),
        dependencies=(Dependency("A", "ghost"),),
        resources=(Resource("q0"),),
    )
    response, sched = plan_overlap(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert sched is None

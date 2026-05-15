"""CP-SAT placement planner end-to-end tests."""

from __future__ import annotations

import pytest

from compgen.solve.placement_planner import (
    Device,
    Edge,
    PlacementPlanInput,
    Region,
    plan_placement,
)
from compgen.solve.solver_types import (
    SolverBackendName,
    SolverStatus,
)


def _two_cpu_regions():
    return PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("cpu0",), memory_bytes=1024, compute_cost_by_device={"cpu0": 1.0}),
            Region("r1", allowed_devices=("cpu0",), memory_bytes=1024, compute_cost_by_device={"cpu0": 1.0}),
        ),
        devices=(Device("cpu0", memory_capacity=1024 * 1024, target_class="host_cpu"),),
    )


def _cpu_plus_gpu():
    return PlacementPlanInput(
        regions=(
            Region(
                "r0",
                allowed_devices=("cpu0", "gpu0"),
                memory_bytes=1024,
                compute_cost_by_device={"cpu0": 10.0, "gpu0": 1.0},
            ),
            Region(
                "r1",
                allowed_devices=("gpu0",),
                memory_bytes=512,
                compute_cost_by_device={"gpu0": 1.0},
            ),
        ),
        devices=(
            Device("cpu0", memory_capacity=1024 * 1024, target_class="host_cpu"),
            Device("gpu0", memory_capacity=1024 * 1024, target_class="cuda_sm75"),
        ),
    )


def test_two_cpu_regions_place_on_cpu():
    response, plan = plan_placement(_two_cpu_regions())
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    assert all(a.device_id == "cpu0" for a in plan.assignments)
    assert response.selected_backend is SolverBackendName.ORTOOLS_CP_SAT


def test_gpu_only_region_placed_on_gpu():
    response, plan = plan_placement(_cpu_plus_gpu())
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    by_id = {a.region_id: a.device_id for a in plan.assignments}
    assert by_id["r0"] == "gpu0"  # GPU is cheaper
    assert by_id["r1"] == "gpu0"  # only allowed on GPU


def test_unsupported_device_combination_returns_infeasible():
    plan_input = PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("gpu0",), memory_bytes=1024, compute_cost_by_device={"gpu0": 1.0}),
        ),
        devices=(Device("cpu0", memory_capacity=1024 * 1024),),
    )
    # gpu0 not in declared devices → validation rejects.
    response, plan = plan_placement(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_transfer_heavy_edge_keeps_producer_consumer_colocated():
    plan_input = PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("cpu0", "gpu0"), memory_bytes=128, compute_cost_by_device={"cpu0": 1.0, "gpu0": 1.0}),
            Region("r1", allowed_devices=("cpu0", "gpu0"), memory_bytes=128, compute_cost_by_device={"cpu0": 1.0, "gpu0": 1.0}),
        ),
        devices=(
            Device("cpu0", memory_capacity=1024 * 1024),
            Device("gpu0", memory_capacity=1024 * 1024),
        ),
        edges=(
            Edge(
                "r0",
                "r1",
                bytes_=10_000_000,
                transfer_cost_by_device_pair={("cpu0", "gpu0"): 1e-3, ("gpu0", "cpu0"): 1e-3},
            ),
        ),
    )
    response, plan = plan_placement(plan_input)
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    devs = {a.region_id: a.device_id for a in plan.assignments}
    assert devs["r0"] == devs["r1"], (
        f"colocation expected to avoid transfer cost; got {devs}"
    )


def test_memory_capacity_forces_split_or_infeasible():
    plan_input = PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("gpu0",), memory_bytes=2048, compute_cost_by_device={"gpu0": 1.0}),
            Region("r1", allowed_devices=("gpu0",), memory_bytes=2048, compute_cost_by_device={"gpu0": 1.0}),
        ),
        devices=(Device("gpu0", memory_capacity=1024),),  # insufficient
    )
    response, plan = plan_placement(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


def test_warm_start_hint_accepted():
    plan_input = _cpu_plus_gpu()
    # Hint r0 to cpu0 (suboptimal); solver should still pick globally
    # optimal (gpu0 for cheaper compute).
    plan_input = PlacementPlanInput(
        regions=plan_input.regions,
        devices=plan_input.devices,
        warm_start={"r0": "cpu0"},
    )
    response, plan = plan_placement(plan_input)
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    # Optimal is still gpu0 for r0; hint was a suggestion, not a constraint.
    assert {a.region_id: a.device_id for a in plan.assignments}["r0"] == "gpu0"


def test_timeout_returns_typed_status():
    # Aggressive 1ms budget on a tiny problem still solves; but the
    # envelope contract is what matters — status must be one of
    # the typed values, never raise.
    plan_input = PlacementPlanInput(
        regions=_two_cpu_regions().regions,
        devices=_two_cpu_regions().devices,
        time_budget_ms=1,
    )
    response, plan = plan_placement(plan_input)
    assert response.status in (
        SolverStatus.OPTIMAL,
        SolverStatus.FEASIBLE,
        SolverStatus.TIMEOUT,
    )


def test_solved_plan_round_trips_through_dict():
    response, plan = plan_placement(_two_cpu_regions())
    assert plan is not None
    body = plan.to_dict()
    assert body["schema_version"] == "placement_plan_solver_v1"
    assert body["solver_backend"] == "ortools_cp_sat"
    assert body["formulation_hash"] == response.formulation_hash
    assert {a["region_id"] for a in body["assignments"]} == {"r0", "r1"}

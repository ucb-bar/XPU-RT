"""Metamorphic invariants on the solver layer (spec §13).

These tests don't pin specific outputs; they pin INVARIANTS that
must hold across small perturbations to the input. Catch subtle
bugs (off-by-one constraints, wrong sign in the objective, etc.)
that point-tests miss.
"""

from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from compgen.solve.bandwidth_planner import (
    BandwidthPlanInput,
    LinkCapacity,
    TransferDemand,
    plan_bandwidth,
)
from compgen.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    plan_memory,
)
from compgen.solve.overlap_planner import (
    Dependency,
    Operation,
    OverlapPlanInput,
    Resource,
    plan_overlap,
)
from compgen.solve.placement_planner import (
    Device,
    Edge,
    PlacementPlanInput,
    Region,
    plan_placement,
)
from compgen.solve.solver_types import SolverStatus
from compgen.solve.z3_obligations import (
    OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
    prove_shape_predicate_implication,
)


# ===========================================================================
# Placement metamorphic invariants
# ===========================================================================


def _placement_base() -> PlacementPlanInput:
    return PlacementPlanInput(
        regions=(
            Region(
                "r0", allowed_devices=("cpu", "gpu"), memory_bytes=1024,
                compute_cost_by_device={"cpu": 10.0, "gpu": 1.0},
            ),
            Region(
                "r1", allowed_devices=("cpu", "gpu"), memory_bytes=2048,
                compute_cost_by_device={"cpu": 2.0, "gpu": 1.0},
            ),
        ),
        devices=(
            Device("cpu", memory_capacity=16 * 1024),
            Device("gpu", memory_capacity=16 * 1024),
        ),
    )


def test_placement_adding_unused_device_does_not_worsen_objective():
    base = _placement_base()
    augmented = PlacementPlanInput(
        regions=tuple(
            Region(
                region_id=r.region_id,
                allowed_devices=r.allowed_devices,  # NOT exposing the new device
                memory_bytes=r.memory_bytes,
                compute_cost_by_device=r.compute_cost_by_device,
            )
            for r in base.regions
        ),
        devices=base.devices + (Device("npu", memory_capacity=4096),),
    )
    r_base, _ = plan_placement(base)
    r_aug, _ = plan_placement(augmented)
    assert r_aug.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    # The objective is a minimization; adding an unreachable device
    # cannot give the solver a worse (higher) optimum.
    assert r_aug.objective_value <= r_base.objective_value + 1e-6


def test_placement_increasing_capacity_keeps_feasible():
    base = _placement_base()
    bigger = PlacementPlanInput(
        regions=base.regions,
        devices=tuple(
            Device(
                device_id=d.device_id,
                memory_capacity=d.memory_capacity * 2,
                target_class=d.target_class,
            )
            for d in base.devices
        ),
        edges=base.edges,
    )
    r_base, _ = plan_placement(base)
    r_big, _ = plan_placement(bigger)
    # If the original was feasible the augmented one MUST be feasible.
    if r_base.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
        assert r_big.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


def test_placement_removing_allowed_device_cannot_improve_objective():
    base = _placement_base()
    restricted = PlacementPlanInput(
        regions=tuple(
            Region(
                region_id=r.region_id,
                allowed_devices=("cpu",) if r.region_id == "r0" else r.allowed_devices,
                memory_bytes=r.memory_bytes,
                compute_cost_by_device=r.compute_cost_by_device,
            )
            for r in base.regions
        ),
        devices=base.devices,
    )
    r_base, _ = plan_placement(base)
    r_restricted, _ = plan_placement(restricted)
    if r_base.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE) and r_restricted.status in (
        SolverStatus.OPTIMAL, SolverStatus.FEASIBLE
    ):
        # Removing options cannot strictly improve a minimization optimum.
        assert r_restricted.objective_value >= r_base.objective_value - 1e-6


# ===========================================================================
# Scheduling metamorphic invariants
# ===========================================================================


def _overlap_base() -> OverlapPlanInput:
    return OverlapPlanInput(
        operations=(
            Operation("A", duration=3, resource_id="q0"),
            Operation("B", duration=5, resource_id="q1"),
            Operation("C", duration=2, resource_id="q0"),
        ),
        dependencies=(Dependency("A", "C"),),
        resources=(Resource("q0"), Resource("q1")),
    )


def test_overlap_increasing_op_duration_cannot_reduce_makespan():
    base = _overlap_base()
    longer = OverlapPlanInput(
        operations=tuple(
            Operation(o.op_id, duration=o.duration + 2 if o.op_id == "A" else o.duration,
                      resource_id=o.resource_id, kind=o.kind)
            for o in base.operations
        ),
        dependencies=base.dependencies,
        resources=base.resources,
    )
    r_base, s_base = plan_overlap(base)
    r_long, s_long = plan_overlap(longer)
    assert s_base is not None and s_long is not None
    assert s_long.makespan >= s_base.makespan


def test_overlap_adding_dependency_cannot_reduce_makespan():
    base = _overlap_base()
    extra_dep = OverlapPlanInput(
        operations=base.operations,
        # B now must wait for A — adds serialization.
        dependencies=base.dependencies + (Dependency("A", "B"),),
        resources=base.resources,
    )
    r_base, s_base = plan_overlap(base)
    r_extra, s_extra = plan_overlap(extra_dep)
    assert s_base is not None and s_extra is not None
    assert s_extra.makespan >= s_base.makespan


def test_overlap_more_resources_cannot_increase_makespan_when_independent():
    """Two independent ops on the same resource get serialized; if we
    split them onto two resources, makespan strictly drops (or stays
    equal for already-balanced cases)."""

    same_resource = OverlapPlanInput(
        operations=(
            Operation("A", duration=4, resource_id="q0"),
            Operation("B", duration=4, resource_id="q0"),
        ),
        resources=(Resource("q0"),),
    )
    split_resource = OverlapPlanInput(
        operations=(
            Operation("A", duration=4, resource_id="q0"),
            Operation("B", duration=4, resource_id="q1"),
        ),
        resources=(Resource("q0"), Resource("q1")),
    )
    r_same, s_same = plan_overlap(same_resource)
    r_split, s_split = plan_overlap(split_resource)
    assert s_same is not None and s_split is not None
    assert s_split.makespan <= s_same.makespan


# ===========================================================================
# Memory metamorphic invariants
# ===========================================================================


def _memory_base() -> MemoryPlanInput:
    return MemoryPlanInput(
        buffers=(
            BufferSpec("a", 1024, 0, 5, ("scratchpad",)),
            BufferSpec("b", 1024, 6, 10, ("scratchpad",)),
            BufferSpec("c", 2048, 0, 10, ("scratchpad", "host")),
        ),
        tier_capacities=(
            TierCapacity("scratchpad", 8 * 1024),
            TierCapacity("host", 1024 * 1024),
        ),
    )


def test_memory_increasing_capacity_keeps_feasible():
    base = _memory_base()
    r_base, p_base = plan_memory(base)
    assert r_base.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)

    bigger = MemoryPlanInput(
        buffers=base.buffers,
        tier_capacities=tuple(
            TierCapacity(t.tier_id, t.capacity_bytes * 2, weight=t.weight)
            for t in base.tier_capacities
        ),
    )
    r_big, _ = plan_memory(bigger)
    assert r_big.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


def test_memory_adding_alias_cannot_make_objective_worse():
    """Alias candidates can only help (or be ignored). With non-zero
    alias incentive, adding alias candidates over disjoint lifetimes
    cannot increase the objective."""

    base = _memory_base()
    with_alias = MemoryPlanInput(
        buffers=base.buffers,
        tier_capacities=base.tier_capacities,
        alias_candidates=(AliasCandidate("a", "b"),),  # disjoint lifetimes
    )
    r_base, _ = plan_memory(base)
    r_alias, _ = plan_memory(with_alias)
    if r_base.objective_value is not None and r_alias.objective_value is not None:
        # With negative alias incentive in the objective, aliasing
        # cannot make the result worse.
        assert r_alias.objective_value <= r_base.objective_value + 1e-3


def test_memory_shortening_lifetime_cannot_increase_peak():
    base = _memory_base()
    r_base, p_base = plan_memory(base)
    assert p_base is not None
    base_peak = p_base.tier_peak_usage.get("scratchpad", 0)

    shorter = MemoryPlanInput(
        buffers=tuple(
            BufferSpec(
                buffer_id=b.buffer_id,
                size_bytes=b.size_bytes,
                lifetime_start=b.lifetime_start,
                lifetime_end=b.lifetime_end if b.buffer_id != "c" else b.lifetime_start + 1,
                allowed_tiers=b.allowed_tiers,
                alignment=b.alignment,
                spill_cost=b.spill_cost,
            )
            for b in base.buffers
        ),
        tier_capacities=base.tier_capacities,
        alias_candidates=base.alias_candidates,
    )
    r_short, p_short = plan_memory(shorter)
    assert p_short is not None
    short_peak = p_short.tier_peak_usage.get("scratchpad", 0)
    assert short_peak <= base_peak


# ===========================================================================
# Bandwidth metamorphic invariants
# ===========================================================================


def test_bandwidth_increasing_link_capacity_cannot_reduce_objective():
    base = BandwidthPlanInput(
        transfers=(
            TransferDemand("t0", bytes_=1, weight=2.0, max_bandwidth=200.0, link_id="L"),
            TransferDemand("t1", bytes_=1, weight=1.0, max_bandwidth=200.0, link_id="L"),
        ),
        links=(LinkCapacity("L", capacity=50.0),),
    )
    bigger = BandwidthPlanInput(
        transfers=base.transfers,
        links=(LinkCapacity("L", capacity=200.0),),
    )
    r_base, _ = plan_bandwidth(base)
    r_big, _ = plan_bandwidth(bigger)
    assert r_base.objective_value <= r_big.objective_value + 1e-6


def test_bandwidth_adding_transfer_with_zero_weight_cannot_reduce_objective():
    base = BandwidthPlanInput(
        transfers=(
            TransferDemand("t0", bytes_=1, weight=1.0, max_bandwidth=100.0, link_id="L"),
        ),
        links=(LinkCapacity("L", capacity=50.0),),
    )
    extended = BandwidthPlanInput(
        transfers=base.transfers + (
            TransferDemand("t_unused", bytes_=1, weight=0.0, max_bandwidth=100.0, link_id="L"),
        ),
        links=base.links,
    )
    r_base, _ = plan_bandwidth(base)
    r_ext, _ = plan_bandwidth(extended)
    assert r_ext.objective_value >= r_base.objective_value - 1e-6


# ===========================================================================
# Z3 metamorphic invariants
# ===========================================================================


def test_z3_strengthening_premise_preserves_proved_status():
    """If ``applies_when=A`` proves the precondition, then a stronger
    ``A ∧ B`` premise must still prove it (logical monotonicity)."""

    weak = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 16}],
        precondition={"op": "divisible_by", "var": "K", "k": 8},
    )
    strong = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[
            {"op": "divisible_by", "var": "K", "k": 16},
            {"op": "ge", "a": "K", "b": 16},
        ],
        precondition={"op": "divisible_by", "var": "K", "k": 8},
    )
    assert weak[0] is SolverStatus.PROVED
    assert strong[0] is SolverStatus.PROVED


def test_z3_weakening_precondition_preserves_proved_status():
    """If a stronger precondition proves, the weaker one must too."""

    strong = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 32}],
        precondition={"op": "divisible_by", "var": "K", "k": 16},
    )
    weak = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 32}],
        precondition={"op": "divisible_by", "var": "K", "k": 8},
    )
    assert strong[0] is SolverStatus.PROVED
    assert weak[0] is SolverStatus.PROVED


def test_z3_weakening_premise_can_invalidate_proof():
    """Removing constraints from ``applies_when`` may turn PROVED
    into SAT_COUNTEREXAMPLE."""

    proved = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[{"op": "divisible_by", "var": "K", "k": 16}],
        precondition={"op": "divisible_by", "var": "K", "k": 16},
    )
    no_premise = prove_shape_predicate_implication(
        variables={"K": {"min": 1, "max": 4096}},
        applies_when=[],
        precondition={"op": "divisible_by", "var": "K", "k": 16},
    )
    assert proved[0] is SolverStatus.PROVED
    assert no_premise[0] is SolverStatus.SAT_COUNTEREXAMPLE

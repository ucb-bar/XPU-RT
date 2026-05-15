"""Consolidated solver negative-control tests (spec §12).

Every cell of the negative-control table from the solver
validation spec lives here. The rule is: each failure mode must
produce a typed status — never raise, never silently succeed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

z3 = pytest.importorskip("z3")

from compgen.solve.backend_registry import SolverBackendRegistry, default_registry
from compgen.solve.backends.highs_backend import HighsBackend
from compgen.solve.backends.mosek_backend import MosekBackend
from compgen.solve.backends.ortools_cp_sat_backend import OrToolsCpSatBackend
from compgen.solve.backends.z3_backend import Z3Backend
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
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)
from compgen.solve.z3_obligations import (
    OBLIGATION_KIND_COPY_IDENTITY,
    OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
    OBLIGATION_KIND_TILE_INDEX_BOUNDS,
)


# ---------------------------------------------------------------------------
# Z3 — invalid rewrite / timeout / unsupported obligation
# ---------------------------------------------------------------------------


def test_z3_invalid_tile_len_returns_counterexample():
    request = SolverRequest(
        problem_id="z3_invalid_tile",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_TILE_INDEX_BOUNDS,
            "params": {"tile": 16, "dim_max": 1024, "use_safe_len": False},
        },
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.SAT_COUNTEREXAMPLE
    assert response.counterexample is not None
    # The counterexample identifies a (dim, iter) pair that breaks
    # the (unsafe) tile-len rule.
    assert "dim" in response.counterexample
    assert "iter" in response.counterexample


def test_z3_invalid_copy_returns_counterexample():
    request = SolverRequest(
        problem_id="z3_invalid_copy",
        problem_kind=SolverProblemKind.PLAN_INVARIANT_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_COPY_IDENTITY,
            "params": {"lo": 0, "hi": 32, "perturb": 1},
        },
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.SAT_COUNTEREXAMPLE


def test_z3_invalid_implication_returns_counterexample():
    request = SolverRequest(
        problem_id="z3_impl_neg",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
            "params": {
                "variables": {"K": {"min": 1, "max": 1024}},
                "applies_when": [{"op": "divisible_by", "var": "K", "k": 8}],
                "precondition": {"op": "divisible_by", "var": "K", "k": 16},
            },
        },
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.SAT_COUNTEREXAMPLE
    # Witness must satisfy K%8==0 but NOT K%16==0; e.g. K=8.
    cex = response.counterexample
    assert cex is not None and cex["K"] % 8 == 0 and cex["K"] % 16 != 0


def test_z3_unsupported_obligation_kind_returns_unsupported():
    request = SolverRequest(
        problem_id="z3_unknown_oblig",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={"obligation_kind": "magic", "params": {}},
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED


def test_z3_timeout_never_returns_proved():
    """1ms budget on a problem that wouldn't normally trip Z3 — we
    can't reliably force a timeout, but if Z3 returns ``unknown``
    the wrapper MUST map to TIMEOUT, never to PROVED.
    """

    request = SolverRequest(
        problem_id="z3_short_budget",
        problem_kind=SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        formulation={
            "obligation_kind": OBLIGATION_KIND_SHAPE_PREDICATE_IMPLICATION,
            "params": {
                "variables": {"K": {"min": 1, "max": 1024}},
                "applies_when": [{"op": "divisible_by", "var": "K", "k": 16}],
                "precondition": {"op": "divisible_by", "var": "K", "k": 8},
                "timeout_ms": 1,
            },
        },
        time_budget_ms=1,
    )
    response = Z3Backend().solve(request)
    assert response.status in {SolverStatus.PROVED, SolverStatus.TIMEOUT}
    if response.status is SolverStatus.TIMEOUT:
        assert response.infeasibility_reason


# ---------------------------------------------------------------------------
# Placement — infeasibility / no-allowed-device / over-capacity
# ---------------------------------------------------------------------------


def test_placement_no_allowed_device_returns_infeasible():
    plan_input = PlacementPlanInput(
        regions=(Region("r0", allowed_devices=(), memory_bytes=0,
                        compute_cost_by_device={}),),
        devices=(Device("cpu0", memory_capacity=1024),),
    )
    response, plan = plan_placement(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_placement_exceeds_memory_returns_infeasible():
    plan_input = PlacementPlanInput(
        regions=(
            Region("r0", allowed_devices=("cpu0",), memory_bytes=2048,
                   compute_cost_by_device={"cpu0": 1.0}),
            Region("r1", allowed_devices=("cpu0",), memory_bytes=2048,
                   compute_cost_by_device={"cpu0": 1.0}),
        ),
        devices=(Device("cpu0", memory_capacity=1024),),
    )
    response, plan = plan_placement(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


def test_placement_disallowed_device_in_input_returns_infeasible():
    """An allowed_devices entry not in the declared devices list must
    cause the input validator to reject upfront."""

    plan_input = PlacementPlanInput(
        regions=(Region("r0", allowed_devices=("gpu0",), memory_bytes=1024,
                        compute_cost_by_device={"gpu0": 1.0}),),
        devices=(Device("cpu0", memory_capacity=1024),),
    )
    response, plan = plan_placement(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# Schedule / overlap — impossible deadline / cycle / unknown op
# ---------------------------------------------------------------------------


def test_overlap_impossible_deadline_returns_infeasible():
    plan_input = OverlapPlanInput(
        operations=(
            Operation("A", duration=10, resource_id="q0"),
            Operation("B", duration=10, resource_id="q0"),
        ),
        resources=(Resource("q0"),),
        deadline=15,
    )
    response, sched = plan_overlap(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert sched is None


def test_overlap_dependency_to_unknown_op_returns_infeasible():
    plan_input = OverlapPlanInput(
        operations=(Operation("A", duration=2, resource_id="q0"),),
        dependencies=(Dependency("A", "ghost"),),
        resources=(Resource("q0"),),
    )
    response, sched = plan_overlap(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


def test_overlap_negative_duration_returns_infeasible():
    plan_input = OverlapPlanInput(
        operations=(Operation("A", duration=-1, resource_id="q0"),),
        resources=(Resource("q0"),),
    )
    response, sched = plan_overlap(plan_input)
    assert response.status is SolverStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# Memory — capacity / alias overlap / no LP backend
# ---------------------------------------------------------------------------


def test_memory_impossible_capacity_returns_infeasible():
    plan_input = MemoryPlanInput(
        buffers=(
            BufferSpec("a", 3000, 0, 5, ("scratchpad",)),
            BufferSpec("b", 3000, 0, 5, ("scratchpad",)),
        ),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
    )
    response, plan = plan_memory(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_memory_alias_with_overlapping_lifetimes_not_collapsed():
    """An alias_candidate over overlapping lifetimes must NOT be
    collapsed — the planner allocates them at distinct offsets."""

    plan_input = MemoryPlanInput(
        buffers=(
            BufferSpec("a", 1024, 0, 10, ("scratchpad",)),
            BufferSpec("b", 1024, 5, 15, ("scratchpad",)),  # overlapping with a
        ),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
        alias_candidates=(AliasCandidate("a", "b"),),
    )
    response, plan = plan_memory(plan_input)
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    by_id = {b.buffer_id: b for b in plan.buffers}
    # Both allocated, NOT collapsed (offsets differ).
    assert by_id["a"].offset_bytes != by_id["b"].offset_bytes
    assert by_id["a"].aliases_with is None
    assert by_id["b"].aliases_with is None


def test_memory_no_lp_milp_backend_returns_blocked():
    """When MOSEK and HiGHS are both unavailable, return BLOCKED
    with infeasibility_reason='no_lp_milp_backend'. Never silently
    fall back to greedy."""

    reg = SolverBackendRegistry()  # empty: no MOSEK, no HiGHS
    plan_input = MemoryPlanInput(
        buffers=(BufferSpec("a", 1024, 0, 5, ("scratchpad",)),),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
    )
    response, plan = plan_memory(plan_input, registry=reg)
    assert response.status is SolverStatus.BLOCKED
    assert plan is None
    assert response.infeasibility_reason == "no_lp_milp_backend"


def test_memory_mosek_forced_unavailable_falls_back_to_highs():
    """Patching MOSEK to ``LICENSE_MISSING`` must route to HiGHS.
    The response.selected_backend must honestly be HIGHS, not
    MOSEK."""

    reg = SolverBackendRegistry()

    class _UnavailableMosek(MosekBackend):
        def probe(self):  # type: ignore[override]
            return BackendProbeResult(
                backend=SolverBackendName.MOSEK,
                availability=BackendAvailabilityStatus.LICENSE_MISSING,
            )

    reg.register(_UnavailableMosek())
    reg.register(HighsBackend())
    plan_input = MemoryPlanInput(
        buffers=(BufferSpec("a", 1024, 0, 5, ("scratchpad",)),),
        tier_capacities=(TierCapacity("scratchpad", 4096),),
    )
    response, plan = plan_memory(plan_input, registry=reg)
    assert response.selected_backend is SolverBackendName.HIGHS
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


# ---------------------------------------------------------------------------
# Routing — preference cannot violate solver-purpose rules
# ---------------------------------------------------------------------------


def test_preference_mosek_on_verification_kind_is_not_honoured():
    """Constructing a request with backend_preference=MOSEK on a
    verification kind must NOT route to MOSEK. Routing falls back
    to the canonical backend (Z3) or returns ``blocked``."""

    from compgen.solve.routing import choose_backend

    reg = default_registry()
    chosen = choose_backend(
        SolverProblemKind.PEEPHOLE_VERIFY,
        reg,
        preference=SolverBackendName.MOSEK,
    )
    assert chosen in (SolverBackendName.Z3, None)


def test_direct_mosek_call_on_verification_kind_returns_unsupported():
    """Even bypassing routing (calling MosekBackend.solve directly)
    a verification kind is hard-rejected. Architecture guard."""

    request = SolverRequest(
        problem_id="bypass_routing_mosek",
        problem_kind=SolverProblemKind.PEEPHOLE_VERIFY,
        formulation={},
    )
    response = MosekBackend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED
    assert response.selected_backend is SolverBackendName.MOSEK


def test_direct_z3_call_on_placement_returns_unsupported():
    request = SolverRequest(
        problem_id="bypass_routing_z3",
        problem_kind=SolverProblemKind.PLACEMENT,
        formulation={},
    )
    response = Z3Backend().solve(request)
    assert response.status is SolverStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# Corrupt artifacts
# ---------------------------------------------------------------------------


def test_corrupt_response_missing_formulation_hash_is_rejected():
    body = {
        "schema_version": "solver_response_v1",
        "problem_id": "x",
        "problem_kind": "placement",
        "selected_backend": "ortools_cp_sat",
        "backend_availability": "available",
        "status": "optimal",
        # missing formulation_hash
        "time_ms": 1.0,
    }
    with pytest.raises(KeyError):
        SolverResponse.from_dict(body)


def test_execution_plan_rejects_overlapping_offsets():
    """ExecutionPlan.validate must reject a memory_plan.solved.json
    that would put two buffers in overlapping byte ranges."""

    from compgen.runtime.execution_plan import (
        BufferDescriptor,
        ExecutionPlan,
        Lifetime,
    )

    plan = ExecutionPlan(workload="x", target="host_cpu")
    plan.buffers.extend(
        [
            BufferDescriptor(
                buffer_id="b0",
                size_bytes=1024,
                memory_space="scratchpad",
                lifetime=Lifetime(0, 10),
                ownership="exclusive",
                offset_bytes=0,
            ),
            BufferDescriptor(
                buffer_id="b1",
                size_bytes=1024,
                memory_space="scratchpad",
                lifetime=Lifetime(0, 10),
                ownership="exclusive",
                offset_bytes=512,  # overlapping
            ),
        ]
    )
    with pytest.raises(ValueError, match="overlapping byte ranges"):
        plan.validate()


# ---------------------------------------------------------------------------
# Unknown / unsupported problem_kind
# ---------------------------------------------------------------------------


def test_unsupported_kind_via_routing_returns_none():
    from compgen.solve.routing import choose_backend

    reg = default_registry()
    # No problem kind is permanently unsupported in the default
    # ROUTING_TABLE, but we can force the no-backend case by passing
    # an empty registry.
    empty = SolverBackendRegistry()
    chosen = choose_backend(SolverProblemKind.PLACEMENT, empty)
    assert chosen is None

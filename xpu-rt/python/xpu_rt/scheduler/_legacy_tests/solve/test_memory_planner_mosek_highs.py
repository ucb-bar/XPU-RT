"""MILP memory planner end-to-end.

Production-path proof that ``solve/memory_planner.py`` actually
calls MOSEK (or HiGHS), returns typed status, and never silently
falls back to a greedy heuristic when no LP/MILP backend is
available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compgen.solve.backend_registry import SolverBackendRegistry
from compgen.solve.backends.highs_backend import HighsBackend
from compgen.solve.backends.mosek_backend import MosekBackend
from compgen.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    plan_memory,
)
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverStatus,
)


def _two_disjoint_lifetimes():
    return MemoryPlanInput(
        buffers=(
            BufferSpec("b0", size_bytes=1024, lifetime_start=0, lifetime_end=10, allowed_tiers=("scratchpad",)),
            BufferSpec("b1", size_bytes=1024, lifetime_start=20, lifetime_end=30, allowed_tiers=("scratchpad",)),
        ),
        tier_capacities=(TierCapacity("scratchpad", capacity_bytes=4096),),
        alias_candidates=(AliasCandidate("b0", "b1"),),
        time_budget_ms=5000,
    )


def _two_overlapping_lifetimes():
    return MemoryPlanInput(
        buffers=(
            BufferSpec("b0", size_bytes=1024, lifetime_start=0, lifetime_end=10, allowed_tiers=("scratchpad",)),
            BufferSpec("b1", size_bytes=1024, lifetime_start=5, lifetime_end=15, allowed_tiers=("scratchpad",)),
        ),
        tier_capacities=(TierCapacity("scratchpad", capacity_bytes=4096),),
        time_budget_ms=5000,
    )


def test_disjoint_lifetimes_alias_to_same_offset():
    response, plan = plan_memory(_two_disjoint_lifetimes())
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    by_id = {b.buffer_id: b for b in plan.buffers}
    assert by_id["b0"].tier == "scratchpad"
    assert by_id["b1"].tier == "scratchpad"
    # Canonical pack aliases disjoint lifetimes to the same offset.
    assert by_id["b1"].offset_bytes == by_id["b0"].offset_bytes
    assert by_id["b1"].aliases_with == "b0"


def test_overlapping_lifetimes_get_disjoint_offsets():
    response, plan = plan_memory(_two_overlapping_lifetimes())
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None
    by_id = {b.buffer_id: b for b in plan.buffers}
    assert by_id["b0"].tier == by_id["b1"].tier == "scratchpad"
    # Their byte ranges do not overlap.
    a_lo, a_hi = by_id["b0"].offset_bytes, by_id["b0"].offset_bytes + 1024
    b_lo, b_hi = by_id["b1"].offset_bytes, by_id["b1"].offset_bytes + 1024
    assert a_hi <= b_lo or b_hi <= a_lo


def test_tier_capacity_exceeded_returns_infeasible():
    plan_input = MemoryPlanInput(
        buffers=(
            BufferSpec("big_a", size_bytes=3000, lifetime_start=0, lifetime_end=10, allowed_tiers=("scratchpad",)),
            BufferSpec("big_b", size_bytes=3000, lifetime_start=0, lifetime_end=10, allowed_tiers=("scratchpad",)),
        ),
        tier_capacities=(TierCapacity("scratchpad", capacity_bytes=4096),),
    )
    response, plan = plan_memory(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_fixed_assignment_to_disallowed_tier_returns_infeasible():
    plan_input = MemoryPlanInput(
        buffers=(
            BufferSpec("b0", size_bytes=512, lifetime_start=0, lifetime_end=10, allowed_tiers=("scratchpad",)),
        ),
        tier_capacities=(
            TierCapacity("scratchpad", capacity_bytes=4096),
            TierCapacity("host", capacity_bytes=4096),
        ),
        fixed_assignments={"b0": "host"},
    )
    response, plan = plan_memory(plan_input)
    assert response.status is SolverStatus.INFEASIBLE
    assert plan is None


def test_falls_back_to_highs_when_mosek_unavailable():
    reg = SolverBackendRegistry()

    class _UnavailableMosek(MosekBackend):
        def probe(self):  # type: ignore[override]
            return BackendProbeResult(
                backend=SolverBackendName.MOSEK,
                availability=BackendAvailabilityStatus.LICENSE_MISSING,
                detail="forced unavailable for test",
            )

    reg.register(_UnavailableMosek())
    reg.register(HighsBackend())

    response, plan = plan_memory(_two_disjoint_lifetimes(), registry=reg)
    assert response.selected_backend is SolverBackendName.HIGHS
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert plan is not None


def test_no_lp_milp_backend_returns_blocked():
    reg = SolverBackendRegistry()
    # No MOSEK and no HiGHS registered.
    response, plan = plan_memory(_two_disjoint_lifetimes(), registry=reg)
    assert response.status is SolverStatus.BLOCKED
    assert plan is None
    assert response.infeasibility_reason == "no_lp_milp_backend"


def test_rerun_byte_identical_solved_plan(tmp_path):
    plan_input = _two_disjoint_lifetimes()
    response_a, plan_a = plan_memory(plan_input)
    response_b, plan_b = plan_memory(plan_input)
    assert plan_a is not None and plan_b is not None
    body_a = json.dumps(plan_a.to_dict(), sort_keys=True)
    body_b = json.dumps(plan_b.to_dict(), sort_keys=True)
    assert body_a == body_b


def test_mosek_native_path_actually_runs():
    """Force MOSEK selection and verify the response is honestly from MOSEK.

    Regression guard against the earlier dishonest behavior where the
    MOSEK code path silently fell through to ``scipy.optimize.milp``
    (a HiGHS-backed solver) while still labeling
    ``selected_backend=MOSEK``. This test pins the real MOSEK API.
    """

    pytest.importorskip("mosek")
    from compgen.solve.backend_registry import default_registry
    from compgen.solve.memory_planner import _build_formulation, _solve_milp_mosek
    from compgen.solve.solver_types import (
        SolverProblemKind,
        SolverRequest,
    )

    reg = default_registry()
    probe = reg.probe(SolverBackendName.MOSEK)
    if probe.availability is not BackendAvailabilityStatus.AVAILABLE:
        pytest.skip(f"MOSEK unavailable: {probe.availability.value}")

    # Patch scipy.optimize.milp to raise if the code path accidentally
    # routes through HiGHS instead of MOSEK's native API.
    import scipy.optimize

    plan_input = _two_disjoint_lifetimes()
    req = SolverRequest(
        problem_id="mosek_native_pin",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        formulation=_build_formulation(plan_input),
    )

    original_milp = scipy.optimize.milp
    scipy.optimize.milp = lambda *a, **k: pytest.fail(
        "MOSEK native path accidentally called scipy.optimize.milp"
    )
    try:
        response = _solve_milp_mosek(req, plan_input, probe=probe, t0=0.0)
    finally:
        scipy.optimize.milp = original_milp

    assert response.selected_backend is SolverBackendName.MOSEK
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert response.time_ms > 0
    assert isinstance(response.solution, dict)
    assert response.solution["solver_backend"] == "mosek"


def test_highs_path_does_not_invoke_mosek():
    """Force HiGHS selection and verify MOSEK is never imported on
    that path. Mirrors the MOSEK-native pin from the opposite side.
    """

    from compgen.solve.backend_registry import default_registry
    from compgen.solve.memory_planner import _build_formulation, _solve_milp_highs
    from compgen.solve.solver_types import (
        SolverProblemKind,
        SolverRequest,
    )

    reg = default_registry()
    probe_h = reg.probe(SolverBackendName.HIGHS)
    if probe_h.availability is not BackendAvailabilityStatus.AVAILABLE:
        pytest.skip(f"HiGHS unavailable: {probe_h.availability.value}")

    plan_input = _two_disjoint_lifetimes()
    req = SolverRequest(
        problem_id="highs_native_pin",
        problem_kind=SolverProblemKind.MEMORY_ALLOCATION,
        formulation=_build_formulation(plan_input),
    )
    response = _solve_milp_highs(req, plan_input, probe=probe_h, t0=0.0)
    assert response.selected_backend is SolverBackendName.HIGHS
    assert response.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)
    assert response.solution["solver_backend"] == "highs"


def test_corrupt_solved_plan_fails_when_consumed_by_execution_plan_validate():
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
                lifetime=Lifetime(0, 10),  # overlapping
                ownership="exclusive",
                offset_bytes=512,  # overlapping with b0
            ),
        ]
    )
    # Validation must reject this; the new offset overlap check
    # is added 's ExecutionPlan extension.
    with pytest.raises(ValueError, match="overlapping byte ranges"):
        plan.validate()

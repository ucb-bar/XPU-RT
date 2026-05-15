"""deterministic problem-kind -> backend routing."""

from __future__ import annotations

import pytest

from compgen.solve.backend_registry import SolverBackendRegistry
from compgen.solve.backends.base import SolverBackend
from compgen.solve.routing import ROUTING_TABLE, choose_backend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)


class _Avail(SolverBackend):
    def __init__(self, name: SolverBackendName, supports: frozenset[SolverProblemKind]):
        self._name = name
        self._supports = supports

    @property
    def name(self) -> SolverBackendName:
        return self._name

    def probe(self) -> BackendProbeResult:
        return BackendProbeResult(backend=self._name, availability=BackendAvailabilityStatus.AVAILABLE)

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in self._supports

    def solve(self, request: SolverRequest) -> SolverResponse:
        raise NotImplementedError


def _all_available_registry() -> SolverBackendRegistry:
    reg = SolverBackendRegistry()
    reg.register(_Avail(SolverBackendName.Z3, frozenset(SolverProblemKind)))
    reg.register(_Avail(SolverBackendName.ORTOOLS_CP_SAT, frozenset(SolverProblemKind)))
    reg.register(_Avail(SolverBackendName.MOSEK, frozenset(SolverProblemKind)))
    reg.register(_Avail(SolverBackendName.HIGHS, frozenset(SolverProblemKind)))
    return reg


def test_proof_kinds_always_route_to_z3():
    reg = _all_available_registry()
    for kind in (
        SolverProblemKind.PEEPHOLE_VERIFY,
        SolverProblemKind.RECIPE_REFINEMENT,
        SolverProblemKind.TRANSLATION_VALIDATION,
        SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        SolverProblemKind.PLAN_INVARIANT_VERIFY,
    ):
        assert choose_backend(kind, reg) is SolverBackendName.Z3


def test_discrete_kinds_always_route_to_cp_sat():
    reg = _all_available_registry()
    for kind in (
        SolverProblemKind.PLACEMENT,
        SolverProblemKind.SCHEDULE,
        SolverProblemKind.NO_OVERLAP_SCHEDULE,
        SolverProblemKind.EVENT_ORDERING,
        SolverProblemKind.OVERLAP_PLANNING,
    ):
        assert choose_backend(kind, reg) is SolverBackendName.ORTOOLS_CP_SAT


def test_numeric_kinds_prefer_mosek_when_available():
    reg = _all_available_registry()
    for kind in (
        SolverProblemKind.MEMORY_ALLOCATION,
        SolverProblemKind.BUFFER_ALIASING,
        SolverProblemKind.BANDWIDTH_ALLOCATION,
        SolverProblemKind.COST_MODEL_FIT,
    ):
        assert choose_backend(kind, reg) is SolverBackendName.MOSEK


def test_numeric_kinds_fall_back_to_highs_when_mosek_missing():
    reg = SolverBackendRegistry()
    reg.register(_Avail(SolverBackendName.HIGHS, frozenset(SolverProblemKind)))
    assert choose_backend(SolverProblemKind.MEMORY_ALLOCATION, reg) is SolverBackendName.HIGHS


def test_numeric_kinds_block_when_no_lp_milp_backend():
    reg = SolverBackendRegistry()
    reg.register(_Avail(SolverBackendName.Z3, frozenset(SolverProblemKind)))
    reg.register(_Avail(SolverBackendName.ORTOOLS_CP_SAT, frozenset(SolverProblemKind)))
    assert choose_backend(SolverProblemKind.MEMORY_ALLOCATION, reg) is None


def test_proof_kinds_never_route_to_mosek_or_highs():
    # Even if a maintainer adds MOSEK to ROUTING_TABLE for a proof
    # kind, the _allowed_for_kind hard separation blocks it.
    reg = SolverBackendRegistry()
    reg.register(_Avail(SolverBackendName.MOSEK, frozenset(SolverProblemKind)))
    reg.register(_Avail(SolverBackendName.HIGHS, frozenset(SolverProblemKind)))
    assert choose_backend(SolverProblemKind.PEEPHOLE_VERIFY, reg) is None


def test_placement_never_routes_to_z3():
    reg = SolverBackendRegistry()
    reg.register(_Avail(SolverBackendName.Z3, frozenset(SolverProblemKind)))
    assert choose_backend(SolverProblemKind.PLACEMENT, reg) is None


def test_preference_honored_when_available():
    reg = _all_available_registry()
    out = choose_backend(
        SolverProblemKind.MEMORY_ALLOCATION,
        reg,
        preference=SolverBackendName.HIGHS,
    )
    assert out is SolverBackendName.HIGHS


def test_preference_ignored_when_not_in_routing_table():
    reg = _all_available_registry()
    out = choose_backend(
        SolverProblemKind.PEEPHOLE_VERIFY,
        reg,
        preference=SolverBackendName.MOSEK,
    )
    assert out is SolverBackendName.Z3


def test_every_problem_kind_has_routing_entry():
    for kind in SolverProblemKind:
        assert kind in ROUTING_TABLE, f"missing routing entry for {kind!r}"

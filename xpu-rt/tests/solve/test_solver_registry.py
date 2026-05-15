"""registry registration, probe caching, deterministic listing."""

from __future__ import annotations

import pytest

from compgen.solve.backend_registry import SolverBackendRegistry, default_registry
from compgen.solve.backends.base import SolverBackend
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)


class _StubBackend(SolverBackend):
    def __init__(
        self,
        name: SolverBackendName,
        availability: BackendAvailabilityStatus,
        supports: tuple[SolverProblemKind, ...] = (),
    ):
        self._name = name
        self._availability = availability
        self._supports = supports
        self._probe_count = 0

    @property
    def name(self) -> SolverBackendName:
        return self._name

    def probe(self) -> BackendProbeResult:
        self._probe_count += 1
        return BackendProbeResult(backend=self._name, availability=self._availability, version="stub-1.0")

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in self._supports

    def solve(self, request: SolverRequest) -> SolverResponse:
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=self._name,
            backend_availability=self._availability,
            status=SolverStatus.PROVED,
            formulation_hash=request.formulation_hash,
            time_ms=0.0,
        )


def test_registry_caches_probe():
    reg = SolverBackendRegistry()
    backend = _StubBackend(SolverBackendName.Z3, BackendAvailabilityStatus.AVAILABLE)
    reg.register(backend)
    reg.probe(SolverBackendName.Z3)
    reg.probe(SolverBackendName.Z3)
    assert backend._probe_count == 1


def test_registry_force_reprobes():
    reg = SolverBackendRegistry()
    backend = _StubBackend(SolverBackendName.Z3, BackendAvailabilityStatus.AVAILABLE)
    reg.register(backend)
    reg.probe(SolverBackendName.Z3)
    reg.probe(SolverBackendName.Z3, force=True)
    assert backend._probe_count == 2


def test_unregistered_backend_returns_typed_missing():
    reg = SolverBackendRegistry()
    res = reg.probe(SolverBackendName.MOSEK)
    assert res.availability is BackendAvailabilityStatus.IMPORT_MISSING
    assert "not registered" in res.detail


def test_available_backends_filters_by_probe():
    reg = SolverBackendRegistry()
    reg.register(_StubBackend(SolverBackendName.Z3, BackendAvailabilityStatus.AVAILABLE))
    reg.register(_StubBackend(SolverBackendName.MOSEK, BackendAvailabilityStatus.LICENSE_MISSING))
    reg.register(
        _StubBackend(SolverBackendName.HIGHS, BackendAvailabilityStatus.AVAILABLE)
    )
    assert set(reg.available_backends()) == {SolverBackendName.Z3, SolverBackendName.HIGHS}


def test_names_sorted_deterministically():
    reg = SolverBackendRegistry()
    reg.register(_StubBackend(SolverBackendName.MOSEK, BackendAvailabilityStatus.AVAILABLE))
    reg.register(_StubBackend(SolverBackendName.HIGHS, BackendAvailabilityStatus.AVAILABLE))
    reg.register(_StubBackend(SolverBackendName.Z3, BackendAvailabilityStatus.AVAILABLE))
    assert reg.names() == (
        SolverBackendName.HIGHS,
        SolverBackendName.MOSEK,
        SolverBackendName.Z3,
    )


def test_supports_for_filters_by_problem_kind():
    reg = SolverBackendRegistry()
    reg.register(
        _StubBackend(
            SolverBackendName.ORTOOLS_CP_SAT,
            BackendAvailabilityStatus.AVAILABLE,
            supports=(SolverProblemKind.PLACEMENT,),
        )
    )
    reg.register(
        _StubBackend(
            SolverBackendName.Z3,
            BackendAvailabilityStatus.AVAILABLE,
            supports=(SolverProblemKind.PEEPHOLE_VERIFY,),
        )
    )
    placement_backends = reg.supports_for(SolverProblemKind.PLACEMENT)
    assert placement_backends == (SolverBackendName.ORTOOLS_CP_SAT,)


def test_default_registry_is_idempotent():
    a = default_registry()
    b = default_registry()
    assert a is b

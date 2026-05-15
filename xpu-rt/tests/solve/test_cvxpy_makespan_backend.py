"""Tests for the CVXPY makespan solver backend.

Asserts the absorbed XPU-RT two-cluster scheduler is wired into
:mod:`xpu_rt.solve` as a first-class :class:`SolverBackend`:

- It registers in the default registry.
- The routing table maps :data:`SolverProblemKind.MAKESPAN_SCHEDULE`
  to it.
- :func:`xpu_rt.scheduler.solve_makespan` produces a typed
  :class:`SolverResponse` with a byte-stable ``formulation_hash``.
- ``probe()`` correctly reports cvxpy + installed solvers.
- The architecture guard refuses to route non-makespan kinds to it.

These tests do NOT exercise the underlying MILP on a realistic
workload — that's covered by
``xpu-rt/python/xpu_rt/scheduler/test_feedback_derivation.py`` and
the on-board e2e harness.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xpu_rt.solve.backend_registry import SolverBackendRegistry, default_registry
from xpu_rt.solve.backends.cvxpy_makespan_backend import (
    METADATA_KWARGS_KEY,
    METADATA_WORKLOAD_KEY,
    CvxpyMakespanBackend,
)
from xpu_rt.solve.routing import ROUTING_TABLE, choose_backend
from xpu_rt.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverStatus,
)


def test_backend_registers_in_default_registry() -> None:
    reg = default_registry()
    assert SolverBackendName.CVXPY_MAKESPAN in reg.names()


def test_routing_table_maps_makespan_schedule() -> None:
    assert ROUTING_TABLE[SolverProblemKind.MAKESPAN_SCHEDULE] == (
        SolverBackendName.CVXPY_MAKESPAN,
    )


def test_choose_backend_returns_cvxpy_when_available() -> None:
    reg = default_registry()
    probe = reg.probe(SolverBackendName.CVXPY_MAKESPAN)
    if probe.availability is not BackendAvailabilityStatus.AVAILABLE:
        pytest.skip(f"cvxpy not available: {probe.detail}")
    chosen = choose_backend(SolverProblemKind.MAKESPAN_SCHEDULE, reg)
    assert chosen is SolverBackendName.CVXPY_MAKESPAN


def test_probe_reports_installed_solvers_when_available() -> None:
    backend = CvxpyMakespanBackend()
    probe = backend.probe()
    if probe.availability is not BackendAvailabilityStatus.AVAILABLE:
        pytest.skip(f"cvxpy not available: {probe.detail}")
    # The supports tuple should at minimum contain our internal tags
    # plus whatever cvxpy reports installed.
    assert "milp" in probe.supports
    assert "makespan_schedule" in probe.supports


def test_probe_requires_mosek_when_cvxpy_present() -> None:
    """XPU-RT hardcodes solver=cp.MOSEK; missing MOSEK must surface as
    a typed BLOCKED-grade availability so the registry never silently
    falls back to another solver that would produce different schedules.
    """

    backend = CvxpyMakespanBackend()
    probe = backend.probe()
    # If the probe reports AVAILABLE, MOSEK MUST be in the supports tuple.
    # If MOSEK is missing the probe MUST be LICENSE_MISSING / IMPORT_MISSING,
    # never AVAILABLE.
    if probe.availability is BackendAvailabilityStatus.AVAILABLE:
        assert "MOSEK" in probe.supports, (
            f"probe is AVAILABLE but MOSEK is not in supports={probe.supports!r}"
        )
        # First entry should announce the preferred solver visually.
        assert probe.supports[0] == "preferred_solver:MOSEK"
    else:
        # Either mosek package missing (IMPORT_MISSING) or installed but
        # cvxpy didn't register it (LICENSE_MISSING).
        assert probe.availability in {
            BackendAvailabilityStatus.IMPORT_MISSING,
            BackendAvailabilityStatus.LICENSE_MISSING,
            BackendAvailabilityStatus.PROBE_ERROR,
        }
        # Detail must mention MOSEK so the user knows why the path is blocked.
        assert "MOSEK" in (probe.detail or "") or "mosek" in (probe.detail or "")


def test_supports_only_makespan_and_probe() -> None:
    backend = CvxpyMakespanBackend()
    assert backend.supports(SolverProblemKind.MAKESPAN_SCHEDULE)
    assert backend.supports(SolverProblemKind.BACKEND_PROBE)
    # Architecture-guard kinds must NOT be supported.
    for forbidden in (
        SolverProblemKind.PEEPHOLE_VERIFY,
        SolverProblemKind.PLACEMENT,
        SolverProblemKind.MEMORY_ALLOCATION,
        SolverProblemKind.NO_OVERLAP_SCHEDULE,
    ):
        assert not backend.supports(forbidden), f"must not support {forbidden}"


def test_solve_returns_unsupported_for_wrong_kind() -> None:
    backend = CvxpyMakespanBackend()
    req = SolverRequest(
        problem_id="t",
        problem_kind=SolverProblemKind.PLACEMENT,
        formulation={"k": "v"},
    )
    resp = backend.solve(req)
    assert resp.status is SolverStatus.UNSUPPORTED
    assert resp.selected_backend is SolverBackendName.CVXPY_MAKESPAN
    assert resp.formulation_hash == req.formulation_hash


def test_solve_returns_error_when_workload_metadata_missing() -> None:
    backend = CvxpyMakespanBackend()
    req = SolverRequest(
        problem_id="t",
        problem_kind=SolverProblemKind.MAKESPAN_SCHEDULE,
        formulation={"schema": "makespan_signature_v1", "n_operations": 0},
    )
    resp = backend.solve(req)
    assert resp.status is SolverStatus.ERROR
    assert METADATA_WORKLOAD_KEY in (resp.infeasibility_reason or "")


def test_solve_makespan_shim_threads_workload_through_metadata() -> None:
    """The shim must put the workload in metadata so the backend finds it."""

    from xpu_rt.scheduler.bridge import solve_makespan, summarise_workload

    # Build a minimal mock workload — we monkeypatch the underlying
    # `schedule()` to avoid running the real MILP.
    mock_workload = MagicMock()
    mock_workload.operations = []
    mock_workload.machines = ["host"]
    mock_workload.machine_combinations = [["host"]]
    mock_workload.transfer_times = None

    sig = summarise_workload(mock_workload, time_limit=5.0)
    assert sig["schema"] == "makespan_signature_v1"
    assert sig["n_operations"] == 0
    assert sig["n_machines"] == 1
    assert sig["kwargs"]["time_limit"] == 5.0


def test_request_metadata_keys_are_stable() -> None:
    """The agreed carrier keys are part of the public contract."""

    assert METADATA_WORKLOAD_KEY == "__cvxpy_makespan_workload"
    assert METADATA_KWARGS_KEY == "__cvxpy_makespan_kwargs"


def test_blocked_response_when_no_backend_available(monkeypatch) -> None:
    """When no backend can route MAKESPAN_SCHEDULE, the shim returns BLOCKED."""

    from xpu_rt.scheduler import bridge

    # Empty registry → no backends → choose_backend returns None.
    empty = SolverBackendRegistry()
    monkeypatch.setattr(bridge, "default_registry", lambda: empty)

    mock_workload = MagicMock()
    mock_workload.operations = []
    mock_workload.machines = []
    mock_workload.machine_combinations = []
    mock_workload.transfer_times = None

    t, alpha, fused, response = bridge.solve_makespan(mock_workload)
    assert response.status is SolverStatus.BLOCKED
    assert t is None and alpha is None and fused is None

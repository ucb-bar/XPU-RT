"""OR-Tools CP-SAT solver backend.

. Probes CP-SAT and dispatches discrete-planning requests
(placement, schedule, no-overlap, event-ordering, overlap-planning).
``solve`` only handles ``BACKEND_PROBE``; real planners
(placement, overlap) call into this backend via the
registry but build their formulations themselves.
"""

from __future__ import annotations

import time

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

__all__ = ["OrToolsCpSatBackend"]


_SUPPORTED_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.PLACEMENT,
        SolverProblemKind.SCHEDULE,
        SolverProblemKind.NO_OVERLAP_SCHEDULE,
        SolverProblemKind.EVENT_ORDERING,
        SolverProblemKind.OVERLAP_PLANNING,
        SolverProblemKind.BACKEND_PROBE,
    }
)


class OrToolsCpSatBackend(SolverBackend):
    @property
    def name(self) -> SolverBackendName:
        return SolverBackendName.ORTOOLS_CP_SAT

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in _SUPPORTED_KINDS

    def probe(self) -> BackendProbeResult:
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail=f"ortools not installed: {exc}",
            )
        try:
            model = cp_model.CpModel()
            x = model.NewBoolVar("x")
            y = model.NewBoolVar("y")
            model.Add(x + y == 1)
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 1.0
            status = solver.Solve(model)
            assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        except Exception as exc:  # pragma: no cover - host-specific
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.PROBE_ERROR,
                detail=str(exc),
            )
        version: str | None
        try:
            import ortools
            version = getattr(ortools, "__version__", None)
        except Exception:  # pragma: no cover
            version = None
        return BackendProbeResult(
            backend=self.name,
            availability=BackendAvailabilityStatus.AVAILABLE,
            version=version,
            supports=("cp_sat", "placement", "schedule", "no_overlap", "event_ordering"),
        )

    def solve(self, request: SolverRequest) -> SolverResponse:
        if not self.supports(request.problem_kind):
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.UNSUPPORTED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=(
                    f"ortools_cp_sat does not support problem_kind="
                    f"{request.problem_kind.value!r}; this is a "
                    f"solver-purpose violation"
                ),
            )
        probe = self.probe()
        if probe.availability is not BackendAvailabilityStatus.AVAILABLE:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.BLOCKED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=f"ortools unavailable: {probe.detail}",
            )
        if request.problem_kind is SolverProblemKind.BACKEND_PROBE:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.OPTIMAL,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                solution={"probe_ok": True, "version": probe.version},
            )
        # Placement / overlap / schedule kinds: planners formulate
        # the problem and call ``self._solve_cp_sat`` directly, OR
        # they pass a fully serialized CP-SAT formulation here.
        #  implement the in-process dispatch path; here
        # we return ``unsupported`` for kinds that did not call us
        # via the planner module.
        t0 = time.perf_counter()
        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=self.name,
            backend_availability=probe.availability,
            status=SolverStatus.UNSUPPORTED,
            formulation_hash=request.formulation_hash,
            time_ms=(time.perf_counter() - t0) * 1000.0,
            infeasibility_reason=(
                "raw CP-SAT formulation dispatch not implemented; "
                "use compgen.solve.placement_planner or overlap_planner"
            ),
        )

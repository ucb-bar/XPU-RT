"""Z3 solver backend.

. Probes Z3 and dispatches proof-flavored ``SolverRequest``s.
``solve`` only handles ``BACKEND_PROBE`` self-checks;
real obligation kinds are wired via
:mod:`compgen.solve.z3_obligations`.
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

__all__ = ["Z3Backend"]


_SUPPORTED_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.PEEPHOLE_VERIFY,
        SolverProblemKind.RECIPE_REFINEMENT,
        SolverProblemKind.TRANSLATION_VALIDATION,
        SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        SolverProblemKind.PLAN_INVARIANT_VERIFY,
        SolverProblemKind.BACKEND_PROBE,
    }
)


class Z3Backend(SolverBackend):
    @property
    def name(self) -> SolverBackendName:
        return SolverBackendName.Z3

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in _SUPPORTED_KINDS

    def probe(self) -> BackendProbeResult:
        try:
            import z3
        except ImportError as exc:
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail=f"z3 not installed: {exc}",
            )
        try:
            # tiny SAT
            x = z3.Int("x")
            s = z3.Solver()
            s.set(timeout=1000)
            s.add(x == 7)
            assert s.check() == z3.sat
            # tiny UNSAT
            s2 = z3.Solver()
            s2.set(timeout=1000)
            y = z3.Int("y")
            s2.add(y == 1, y == 2)
            assert s2.check() == z3.unsat
        except Exception as exc:  # pragma: no cover - host-specific
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.PROBE_ERROR,
                detail=str(exc),
            )
        version: str | None
        try:
            version = z3.get_version_string()
        except Exception:  # pragma: no cover
            version = None
        return BackendProbeResult(
            backend=self.name,
            availability=BackendAvailabilityStatus.AVAILABLE,
            version=version,
            supports=("smt_qfbv", "smt_lia", "smt_arrays"),
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
                    f"z3 does not support problem_kind={request.problem_kind.value!r}; "
                    f"this is a solver-purpose violation (placement/schedule kinds "
                    f"must not route to a proof backend)"
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
                infeasibility_reason=f"z3 unavailable: {probe.detail}",
            )
        if request.problem_kind is SolverProblemKind.BACKEND_PROBE:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.PROVED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                solution={"probe_ok": True, "version": probe.version},
            )
        # Real obligation kinds: delegate to z3_obligations.
        from compgen.solve import z3_obligations

        t0 = time.perf_counter()
        try:
            return z3_obligations.solve_request(request, probe=probe)
        except Exception as exc:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.ERROR,
                formulation_hash=request.formulation_hash,
                time_ms=(time.perf_counter() - t0) * 1000.0,
                infeasibility_reason=f"z3 obligation harness raised: {exc}",
            )

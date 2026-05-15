"""HiGHS solver backend (open-source LP/MILP fallback).

. Prefers ``highspy``; falls back to ``scipy.optimize.linprog``
with ``method="highs"`` when ``highspy`` is missing.
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

__all__ = ["HighsBackend"]


_SUPPORTED_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.MEMORY_ALLOCATION,
        SolverProblemKind.BUFFER_ALIASING,
        SolverProblemKind.BANDWIDTH_ALLOCATION,
        SolverProblemKind.COST_MODEL_FIT,
        SolverProblemKind.BACKEND_PROBE,
    }
)


class HighsBackend(SolverBackend):
    @property
    def name(self) -> SolverBackendName:
        return SolverBackendName.HIGHS

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in _SUPPORTED_KINDS

    def _probe_highspy(self) -> BackendProbeResult | None:
        try:
            import highspy  # type: ignore[import-not-found]
        except ImportError:
            return None
        try:
            h = highspy.Highs()
            h.silent()
            lp = highspy.HighsLp()
            lp.num_col_ = 2
            lp.num_row_ = 0
            lp.col_cost_ = [1.0, 1.0]
            lp.col_lower_ = [1.0, 1.0]
            lp.col_upper_ = [1.0e30, 1.0e30]
            h.passModel(lp)
            status = h.run()
            if str(status) not in ("HighsStatus.kOk", "kOk", "HighsStatus.Ok"):
                return BackendProbeResult(
                    backend=self.name,
                    availability=BackendAvailabilityStatus.PROBE_ERROR,
                    detail=f"highspy probe returned {status}",
                )
            version: str | None
            try:
                from importlib import metadata as _md

                version = _md.version("highspy")
            except Exception:  # pragma: no cover
                version = None
            if version is None:
                major = getattr(highspy, "HIGHS_VERSION_MAJOR", None)
                minor = getattr(highspy, "HIGHS_VERSION_MINOR", None)
                patch = getattr(highspy, "HIGHS_VERSION_PATCH", None)
                if major is not None:
                    version = f"{major}.{minor}.{patch}"
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.AVAILABLE,
                version=version,
                supports=("lp", "milp"),
                detail="via highspy",
            )
        except Exception as exc:  # pragma: no cover - host-specific
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.PROBE_ERROR,
                detail=f"highspy: {exc}",
            )

    def _probe_scipy(self) -> BackendProbeResult:
        try:
            from scipy.optimize import linprog
        except ImportError as exc:
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail=f"scipy not installed: {exc}",
            )
        try:
            res = linprog(c=[1.0, 1.0], bounds=[(1.0, None), (1.0, None)], method="highs")
            if not res.success:
                return BackendProbeResult(
                    backend=self.name,
                    availability=BackendAvailabilityStatus.PROBE_ERROR,
                    detail=f"scipy.linprog highs probe failed: {res.message}",
                )
        except Exception as exc:  # pragma: no cover - host-specific
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.PROBE_ERROR,
                detail=f"scipy: {exc}",
            )
        try:
            import scipy
            version = getattr(scipy, "__version__", None)
        except Exception:  # pragma: no cover
            version = None
        return BackendProbeResult(
            backend=self.name,
            availability=BackendAvailabilityStatus.AVAILABLE,
            version=f"scipy/{version}",
            supports=("lp", "milp"),
            detail="via scipy.optimize.linprog method=highs",
        )

    def probe(self) -> BackendProbeResult:
        result = self._probe_highspy()
        if result is not None:
            return result
        return self._probe_scipy()

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
                    f"highs does not support problem_kind="
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
                infeasibility_reason=f"highs unavailable: {probe.detail}",
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
                solution={"probe_ok": True, "version": probe.version, "detail": probe.detail},
            )
        t0 = time.perf_counter()
        from compgen.solve import _highs_solve_impl

        try:
            return _highs_solve_impl.solve(request, probe=probe)
        except NotImplementedError as exc:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.UNSUPPORTED,
                formulation_hash=request.formulation_hash,
                time_ms=(time.perf_counter() - t0) * 1000.0,
                infeasibility_reason=f"highs backend kind not implemented: {exc}",
                caveats=("highs_milp_fallback_limited",),
            )
        except Exception as exc:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.ERROR,
                formulation_hash=request.formulation_hash,
                time_ms=(time.perf_counter() - t0) * 1000.0,
                infeasibility_reason=f"highs solve raised: {exc}",
            )

"""MOSEK solver backend.

. MOSEK is the licensed numeric backend; HiGHS is the
open-source fallback. On first probe, if ``MOSEKLM_LICENSE_FILE`` is
unset and ``<repo_root>/mosek.lic`` exists, this module sets the env
var so MOSEK can pick up the repo-local license without the user
having to export it. The license body is never read or printed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

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

__all__ = ["MosekBackend", "ensure_mosek_license_env"]


_SUPPORTED_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.MEMORY_ALLOCATION,
        SolverProblemKind.BUFFER_ALIASING,
        SolverProblemKind.BANDWIDTH_ALLOCATION,
        SolverProblemKind.COST_MODEL_FIT,
        SolverProblemKind.BACKEND_PROBE,
    }
)


def _repo_root() -> Path:
    """Best-effort repo root: this file is at <repo>/python/compgen/solve/backends/."""

    return Path(__file__).resolve().parents[4]


def ensure_mosek_license_env() -> str | None:
    """If ``MOSEKLM_LICENSE_FILE`` is unset, point it at ``<repo>/mosek.lic``.

    Returns the chosen license path (set or pre-existing), or ``None``
    if no license is configured. Never reads the license body.
    """

    existing = os.environ.get("MOSEKLM_LICENSE_FILE")
    if existing:
        return existing
    candidate = _repo_root() / "mosek.lic"
    if candidate.is_file():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(candidate)
        return str(candidate)
    return None


def _classify_mosek_error(message: str) -> BackendAvailabilityStatus:
    msg = message.lower()
    if "license" in msg or "lic" in msg or "no licens" in msg or "no_licens" in msg:
        if "token" in msg or "flexlm" in msg or "feature" in msg:
            return BackendAvailabilityStatus.LICENSE_TOKEN_UNAVAILABLE
        return BackendAvailabilityStatus.LICENSE_MISSING
    return BackendAvailabilityStatus.PROBE_ERROR


class MosekBackend(SolverBackend):
    @property
    def name(self) -> SolverBackendName:
        return SolverBackendName.MOSEK

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in _SUPPORTED_KINDS

    def probe(self) -> BackendProbeResult:
        ensure_mosek_license_env()
        try:
            import mosek  # type: ignore[import-not-found]
        except ImportError as exc:
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail=f"mosek not installed: {exc}",
            )
        try:
            with mosek.Env() as env:
                with env.Task(0, 0) as task:
                    # 2-variable LP: min x+y, s.t. x>=1, y>=1
                    task.appendvars(2)
                    task.appendcons(2)
                    for j in range(2):
                        task.putcj(j, 1.0)
                        task.putvarbound(j, mosek.boundkey.lo, 1.0, +1.0e30)
                    task.putconbound(0, mosek.boundkey.lo, 1.0, +1.0e30)
                    task.putconbound(1, mosek.boundkey.lo, 1.0, +1.0e30)
                    task.putaij(0, 0, 1.0)
                    task.putaij(1, 1, 1.0)
                    task.putobjsense(mosek.objsense.minimize)
                    task.optimize()
                    sol_sta = task.getsolsta(mosek.soltype.bas)
                    if sol_sta != mosek.solsta.optimal:
                        return BackendProbeResult(
                            backend=self.name,
                            availability=BackendAvailabilityStatus.PROBE_ERROR,
                            detail=f"probe LP not optimal: {sol_sta}",
                        )
        except Exception as exc:  # mosek.Error is also caught here
            cls = _classify_mosek_error(str(exc))
            return BackendProbeResult(
                backend=self.name,
                availability=cls,
                detail=str(exc),
            )
        version: str | None
        try:
            parts = mosek.Env.getversion()
            version = ".".join(str(p) for p in parts)
        except Exception:  # pragma: no cover
            version = None
        return BackendProbeResult(
            backend=self.name,
            availability=BackendAvailabilityStatus.AVAILABLE,
            version=version,
            supports=("lp", "qp", "conic", "milp"),
        )

    def solve(self, request: SolverRequest) -> SolverResponse:
        if not self.supports(request.problem_kind):
            # Architecture guard: MOSEK must never handle semantic-proof
            # / discrete-scheduling kinds even if a misrouted call
            # arrives directly. Hard-typed UNSUPPORTED.
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.UNSUPPORTED,
                formulation_hash=request.formulation_hash,
                time_ms=0.0,
                infeasibility_reason=(
                    f"mosek does not support problem_kind={request.problem_kind.value!r}; "
                    f"this is a solver-purpose violation (proof/discrete kinds must not "
                    f"route to a numeric LP/MILP backend)"
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
                infeasibility_reason=f"mosek unavailable: {probe.detail}",
                caveats=("mosek_license_unavailable",)
                if probe.availability
                in {
                    BackendAvailabilityStatus.LICENSE_MISSING,
                    BackendAvailabilityStatus.LICENSE_TOKEN_UNAVAILABLE,
                }
                else (),
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
        t0 = time.perf_counter()
        # memory_planner / cost_model_fit submit MILP / LP
        # formulations directly via this backend.
        from compgen.solve import _mosek_solve_impl

        try:
            return _mosek_solve_impl.solve(request, probe=probe)
        except NotImplementedError as exc:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=probe.availability,
                status=SolverStatus.UNSUPPORTED,
                formulation_hash=request.formulation_hash,
                time_ms=(time.perf_counter() - t0) * 1000.0,
                infeasibility_reason=f"mosek backend kind not implemented: {exc}",
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
                infeasibility_reason=f"mosek solve raised: {exc}",
            )

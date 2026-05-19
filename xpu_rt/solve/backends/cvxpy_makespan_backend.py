"""CVXPY-backed makespan scheduling backend.

Wraps the absorbed XPU-RT two-cluster scheduler (`xpu_rt.scheduler.schedule`)
as a first-class :class:`SolverBackend`. With this in place every solver call
in the compiler — placement, memory planning, makespan scheduling, semantic
proofs — flows through the same `SolverRequest`/`SolverResponse` envelope
and probe-aware routing layer.

The underlying scheduler builds a Mixed-Integer Linear Program in
CVXPY and dispatches it to whichever MILP solver CVXPY can find. We
prefer MOSEK (license auto-discovered by
:func:`xpu_rt.solve.backends.mosek_backend.ensure_mosek_license_env`)
and accept HiGHS / SCIP / GLPK as open-source fallbacks.

Carrier convention:

The `Workload` and the rich kwargs of `schedule()` carry numpy
arrays, dict-of-tuple structures, and other non-trivially-
JSON-serialisable shapes. To keep `SolverRequest.formulation`
JSON-friendly (so `formulation_hash` is byte-stable) we put a
compact problem *signature* (op count, machine count, fusion
threshold, time limit, ...) in `formulation` and pass the live
Python objects via two reserved `metadata` keys:

    metadata["__cvxpy_makespan_workload"]  ->  Workload
    metadata["__cvxpy_makespan_kwargs"]    ->  dict of schedule() kwargs

The shim :func:`xpu_rt.scheduler.solve_makespan` does this wrapping
so callers never touch the registry directly.
"""

from __future__ import annotations

import time
from typing import Any

from xpu_rt.solve.backends.base import SolverBackend
from xpu_rt.solve.backends.mosek_backend import ensure_mosek_license_env
from xpu_rt.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
)

__all__ = [
    "CvxpyMakespanBackend",
    "METADATA_WORKLOAD_KEY",
    "METADATA_KWARGS_KEY",
]


METADATA_WORKLOAD_KEY = "__cvxpy_makespan_workload"
METADATA_KWARGS_KEY = "__cvxpy_makespan_kwargs"


_SUPPORTED_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.MAKESPAN_SCHEDULE,
        SolverProblemKind.BACKEND_PROBE,
    }
)


class CvxpyMakespanBackend(SolverBackend):
    """SolverBackend wrapper around `xpu_rt.scheduler.schedule()`."""

    @property
    def name(self) -> SolverBackendName:
        return SolverBackendName.CVXPY_MAKESPAN

    def supports(self, problem_kind: SolverProblemKind) -> bool:
        return problem_kind in _SUPPORTED_KINDS

    def probe(self) -> BackendProbeResult:
        # XPU-RT's two-cluster scheduler (xpu_rt.scheduler.scheduler.schedule)
        # hard-codes ``solver=cp.MOSEK`` at every cvxpy.Problem.solve() site
        # (4 occurrences). That's a deliberate choice from the original
        # XPU-RT design — MOSEK's interior-point solver is what produced
        # the reference scheduling results. We mirror that requirement here:
        # if MOSEK can't be reached, the registry MUST return BLOCKED rather
        # than silently letting cvxpy fall back to a different solver that
        # would produce subtly different schedules.

        # Make the repo-local MOSEK license visible to cvxpy without
        # forcing users to export the env var. Mirrors what the MOSEK
        # backend does in its own probe.
        ensure_mosek_license_env()

        try:
            import cvxpy  # type: ignore[import-not-found]
        except ImportError as exc:
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.IMPORT_MISSING,
                detail=f"cvxpy not installed: {exc}",
            )

        # Tiny LP to confirm cvxpy itself works.
        try:
            x = cvxpy.Variable()
            prob = cvxpy.Problem(cvxpy.Minimize(x), [x >= 1])
            prob.solve()
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return BackendProbeResult(
                    backend=self.name,
                    availability=BackendAvailabilityStatus.PROBE_ERROR,
                    detail=f"cvxpy probe LP not optimal: {prob.status}",
                )
        except Exception as exc:  # pragma: no cover - cvxpy install issue
            return BackendProbeResult(
                backend=self.name,
                availability=BackendAvailabilityStatus.PROBE_ERROR,
                detail=str(exc),
            )

        # Enumerate cvxpy's installed solvers — MOSEK MUST be in here.
        installed: tuple[str, ...] = ()
        try:
            installed = tuple(sorted(cvxpy.installed_solvers()))
        except Exception:  # pragma: no cover
            pass

        if "MOSEK" not in installed:
            # Distinguish "mosek Python package missing" from "license
            # missing" so the audit log says the right thing.
            try:
                import mosek  # type: ignore[import-not-found]  # noqa: F401

                availability = BackendAvailabilityStatus.LICENSE_MISSING
                detail = (
                    "mosek package importable but cvxpy did not register the "
                    "MOSEK solver (likely license issue). XPU-RT's scheduler "
                    "hard-codes solver=cp.MOSEK, so other solvers will not "
                    f"substitute. Installed cvxpy solvers: {installed!r}"
                )
            except ImportError:
                availability = BackendAvailabilityStatus.IMPORT_MISSING
                detail = (
                    "mosek not installed. XPU-RT's two-cluster scheduler "
                    "hard-codes solver=cp.MOSEK; install with "
                    "`uv pip install -e \".[solve-mosek]\"`. Installed "
                    f"cvxpy solvers: {installed!r}"
                )
            return BackendProbeResult(
                backend=self.name,
                availability=availability,
                version=getattr(cvxpy, "__version__", None),
                supports=("milp", "makespan_schedule") + installed,
                detail=detail,
            )

        # MOSEK present — preferred solver is wired up. Report it FIRST in
        # the supports tuple so audit / probe output makes the choice
        # visually obvious.
        return BackendProbeResult(
            backend=self.name,
            availability=BackendAvailabilityStatus.AVAILABLE,
            version=getattr(cvxpy, "__version__", None),
            supports=("preferred_solver:MOSEK", "milp", "makespan_schedule") + installed,
            detail=(
                "cvxpy with MOSEK; XPU-RT's scheduler hard-codes "
                "solver=cp.MOSEK at every Problem.solve() site"
            ),
        )

    def solve(self, request: SolverRequest) -> SolverResponse:
        started = time.monotonic()

        if not self.supports(request.problem_kind):
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.UNSUPPORTED,
                formulation_hash=request.formulation_hash,
                time_ms=(time.monotonic() - started) * 1000.0,
                infeasibility_reason=(
                    f"cvxpy_makespan does not support problem_kind="
                    f"{request.problem_kind.value!r}"
                ),
            )

        workload = request.metadata.get(METADATA_WORKLOAD_KEY)
        if workload is None:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.ERROR,
                formulation_hash=request.formulation_hash,
                time_ms=(time.monotonic() - started) * 1000.0,
                infeasibility_reason=(
                    f"missing {METADATA_WORKLOAD_KEY!r} in request.metadata; "
                    f"caller must use xpu_rt.scheduler.solve_makespan()"
                ),
            )

        sched_kwargs: dict[str, Any] = dict(
            request.metadata.get(METADATA_KWARGS_KEY) or {}
        )

        # Map the envelope's time_budget_ms onto the scheduler's
        # time_limit (seconds). Caller-supplied kwarg wins if set.
        if "time_limit" not in sched_kwargs and request.time_budget_ms:
            sched_kwargs["time_limit"] = request.time_budget_ms / 1000.0

        # Late import — avoids a hard cvxpy dependency at import time
        # for callers who only want the typed envelope.
        try:
            from xpu_rt.scheduler.scheduler import schedule as _cvxpy_schedule
        except ImportError as exc:
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.IMPORT_MISSING,
                status=SolverStatus.BLOCKED,
                formulation_hash=request.formulation_hash,
                time_ms=(time.monotonic() - started) * 1000.0,
                infeasibility_reason=(
                    f"xpu_rt.scheduler.scheduler import failed: {exc}"
                ),
            )

        try:
            t_arr, alpha_arr, fused_workload, meta = _cvxpy_schedule(
                workload, **sched_kwargs
            )
        except Exception as exc:  # cvxpy / MOSEK runtime failure
            return SolverResponse(
                problem_id=request.problem_id,
                problem_kind=request.problem_kind,
                selected_backend=self.name,
                backend_availability=BackendAvailabilityStatus.AVAILABLE,
                status=SolverStatus.ERROR,
                formulation_hash=request.formulation_hash,
                time_ms=(time.monotonic() - started) * 1000.0,
                infeasibility_reason=f"{type(exc).__name__}: {exc}",
            )

        time_ms = (time.monotonic() - started) * 1000.0

        # `meta` is a dict from `schedule()` when the underlying MOSEK
        # call returned a status; map it onto the typed envelope. When
        # `meta` is None we treat success as OPTIMAL (the scheduler
        # only returns when MOSEK reports optimal under default
        # settings; honest fallback to FEASIBLE if a time_limit cut
        # the search short).
        if meta is None:
            status = (
                SolverStatus.FEASIBLE
                if sched_kwargs.get("time_limit") is not None
                else SolverStatus.OPTIMAL
            )
            objective_value = float(t_arr.max()) if t_arr.size else None
            infeasibility_reason = None
        else:
            mosek_status = str(meta.get("status", "")).lower()
            if mosek_status in {"optimal", "integer_optimal"}:
                status = SolverStatus.OPTIMAL
            elif mosek_status in {"feasible", "near_optimal"}:
                status = SolverStatus.FEASIBLE
            elif mosek_status == "infeasible":
                status = SolverStatus.INFEASIBLE
            elif mosek_status == "timeout":
                status = SolverStatus.TIMEOUT
            else:
                status = SolverStatus.FEASIBLE if t_arr.size else SolverStatus.ERROR
            objective_value = (
                float(meta.get("makespan"))
                if meta.get("makespan") is not None
                else (float(t_arr.max()) if t_arr.size else None)
            )
            infeasibility_reason = meta.get("infeasibility_reason")

        # Solution payload — keep it dict-shaped so downstream tooling
        # (e.g. caveat ledger, replay harness) can read it without
        # numpy. The Workload reference stays in metadata; we do NOT
        # echo it here.
        solution = {
            "t": [list(map(float, row)) for row in t_arr.tolist()]
            if hasattr(t_arr, "tolist")
            else list(t_arr),
            "alpha": [list(map(float, row)) for row in alpha_arr.tolist()]
            if hasattr(alpha_arr, "tolist")
            else list(alpha_arr),
            "fused": fused_workload is not None,
            "scheduler_meta": meta,
        }

        # Surface the actual MILP solver cvxpy ran for audit purposes. The
        # original scheduler hard-codes MOSEK, so under healthy probe state
        # this is always "MOSEK"; if downstream code is ever changed to
        # accept a fallback, this caveat will flag the substitution.
        caveats: tuple[str, ...] = ()
        if isinstance(meta, dict):
            actual_solver = meta.get("solver_name") or meta.get("solver")
            if actual_solver and str(actual_solver).upper() != "MOSEK":
                caveats = (
                    f"cvxpy_makespan ran on solver={actual_solver!r}, NOT MOSEK; "
                    "XPU-RT's reference results were produced with MOSEK — "
                    "the schedule may differ from the canonical run",
                )

        return SolverResponse(
            problem_id=request.problem_id,
            problem_kind=request.problem_kind,
            selected_backend=self.name,
            backend_availability=BackendAvailabilityStatus.AVAILABLE,
            status=status,
            formulation_hash=request.formulation_hash,
            time_ms=time_ms,
            objective_value=objective_value,
            solution=solution,
            infeasibility_reason=infeasibility_reason,
            caveats=caveats,
        )

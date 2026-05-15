"""XPU-RT two-cluster makespan scheduler (CVXPY MILP / MOSEK).

This subpackage is the absorbed XPU-RT scheduling stack: workload model,
CVXPY MILP scheduler, fusion, packing, postprocessing, and the on-board
telemetry → dispatch-hint feedback path.

Two entry points:

- :func:`xpu_rt.scheduler.scheduler.schedule` — the original direct
  function (cvxpy + MOSEK), kept for callers that already wire it up.
- :func:`solve_makespan` — typed-envelope wrapper that routes the same
  computation through :mod:`xpu_rt.solve` so the call is registered,
  probe-aware, license-aware, and audit-trackable (formulation_hash,
  selected_backend, time_ms, typed SolverStatus).

The original ``schedule()`` hard-coded MOSEK as the MILP solver; the
envelope path preserves that default (CVXPY routes to MOSEK when its
license is auto-discovered by
:func:`xpu_rt.solve.backends.mosek_backend.ensure_mosek_license_env`)
and only falls back to HiGHS / SCIP / GLPK when no MOSEK license is
present. Both paths share the same repo-local ``mosek.lic`` file.
"""

from __future__ import annotations

from xpu_rt.scheduler.bridge import solve_makespan

__all__ = ["solve_makespan"]

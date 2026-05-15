"""MOSEK solve dispatch.

Routes :class:`SolverRequest` to the per-kind MOSEK formulation
modules. implements ``MEMORY_ALLOCATION``; other kinds raise
``NotImplementedError`` which the backend converts to a typed
``unsupported`` response.
"""

from __future__ import annotations

from compgen.solve.solver_types import (
    BackendProbeResult,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
)

__all__ = ["solve"]


def solve(request: SolverRequest, *, probe: BackendProbeResult) -> SolverResponse:
    if request.problem_kind is SolverProblemKind.MEMORY_ALLOCATION:
        from compgen.solve import memory_planner

        return memory_planner.solve_via_mosek(request, probe=probe)
    if request.problem_kind is SolverProblemKind.BANDWIDTH_ALLOCATION:
        # Bandwidth solves come via plan_bandwidth which calls into
        # _solve_lp_mosek directly; routing-via-Mosek-backend is only
        # used for the probe self-check.
        raise NotImplementedError(
            f"bandwidth_allocation routed through plan_bandwidth, not via MosekBackend.solve"
        )
    raise NotImplementedError(request.problem_kind.value)

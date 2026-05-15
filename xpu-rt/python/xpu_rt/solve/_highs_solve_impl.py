"""HiGHS solve dispatch.

Routes :class:`SolverRequest` to per-kind HiGHS formulation modules.
implements ``MEMORY_ALLOCATION`` via either ``highspy`` or
``scipy.optimize.linprog`` with HiGHS. Other kinds raise
``NotImplementedError`` for the backend to convert to a typed
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

        return memory_planner.solve_via_highs(request, probe=probe)
    raise NotImplementedError(request.problem_kind.value)

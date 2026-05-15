"""Deterministic problem-kind -> backend routing.

. The table is fixed at design time: semantic-proof kinds
ALWAYS go to Z3, scheduling / placement / no-overlap ALWAYS go to
OR-Tools CP-SAT, and numeric LP / MILP prefers MOSEK with HiGHS as
the open-source fallback.

Rules:

* No semantic-proof kind ever routes to MOSEK or HiGHS.
* No placement / schedule / overlap kind ever routes to Z3.
* When the preferred backend is unavailable, the table consults its
  declared fallback; if neither is available, returns ``None`` and
  the caller MUST emit a typed ``SolverStatus.BLOCKED`` response —
  never a greedy heuristic.
"""

from __future__ import annotations

from compgen.solve.backend_registry import SolverBackendRegistry, default_registry
from compgen.solve.solver_types import (
    SolverBackendName,
    SolverProblemKind,
)

__all__ = ["choose_backend", "ROUTING_TABLE"]


# Each entry is the ordered preference list for that problem kind.
ROUTING_TABLE: dict[SolverProblemKind, tuple[SolverBackendName, ...]] = {
    # Z3 proof kinds
    SolverProblemKind.PEEPHOLE_VERIFY: (SolverBackendName.Z3,),
    SolverProblemKind.RECIPE_REFINEMENT: (SolverBackendName.Z3,),
    SolverProblemKind.TRANSLATION_VALIDATION: (SolverBackendName.Z3,),
    SolverProblemKind.SHAPE_PREDICATE_VERIFY: (SolverBackendName.Z3,),
    SolverProblemKind.PLAN_INVARIANT_VERIFY: (SolverBackendName.Z3,),
    # OR-Tools CP-SAT discrete kinds
    SolverProblemKind.PLACEMENT: (SolverBackendName.ORTOOLS_CP_SAT,),
    SolverProblemKind.SCHEDULE: (SolverBackendName.ORTOOLS_CP_SAT,),
    SolverProblemKind.NO_OVERLAP_SCHEDULE: (SolverBackendName.ORTOOLS_CP_SAT,),
    SolverProblemKind.EVENT_ORDERING: (SolverBackendName.ORTOOLS_CP_SAT,),
    SolverProblemKind.OVERLAP_PLANNING: (SolverBackendName.ORTOOLS_CP_SAT,),
    # MILP / LP numeric kinds — MOSEK preferred, HiGHS fallback
    SolverProblemKind.MEMORY_ALLOCATION: (SolverBackendName.MOSEK, SolverBackendName.HIGHS),
    SolverProblemKind.BUFFER_ALIASING: (SolverBackendName.MOSEK, SolverBackendName.HIGHS),
    SolverProblemKind.BANDWIDTH_ALLOCATION: (SolverBackendName.MOSEK, SolverBackendName.HIGHS),
    SolverProblemKind.COST_MODEL_FIT: (SolverBackendName.MOSEK, SolverBackendName.HIGHS),
    # Meta
    SolverProblemKind.BACKEND_PROBE: (
        SolverBackendName.Z3,
        SolverBackendName.ORTOOLS_CP_SAT,
        SolverBackendName.MOSEK,
        SolverBackendName.HIGHS,
    ),
}


# Hard separation: these kinds must never run on the wrong family of
# backends, even if a contributor adds a stray entry to ``ROUTING_TABLE``.
_PROOF_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.PEEPHOLE_VERIFY,
        SolverProblemKind.RECIPE_REFINEMENT,
        SolverProblemKind.TRANSLATION_VALIDATION,
        SolverProblemKind.SHAPE_PREDICATE_VERIFY,
        SolverProblemKind.PLAN_INVARIANT_VERIFY,
    }
)
_DISCRETE_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.PLACEMENT,
        SolverProblemKind.SCHEDULE,
        SolverProblemKind.NO_OVERLAP_SCHEDULE,
        SolverProblemKind.EVENT_ORDERING,
        SolverProblemKind.OVERLAP_PLANNING,
    }
)
_NUMERIC_KINDS: frozenset[SolverProblemKind] = frozenset(
    {
        SolverProblemKind.MEMORY_ALLOCATION,
        SolverProblemKind.BUFFER_ALIASING,
        SolverProblemKind.BANDWIDTH_ALLOCATION,
        SolverProblemKind.COST_MODEL_FIT,
    }
)


def _allowed_for_kind(kind: SolverProblemKind, backend: SolverBackendName) -> bool:
    if kind in _PROOF_KINDS:
        return backend is SolverBackendName.Z3
    if kind in _DISCRETE_KINDS:
        return backend is SolverBackendName.ORTOOLS_CP_SAT
    if kind in _NUMERIC_KINDS:
        return backend in {
            SolverBackendName.MOSEK,
            SolverBackendName.HIGHS,
            SolverBackendName.OSQP_OPTIONAL,
            SolverBackendName.CLARABEL_OPTIONAL,
        }
    return True


def choose_backend(
    problem_kind: SolverProblemKind,
    registry: SolverBackendRegistry | None = None,
    *,
    preference: SolverBackendName | None = None,
) -> SolverBackendName | None:
    """Pick the backend for ``problem_kind`` given current availability.

    Returns ``None`` when no available backend can handle the kind;
    the caller MUST emit ``SolverStatus.BLOCKED`` (never silently
    swap in a greedy heuristic).

    ``preference``, when supplied, is honored only if it is in the
    routing table for ``problem_kind`` AND is available.
    """

    reg = registry if registry is not None else default_registry()
    available = set(reg.available_backends())

    preferences = ROUTING_TABLE.get(problem_kind, ())
    if preference is not None and preference in preferences and preference in available:
        if not _allowed_for_kind(problem_kind, preference):
            return None
        return preference

    for candidate in preferences:
        if not _allowed_for_kind(problem_kind, candidate):
            continue
        if candidate in available:
            return candidate
    return None

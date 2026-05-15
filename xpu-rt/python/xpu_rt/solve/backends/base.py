"""Abstract base class for solver backends.

. Every backend (Z3 / OR-Tools CP-SAT / MOSEK / HiGHS)
implements this interface so the registry can probe availability,
filter by supported problem kind, and dispatch a typed
``SolverRequest`` to a typed ``SolverResponse``.

A backend MUST:

* ``probe()`` without raising. Map every failure to
  :class:`BackendAvailabilityStatus`.
* Enforce ``request.time_budget_ms``.
* Set ``response.status`` from the typed enum; never invent strings.
* Compute ``response.formulation_hash == request.formulation_hash``;
  the registry verifies this on return.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from compgen.solve.solver_types import (
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
)

__all__ = ["SolverBackend"]


class SolverBackend(ABC):
    """Abstract solver backend.

    Subclasses live in ``compgen.solve.backends.*_backend``.
    """

    @property
    @abstractmethod
    def name(self) -> SolverBackendName:
        """The backend's canonical name."""

    @abstractmethod
    def probe(self) -> BackendProbeResult:
        """Check availability without raising.

        Implementations import their dependency lazily, run a tiny
        self-check, and map any failure to a typed
        :class:`BackendAvailabilityStatus`.
        """

    @abstractmethod
    def supports(self, problem_kind: SolverProblemKind) -> bool:
        """Whether this backend handles ``problem_kind`` at all."""

    @abstractmethod
    def solve(self, request: SolverRequest) -> SolverResponse:
        """Run the solver on ``request``.

        Returns a typed :class:`SolverResponse`. MUST NOT raise on
        infeasibility/timeout — set ``status`` accordingly.
        """

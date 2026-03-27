"""MILP solver backend.

Used for memory allocation, cost optimization, and problems with
linear constraints and objectives.

Invariants:
    - Solver is imported at call time (optional dep).
    - Supports both OR-Tools linear solver and other MILP backends.

TODO: Implement MILPSolver with memory allocation method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MILPSolver:
    """MILP solver backend.

    Attributes:
        timeout_ms: Solver timeout.
        gap_tolerance: Acceptable optimality gap.

    TODO: Implement solve() for MILP problems.
    """

    timeout_ms: int = 30000
    gap_tolerance: float = 0.01

    def solve(self, problem: Any) -> Any:
        """Solve a MILP problem.

        TODO: Build linear model from problem.
        TODO: Solve with timeout and gap tolerance.
        """
        raise NotImplementedError("MILPSolver.solve is not yet implemented")


__all__ = ["MILPSolver"]

"""SMT solver backend (Z3).

Used for legality checking and semantic verification:
    - Translation validation queries
    - Peephole rewrite verification
    - Dataflow analysis soundness
    - Synthesized-guard soundness checks

Invariants:
    - z3 is imported at call time (optional dep).
    - Timeout is always set.
    - "unknown" results are distinguished from "unsat" and "sat".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SMTSolver:
    """SMT solver backend using Z3.

    Attributes:
        timeout_ms: Solver timeout.

    TODO: Implement check_sat() for satisfiability queries.
    TODO: Implement prove() for validity queries.
    TODO: Implement get_model() for counterexample extraction.
    """

    timeout_ms: int = 30000

    def check_sat(self, formula: Any) -> str:
        """Check satisfiability of a formula.

        Returns: "sat", "unsat", or "unknown".
        """
        try:
            import z3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("z3 is required for SMT solving") from exc

        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(formula)
        result = solver.check()
        if result == z3.sat:
            return "sat"
        if result == z3.unsat:
            return "unsat"
        return "unknown"

    def prove(self, formula: Any) -> str:
        """Prove validity of a formula (check unsat of negation).

        Returns: "valid", "invalid", or "unknown".
        """
        try:
            import z3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("z3 is required for SMT solving") from exc

        status = self.check_sat(z3.Not(formula))
        if status == "unsat":
            return "valid"
        if status == "sat":
            return "invalid"
        return "unknown"

    def get_model(self, formula: Any) -> dict[str, Any] | None:
        """Get a satisfying model (counterexample).

        Returns: Model dict if sat, None otherwise.
        """
        try:
            import z3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("z3 is required for SMT solving") from exc

        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(formula)
        result = solver.check()
        if result != z3.sat:
            return None

        model = solver.model()
        extracted: dict[str, Any] = {}
        for decl in model.decls():
            value = model[decl]
            if value is None:
                continue
            if z3.is_true(value):
                extracted[decl.name()] = True
            elif z3.is_false(value):
                extracted[decl.name()] = False
            elif z3.is_int_value(value):
                extracted[decl.name()] = value.as_long()
            elif z3.is_rational_value(value):
                extracted[decl.name()] = float(value.as_fraction())
            else:
                extracted[decl.name()] = str(value)
        return extracted


__all__ = ["SMTSolver"]

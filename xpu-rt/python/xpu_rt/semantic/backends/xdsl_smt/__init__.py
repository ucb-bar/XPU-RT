"""xDSL-SMT verification backend.

Provides translation validation, PDL rewrite verification, and transfer
function verification using xDSL's SMT dialect and Z3 as the solver.

The architecture follows the paper "First-Class Verification Dialects
for MLIR" (PLDI'25): lower program dialects to semantic dialects,
optimize the semantic IR, then solve via SMT.

This package uses ``xdsl.dialects.smt`` (upstream) for the core SMT
types/ops and ``z3-solver`` for solving. It does NOT depend on the
separate ``xdsl-smt`` pip package.
"""

from __future__ import annotations

__all__: list[str] = []

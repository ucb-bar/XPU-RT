"""Solver subsystem for CompGen.

Sits between Recipe IR and Plan IR. Uses mathematical solvers to make
globally consistent placement, scheduling, and memory allocation decisions.

Division of labor:
    - LLM (via Recipe IR): proposes legal choices, narrows search space
    - Solver: chooses globally optimal assignments under constraints
    - Compiler: extracts dispatch DAGs and cost estimates
    - Verifier: checks correctness of results

Solver backends:
    - CP-SAT (Google OR-Tools): placement, scheduling, combinatorial
    - MILP: cost optimization, memory allocation with linear constraints
    - SMT (Z3): legality and semantic verification

The solver sees a compressed problem extracted from Recipe IR + target profile,
NOT the full raw graph.
"""

from __future__ import annotations

__all__: list[str] = []

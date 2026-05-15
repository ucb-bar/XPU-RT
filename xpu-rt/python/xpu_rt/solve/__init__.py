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

from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    BackendProbeResult,
    SolverBackendName,
    SolverProblemKind,
    SolverRequest,
    SolverResponse,
    SolverStatus,
    compute_formulation_hash,
)
from compgen.solve.backend_registry import SolverBackendRegistry, default_registry
from compgen.solve.routing import ROUTING_TABLE, choose_backend
from compgen.solve.reports import (
    summarize_response_md,
    write_solver_request,
    write_solver_response,
)

__all__ = [
    "BackendAvailabilityStatus",
    "BackendProbeResult",
    "SolverBackendName",
    "SolverBackendRegistry",
    "SolverProblemKind",
    "SolverRequest",
    "SolverResponse",
    "SolverStatus",
    "ROUTING_TABLE",
    "choose_backend",
    "compute_formulation_hash",
    "default_registry",
    "summarize_response_md",
    "write_solver_request",
    "write_solver_response",
]

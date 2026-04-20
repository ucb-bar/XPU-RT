"""MILP solver backend.

Used for memory allocation, cost optimization, and problems with
linear constraints and objectives.

Invariants:
    - Solver is imported at call time (optional dep).
    - Formulates placement as a binary integer linear program.
    - Falls back to a clear ImportError when scipy is missing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.solve.contracts import SolverProblem

log = structlog.get_logger()


@dataclass(frozen=True)
class MILPResult:
    """Result from the MILP solver backend.

    Attributes:
        placement: Dict mapping partition_id -> device index.
        schedule: Placeholder (MILP solves placement only; scheduling
            can be layered on top).
        memory: Placeholder for memory solution.
        feasible: Whether a feasible solution was found.
        solve_time_ms: Wall-clock solve time.
        objective_value: Total cost of the placement.
    """

    placement: dict[str, int] = field(default_factory=dict)
    schedule: Any = None
    memory: Any = None
    feasible: bool = False
    solve_time_ms: float = 0.0
    objective_value: float = float("inf")


@dataclass
class MILPSolver:
    """MILP solver backend.

    Formulates device placement as a binary integer linear program and
    solves it with ``scipy.optimize.milp``.

    Variables:
        x[i, j] = 1 if partition *i* is assigned to device *j*.

    Constraints:
        - Each partition assigned to exactly one device.
        - Memory capacity per device is not exceeded.

    Objective:
        Minimize total estimated cost (partition cost / device speed,
        defaulting to uniform speed).

    Attributes:
        timeout_ms: Solver timeout.
        gap_tolerance: Acceptable optimality gap.
    """

    timeout_ms: int = 30000
    gap_tolerance: float = 0.01

    def solve(self, problem: SolverProblem) -> MILPResult:
        """Solve device placement via MILP using scipy.

        Args:
            problem: A ``SolverProblem`` extracted from module + target.

        Returns:
            ``MILPResult`` with placement assignments and cost.

        Raises:
            ImportError: If ``scipy`` is not installed.
        """
        try:
            import numpy as np
            from scipy.optimize import LinearConstraint, milp
        except ImportError as exc:
            raise ImportError("scipy is required for MILPSolver. Install it with: pip install scipy") from exc

        t0 = time.perf_counter()

        # --- Empty problem fast-path ---
        if not problem.partitions:
            log.info("milp.solve.empty", target=problem.target_name)
            return MILPResult(feasible=True, solve_time_ms=0.0, objective_value=0.0)

        num_partitions = len(problem.partitions)
        num_devices = max(len(problem.device_capacities), 1)
        num_vars = num_partitions * num_devices  # x[i*D + j]

        def _idx(i: int, j: int) -> int:
            """Flat index for x[i, j]."""
            return i * num_devices + j

        # --- Objective: minimize total cost ---
        # c[i*D + j] = cost of assigning partition i to device j
        c = np.zeros(num_vars)
        for i, p in enumerate(problem.partitions):
            for j in range(num_devices):
                # Uniform device speed for now (cost = partition cost)
                c[_idx(i, j)] = p.estimated_cost_us

        # --- Constraint 1: each partition on exactly one device ---
        # For each i: sum_j x[i,j] = 1
        A_assign = np.zeros((num_partitions, num_vars))
        for i in range(num_partitions):
            for j in range(num_devices):
                A_assign[i, _idx(i, j)] = 1.0

        assign_constraint = LinearConstraint(
            A_assign,
            lb=np.ones(num_partitions),
            ub=np.ones(num_partitions),
        )

        constraints_list = [assign_constraint]

        # --- Constraint 2: memory capacity per device ---
        if problem.device_capacities:
            A_mem = np.zeros((num_devices, num_vars))
            lb_mem = np.full(num_devices, -np.inf)
            ub_mem = np.zeros(num_devices)

            for j in range(num_devices):
                cap = problem.device_capacities.get(j, 2**63)
                ub_mem[j] = float(cap)
                for i, p in enumerate(problem.partitions):
                    A_mem[j, _idx(i, j)] = float(p.memory_bytes)

            mem_constraint = LinearConstraint(A_mem, lb=lb_mem, ub=ub_mem)
            constraints_list.append(mem_constraint)

        # --- Variable bounds: 0 <= x[i,j] <= 1, integrality = 1 (binary) ---
        from scipy.optimize import Bounds

        bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))
        integrality = np.ones(num_vars)  # all variables are binary

        # --- Solve ---
        options = {"time_limit": self.timeout_ms / 1000.0, "mip_rel_gap": self.gap_tolerance}

        result = milp(
            c=c,
            constraints=constraints_list,
            integrality=integrality,
            bounds=bounds,
            options=options,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        if result.success and result.x is not None:
            # Extract assignments
            assignments: dict[str, int] = {}
            for i, p in enumerate(problem.partitions):
                for j in range(num_devices):
                    if result.x[_idx(i, j)] > 0.5:
                        assignments[p.partition_id] = j
                        break

            log.info(
                "milp.solve.done",
                target=problem.target_name,
                feasible=True,
                objective=round(result.fun, 4),
                solve_time_ms=round(elapsed, 2),
                num_partitions=num_partitions,
                num_devices=num_devices,
            )

            return MILPResult(
                placement=assignments,
                feasible=True,
                solve_time_ms=elapsed,
                objective_value=float(result.fun),
            )

        log.warning(
            "milp.solve.infeasible",
            target=problem.target_name,
            message=result.message,
            solve_time_ms=round(elapsed, 2),
        )
        return MILPResult(feasible=False, solve_time_ms=elapsed)


__all__ = ["MILPResult", "MILPSolver"]

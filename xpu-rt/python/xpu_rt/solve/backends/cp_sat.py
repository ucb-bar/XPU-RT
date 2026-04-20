"""Google OR-Tools CP-SAT solver backend.

Used for combinatorial placement and scheduling problems.

Invariants:
    - ortools is imported at call time, not at module level (optional dep).
    - ImportError produces a clear diagnostic.
    - Solver timeout is always set (no unbounded solves).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from compgen.solve.contracts import SolverProblem

log = structlog.get_logger()


@dataclass(frozen=True)
class CPSatResult:
    """Result from the CP-SAT solver backend.

    Attributes:
        placement: Placement solution from ``solve_placement``.
        schedule: Schedule solution from ``solve_schedule``.
        memory: Memory allocation solution from ``solve_memory``.
        feasible: Whether all sub-problems found feasible solutions.
        solve_time_ms: Total wall-clock time across all phases.
    """

    placement: Any = None
    schedule: Any = None
    memory: Any = None
    feasible: bool = False
    solve_time_ms: float = 0.0


@dataclass
class CPSatSolver:
    """CP-SAT solver backend.

    Orchestrates placement, scheduling, and memory allocation using the
    existing CP-SAT-based solvers in ``compgen.solve``.

    Attributes:
        timeout_ms: Solver timeout in milliseconds (per sub-problem).
        num_workers: Number of parallel solver workers.
    """

    timeout_ms: int = 10000
    num_workers: int = 4

    def solve(self, problem: SolverProblem) -> CPSatResult:
        """Solve a placement/scheduling/memory problem via CP-SAT.

        Runs the three solver phases sequentially:
        1. Placement -- assign partitions to devices.
        2. Scheduling -- determine start times respecting placement.
        3. Memory -- allocate buffer offsets within device capacities.

        Args:
            problem: A ``SolverProblem`` extracted from module + target.

        Returns:
            ``CPSatResult`` with sub-solutions and overall feasibility.

        Raises:
            ImportError: If ``ortools`` is not installed.
        """
        try:
            from ortools.sat.python import cp_model as _cp_model  # noqa: F401
        except ImportError as exc:
            raise ImportError("ortools is required for CPSatSolver. Install it with: pip install ortools") from exc

        from compgen.solve.memory import BufferLifetime, solve_memory
        from compgen.solve.placement import solve_placement
        from compgen.solve.schedule import solve_schedule

        t0 = time.perf_counter()

        # --- Empty problem fast-path ---
        if not problem.partitions:
            log.info("cp_sat.solve.empty", target=problem.target_name)
            return CPSatResult(feasible=True, solve_time_ms=0.0)

        num_devices = max(len(problem.device_capacities), 1)

        # --- Phase 1: Placement ---
        device_memory_caps = (
            [problem.device_capacities[d] for d in range(num_devices)] if problem.device_capacities else None
        )

        placement = solve_placement(
            partitions=problem.partitions,
            num_devices=num_devices,
            device_memory_caps=device_memory_caps,
            transfer_cost_matrix=problem.transfer_costs or None,
            constraints=problem.placement_constraints or None,
            timeout_ms=self.timeout_ms,
        )

        if not placement.feasible:
            elapsed = (time.perf_counter() - t0) * 1000
            log.warning("cp_sat.solve.placement_infeasible", target=problem.target_name)
            return CPSatResult(
                placement=placement,
                feasible=False,
                solve_time_ms=elapsed,
            )

        # --- Phase 2: Scheduling ---
        partition_ids = [p.partition_id for p in problem.partitions]
        durations = {p.partition_id: p.estimated_cost_us for p in problem.partitions}
        deps = {p.partition_id: list(p.dependencies) for p in problem.partitions}

        schedule = solve_schedule(
            partition_ids=partition_ids,
            durations_us=durations,
            device_assignments=placement.assignments,
            dependencies=deps,
            num_devices=num_devices,
            constraints=problem.schedule_constraints or None,
            timeout_ms=self.timeout_ms,
        )

        if not schedule.feasible:
            elapsed = (time.perf_counter() - t0) * 1000
            log.warning("cp_sat.solve.schedule_infeasible", target=problem.target_name)
            return CPSatResult(
                placement=placement,
                schedule=schedule,
                feasible=False,
                solve_time_ms=elapsed,
            )

        # --- Phase 3: Memory allocation ---
        lifetimes: list[BufferLifetime] = []
        for p in problem.partitions:
            pid = p.partition_id
            device_idx = placement.assignments.get(pid, 0)
            start = schedule.start_times.get(pid, 0.0)
            end = schedule.end_times.get(pid, start + p.estimated_cost_us)

            if p.memory_bytes > 0:
                lifetimes.append(
                    BufferLifetime(
                        buffer_name=pid,
                        size_bytes=p.memory_bytes,
                        device_index=device_idx,
                        start_us=start,
                        end_us=end,
                    )
                )

        device_caps = dict(problem.device_capacities) if problem.device_capacities else {}
        memory = solve_memory(
            lifetimes=lifetimes,
            device_capacities=device_caps,
            timeout_ms=self.timeout_ms,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        overall_feasible = placement.feasible and schedule.feasible and memory.feasible

        log.info(
            "cp_sat.solve.done",
            target=problem.target_name,
            feasible=overall_feasible,
            solve_time_ms=round(elapsed, 2),
            num_partitions=len(problem.partitions),
            makespan_us=schedule.makespan_us,
        )

        return CPSatResult(
            placement=placement,
            schedule=schedule,
            memory=memory,
            feasible=overall_feasible,
            solve_time_ms=elapsed,
        )


__all__ = ["CPSatResult", "CPSatSolver"]

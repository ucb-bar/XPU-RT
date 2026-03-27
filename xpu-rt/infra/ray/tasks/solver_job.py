"""Solver job task — run CP-SAT/MILP/SMT solver remotely.

Wraps the solver backends as Ray remote tasks.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@ray.remote(num_cpus=2)
def solver_job(
    problem: dict[str, Any],
    solver_backend: str = "cp_sat",
    timeout_ms: int = 30000,
) -> dict[str, Any]:
    """Run a solver remotely.

    Args:
        problem: Problem specification dict with keys:
            "type" (placement/schedule/memory),
            "partitions", "num_devices", etc.
        solver_backend: Backend to use ("cp_sat", "milp", "smt").
        timeout_ms: Solver timeout in milliseconds.

    Returns:
        Solution dict.
    """
    problem_type = problem.get("type", "placement")

    if problem_type == "placement":
        from compgen.solve.placement import solve_placement

        result = solve_placement(
            partitions=problem.get("partitions", []),
            num_devices=problem.get("num_devices", 1),
            device_compute_rates=problem.get("device_compute_rates", [1.0]),
            device_memory_caps=problem.get("device_memory_caps", []),
            transfer_cost_matrix=problem.get("transfer_cost_matrix", {}),
        )
        return {
            "feasible": result.feasible,
            "assignments": result.assignments,
            "gap": result.gap,
            "solve_time_ms": result.solve_time_ms,
        }

    if problem_type == "schedule":
        from compgen.solve.schedule import solve_schedule

        result = solve_schedule(
            partition_ids=problem.get("partition_ids", []),
            durations_us=problem.get("durations_us", {}),
            device_assignments=problem.get("device_assignments", {}),
            dependencies=problem.get("dependencies", {}),
            num_devices=problem.get("num_devices", 1),
        )
        return {
            "feasible": result.feasible,
            "makespan_us": result.makespan_us,
            "start_times": result.start_times,
            "solve_time_ms": result.solve_time_ms,
        }

    return {"error": f"Unknown problem type: {problem_type}"}


__all__ = ["solver_job"]

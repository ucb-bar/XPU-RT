"""Region-to-device placement via CP-SAT.

Assigns regions to devices minimizing total estimated latency + transfer cost,
subject to memory capacity constraints and device capability constraints.

Uses Google OR-Tools CP-SAT solver — exact combinatorial optimization,
not heuristics. The agent invokes this as a tool via SolveAction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from compgen.solve.partition import Partition


@dataclass(frozen=True)
class PlacementConstraint:
    """A constraint on placement.

    Attributes:
        partition_id: Partition this constrains.
        allowed_devices: Set of device indices this partition may run on.
        required_device: Specific device (if forced by the agent).
        reason: Why this constraint exists.
    """

    partition_id: str
    allowed_devices: set[int] = field(default_factory=set)
    required_device: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class PlacementSolution:
    """Solution from the placement solver.

    Attributes:
        assignments: Dict mapping partition_id -> device_index.
        feasible: Whether a feasible solution was found.
        objective_value: Total cost (lower is better).
        solve_time_ms: Solver wall-clock time.
        gap: Optimality gap (0.0 = proven optimal).
        transfer_cost: Total estimated cross-device transfer cost.
    """

    assignments: dict[str, int] = field(default_factory=dict)
    feasible: bool = False
    objective_value: float = float("inf")
    solve_time_ms: float = 0.0
    gap: float = 1.0
    transfer_cost: float = 0.0


def solve_placement(
    partitions: list[Partition],
    num_devices: int,
    device_compute_rates: list[float] | None = None,
    device_memory_caps: list[int] | None = None,
    transfer_cost_matrix: dict[tuple[int, int], float] | None = None,
    constraints: list[PlacementConstraint] | None = None,
    timeout_ms: int = 10000,
) -> PlacementSolution:
    """Solve device placement using CP-SAT.

    Minimizes: sum(partition_latency[p] on device[d]) + sum(transfer_costs)

    Args:
        partitions: Regions to place.
        num_devices: Number of available devices.
        device_compute_rates: Relative compute speed per device (higher = faster).
            Default [1.0] * num_devices.
        device_memory_caps: Memory capacity per device in bytes. None = unlimited.
        transfer_cost_matrix: Cost to transfer 1 byte between devices (i,j) -> cost.
        constraints: Agent-specified placement constraints.
        timeout_ms: Solver timeout.

    Returns:
        PlacementSolution with assignments and cost.
    """
    import time

    from ortools.sat.python import cp_model

    if not partitions:
        return PlacementSolution(feasible=True, objective_value=0.0)

    if device_compute_rates is None:
        device_compute_rates = [1.0] * num_devices
    if transfer_cost_matrix is None:
        transfer_cost_matrix = {}

    t0 = time.perf_counter()

    model = cp_model.CpModel()

    # Decision variables: x[p][d] = 1 if partition p is on device d
    x: dict[tuple[str, int], cp_model.IntVar] = {}
    for p in partitions:
        for d in range(num_devices):
            x[(p.partition_id, d)] = model.new_bool_var(f"x_{p.partition_id}_{d}")

    # Each partition on exactly one device
    for p in partitions:
        model.add_exactly_one(x[(p.partition_id, d)] for d in range(num_devices))

    # Agent constraints
    constraint_list = constraints or []
    for c in constraint_list:
        if c.required_device is not None:
            # Force specific device
            model.add(x[(c.partition_id, c.required_device)] == 1)
        elif c.allowed_devices:
            # Restrict to allowed devices
            for d in range(num_devices):
                if d not in c.allowed_devices:
                    model.add(x[(c.partition_id, d)] == 0)

    # Memory capacity constraints
    if device_memory_caps:
        for d in range(num_devices):
            model.add(
                sum(
                    p.memory_bytes * x[(p.partition_id, d)]
                    for p in partitions
                ) <= device_memory_caps[d]
            )

    # Objective: minimize compute cost + transfer cost
    # Scale to integers for CP-SAT (multiply by 1000 for microseconds precision)
    scale = 1000

    # Compute cost: partition_cost / device_speed
    compute_terms = []
    for p in partitions:
        for d in range(num_devices):
            cost_scaled = int(p.estimated_cost_us * scale / max(device_compute_rates[d], 0.001))
            compute_terms.append(cost_scaled * x[(p.partition_id, d)])

    # Transfer cost: for each dependency edge, if src and dst on different devices
    transfer_terms = []
    part_by_id = {p.partition_id: p for p in partitions}
    for p in partitions:
        for dep_id in p.dependencies:
            if dep_id not in part_by_id:
                continue
            dep = part_by_id[dep_id]
            # Estimate transfer bytes (use the smaller partition's memory as proxy)
            transfer_bytes = min(p.memory_bytes, dep.memory_bytes)

            for d1 in range(num_devices):
                for d2 in range(num_devices):
                    if d1 == d2:
                        continue
                    rate = transfer_cost_matrix.get((d1, d2), 1e-6)  # cost per byte
                    cost_scaled = int(transfer_bytes * rate * scale)
                    # Both on different devices
                    both = model.new_bool_var(f"xfer_{p.partition_id}_{dep_id}_{d1}_{d2}")
                    model.add_implication(both, x[(p.partition_id, d1)])
                    model.add_implication(both, x[(dep_id, d2)])
                    transfer_terms.append(cost_scaled * both)

    model.minimize(sum(compute_terms) + sum(transfer_terms))

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_ms / 1000.0

    status = solver.solve(model)
    solve_time = (time.perf_counter() - t0) * 1000

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = {}
        for p in partitions:
            for d in range(num_devices):
                if solver.value(x[(p.partition_id, d)]) == 1:
                    assignments[p.partition_id] = d

        return PlacementSolution(
            assignments=assignments,
            feasible=True,
            objective_value=solver.objective_value / scale,
            solve_time_ms=solve_time,
            gap=0.0 if status == cp_model.OPTIMAL else solver.best_objective_bound / max(solver.objective_value, 1),
        )

    return PlacementSolution(
        feasible=False,
        solve_time_ms=solve_time,
    )


__all__ = ["PlacementConstraint", "PlacementSolution", "solve_placement"]

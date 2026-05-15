"""Solver-ready problem extraction from module + target profiles.

Converts module ops and target hardware constraints into the compact
problem representation that solvers consume.

Invariants:
    - Extraction is deterministic.
    - All costs come from profiled data or target profile cost model.
    - Missing costs default to conservative estimates.
    - When calibration data is available, cost estimates are corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.solve.objectives import CompositeCost
from compgen.solve.partition import Partition, partition_graph
from compgen.solve.placement import PlacementConstraint
from compgen.solve.schedule import ScheduleConstraint
from compgen.targets.schema import TargetProfile
from compgen.targets.utils import extract_device_memory, extract_transfer_cost_matrix

log = structlog.get_logger()


@runtime_checkable
class CostCalibrationProtocol(Protocol):
    """Structural protocol matching ``agent.memory.CostCalibration``."""

    def get_factor(self, device_name: str, op_type: str) -> float: ...


@dataclass(frozen=True)
class SolverProblem:
    """Compressed solver problem extracted from module + target.

    Attributes:
        partitions: Graph partitions.
        placement_constraints: Placement constraints.
        schedule_constraints: Scheduling constraints.
        device_capacities: Memory capacity per device (bytes).
        transfer_costs: Transfer cost matrix (device_i, device_j) -> us/byte.
        objective: Cost function to minimize.
        target_name: Target profile name (for logging).
    """

    partitions: list[Partition] = field(default_factory=list)
    placement_constraints: list[PlacementConstraint] = field(default_factory=list)
    schedule_constraints: list[ScheduleConstraint] = field(default_factory=list)
    device_capacities: dict[int, int] = field(default_factory=dict)
    transfer_costs: dict[tuple[int, int], float] = field(default_factory=dict)
    objective: CompositeCost = field(default_factory=CompositeCost)
    target_name: str = ""


def extract_solver_problem(
    module: ModuleOp,
    target: TargetProfile,
    cost_data: dict[str, float] | None = None,
    calibration: CostCalibrationProtocol | None = None,
) -> SolverProblem:
    """Extract a solver problem from module + target profile.

    Args:
        module: xDSL ModuleOp.
        target: Target hardware profile.
        cost_data: Optional profiled cost data (op_name -> latency_us).
        calibration: Optional calibration instance (duck-typed to
            ``agent.memory.CostCalibration``). When provided, partition cost
            estimates are multiplied by the per-(device, op_type) correction
            factor.

    Returns:
        SolverProblem ready for the solver backends.
    """
    partitions = partition_graph(module)

    if calibration is not None:
        partitions = _apply_calibration(partitions, target, calibration)
        log.info("solver.contracts.calibration_applied", num_partitions=len(partitions))

    device_capacities = {i: cap for i, cap in enumerate(extract_device_memory(target))}
    transfer_costs = extract_transfer_cost_matrix(target)

    return SolverProblem(
        partitions=partitions,
        device_capacities=device_capacities,
        transfer_costs=transfer_costs,
        target_name=target.name,
    )


def _apply_calibration(
    partitions: list[Partition],
    target: TargetProfile,
    calibration: CostCalibrationProtocol,
) -> list[Partition]:
    """Apply calibration correction factors to partition cost estimates.

    For each partition, looks up the correction factor based on the first
    target device name and the partition's primary op type.

    Args:
        partitions: Original partitions with uncalibrated cost estimates.
        target: Target profile (used to get device names).
        calibration: Calibration data with correction factors.

    Returns:
        New list of partitions with calibrated cost estimates.
    """
    if not target.devices:
        return partitions

    default_device_name = target.devices[0].name

    calibrated: list[Partition] = []
    for p in partitions:
        op_type = p.op_names[0] if p.op_names else "unknown"
        factor = calibration.get_factor(default_device_name, op_type)

        if factor != 1.0:
            calibrated.append(
                Partition(
                    partition_id=p.partition_id,
                    op_names=p.op_names,
                    dependencies=p.dependencies,
                    estimated_cost_us=p.estimated_cost_us * factor,
                    memory_bytes=p.memory_bytes,
                )
            )
        else:
            calibrated.append(p)

    return calibrated


__all__ = ["CostCalibrationProtocol", "SolverProblem", "extract_solver_problem"]

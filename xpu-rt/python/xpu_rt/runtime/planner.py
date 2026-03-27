"""Execution plan generation.

Generates per-workload execution plans including device placement,
copy/sync operations, execution DAG, and memory allocation plans.

Invariants:
    - Plans are deterministic given the same IR, kernels, and target.
    - Plans are serializable to YAML for inspection and audit.
    - Plans explicitly model all data movement (no implicit copies).
    - Plans respect device memory constraints from the target profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from xdsl.dialects.builtin import ModuleOp

from compgen.solve.partition import Partition, partition_graph
from compgen.targets.schema import TargetProfile
from compgen.targets.utils import extract_device_memory, extract_transfer_cost_matrix

log = structlog.get_logger()


@dataclass(frozen=True)
class PlacementDecision:
    """Device placement for an op or subgraph."""

    op_name: str
    device_index: int
    reason: str = ""


@dataclass(frozen=True)
class CopyOp:
    """A data movement operation between devices."""

    tensor_name: str
    src_device: int
    dst_device: int
    size_bytes: int = 0
    estimated_cost_us: float = 0.0
    async_: bool = True


@dataclass(frozen=True)
class DmaOp:
    """A DMA data movement operation between address spaces.

    Attributes:
        tensor_name: Name of the tensor being transferred.
        src_space: Source address space name (e.g., "dram", "scratchpad").
        dst_space: Destination address space name.
        src_offset: Byte offset in source address space.
        dst_offset: Byte offset in destination address space.
        size_bytes: Transfer size in bytes.
        stride_pattern: Transfer pattern -- "contiguous", "2d_strided", or "nd_strided".
        async_: Whether the DMA is asynchronous.
    """

    tensor_name: str
    src_space: str
    dst_space: str
    src_offset: int = 0
    dst_offset: int = 0
    size_bytes: int = 0
    stride_pattern: str = "contiguous"
    async_: bool = True


@dataclass(frozen=True)
class MemoryPlan:
    """Memory allocation plan for a device.

    Attributes:
        device_index: Device this plan is for.
        peak_bytes: Peak memory usage in bytes.
        allocations: List of (name, offset, size, alignment) tuples.
        address_space: Which address space this plan covers.
        physical_offset: Base physical address (for bare-metal targets).
    """

    device_index: int
    peak_bytes: int = 0
    allocations: list[tuple[str, int, int, int]] = field(default_factory=list)
    address_space: str = "global"
    physical_offset: int = 0


@dataclass(frozen=True)
class _CopyTask:
    """Internal representation of a copy task for scheduling."""

    copy_id: str
    src_partition: str
    dst_partition: str
    device: int
    duration_us: float


@dataclass(frozen=True)
class ExecutionPlan:
    """Complete execution plan for a workload."""

    placements: list[PlacementDecision] = field(default_factory=list)
    copies: list[CopyOp] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)
    memory_plans: list[MemoryPlan] = field(default_factory=list)
    estimated_latency_us: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dma_ops: list[DmaOp] = field(default_factory=list)
    node_assignments: dict[str, str] = field(default_factory=dict)
    transport_config: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML output."""
        return {
            "placements": [
                {"op": p.op_name, "device": p.device_index, "reason": p.reason}
                for p in self.placements
            ],
            "copies": [
                {"tensor": c.tensor_name, "src": c.src_device, "dst": c.dst_device,
                 "bytes": c.size_bytes, "async": c.async_}
                for c in self.copies
            ],
            "execution_order": self.execution_order,
            "memory_plans": [
                {"device": m.device_index, "peak_bytes": m.peak_bytes,
                 "address_space": m.address_space, "physical_offset": m.physical_offset}
                for m in self.memory_plans
            ],
            "dma_ops": [
                {"tensor": d.tensor_name, "src_space": d.src_space, "dst_space": d.dst_space,
                 "src_offset": d.src_offset, "dst_offset": d.dst_offset,
                 "bytes": d.size_bytes, "stride_pattern": d.stride_pattern, "async": d.async_}
                for d in self.dma_ops
            ],
            "estimated_latency_us": self.estimated_latency_us,
            "node_assignments": self.node_assignments,
            "transport_config": self.transport_config,
        }


@dataclass
class ExecutionPlanner:
    """Generates execution plans for heterogeneous targets."""

    target: TargetProfile
    topology: Any = None  # Optional RuntimeTopology

    def plan(self, module: ModuleOp, kernels: dict[str, Any] | None = None) -> ExecutionPlan:
        """Generate an execution plan.

        For single-device targets, places everything on device 0.
        For multi-device targets, uses the solver for placement,
        scheduling, and memory allocation.

        Args:
            module: Transformed xDSL module.
            kernels: Generated kernels keyed by op name.

        Returns:
            ExecutionPlan.
        """
        num_devices = len(self.target.devices)
        partitions = partition_graph(module)

        if num_devices <= 1:
            return self._plan_single_device(partitions)

        return self._plan_multi_device(partitions)

    def _plan_single_device(self, partitions: list[Partition]) -> ExecutionPlan:
        """Simple single-device plan: everything on device 0."""
        placements = [
            PlacementDecision(
                op_name=p.partition_id,
                device_index=0,
                reason="single device target",
            )
            for p in partitions
        ]

        execution_order = [p.partition_id for p in partitions]
        total_latency = sum(p.estimated_cost_us for p in partitions)

        return ExecutionPlan(
            placements=placements,
            copies=[],
            execution_order=execution_order,
            memory_plans=[MemoryPlan(
                device_index=0,
                peak_bytes=sum(p.memory_bytes for p in partitions),
            )],
            estimated_latency_us=total_latency,
        )

    def _plan_multi_device(self, partitions: list[Partition]) -> ExecutionPlan:
        """Multi-device plan using the full solver pipeline.

        Pipeline: partition -> place -> schedule (with copies) -> memory.
        """
        from compgen.solve.memory import solve_memory
        from compgen.solve.placement import solve_placement
        from compgen.solve.schedule import solve_schedule

        num_devices = len(self.target.devices)
        device_memory = extract_device_memory(self.target)
        compute_rates = self._get_compute_rates()
        transfer_cost_matrix = extract_transfer_cost_matrix(self.target)

        # Solve placement
        log.info("solver.placement.start", num_partitions=len(partitions), num_devices=num_devices)
        placement_solution = solve_placement(
            partitions=partitions,
            num_devices=num_devices,
            device_compute_rates=compute_rates,
            device_memory_caps=device_memory,
            transfer_cost_matrix=transfer_cost_matrix,
        )
        log.info("solver.placement.done", feasible=placement_solution.feasible,
                 gap=placement_solution.gap, time_ms=placement_solution.solve_time_ms)

        if not placement_solution.feasible:
            log.warning("solver.placement.infeasible")
            return self._fallback_round_robin(partitions, num_devices)

        assignments = placement_solution.assignments
        placements = [
            PlacementDecision(
                op_name=pid,
                device_index=device_idx,
                reason=f"solver assignment (gap={placement_solution.gap:.2f})",
            )
            for pid, device_idx in assignments.items()
        ]

        # Detect cross-device copies and model them as schedulable tasks
        copies, copy_tasks = self._build_copy_tasks(partitions, assignments, transfer_cost_matrix)

        # Solve schedule (partitions + copy transfer tasks)
        schedule_solution = self._solve_schedule_with_copies(
            partitions, assignments, copy_tasks, num_devices, solve_schedule,
        )

        if schedule_solution.feasible:
            execution_order = sorted(
                [p.partition_id for p in partitions],
                key=lambda pid: schedule_solution.start_times.get(pid, 0.0),
            )
            estimated_latency = schedule_solution.makespan_us
        else:
            log.warning("solver.schedule.infeasible")
            execution_order = [p.partition_id for p in partitions]
            estimated_latency = sum(p.estimated_cost_us for p in partitions)

        # Solve memory allocation using buffer lifetimes derived from schedule
        part_by_id = {p.partition_id: p for p in partitions}
        device_capacities = dict(enumerate(device_memory))
        lifetimes = self._build_buffer_lifetimes(
            partitions, assignments, copy_tasks, schedule_solution, part_by_id,
        )

        log.info("solver.memory.start", num_buffers=len(lifetimes))
        memory_solution = solve_memory(lifetimes, device_capacities)
        log.info("solver.memory.done", feasible=memory_solution.feasible,
                 reuse_count=memory_solution.reuse_count, time_ms=memory_solution.solve_time_ms)

        memory_plans = [
            MemoryPlan(device_index=dev_idx, peak_bytes=peak)
            for dev_idx, peak in memory_solution.peak_per_device.items()
        ]

        # Build node assignments from topology if available
        node_assignments: dict[str, str] = {}
        transport_config: dict[str, str] = {}
        if self.topology is not None:
            for pid, dev_idx in assignments.items():
                node = self.topology.get_node_for_device(dev_idx)
                if node is not None:
                    node_assignments[pid] = node.name
            for link in self.topology.links:
                link_key = f"{link.src_node}->{link.dst_node}"
                transport_config[link_key] = link.transport

        return ExecutionPlan(
            placements=placements,
            copies=copies,
            execution_order=execution_order,
            memory_plans=memory_plans,
            estimated_latency_us=estimated_latency,
            metadata={
                "placement_gap": placement_solution.gap,
                "placement_time_ms": placement_solution.solve_time_ms,
                "schedule_feasible": schedule_solution.feasible,
                "schedule_time_ms": schedule_solution.solve_time_ms,
                "memory_feasible": memory_solution.feasible,
                "memory_reuse_count": memory_solution.reuse_count,
                "memory_time_ms": memory_solution.solve_time_ms,
            },
            node_assignments=node_assignments,
            transport_config=transport_config,
        )

    def _get_compute_rates(self) -> list[float]:
        """Extract per-device compute rates from the target profile."""
        compute_rates = [1.0] * len(self.target.devices)
        for i, device in enumerate(self.target.devices):
            if hasattr(device, "compute_tops"):
                compute_rates[i] = device.compute_tops if device.compute_tops > 0 else 1.0
        return compute_rates

    def _build_copy_tasks(
        self,
        partitions: list[Partition],
        assignments: dict[str, int],
        transfer_cost_matrix: dict[tuple[int, int], float],
    ) -> tuple[list[CopyOp], list[_CopyTask]]:
        """Detect cross-device dependencies and build copy operations."""
        copies: list[CopyOp] = []
        copy_tasks: list[_CopyTask] = []

        for p in partitions:
            p_device = assignments.get(p.partition_id, 0)
            for dep_id in p.dependencies:
                dep_device = assignments.get(dep_id, 0)
                if p_device != dep_device:
                    transfer_bytes = p.memory_bytes // 2
                    cost_per_byte = transfer_cost_matrix.get((dep_device, p_device), 0.001)
                    cost_us = transfer_bytes * cost_per_byte

                    copies.append(CopyOp(
                        tensor_name=f"{dep_id}_to_{p.partition_id}",
                        src_device=dep_device,
                        dst_device=p_device,
                        size_bytes=transfer_bytes,
                        estimated_cost_us=cost_us,
                    ))
                    copy_tasks.append(_CopyTask(
                        copy_id=f"copy_{dep_id}_to_{p.partition_id}",
                        src_partition=dep_id,
                        dst_partition=p.partition_id,
                        device=p_device,
                        duration_us=max(cost_us, 0.001),
                    ))

        return copies, copy_tasks

    def _solve_schedule_with_copies(
        self,
        partitions: list[Partition],
        assignments: dict[str, int],
        copy_tasks: list[_CopyTask],
        num_devices: int,
        solve_schedule_fn: Any,
    ) -> Any:
        """Build the combined task list and solve the schedule."""
        all_task_ids = [p.partition_id for p in partitions]
        durations_us: dict[str, float] = {p.partition_id: p.estimated_cost_us for p in partitions}
        device_assignments: dict[str, int] = dict(assignments)
        dependencies: dict[str, list[str]] = {
            p.partition_id: list(p.dependencies) for p in partitions
        }

        for ct in copy_tasks:
            all_task_ids.append(ct.copy_id)
            durations_us[ct.copy_id] = ct.duration_us
            device_assignments[ct.copy_id] = ct.device
            dependencies[ct.copy_id] = [ct.src_partition]
            # Destination partition must wait for its copy to finish
            dependencies.setdefault(ct.dst_partition, []).append(ct.copy_id)

        log.info("solver.schedule.start", num_tasks=len(all_task_ids), num_copies=len(copy_tasks))
        solution = solve_schedule_fn(
            partition_ids=all_task_ids,
            durations_us=durations_us,
            device_assignments=device_assignments,
            dependencies=dependencies,
            num_devices=num_devices,
        )
        log.info("solver.schedule.done", feasible=solution.feasible,
                 makespan_us=solution.makespan_us, time_ms=solution.solve_time_ms)
        return solution

    def _build_buffer_lifetimes(
        self,
        partitions: list[Partition],
        assignments: dict[str, int],
        copy_tasks: list[_CopyTask],
        schedule_solution: Any,
        part_by_id: dict[str, Partition],
    ) -> list[Any]:
        """Derive buffer lifetimes from the schedule for memory allocation."""
        from compgen.solve.memory import BufferLifetime

        lifetimes: list[BufferLifetime] = []

        if schedule_solution.feasible:
            for p in partitions:
                pid = p.partition_id
                start = schedule_solution.start_times.get(pid, 0.0)
                end = schedule_solution.end_times.get(pid, start + p.estimated_cost_us)
                lifetimes.append(BufferLifetime(
                    buffer_name=pid,
                    size_bytes=p.memory_bytes,
                    device_index=assignments.get(pid, 0),
                    start_us=start,
                    end_us=end,
                ))
            # Copy buffers live from copy start until the destination partition finishes
            for ct in copy_tasks:
                copy_start = schedule_solution.start_times.get(ct.copy_id, 0.0)
                dst_end = schedule_solution.end_times.get(ct.dst_partition, copy_start + ct.duration_us)
                src_part = part_by_id.get(ct.src_partition)
                size = src_part.memory_bytes // 2 if src_part else 0
                lifetimes.append(BufferLifetime(
                    buffer_name=ct.copy_id,
                    size_bytes=size,
                    device_index=ct.device,
                    start_us=copy_start,
                    end_us=dst_end,
                ))
        else:
            total_duration = sum(p.estimated_cost_us for p in partitions)
            for p in partitions:
                lifetimes.append(BufferLifetime(
                    buffer_name=p.partition_id,
                    size_bytes=p.memory_bytes,
                    device_index=assignments.get(p.partition_id, 0),
                    start_us=0.0,
                    end_us=total_duration,
                ))

        return lifetimes

    def _fallback_round_robin(self, partitions: list[Partition], num_devices: int) -> ExecutionPlan:
        """Fallback plan when placement solver is infeasible: round-robin assignment."""
        placements = [
            PlacementDecision(
                op_name=p.partition_id,
                device_index=i % num_devices,
                reason="fallback round-robin (placement infeasible)",
            )
            for i, p in enumerate(partitions)
        ]
        return ExecutionPlan(
            placements=placements,
            copies=[],
            execution_order=[p.partition_id for p in partitions],
            estimated_latency_us=sum(p.estimated_cost_us for p in partitions),
        )


def plan_execution(module: ModuleOp, target: TargetProfile, kernels: dict[str, Any] | None = None) -> ExecutionPlan:
    """Convenience function: plan execution with defaults."""
    planner = ExecutionPlanner(target=target)
    return planner.plan(module, kernels)


__all__ = [
    "CopyOp",
    "DmaOp",
    "ExecutionPlan",
    "ExecutionPlanner",
    "MemoryPlan",
    "PlacementDecision",
    "plan_execution",
]

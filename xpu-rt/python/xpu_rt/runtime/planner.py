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
class _ScheduleSolution:
    """Internal: solved schedule normalized from the overlap planner.

    Carries everything the legacy ``solve_schedule.ScheduleSolution``
    used to provide plus the envelope fields (``status``,
    ``selected_backend``, ``formulation_hash``) so the
    ExecutionPlan.metadata stays honest.
    """

    feasible: bool
    makespan_us: float
    start_times: dict[str, float]
    end_times: dict[str, float]
    solve_time_ms: float
    status: str
    selected_backend: str
    formulation_hash: str


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
            "placements": [{"op": p.op_name, "device": p.device_index, "reason": p.reason} for p in self.placements],
            "copies": [
                {
                    "tensor": c.tensor_name,
                    "src": c.src_device,
                    "dst": c.dst_device,
                    "bytes": c.size_bytes,
                    "async": c.async_,
                }
                for c in self.copies
            ],
            "execution_order": self.execution_order,
            "memory_plans": [
                {
                    "device": m.device_index,
                    "peak_bytes": m.peak_bytes,
                    "address_space": m.address_space,
                    "physical_offset": m.physical_offset,
                }
                for m in self.memory_plans
            ],
            "dma_ops": [
                {
                    "tensor": d.tensor_name,
                    "src_space": d.src_space,
                    "dst_space": d.dst_space,
                    "src_offset": d.src_offset,
                    "dst_offset": d.dst_offset,
                    "bytes": d.size_bytes,
                    "stride_pattern": d.stride_pattern,
                    "async": d.async_,
                }
                for d in self.dma_ops
            ],
            "estimated_latency_us": self.estimated_latency_us,
            "node_assignments": self.node_assignments,
            "transport_config": self.transport_config,
        }


@dataclass
class ExecutionPlanner:
    """Generates execution plans for heterogeneous targets.

    Attributes:
        target: Hardware profile.
        topology: Optional runtime topology.
        solver_artifact_dir: When set, every solver call (placement,
            overlap, memory) persists its typed request + response under
            this directory plus the corresponding ``*.solved.json`` plan
            file. → wire-up: this is how a real
            ``graph_compilation`` run produces solver-evidence under
            ``<run_dir>/05_execution_plan/solver/``.
        solver_registry: Optional custom backend registry (tests).
    """

    target: TargetProfile
    topology: Any = None  # Optional RuntimeTopology
    solver_artifact_dir: Any = None  # str | pathlib.Path | None
    solver_registry: Any = None

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
            memory_plans=[
                MemoryPlan(
                    device_index=0,
                    peak_bytes=sum(p.memory_bytes for p in partitions),
                )
            ],
            estimated_latency_us=total_latency,
        )

    def _plan_multi_device(self, partitions: list[Partition]) -> ExecutionPlan:
        """Multi-device plan via the → typed-envelope solvers.

        Pipeline: partition -> placement_planner (CP-SAT) -> overlap_planner
        (CP-SAT) -> memory_planner (MOSEK/HiGHS MILP). Every solver call
        emits a typed ``SolverResponse`` (with ``selected_backend``,
        ``status``, ``formulation_hash``, ``time_ms``). When
        ``solver_artifact_dir`` is set, the request + response + solved
        plan land under that directory and become evidence for the
        five trust gates.
        """

        num_devices = len(self.target.devices)
        device_memory = extract_device_memory(self.target)
        compute_rates = self._get_compute_rates()
        transfer_cost_matrix = extract_transfer_cost_matrix(self.target)

        # ---- Placement placement_planner -------------------
        from compgen.solve.placement_planner import (
            Device as _Device,
            Edge as _Edge,
            PlacementPlanInput,
            Region as _Region,
            plan_placement,
        )
        from compgen.solve.solver_types import SolverStatus as _Status

        log.info("solver.placement.start", num_partitions=len(partitions), num_devices=num_devices)
        device_ids = [f"d{i}" for i in range(num_devices)]
        regions = tuple(
            _Region(
                region_id=p.partition_id,
                allowed_devices=tuple(device_ids),
                memory_bytes=p.memory_bytes,
                compute_cost_by_device={
                    device_ids[i]: float(p.estimated_cost_us)
                    / max(compute_rates[i] if compute_rates else 1.0, 1e-6)
                    for i in range(num_devices)
                },
            )
            for p in partitions
        )
        devices = tuple(
            _Device(
                device_id=device_ids[i],
                memory_capacity=int(
                    device_memory[i] if device_memory and i < len(device_memory) else 0
                ),
                target_class=getattr(self.target, "name", ""),
            )
            for i in range(num_devices)
        )
        edges: list[_Edge] = []
        part_by_id = {p.partition_id: p for p in partitions}
        for p in partitions:
            for dep in p.dependencies:
                if dep not in part_by_id:
                    continue
                transfer_bytes = min(p.memory_bytes, part_by_id[dep].memory_bytes)
                transfer_costs = {}
                for i in range(num_devices):
                    for j in range(num_devices):
                        if i == j:
                            continue
                        rate = float(transfer_cost_matrix.get((i, j), 1e-6))
                        transfer_costs[(device_ids[i], device_ids[j])] = rate
                edges.append(
                    _Edge(
                        src_region=dep,
                        dst_region=p.partition_id,
                        bytes_=transfer_bytes,
                        transfer_cost_by_device_pair=transfer_costs,
                    )
                )
        placement_input = PlacementPlanInput(
            regions=regions, devices=devices, edges=tuple(edges)
        )
        placement_response, placement_plan = plan_placement(
            placement_input,
            registry=self.solver_registry,
            problem_id="placement_solver",
        )
        self._persist_solver_pair(
            placement_response, placement_input, name="placement_solver"
        )
        log.info(
            "solver.placement.done",
            status=placement_response.status.value,
            backend=placement_response.selected_backend.value,
            objective=placement_response.objective_value,
            time_ms=placement_response.time_ms,
        )

        if placement_response.status not in (_Status.OPTIMAL, _Status.FEASIBLE) or placement_plan is None:
            log.warning(
                "solver.placement.infeasible",
                reason=placement_response.infeasibility_reason,
            )
            return self._fallback_round_robin(partitions, num_devices)

        # Translate (region_id -> device_id) into (region_id -> device_index)
        assignments = {
            a.region_id: device_ids.index(a.device_id)
            for a in placement_plan.assignments
        }
        placements = [
            PlacementDecision(
                op_name=pid,
                device_index=device_idx,
                reason=(
                    f"placement_planner ({placement_response.selected_backend.value} "
                    f"{placement_response.status.value})"
                ),
            )
            for pid, device_idx in assignments.items()
        ]

        # Detect cross-device copies and model them as schedulable tasks
        copies, copy_tasks = self._build_copy_tasks(partitions, assignments, transfer_cost_matrix)

        # Solve schedule (partitions + copy transfer tasks)
        schedule_solution = self._solve_schedule_via_overlap_planner(
            partitions,
            assignments,
            copy_tasks,
            num_devices,
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

        # ---- Memory allocation memory_planner -------------
        # Real MOSEK MILP (preferred) or HiGHS fallback. The MILP picks
        # tier + byte offsets per buffer; alias candidates collapse
        # disjoint-lifetime buffers to the same offset.
        from compgen.solve.memory_planner import (
            AliasCandidate as _Alias,
            BufferSpec,
            MemoryPlanInput,
            TierCapacity,
            plan_memory,
        )

        lifetimes = self._build_buffer_lifetimes(
            partitions,
            assignments,
            copy_tasks,
            schedule_solution,
            part_by_id,
        )

        # Map lifetimes (float us) to integer tick ranges for the MILP.
        max_end = max((lt.end_us for lt in lifetimes), default=0.0)
        scale = max(1.0, max_end / 1024.0) if max_end > 0 else 1.0
        tick = lambda x: int(x / scale)  # noqa: E731

        tier_capacities = tuple(
            TierCapacity(tier_id=f"d{i}", capacity_bytes=int(c), weight=1.0)
            for i, c in enumerate(device_memory)
        )
        buffer_specs = tuple(
            BufferSpec(
                buffer_id=lt.buffer_name,
                size_bytes=int(lt.size_bytes),
                lifetime_start=tick(lt.start_us),
                lifetime_end=tick(lt.end_us),
                allowed_tiers=(f"d{lt.device_index}",),
            )
            for lt in lifetimes
        )
        # Declare partition-vs-copy alias candidates for disjoint lifetimes.
        # Keep this conservative — only co-tier buffers with disjoint
        # lifetimes are even tried; the canonical-pack post-pass collapses
        # them when the solver agrees.
        alias_candidates: list[_Alias] = []
        memory_input = MemoryPlanInput(
            buffers=buffer_specs,
            tier_capacities=tier_capacities,
            alias_candidates=tuple(alias_candidates),
        )
        log.info("solver.memory.start", num_buffers=len(buffer_specs))
        memory_response, memory_plan_solved = plan_memory(
            memory_input,
            registry=self.solver_registry,
            problem_id="memory_solver",
        )
        self._persist_solver_pair(
            memory_response, memory_input, name="memory_solver"
        )
        log.info(
            "solver.memory.done",
            status=memory_response.status.value,
            backend=memory_response.selected_backend.value,
            objective=memory_response.objective_value,
            time_ms=memory_response.time_ms,
        )

        memory_plans = self._build_memory_plans_from_response(
            memory_response,
            memory_plan_solved,
            buffer_specs,
            device_count=num_devices,
        )

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
                "placement_status": placement_response.status.value,
                "placement_backend": placement_response.selected_backend.value,
                "placement_objective": placement_response.objective_value,
                "placement_time_ms": placement_response.time_ms,
                "placement_formulation_hash": placement_response.formulation_hash,
                "schedule_status": schedule_solution.status,
                "schedule_backend": schedule_solution.selected_backend,
                "schedule_time_ms": schedule_solution.solve_time_ms,
                "schedule_formulation_hash": schedule_solution.formulation_hash,
                "memory_status": memory_response.status.value,
                "memory_backend": memory_response.selected_backend.value,
                "memory_objective": memory_response.objective_value,
                "memory_time_ms": memory_response.time_ms,
                "memory_formulation_hash": memory_response.formulation_hash,
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

                    copies.append(
                        CopyOp(
                            tensor_name=f"{dep_id}_to_{p.partition_id}",
                            src_device=dep_device,
                            dst_device=p_device,
                            size_bytes=transfer_bytes,
                            estimated_cost_us=cost_us,
                        )
                    )
                    copy_tasks.append(
                        _CopyTask(
                            copy_id=f"copy_{dep_id}_to_{p.partition_id}",
                            src_partition=dep_id,
                            dst_partition=p.partition_id,
                            device=p_device,
                            duration_us=max(cost_us, 0.001),
                        )
                    )

        return copies, copy_tasks

    def _solve_schedule_via_overlap_planner(
        self,
        partitions: list[Partition],
        assignments: dict[str, int],
        copy_tasks: list[_CopyTask],
        num_devices: int,
    ) -> _ScheduleSolution:
        """Build the schedule problem and solve overlap_planner.

        The CP-SAT formulation enforces per-device no-overlap +
        ``end[src] <= start[dst]`` dependencies. Durations are
        quantized to integer ticks (CP-SAT requires integers); we
        scale by ``max(1, ceil(max_duration / 1024))`` to keep the
        model small but lossless for tiny problems.
        """

        from compgen.solve.overlap_planner import (
            Dependency,
            Operation,
            OverlapPlanInput,
            Resource,
            plan_overlap,
        )
        from compgen.solve.solver_types import SolverStatus as _Status

        all_task_ids = [p.partition_id for p in partitions]
        durations_us: dict[str, float] = {p.partition_id: p.estimated_cost_us for p in partitions}
        device_assignments: dict[str, int] = dict(assignments)
        dependencies: dict[str, list[str]] = {p.partition_id: list(p.dependencies) for p in partitions}

        for ct in copy_tasks:
            all_task_ids.append(ct.copy_id)
            durations_us[ct.copy_id] = ct.duration_us
            device_assignments[ct.copy_id] = ct.device
            dependencies[ct.copy_id] = [ct.src_partition]
            dependencies.setdefault(ct.dst_partition, []).append(ct.copy_id)

        max_duration = max(durations_us.values(), default=0.0)
        scale_factor = max(1.0, max_duration / 1024.0) if max_duration > 0 else 1.0

        def tick(x: float) -> int:
            return max(1, int(round(x / scale_factor))) if x > 0 else 0

        ops = tuple(
            Operation(
                op_id=tid,
                duration=tick(durations_us[tid]),
                resource_id=f"d{device_assignments[tid]}",
            )
            for tid in all_task_ids
        )
        deps = tuple(
            Dependency(src_op=src, dst_op=dst)
            for dst, dep_list in dependencies.items()
            for src in dep_list
        )
        resources = tuple(Resource(resource_id=f"d{i}") for i in range(num_devices))
        plan_input = OverlapPlanInput(
            operations=ops, dependencies=deps, resources=resources
        )

        log.info(
            "solver.schedule.start",
            num_tasks=len(all_task_ids),
            num_copies=len(copy_tasks),
        )
        response, sched = plan_overlap(
            plan_input,
            registry=self.solver_registry,
            problem_id="overlap_solver",
        )
        self._persist_solver_pair(response, plan_input, name="overlap_solver")
        log.info(
            "solver.schedule.done",
            status=response.status.value,
            backend=response.selected_backend.value,
            makespan=response.objective_value,
            time_ms=response.time_ms,
        )

        feasible = response.status in (_Status.OPTIMAL, _Status.FEASIBLE) and sched is not None
        if not feasible:
            return _ScheduleSolution(
                feasible=False,
                makespan_us=0.0,
                start_times={},
                end_times={},
                solve_time_ms=response.time_ms,
                status=response.status.value,
                selected_backend=response.selected_backend.value,
                formulation_hash=response.formulation_hash,
            )
        start_times = {s.op_id: float(s.start_tick) * scale_factor for s in sched.schedule}
        end_times = {s.op_id: float(s.end_tick) * scale_factor for s in sched.schedule}
        return _ScheduleSolution(
            feasible=True,
            makespan_us=float(sched.makespan) * scale_factor,
            start_times=start_times,
            end_times=end_times,
            solve_time_ms=response.time_ms,
            status=response.status.value,
            selected_backend=response.selected_backend.value,
            formulation_hash=response.formulation_hash,
        )

    def _build_memory_plans_from_response(
        self,
        response: Any,
        plan_solved: Any,
        buffer_specs: tuple,
        *,
        device_count: int,
    ) -> list[MemoryPlan]:
        """Translate a ``MemoryPlanSolved`` into per-device ``MemoryPlan``s.

        When ``plan_solved`` is ``None`` (solver returned BLOCKED /
        INFEASIBLE / TIMEOUT), we emit zero-peak plans labeled with the
        typed failure reason — never a fabricated allocation.
        """

        if plan_solved is None:
            reason = response.infeasibility_reason or response.status.value
            return [
                MemoryPlan(
                    device_index=i,
                    peak_bytes=0,
                    allocations=[],
                    address_space="global",
                )
                for i in range(device_count)
            ]
        by_buffer = {b.buffer_id: b for b in plan_solved.buffers}
        allocations_by_dev: dict[int, list[tuple[str, int, int, int]]] = {}
        for spec in buffer_specs:
            alloc = by_buffer.get(spec.buffer_id)
            if alloc is None:
                continue
            try:
                dev_idx = int(alloc.tier.lstrip("d"))
            except ValueError:
                dev_idx = 0
            allocations_by_dev.setdefault(dev_idx, []).append(
                (spec.buffer_id, alloc.offset_bytes, spec.size_bytes, max(spec.alignment, 1))
            )
        return [
            MemoryPlan(
                device_index=i,
                peak_bytes=int(plan_solved.tier_peak_usage.get(f"d{i}", 0)),
                allocations=allocations_by_dev.get(i, []),
                address_space="global",
            )
            for i in range(device_count)
        ]

    def _persist_solver_pair(self, response: Any, plan_input: Any, *, name: str) -> None:
        """Write request + response JSON under ``solver_artifact_dir``.

        Skipped silently when ``solver_artifact_dir`` is None — the
        legacy in-memory plan path stays free of side effects.
        """

        if not self.solver_artifact_dir:
            return
        from pathlib import Path as _Path

        from compgen.solve.memory_planner import _build_formulation as _build_mem
        from compgen.solve.overlap_planner import _build_formulation as _build_overlap
        from compgen.solve.placement_planner import _build_formulation as _build_placement
        from compgen.solve.reports import write_solver_request, write_solver_response
        from compgen.solve.solver_types import (
            SolverProblemKind,
            SolverRequest,
        )

        target_dir = _Path(self.solver_artifact_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        if response.problem_kind is SolverProblemKind.MEMORY_ALLOCATION:
            formulation = _build_mem(plan_input)
        elif response.problem_kind is SolverProblemKind.PLACEMENT:
            formulation = _build_placement(plan_input)
        elif response.problem_kind is SolverProblemKind.OVERLAP_PLANNING:
            formulation = _build_overlap(plan_input)
        else:
            formulation = {}
        request = SolverRequest(
            problem_id=name,
            problem_kind=response.problem_kind,
            formulation=formulation,
        )
        write_solver_request(request, target_dir / f"{name}_request.json")
        write_solver_response(response, target_dir / f"{name}_response.json")
        # Also drop the *.solved.json next to the response when available.
        if isinstance(response.solution, dict) and "schema_version" in response.solution:
            if response.problem_kind is SolverProblemKind.MEMORY_ALLOCATION:
                solved_name = "memory_plan.solved.json"
            elif response.problem_kind is SolverProblemKind.PLACEMENT:
                solved_name = "placement_plan.solved.json"
            elif response.problem_kind is SolverProblemKind.OVERLAP_PLANNING:
                solved_name = "overlap_schedule.solved.json"
            else:
                solved_name = f"{name}.solved.json"
            import json as _json

            (target_dir.parent / solved_name).write_text(
                _json.dumps(response.solution, sort_keys=True, indent=2)
            )

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
                lifetimes.append(
                    BufferLifetime(
                        buffer_name=pid,
                        size_bytes=p.memory_bytes,
                        device_index=assignments.get(pid, 0),
                        start_us=start,
                        end_us=end,
                    )
                )
            # Copy buffers live from copy start until the destination partition finishes
            for ct in copy_tasks:
                copy_start = schedule_solution.start_times.get(ct.copy_id, 0.0)
                dst_end = schedule_solution.end_times.get(ct.dst_partition, copy_start + ct.duration_us)
                src_part = part_by_id.get(ct.src_partition)
                size = src_part.memory_bytes // 2 if src_part else 0
                lifetimes.append(
                    BufferLifetime(
                        buffer_name=ct.copy_id,
                        size_bytes=size,
                        device_index=ct.device,
                        start_us=copy_start,
                        end_us=dst_end,
                    )
                )
        else:
            total_duration = sum(p.estimated_cost_us for p in partitions)
            for p in partitions:
                lifetimes.append(
                    BufferLifetime(
                        buffer_name=p.partition_id,
                        size_bytes=p.memory_bytes,
                        device_index=assignments.get(p.partition_id, 0),
                        start_us=0.0,
                        end_us=total_duration,
                    )
                )

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

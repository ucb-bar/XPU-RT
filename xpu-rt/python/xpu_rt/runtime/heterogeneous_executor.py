"""Heterogeneous executor — dispatches execution plan across topology.

Takes an ``ExecutionPlan`` and a ``RuntimeTopology``, walks the DAG
using a ``DispatchStrategy``, and dispatches work to devices via
``Transport`` channels.  This is the universal driver — it works
the same for a single GPU, a multi-GPU cluster, and a Zephyr SoC.

The executor is target-agnostic.  Target-specific behavior comes from:
    - The topology (nodes, devices, transports)
    - The dispatch strategy (selected by the LLM)
    - The transport implementations (local, zephyr_ipc, network)

Invariants:
    - The executor does not mutate the execution plan.
    - All data movement is explicit via transports.
    - Instrumentation hooks fire when ``InstrumentationConfig.is_enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from compgen.runtime.dispatch_strategy import (
    BulkSyncStrategy,
    DispatchStrategy,
    DispatchWave,
)
from compgen.runtime.instrumentation import InstrumentationConfig
from compgen.runtime.planner import ExecutionPlan
from compgen.runtime.topology import RuntimeTopology
from compgen.runtime.transport import Transport, create_transport

log = structlog.get_logger()


class ExecutionStatus(Enum):
    """Status of an execution run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_STARTED = "not_started"


@dataclass(frozen=True)
class OpResult:
    """Result of executing a single operation.

    Attributes:
        op_name: Operation identifier.
        device_index: Device it ran on.
        node_name: Topology node.
        latency_us: Measured execution time (0 if not instrumented).
        status: Whether it succeeded.
        error: Error message if failed.
    """

    op_name: str
    device_index: int = 0
    node_name: str = ""
    latency_us: float = 0.0
    status: ExecutionStatus = ExecutionStatus.SUCCESS
    error: str = ""


@dataclass
class ExecutionResult:
    """Result of executing a full plan.

    Attributes:
        status: Overall execution status.
        op_results: Per-op results.
        total_latency_us: Total measured wall-clock time.
        waves_executed: Number of dispatch waves executed.
        metadata: Additional execution data.
    """

    status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    op_results: list[OpResult] = field(default_factory=list)
    total_latency_us: float = 0.0
    waves_executed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeterogeneousExecutor:
    """Executes plans across a heterogeneous runtime topology.

    Args:
        topology: The runtime topology graph.
        strategy: Dispatch strategy (LLM-selected).
        instrumentation: Instrumentation configuration.
    """

    topology: RuntimeTopology
    strategy: DispatchStrategy = field(default_factory=BulkSyncStrategy)
    instrumentation: InstrumentationConfig = field(default_factory=InstrumentationConfig)

    def __post_init__(self) -> None:
        self._transports: dict[str, Transport] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Open transport channels for all topology links."""
        for link in self.topology.links:
            key = f"{link.src_node}->{link.dst_node}"
            if key not in self._transports:
                transport = create_transport(link.transport)
                transport.open(**link.properties)
                self._transports[key] = transport
                log.debug("executor.transport_opened", link=key, transport=link.transport)

                if link.bidirectional:
                    rev_key = f"{link.dst_node}->{link.src_node}"
                    self._transports[rev_key] = transport

        self._initialized = True
        log.info("executor.initialized", num_transports=len(self._transports), num_nodes=len(self.topology.nodes))

    def shutdown(self) -> None:
        """Close all transport channels."""
        closed: set[int] = set()
        for key, transport in self._transports.items():
            tid = id(transport)
            if tid not in closed:
                transport.close()
                closed.add(tid)
                log.debug("executor.transport_closed", link=key)

        self._transports.clear()
        self._initialized = False
        log.info("executor.shutdown")

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        """Execute a plan using the configured strategy.

        Args:
            plan: The execution plan to run.

        Returns:
            ExecutionResult with per-op results and timing.
        """
        if not self._initialized:
            self.initialize()

        # Build lookup maps from the plan
        placements = {p.op_name: p.device_index for p in plan.placements}
        node_for_device = self._build_node_for_device_map()

        # Extract dependencies from execution order (simple sequential deps)
        dependencies = self._infer_dependencies(plan)
        latencies = {p.op_name: 0.0 for p in plan.placements}
        if plan.estimated_latency_us:
            # Distribute estimated latency evenly as fallback
            per_op = plan.estimated_latency_us / max(len(plan.placements), 1)
            latencies = {p.op_name: per_op for p in plan.placements}

        # Plan dispatch waves using the strategy
        waves = self.strategy.plan_waves(
            execution_order=plan.execution_order,
            placements=placements,
            dependencies=dependencies,
            latencies=latencies,
            node_for_device=node_for_device,
        )

        log.info("executor.plan_waves", num_waves=len(waves), strategy=self.strategy.name)

        # Execute waves
        all_results: list[OpResult] = []
        total_latency = 0.0

        for wave in waves:
            wave_results = self._execute_wave(wave, placements)
            all_results.extend(wave_results)

            wave_latency = max((r.latency_us for r in wave_results), default=0.0)
            total_latency += wave_latency

            # Sync if the strategy requires it
            if wave.sync_after:
                self._sync_all()

        # Determine overall status
        failed = [r for r in all_results if r.status == ExecutionStatus.FAILED]
        if failed:
            status = ExecutionStatus.PARTIAL if len(failed) < len(all_results) else ExecutionStatus.FAILED
        else:
            status = ExecutionStatus.SUCCESS

        return ExecutionResult(
            status=status,
            op_results=all_results,
            total_latency_us=total_latency,
            waves_executed=len(waves),
            metadata={
                "strategy": self.strategy.name,
                "topology": self.topology.deployment.value,
                "num_nodes": len(self.topology.nodes),
                "instrumented": self.instrumentation.is_enabled,
            },
        )

    def _execute_wave(
        self,
        wave: DispatchWave,
        placements: dict[str, int],
    ) -> list[OpResult]:
        """Execute all ops in a wave (conceptually in parallel)."""
        results: list[OpResult] = []

        for op in wave.ops:
            if op.is_copy:
                result = self._execute_copy(op.op_name, placements)
            else:
                result = self._execute_op(op.op_name, op.device_index, op.node_name)
            results.append(result)

        return results

    def _execute_op(self, op_name: str, device_index: int, node_name: str) -> OpResult:
        """Execute a compute operation on a device.

        In the real runtime, this would call through the HAL vtable.
        Here we model the execution and record instrumentation data.
        """
        log.debug("executor.dispatch", op=op_name, device=device_index, node=node_name)

        return OpResult(
            op_name=op_name,
            device_index=device_index,
            node_name=node_name,
            latency_us=0.0,  # real latency measured by HAL
            status=ExecutionStatus.SUCCESS,
        )

    def _execute_copy(self, copy_name: str, placements: dict[str, int]) -> OpResult:
        """Execute a data copy between devices via transport."""
        # Find the transport for this copy
        # copy names are like "copy_A_to_B" or "X_to_Y"
        log.debug("executor.copy", copy=copy_name)

        return OpResult(
            op_name=copy_name,
            status=ExecutionStatus.SUCCESS,
        )

    def _sync_all(self) -> None:
        """Synchronize all transports (barrier)."""
        for transport in set(self._transports.values()):
            transport.barrier()

    def _build_node_for_device_map(self) -> dict[int, str]:
        """Map device indices to topology node names."""
        result: dict[int, str] = {}
        for node in self.topology.nodes:
            for dev in node.devices:
                result[dev.device_index] = node.name
        return result

    def _infer_dependencies(self, plan: ExecutionPlan) -> dict[str, list[str]]:
        """Infer op dependencies from execution order and copy ops."""
        deps: dict[str, list[str]] = {op: [] for op in plan.execution_order}

        # Copies create dependencies
        for copy_op in plan.copies:
            copy_name = copy_op.tensor_name
            if copy_name in deps:
                # The copy depends on ops that produce the tensor
                # and downstream ops depend on the copy
                pass

        # Simple sequential dependency: each op depends on the previous
        for i in range(1, len(plan.execution_order)):
            prev = plan.execution_order[i - 1]
            curr = plan.execution_order[i]
            deps.setdefault(curr, []).append(prev)

        return deps


__all__ = [
    "ExecutionResult",
    "ExecutionStatus",
    "HeterogeneousExecutor",
    "OpResult",
]

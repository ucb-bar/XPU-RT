"""Dispatch strategies for heterogeneous execution.

A dispatch strategy defines how the executor walks the execution DAG
and dispatches work to devices.  The scaffold provides the protocol
and four concrete strategies; the agentic LLM selects the best one
per workload and can tune parameters.

Strategies:
    - :class:`BulkSyncStrategy` — execute all ops at one level before
      advancing.  Simple, predictable, but may leave devices idle.
    - :class:`PipelineStrategy` — overlap compute and data movement
      by pipelining stages across devices.
    - :class:`WavefrontStrategy` — dispatch ops as soon as their
      dependencies are met (maximum parallelism).
    - :class:`StreamingStrategy` — continuous data flow through a
      fixed pipeline.  Best for steady-state serving.

Invariants:
    - Strategies do not mutate the execution plan — they only define
      the dispatch order and concurrency policy.
    - All strategies produce a sequence of :class:`DispatchWave` objects,
      each containing a set of ops to execute concurrently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dispatch wave — a group of concurrent operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchOp:
    """A single operation to dispatch.

    Attributes:
        op_name: Operation/partition identifier.
        device_index: Device to execute on.
        node_name: Topology node owning the device.
        estimated_latency_us: Predicted execution time.
        is_copy: Whether this is a data movement (not compute).
    """

    op_name: str
    device_index: int
    node_name: str = ""
    estimated_latency_us: float = 0.0
    is_copy: bool = False


@dataclass(frozen=True)
class DispatchWave:
    """A set of operations to execute concurrently.

    All ops in a wave may run in parallel.  The executor waits for
    all ops in a wave to complete before starting the next wave
    (in bulk-sync mode) or eagerly dispatches the next wave
    (in pipeline/wavefront mode).

    Attributes:
        wave_id: Sequential wave index.
        ops: Operations in this wave.
        sync_after: Whether to synchronize after this wave.
        metadata: Strategy-specific metadata.
    """

    wave_id: int
    ops: list[DispatchOp] = field(default_factory=list)
    sync_after: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


class StrategyKind(Enum):
    """Dispatch strategy classification."""

    BULK_SYNC = "bulk_sync"
    PIPELINE = "pipeline"
    WAVEFRONT = "wavefront"
    STREAMING = "streaming"


class DispatchStrategy(ABC):
    """Base class for dispatch strategies.

    Subclasses implement :meth:`plan_waves` to convert an execution plan
    into a sequence of dispatch waves.
    """

    @property
    @abstractmethod
    def kind(self) -> StrategyKind:
        """The strategy classification."""
        ...

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return self.kind.value

    @abstractmethod
    def plan_waves(
        self,
        execution_order: list[str],
        placements: dict[str, int],
        dependencies: dict[str, list[str]],
        latencies: dict[str, float],
        *,
        node_for_device: dict[int, str] | None = None,
    ) -> list[DispatchWave]:
        """Plan dispatch waves from an execution plan.

        Args:
            execution_order: Op names in planned execution order.
            placements: Op name → device index.
            dependencies: Op name → list of dependency op names.
            latencies: Op name → estimated latency in microseconds.
            node_for_device: Device index → topology node name.

        Returns:
            Ordered list of dispatch waves.
        """
        ...

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for LLM prompts."""
        return {"kind": self.kind.value, "name": self.name}


# ---------------------------------------------------------------------------
# Bulk-synchronous strategy
# ---------------------------------------------------------------------------


class BulkSyncStrategy(DispatchStrategy):
    """Execute all ops at one DAG level before advancing.

    Simple and predictable.  Good baseline.  May leave devices idle
    when levels are unbalanced.
    """

    @property
    def kind(self) -> StrategyKind:
        return StrategyKind.BULK_SYNC

    def plan_waves(
        self,
        execution_order: list[str],
        placements: dict[str, int],
        dependencies: dict[str, list[str]],
        latencies: dict[str, float],
        *,
        node_for_device: dict[int, str] | None = None,
    ) -> list[DispatchWave]:
        node_map = node_for_device or {}
        # Topological level assignment
        levels = _compute_levels(execution_order, dependencies)

        # Group by level
        level_groups: dict[int, list[str]] = {}
        for op, level in levels.items():
            level_groups.setdefault(level, []).append(op)

        waves: list[DispatchWave] = []
        for level_idx in sorted(level_groups):
            ops = [
                DispatchOp(
                    op_name=op,
                    device_index=placements.get(op, 0),
                    node_name=node_map.get(placements.get(op, 0), ""),
                    estimated_latency_us=latencies.get(op, 0.0),
                    is_copy=op.startswith("copy_"),
                )
                for op in level_groups[level_idx]
            ]
            waves.append(DispatchWave(
                wave_id=level_idx,
                ops=ops,
                sync_after=True,
            ))

        log.debug("dispatch.bulk_sync.planned", num_waves=len(waves))
        return waves


# ---------------------------------------------------------------------------
# Pipeline strategy
# ---------------------------------------------------------------------------


class PipelineStrategy(DispatchStrategy):
    """Overlap compute and data movement across pipeline stages.

    Assigns ops to pipeline stages and overlaps stage N+1's data
    movement with stage N's compute.  Best when the DAG has a
    clear sequential spine with cross-device transfers.

    Attributes:
        num_stages: Number of pipeline stages (0 = auto-detect).
    """

    def __init__(self, num_stages: int = 0) -> None:
        self._num_stages = num_stages

    @property
    def kind(self) -> StrategyKind:
        return StrategyKind.PIPELINE

    def plan_waves(
        self,
        execution_order: list[str],
        placements: dict[str, int],
        dependencies: dict[str, list[str]],
        latencies: dict[str, float],
        *,
        node_for_device: dict[int, str] | None = None,
    ) -> list[DispatchWave]:
        node_map = node_for_device or {}

        # Separate copies from compute
        copies = [op for op in execution_order if op.startswith("copy_")]
        computes = [op for op in execution_order if not op.startswith("copy_")]

        waves: list[DispatchWave] = []
        wave_id = 0

        # Interleave: dispatch copies (async), then compute
        # Group by dependency depth to find natural pipeline stages
        levels = _compute_levels(execution_order, dependencies)
        level_groups: dict[int, list[str]] = {}
        for op, level in levels.items():
            level_groups.setdefault(level, []).append(op)

        for level_idx in sorted(level_groups):
            group = level_groups[level_idx]
            copy_ops = [op for op in group if op in copies]
            compute_ops = [op for op in group if op in computes]

            # Emit copies first (async, no sync)
            if copy_ops:
                ops = [
                    DispatchOp(
                        op_name=op,
                        device_index=placements.get(op, 0),
                        node_name=node_map.get(placements.get(op, 0), ""),
                        estimated_latency_us=latencies.get(op, 0.0),
                        is_copy=True,
                    )
                    for op in copy_ops
                ]
                waves.append(DispatchWave(
                    wave_id=wave_id,
                    ops=ops,
                    sync_after=False,  # overlap with next compute wave
                    metadata={"pipeline_stage": level_idx, "phase": "transfer"},
                ))
                wave_id += 1

            # Then compute
            if compute_ops:
                ops = [
                    DispatchOp(
                        op_name=op,
                        device_index=placements.get(op, 0),
                        node_name=node_map.get(placements.get(op, 0), ""),
                        estimated_latency_us=latencies.get(op, 0.0),
                        is_copy=False,
                    )
                    for op in compute_ops
                ]
                waves.append(DispatchWave(
                    wave_id=wave_id,
                    ops=ops,
                    sync_after=True,  # sync before next stage
                    metadata={"pipeline_stage": level_idx, "phase": "compute"},
                ))
                wave_id += 1

        log.debug("dispatch.pipeline.planned", num_waves=len(waves))
        return waves

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), "num_stages": self._num_stages}


# ---------------------------------------------------------------------------
# Wavefront strategy
# ---------------------------------------------------------------------------


class WavefrontStrategy(DispatchStrategy):
    """Dispatch ops as soon as dependencies are met.

    Maximum parallelism.  Each wave contains all ops whose dependencies
    have been satisfied by previous waves.  No explicit sync between
    waves — uses fine-grained dependency tracking.
    """

    @property
    def kind(self) -> StrategyKind:
        return StrategyKind.WAVEFRONT

    def plan_waves(
        self,
        execution_order: list[str],
        placements: dict[str, int],
        dependencies: dict[str, list[str]],
        latencies: dict[str, float],
        *,
        node_for_device: dict[int, str] | None = None,
    ) -> list[DispatchWave]:
        node_map = node_for_device or {}

        # Kahn's algorithm to find natural wavefronts
        in_degree: dict[str, int] = {op: 0 for op in execution_order}
        for op, deps in dependencies.items():
            if op in in_degree:
                in_degree[op] = len([d for d in deps if d in in_degree])

        remaining = set(execution_order)
        waves: list[DispatchWave] = []
        wave_id = 0

        while remaining:
            # Find all ops with zero in-degree
            ready = [
                op for op in execution_order
                if op in remaining and in_degree.get(op, 0) == 0
            ]

            if not ready:
                # Break cycle — force-dispatch remaining
                log.warning("dispatch.wavefront.cycle_detected",
                            remaining=len(remaining))
                ready = list(remaining)

            ops = [
                DispatchOp(
                    op_name=op,
                    device_index=placements.get(op, 0),
                    node_name=node_map.get(placements.get(op, 0), ""),
                    estimated_latency_us=latencies.get(op, 0.0),
                    is_copy=op.startswith("copy_"),
                )
                for op in ready
            ]

            # Wavefront: only sync when crossing device boundaries
            devices_in_wave = {placements.get(op, 0) for op in ready}
            waves.append(DispatchWave(
                wave_id=wave_id,
                ops=ops,
                sync_after=len(devices_in_wave) > 1,
                metadata={"wavefront_width": len(ready)},
            ))

            # Update in-degrees
            for op in ready:
                remaining.discard(op)
                # Decrement in-degree of successors
                for succ in execution_order:
                    if succ in remaining and op in dependencies.get(succ, []):
                        in_degree[succ] = max(0, in_degree.get(succ, 1) - 1)

            wave_id += 1

        log.debug("dispatch.wavefront.planned", num_waves=len(waves))
        return waves


# ---------------------------------------------------------------------------
# Streaming strategy
# ---------------------------------------------------------------------------


class StreamingStrategy(DispatchStrategy):
    """Continuous data flow through a fixed pipeline.

    Best for steady-state serving.  Ops are grouped by device and
    dispatched in a streaming fashion with double-buffered data
    movement.

    Attributes:
        double_buffer: Whether to use double-buffering for transfers.
    """

    def __init__(self, *, double_buffer: bool = True) -> None:
        self._double_buffer = double_buffer

    @property
    def kind(self) -> StrategyKind:
        return StrategyKind.STREAMING

    def plan_waves(
        self,
        execution_order: list[str],
        placements: dict[str, int],
        dependencies: dict[str, list[str]],
        latencies: dict[str, float],
        *,
        node_for_device: dict[int, str] | None = None,
    ) -> list[DispatchWave]:
        node_map = node_for_device or {}

        # Group ops by device
        device_ops: dict[int, list[str]] = {}
        for op in execution_order:
            dev = placements.get(op, 0)
            device_ops.setdefault(dev, []).append(op)

        waves: list[DispatchWave] = []
        wave_id = 0

        # Create one wave per device "slice" in execution order
        # This gives maximum overlap between devices
        max_ops = max((len(ops) for ops in device_ops.values()), default=0)

        for slot in range(max_ops):
            slot_ops: list[DispatchOp] = []
            for dev, ops in device_ops.items():
                if slot < len(ops):
                    op = ops[slot]
                    slot_ops.append(DispatchOp(
                        op_name=op,
                        device_index=dev,
                        node_name=node_map.get(dev, ""),
                        estimated_latency_us=latencies.get(op, 0.0),
                        is_copy=op.startswith("copy_"),
                    ))

            if slot_ops:
                waves.append(DispatchWave(
                    wave_id=wave_id,
                    ops=slot_ops,
                    sync_after=False,  # streaming: no sync between slots
                    metadata={
                        "slot": slot,
                        "double_buffer": self._double_buffer,
                    },
                ))
                wave_id += 1

        # Final sync wave
        if waves:
            waves[-1] = DispatchWave(
                wave_id=waves[-1].wave_id,
                ops=waves[-1].ops,
                sync_after=True,  # sync at end
                metadata=waves[-1].metadata,
            )

        log.debug("dispatch.streaming.planned", num_waves=len(waves))
        return waves

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), "double_buffer": self._double_buffer}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_levels(
    ops: list[str],
    dependencies: dict[str, list[str]],
) -> dict[str, int]:
    """Compute topological level for each op.

    Level 0 = no dependencies.  Level N = max(dep levels) + 1.
    """
    levels: dict[str, int] = {}
    op_set = set(ops)

    def _level(op: str) -> int:
        if op in levels:
            return levels[op]
        deps = [d for d in dependencies.get(op, []) if d in op_set]
        if not deps:
            levels[op] = 0
            return 0
        result = max(_level(d) for d in deps) + 1
        levels[op] = result
        return result

    for op in ops:
        _level(op)

    return levels


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


_STRATEGY_REGISTRY: dict[str, type[DispatchStrategy]] = {
    "bulk_sync": BulkSyncStrategy,
    "pipeline": PipelineStrategy,
    "wavefront": WavefrontStrategy,
    "streaming": StreamingStrategy,
}


def create_strategy(name: str, **kwargs: Any) -> DispatchStrategy:
    """Create a dispatch strategy by name.

    Args:
        name: Strategy name (``"bulk_sync"``, ``"pipeline"``,
            ``"wavefront"``, ``"streaming"``).
        **kwargs: Strategy-specific parameters.

    Returns:
        A strategy instance.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        msg = (
            f"Unknown dispatch strategy {name!r}. "
            f"Available: {sorted(_STRATEGY_REGISTRY)}"
        )
        raise ValueError(msg)
    return cls(**kwargs)


def register_strategy(name: str, cls: type[DispatchStrategy]) -> None:
    """Register a custom dispatch strategy.

    The agentic LLM can use this to register target-specific strategies.

    Args:
        name: Strategy name.
        cls: Strategy class (must subclass :class:`DispatchStrategy`).
    """
    _STRATEGY_REGISTRY[name] = cls
    log.info("dispatch_strategy.registered", name=name, cls=cls.__name__)


__all__ = [
    "BulkSyncStrategy",
    "DispatchOp",
    "DispatchStrategy",
    "DispatchWave",
    "PipelineStrategy",
    "StrategyKind",
    "StreamingStrategy",
    "WavefrontStrategy",
    "create_strategy",
    "register_strategy",
]

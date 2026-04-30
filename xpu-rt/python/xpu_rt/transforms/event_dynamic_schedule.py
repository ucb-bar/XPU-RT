"""Dynamic scheduling compiler pass — Paper §3.2 on-GPU push/pop queue.

Produces a :class:`DynamicSchedule` for workloads whose tile runtimes
or dependency structures are unpredictable at compile time (MoE,
irregular reductions, data-dependent fan-outs via
:class:`~compgen.ir.event.ops.TriggerOp`). Phase 5 turns the schedule
into a fused persistent-megakernel CUDA source whose main loop pops
tasks from a global ready queue and pushes dependents when events
fire.

Key differences from :mod:`event_static_schedule`:

- **No per-SM queues** baked at compile time. Every SM pulls from a
  shared ring buffer protected by atomic head/tail indices.
- **Initial ready set**: tasks with zero predecessors at compile
  time. These seed the queue before the persistent launch.
- **Dynamic pushes**: when a notify/trigger satisfies a successor's
  last predecessor, the runtime pushes that task. The schedule
  records the successor→predecessor adjacency so the emitter
  inlines the push logic.
- **Queue sizing**: circular buffer capacity = total tasks + small
  slack (×1.5). The paper's centralized queue contends on atomics
  at O(sm_count); we surface the capacity + batch hints in the
  launch config.

The schedule is gated on ``DeviceTraits.supports_ondevice_scheduler``
(Phase 6). When gated off the compiler falls back to Phase 2's
static pass or raises ``DynamicSchedulingUnavailable`` depending on
whether a trigger / symbolic structure forces dynamic semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.megakernel import DeviceCall, MegakernelGraph, _Task
from compgen.transforms.event_static_schedule import (
    EventTensorAllocSpec,
    LaunchConfig,
    TaskDescriptor,
    _canon_dtype,
    _task_to_descriptor,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DynamicTaskRecord:
    """Per-task metadata the dynamic-scheduler CUDA main loop needs.

    The emitted persistent kernel uses these to:

    1. Wait for ``predecessor_count`` notifications on this task
       before executing it (atomic decrement on a per-task predecessor
       counter living in the dynamic-queue metadata region).
    2. After executing, for every ``(event_name, cell, decrement)`` in
       ``out_cells`` perform notify; each successful notify that
       reaches zero pushes the downstream task's ``task_id`` onto the
       ready queue.
    3. Record this task_id's ``successor_task_ids`` so the runtime
       can enumerate dependents without re-walking the event graph.
    """

    descriptor: TaskDescriptor
    predecessor_count: int
    successor_task_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReadyQueueSpec:
    """Ring-buffer spec for the on-GPU ready queue.

    Layout (allocated in global memory by the CUDA launcher before
    launch; the exact CUDA type is `int32_t` with -1 as sentinel):

    - ``slots``: ``capacity`` int32 task_ids, padded to avoid false
      sharing.
    - ``head`` / ``tail``: two int32 atomic indices (wrapped modulo
      capacity).

    Pop = atomic load tail, atomic sub head; slot returned if
    head < tail before the sub.
    Push = atomic add tail, store into slot.
    """

    capacity: int
    initial_task_ids: tuple[int, ...]
    pop_batch_hint: int = 8  # tasks per dequeue attempt; matches paper §3.2


@dataclass(frozen=True)
class TriggerGenerator:
    """Records a :class:`~compgen.ir.event.ops.TriggerOp` site the
    dynamic scheduler must service at runtime.

    The paper's MoE case triggers a variable number of GroupGEMM
    tiles per expert from a CSR-style ``exp_indptr`` tensor. The
    emitter inlines a loop that, after the trigger tensor is
    populated (upstream in the graph), pushes
    ``indptr[i+1] - indptr[i]`` instances of the downstream task
    onto the ready queue.

    Schedule-time this is metadata; actual enumeration happens on
    the GPU.
    """

    target_event: str
    source_tensor: str
    target_device_func: str
    task_shape: tuple[int, ...]


@dataclass(frozen=True)
class DynamicSchedule:
    """Full Phase-3 output. Phase 5 emits CUDA from this structure."""

    graph_name: str
    sm_count: int
    tasks: tuple[DynamicTaskRecord, ...]
    ready_queue: ReadyQueueSpec
    event_tensor_allocs: tuple[EventTensorAllocSpec, ...]
    launch_config: LaunchConfig
    trigger_generators: tuple[TriggerGenerator, ...] = ()
    scheduling_metadata: dict[str, Any] = field(default_factory=dict)

    def as_yaml(self) -> str:
        """Serialize to YAML for bundle/megakernel/manifest.yaml."""
        import yaml

        def _dump(x: Any) -> Any:
            if hasattr(x, "__dataclass_fields__"):
                return {k: _dump(getattr(x, k)) for k in x.__dataclass_fields__}
            if isinstance(x, tuple):
                return [_dump(i) for i in x]
            if isinstance(x, dict):
                return {k: _dump(v) for k, v in x.items()}
            return x

        return yaml.safe_dump(_dump(self), sort_keys=False)


# ---------------------------------------------------------------------------
# Pass
# ---------------------------------------------------------------------------


class DynamicSchedulingUnavailable(RuntimeError):
    """Raised when a graph requires dynamic scheduling (e.g. has a
    TriggerOp) but the target doesn't expose an on-device scheduler."""


def compute_dynamic_schedule(
    graph: MegakernelGraph,
    *,
    sm_count: int,
    supports_ondevice_scheduler: bool = True,
    queue_capacity_factor: float = 1.5,
    pop_batch_hint: int = 8,
    cost_hints_us: dict[str, float] | None = None,
    cost_fn: Callable[[DeviceCall, tuple[int, ...]], float] | None = None,
    shared_mem_per_block_bytes: int = 0,
    block_dim: tuple[int, int, int] = (128, 1, 1),
    supports_clusters: bool = False,
    cluster_dim: tuple[int, int, int] | None = None,
    trigger_generators: tuple[TriggerGenerator, ...] = (),
) -> DynamicSchedule:
    """Compute the dynamic schedule for ``graph``.

    Args:
        graph: A :class:`MegakernelGraph` (usually from
            :func:`compgen.ir.event.lower.lower_graph_op`).
        sm_count: Number of SMs that will run the persistent
            megakernel. Every SM executes the same main-loop body.
        supports_ondevice_scheduler: Gate. When False and the graph
            has trigger generators → :class:`DynamicSchedulingUnavailable`.
            When False and the graph has no dynamic structure the
            caller should use the static pass instead.
        queue_capacity_factor: Ring-buffer capacity =
            ``ceil(total_tasks * factor)``. 1.5× is the paper's
            recommended slack — large enough that push-side
            contention stays bounded without over-allocating.
        pop_batch_hint: SMs pop up to this many tasks per queue
            round-trip to amortize atomic overhead. Paper §3.2
            default is 8 for the workloads they evaluate.
        cost_hints_us: Same semantics as
            :func:`compute_static_schedule`. Only used for the
            per-task cost metadata the Phase-5 emitter threads into
            the trace.
        cost_fn: Same semantics as
            :func:`compute_static_schedule`.
        shared_mem_per_block_bytes: Dynamic smem budget.
        block_dim: Thread-block shape.
        supports_clusters: Cluster launch opt-in.
        cluster_dim: Cluster dim.
        trigger_generators: Records of :class:`TriggerOp` sites
            supplied by the upstream IR → schedule translation.

    Returns:
        A :class:`DynamicSchedule` ready for Phase-5 emission.

    Raises:
        DynamicSchedulingUnavailable: target doesn't expose an
            on-device scheduler yet the graph requires one.
        ValueError: ``sm_count <= 0`` or empty graph.
    """
    if sm_count <= 0:
        raise ValueError(f"sm_count must be > 0, got {sm_count}")
    if not graph._tasks:
        raise ValueError(f"graph {graph.name!r} has no tasks to schedule")
    if not supports_ondevice_scheduler and trigger_generators:
        raise DynamicSchedulingUnavailable(
            f"graph {graph.name!r} has {len(trigger_generators)} trigger "
            "generator(s) that require an on-device scheduler, but the "
            "target's DeviceTraits.supports_ondevice_scheduler is False. "
            "Either route this graph through a target that exposes one, "
            "or rewrite the trigger pattern into a static fan-out."
        )

    # --- 1. Dependency graph (reuse). -----------------------------------
    successors, pred_count = graph._build_dependency_graph()
    tasks_by_id = {t.task_id: t for t in graph._tasks}

    # --- 2. Per-task cost. ----------------------------------------------
    def task_cost(t: _Task) -> float:
        if cost_fn is not None:
            return max(0.0, float(cost_fn(t.call, t.coord)))
        if cost_hints_us is not None and t.call.name in cost_hints_us:
            return max(0.0, float(cost_hints_us[t.call.name]))
        return 1.0

    # --- 3. Build per-task records. -------------------------------------
    records: list[DynamicTaskRecord] = []
    initial_ready: list[int] = []
    for tid, count in pred_count.items():
        t = tasks_by_id[tid]
        desc = _task_to_descriptor(t, cost_us=task_cost(t))
        rec = DynamicTaskRecord(
            descriptor=desc,
            predecessor_count=count,
            successor_task_ids=tuple(sorted(successors.get(tid, ()))),
        )
        records.append(rec)
        if count == 0:
            initial_ready.append(tid)
    records.sort(key=lambda r: r.descriptor.task_id)
    initial_ready.sort()

    # --- 4. Ring-buffer sizing. -----------------------------------------
    import math

    total = len(records)
    capacity = max(total, int(math.ceil(total * queue_capacity_factor)))
    ready_queue = ReadyQueueSpec(
        capacity=capacity,
        initial_task_ids=tuple(initial_ready),
        pop_batch_hint=pop_batch_hint,
    )

    # --- 5. Event-tensor allocation specs. -------------------------------
    allocs = tuple(
        sorted(
            (
                EventTensorAllocSpec(
                    name=name,
                    shape=tuple(et.shape),
                    wait_count_default=int(et.wait_count_default),
                    dtype=_canon_dtype(et),
                    scope=et.scope,
                )
                for name, et in graph.event_tensors.items()
            ),
            key=lambda a: a.name,
        )
    )

    # --- 6. Launch config. ----------------------------------------------
    launch_config = LaunchConfig(
        grid_dim=(sm_count, 1, 1),
        block_dim=block_dim,
        cluster_dim=cluster_dim if (supports_clusters and cluster_dim is not None) else None,
        shared_mem_bytes=int(shared_mem_per_block_bytes),
        cooperative=True,
    )

    # --- 7. Metadata. ---------------------------------------------------
    total_cost = sum(r.descriptor.cost_us for r in records)
    metadata: dict[str, Any] = {
        "total_cost_us": total_cost,
        "num_events": len(graph.event_tensors),
        "initial_ready_count": len(initial_ready),
        "max_fanout": max((len(r.successor_task_ids) for r in records), default=0),
        "policy": "dynamic",
        "num_triggers": len(trigger_generators),
    }

    schedule = DynamicSchedule(
        graph_name=graph.name,
        sm_count=sm_count,
        tasks=tuple(records),
        ready_queue=ready_queue,
        event_tensor_allocs=allocs,
        launch_config=launch_config,
        trigger_generators=tuple(trigger_generators),
        scheduling_metadata=metadata,
    )

    log.info(
        "event_dynamic_schedule.done",
        graph=graph.name,
        sm_count=sm_count,
        total_tasks=total,
        initial_ready=len(initial_ready),
        queue_capacity=capacity,
        num_triggers=len(trigger_generators),
    )
    return schedule


__all__ = [
    "DynamicSchedule",
    "DynamicSchedulingUnavailable",
    "DynamicTaskRecord",
    "ReadyQueueSpec",
    "TriggerGenerator",
    "compute_dynamic_schedule",
]

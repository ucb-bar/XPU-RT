"""Static scheduling compiler pass — Paper Algorithm 1.

Takes an ``event.graph`` op (or its lowered :class:`MegakernelGraph`)
plus a :class:`TargetProfile` and produces a :class:`StaticSchedule`
containing the per-SM task queues, event-tensor allocation metadata,
and notify/wait placement that Phase-5 code emission will consume
when it writes the fused persistent-kernel CUDA source.

Design:

- The Python-side schedule is the IR of this phase. Emitting xDSL
  ops for per-SM queue constants is premature — the real consumers
  (Tile-IR emitter, NVRTC wrapper) want structured Python data.
  Phase 5 turns this schedule into constant-memory tables in the
  emitted CUDA source.
- Scheduling reuses :meth:`MegakernelGraph._build_dependency_graph`
  and :meth:`MegakernelGraph._topo_sort` via the new
  :meth:`MegakernelGraph.plan_static` helper — no logic duplication.
- Task costs: caller supplies ``cost_hint_us`` per device-function
  name (typically from
  :func:`compgen.kernels.cost.roofline.predict`); absent hints
  default to 1.0 µs so every task weighs equally. Round-robin
  partitioning respects cumulative cost so SM loads stay balanced.

Paper correspondence (Jin et al., MLSys '26, §3.1 + Algorithm 1):

1. Input: a tile-level task DAG with explicit Event Tensor edges.
2. Partition tasks across SMs ahead of time (here: cost-weighted
   round-robin over the topological order).
3. Generate per-SM execution queues as a constant table in the
   emitted PTX.
4. Lower ETensor dependencies to explicit notify/wait — deferred
   to Phase 5 emission; this pass records placement directives.

Inputs + outputs::

    schedule = compute_static_schedule(
        graph=megakernel_graph,           # or the event.graph op
        target=blackwell_b200_profile,
        cost_hints_us={"gemm_tile": 4.2, "reduce_scatter_row": 1.1},
    )
    # schedule.sm_queues: one TaskQueue per SM, ordered.
    # schedule.event_tensor_allocs: {name → AllocSpec} for the CUDA launcher.
    # schedule.launch_config: grid, block, cluster, shared mem, cooperative.
    # schedule.as_yaml(): full manifest for bundle/megakernel/manifest.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

from compgen.runtime.event_tensor import EventTensor
from compgen.runtime.megakernel import DeviceCall, EventEdge, MegakernelGraph, _Task

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Schedule result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskDescriptor:
    """One task as it appears in an SM queue.

    The Phase-5 emitter reads these to produce the fused C++ main
    loop. ``in_edges`` / ``out_edges`` carry the information needed
    to emit ``cg_rt_cuda_etensor_wait`` / ``cg_rt_cuda_etensor_notify``
    calls with compile-time-resolvable cells (for truly static
    schedules, every cell resolves at compile time; data-dependent
    ones get the indexing deferred to runtime — but those go through
    the Phase-3 dynamic pass instead).
    """

    task_id: int
    device_func: str
    coord: tuple[int, ...]
    # ``(event_name, cell, decrement, peer_rank)`` per edge. ``peer_rank``
    # is ``None`` for intra-rank edges (the common case) and a
    # non-negative rank index for cross-rank edges (Phase-4b v2). The
    # emitter dispatches local vs peer notify/wait based on this
    # field; the launcher arranges peer-mapped event-tensor pointers
    # so the kernel can address peer cells via the same UVA-coherent
    # pointer space.
    in_cells: tuple[tuple[str, tuple[int, ...], int, int | None], ...]
    out_cells: tuple[tuple[str, tuple[int, ...], int, int | None], ...]
    cost_us: float = 1.0
    # Wave 1.6b — per-cell intra-cluster classification. One bool per
    # entry in ``in_cells`` (and one per entry in ``out_cells``),
    # parallel to those tuples. ``True`` means EVERY peer task
    # connected via that cell is on an SM in the same Blackwell
    # cluster as this task (and thus eligible for cluster-DSM
    # signalling); ``False`` means at least one peer is on a
    # different cluster (or the schedule is cluster-agnostic), so
    # the global-atomic notify/wait path is mandatory. When
    # cluster-aware partitioning is OFF, both masks are all-False
    # (the cluster path is never emitted).
    in_cluster_mask: tuple[bool, ...] = ()
    out_cluster_mask: tuple[bool, ...] = ()


@dataclass(frozen=True)
class SMQueue:
    """Per-SM task queue with its cumulative cost.

    The Phase-5 emitter lays out these queues in constant memory as
    a single flat table indexed by ``sm_id``. The total cost is
    exposed so the scheduler's load-balance decision is auditable.

    ``cluster_id`` is set when cluster-aware partitioning ran. SMs
    in the same cluster share distributed shared memory (Blackwell
    cluster.dsmem), so co-locating tasks linked by event-tensor
    edges within a single cluster lets Wave 1.6b's emitter
    eventually replace global-atomic notify/wait with cluster-DSM
    signals on intra-cluster edges. ``None`` when the schedule is
    cluster-agnostic.
    """

    sm_id: int
    tasks: tuple[TaskDescriptor, ...]
    total_cost_us: float
    cluster_id: int | None = None


@dataclass(frozen=True)
class EventTensorAllocSpec:
    """Allocation metadata for one Event Tensor.

    The CUDA launcher reads this to allocate a single integer tensor
    per Event Tensor before the persistent kernel launches. Symbolic
    dims are resolved at schedule-time (via upstream
    MaterializeViewOp / caller-provided concrete tensors).
    """

    name: str
    shape: tuple[int, ...]
    wait_count_default: int
    dtype: str
    scope: str


@dataclass(frozen=True)
class LaunchConfig:
    """Cooperative-launch parameters for the persistent megakernel.

    ``cluster_dim`` is None on devices without cluster support
    (SM_90-; we gate it via ``DeviceTraits.supports_clusters`` in
    Phase 6). ``cooperative=True`` is the static-scheduler default
    because the whole point is that every SM is simultaneously
    active for the duration of the forward pass.
    """

    grid_dim: tuple[int, int, int]
    block_dim: tuple[int, int, int]
    cluster_dim: tuple[int, int, int] | None
    shared_mem_bytes: int
    cooperative: bool = True


@dataclass(frozen=True)
class StaticSchedule:
    """Full Phase-2 output. Phase 5 emits CUDA from this structure."""

    graph_name: str
    sm_count: int
    sm_queues: tuple[SMQueue, ...]
    event_tensor_allocs: tuple[EventTensorAllocSpec, ...]
    launch_config: LaunchConfig
    total_tasks: int
    scheduling_metadata: dict[str, Any] = field(default_factory=dict)

    def task_count_per_sm(self) -> tuple[int, ...]:
        """Convenience: queue depth by SM index."""
        return tuple(len(q.tasks) for q in self.sm_queues)

    def cost_per_sm_us(self) -> tuple[float, ...]:
        """Convenience: total cost by SM index."""
        return tuple(q.total_cost_us for q in self.sm_queues)

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
# Scheduling pass
# ---------------------------------------------------------------------------


def compute_static_schedule(
    graph: MegakernelGraph,
    *,
    sm_count: int,
    cost_hints_us: dict[str, float] | None = None,
    cost_fn: Callable[[DeviceCall, tuple[int, ...]], float] | None = None,
    shared_mem_per_block_bytes: int = 0,
    block_dim: tuple[int, int, int] = (128, 1, 1),
    supports_clusters: bool = False,
    cluster_dim: tuple[int, int, int] | None = None,
) -> StaticSchedule:
    """Compute the full static schedule for ``graph`` on ``sm_count`` SMs.

    Args:
        graph: A :class:`MegakernelGraph` (usually from
            :func:`compgen.ir.event.lower.lower_graph_op`).
        sm_count: Number of SMs on the target device. Every task
            lands on exactly one SM; partitioning is cost-weighted
            round-robin over the topo order.
        cost_hints_us: Map ``device_func_name → microseconds`` used
            as the per-task cost. Cheaper than computing roofline
            per-call. Missing entries default to ``1.0``.
        cost_fn: Alternative: a callable ``(DeviceCall, task_coord)
            → float`` for callers with per-coord cost needs. Takes
            precedence over ``cost_hints_us`` when supplied.
        shared_mem_per_block_bytes: Dynamic shared-memory budget the
            launcher reserves per block. Phase 5 fills this in from
            Tile-IR introspection; plain ``0`` is acceptable for
            tasks that don't use dynamic smem.
        block_dim: Persistent-kernel thread-block shape.
        supports_clusters: When True and ``cluster_dim`` is set, the
            launch config opts into cluster launch (Blackwell
            distributed shared memory). Gated in Phase 6 by
            :class:`DeviceTraits`.
        cluster_dim: Cluster dim when ``supports_clusters`` is True.

    Returns:
        A :class:`StaticSchedule` ready for Phase-5 emission.

    Raises:
        ValueError: ``sm_count <= 0`` or the graph has zero tasks.
        ValueError: The graph has an event-tensor dependency cycle.
    """
    if sm_count is None:
        raise ValueError(
            "compute_static_schedule(sm_count=...) is required; pass an "
            "int (typically from compgen.runtime.autotune.probe_device "
            "or compgen.api._resolve_sm_count_for_target). Per bridge "
            "#131: a fresh-lowered MegakernelGraph doesn't carry an "
            "sm_count — that's resolved at schedule time, not at lower."
        )
    if sm_count <= 0:
        raise ValueError(f"sm_count must be > 0, got {sm_count}")
    if not graph._tasks:
        raise ValueError(f"graph {graph.name!r} has no tasks to schedule")

    # --- 1. Dependency graph + topological order (reuse MegakernelGraph). --
    successors, pred_count = graph._build_dependency_graph()
    order = graph._topo_sort(successors, pred_count)
    tasks_by_id = {t.task_id: t for t in graph._tasks}

    # --- 2. Cost-weighted round-robin over the topo order. ----------------
    def task_cost(t: _Task) -> float:
        if cost_fn is not None:
            return max(0.0, float(cost_fn(t.call, t.coord)))
        if cost_hints_us is not None and t.call.name in cost_hints_us:
            return max(0.0, float(cost_hints_us[t.call.name]))
        return 1.0

    # Use a simple greedy: pick the SM with the smallest cumulative
    # cost, breaking ties by sm_id (deterministic). This is
    # Longest-Processing-Time-first when ``order`` happens to be
    # cost-sorted; for a topo order it's still a reasonable
    # approximation and keeps the pass deterministic + trace-auditable.
    # Wave 1.6b — cluster-aware partitioning. When clusters are
    # enabled, prefer SMs in clusters that already host one of the
    # task's predecessors (so the predecessor → successor event-tensor
    # edge becomes intra-cluster, eligible for cluster-DSM signaling
    # in the Phase-5 emitter once that lands). Falls back to the
    # global cost-min SM when the preferred cluster is overloaded
    # past ``cluster_load_tolerance`` × the global min — keeping
    # load balance from breaking under graphs with very long chains.
    cluster_size = (
        cluster_dim[0] * cluster_dim[1] * cluster_dim[2] if (supports_clusters and cluster_dim is not None) else 1
    )
    cluster_aware = cluster_size > 1
    sm_to_cluster: list[int | None] = (
        [i // cluster_size for i in range(sm_count)] if cluster_aware else [None] * sm_count
    )
    cluster_load_tolerance = 1.5  # accept up to 1.5× the global min

    # Predecessor lookup: invert ``successors``. Cheap; we already
    # built ``successors`` above.
    predecessors_of: dict[int, list[int]] = {t.task_id: [] for t in graph._tasks}
    for src_tid, succ_list in successors.items():
        for dst_tid in succ_list:
            predecessors_of[dst_tid].append(src_tid)

    per_sm_tasks: list[list[TaskDescriptor]] = [[] for _ in range(sm_count)]
    per_sm_cost: list[float] = [0.0] * sm_count
    task_to_sm: dict[int, int] = {}
    intra_cluster_edges = 0
    cross_cluster_edges = 0

    for tid in order:
        t = tasks_by_id[tid]
        cost = task_cost(t)

        # Global argmin (used as the baseline + the fallback when the
        # cluster preference is too loaded).
        global_best = min(range(sm_count), key=lambda i: (per_sm_cost[i], i))

        if cluster_aware:
            # Find clusters that already host one of this task's
            # predecessors. The first time a task with no scheduled
            # predecessors lands, the cluster is unconstrained — use
            # the global argmin so cost balance dominates.
            preferred_clusters: set[int] = set()
            for pred_tid in predecessors_of.get(tid, ()):
                pred_sm = task_to_sm.get(pred_tid)
                if pred_sm is not None and sm_to_cluster[pred_sm] is not None:
                    preferred_clusters.add(sm_to_cluster[pred_sm])  # type: ignore[arg-type]

            if preferred_clusters:
                preferred_sms = [s for s in range(sm_count) if sm_to_cluster[s] in preferred_clusters]
                cluster_best = min(preferred_sms, key=lambda i: (per_sm_cost[i], i))
                # Accept the cluster preference only if its cost stays
                # within the load-tolerance band; otherwise fall back
                # to the global argmin so load balance is preserved.
                global_min_cost = per_sm_cost[global_best]
                if per_sm_cost[cluster_best] <= max(global_min_cost * cluster_load_tolerance, global_min_cost + cost):
                    best_sm = cluster_best
                else:
                    best_sm = global_best
            else:
                best_sm = global_best
        else:
            best_sm = global_best

        desc = _task_to_descriptor(t, cost_us=cost)
        per_sm_tasks[best_sm].append(desc)
        per_sm_cost[best_sm] += cost
        task_to_sm[tid] = best_sm

        # Audit edge locality. Counts each (predecessor → this task)
        # edge once; matches the dependency-graph edge count.
        if cluster_aware:
            this_cluster = sm_to_cluster[best_sm]
            for pred_tid in predecessors_of.get(tid, ()):
                pred_sm = task_to_sm.get(pred_tid)
                if pred_sm is None:
                    continue
                if sm_to_cluster[pred_sm] == this_cluster:
                    intra_cluster_edges += 1
                else:
                    cross_cluster_edges += 1

    # --- 2b. Per-cell intra-cluster mask (Wave 1.6b emitter half). --------
    # We need to know — per *cell* on each TaskDescriptor — whether
    # every peer task connected to that cell is on the same cluster.
    # The dependency graph is cell-granular (see
    # ``MegakernelGraph._build_dependency_graph``): an edge exists
    # between predecessor P and successor S whenever P has an
    # out-edge and S has an in-edge that resolve to the same
    # ``(event_name, cell)``.
    # Algorithm:
    #   1. For each (event_name, cell) build {producers, consumers}.
    #   2. For a successor's in-cell, mark intra-cluster IFF every
    #      producer of that cell sits on the same cluster as the
    #      successor's SM.
    #   3. For a predecessor's out-cell, mark intra-cluster IFF every
    #      consumer of that cell sits on the same cluster as the
    #      predecessor's SM.
    #   4. Cells with no peers (e.g. initial out-edges with no
    #      consumer in this graph) default to False — keep the safe
    #      global path.
    # When ``cluster_aware`` is False the masks are all-False so the
    # emitter falls back to the existing pure-global-atomic path.
    in_cluster_mask_for: dict[int, tuple[bool, ...]] = {}
    out_cluster_mask_for: dict[int, tuple[bool, ...]] = {}

    if cluster_aware:
        from collections import defaultdict

        producers_of_cell: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
        consumers_of_cell: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
        for t in graph._tasks:
            for e in t.call.out_edges:
                # Cross-rank edges target a peer rank's tensor — they
                # are never intra-cluster (different rank entirely),
                # so we skip the producer-bookkeeping for them.
                if e.peer_rank is not None:
                    continue
                producers_of_cell[(e.event_name, e.resolve(t.coord))].append(t.task_id)
            for e in t.call.in_edges:
                if e.peer_rank is not None:
                    continue
                consumers_of_cell[(e.event_name, e.resolve(t.coord))].append(t.task_id)

        for tid, sm in task_to_sm.items():
            this_cluster = sm_to_cluster[sm]
            t = tasks_by_id[tid]

            # In-cell mask: a wait on cell C is intra-cluster only if
            # every producer of C is on the same cluster as us.
            in_mask: list[bool] = []
            for e in t.call.in_edges:
                if e.peer_rank is not None:
                    in_mask.append(False)
                    continue
                cell = e.resolve(t.coord)
                producers = producers_of_cell.get((e.event_name, cell), [])
                # Drop self-edges (a task that lists itself as both
                # producer + consumer of one cell — degenerate but
                # legal). We keep the safe path for self-edges to
                # avoid weird intra-block cluster-sync interactions.
                producers = [p for p in producers if p != tid]
                if not producers:
                    in_mask.append(False)
                    continue
                all_intra = all(sm_to_cluster[task_to_sm[p]] == this_cluster for p in producers)
                in_mask.append(bool(all_intra))
            in_cluster_mask_for[tid] = tuple(in_mask)

            # Out-cell mask: a notify on cell C is intra-cluster only
            # if every consumer of C is on the same cluster as us AND
            # this task is the unique producer of C within the cluster.
            # The second condition is a correctness gate: the
            # intra-cluster notify path is a *relaxed* (non-atomic)
            # decrement of the local cluster-DSM view. When two
            # producers in the same cluster decrement the same cell
            # in the same wave, they race and the consumer's wait can
            # spin forever (the cell never reaches the expected count).
            # We saw this with Wave 2.5 epilogue fusion (bridge #146):
            # the fused FFN topology placed two ``linear_up_relu``
            # producers in cluster 0 both folding onto the same
            # ``ev_relu`` cell. Forcing those producers to the global
            # atomic path (``out_mask=False``) restores correctness;
            # the consumer's wait spin is fine either way (volatile
            # read sees both global-atomic and cluster-DSM writes).
            out_mask: list[bool] = []
            for e in t.call.out_edges:
                if e.peer_rank is not None:
                    out_mask.append(False)
                    continue
                cell = e.resolve(t.coord)
                consumers = consumers_of_cell.get((e.event_name, cell), [])
                consumers = [c for c in consumers if c != tid]
                if not consumers:
                    out_mask.append(False)
                    continue
                all_consumers_intra = all(sm_to_cluster[task_to_sm[c]] == this_cluster for c in consumers)
                if not all_consumers_intra:
                    out_mask.append(False)
                    continue
                co_producers = producers_of_cell.get((e.event_name, cell), [])
                co_producers_in_cluster = [
                    p for p in co_producers if p != tid and sm_to_cluster[task_to_sm[p]] == this_cluster
                ]
                if co_producers_in_cluster:
                    # Multiple producers in this cluster on the same
                    # cell — relaxed decrement would race. Fall back
                    # to the safe global atomic path.
                    out_mask.append(False)
                    continue
                out_mask.append(True)
            out_cluster_mask_for[tid] = tuple(out_mask)

    # --- 2c. Stitch the masks into the placed TaskDescriptors. ------------
    placed_per_sm: list[list[TaskDescriptor]] = [[] for _ in range(sm_count)]
    for sm_idx, descs in enumerate(per_sm_tasks):
        for desc in descs:
            in_mask = in_cluster_mask_for.get(desc.task_id, tuple(False for _ in desc.in_cells))
            out_mask = out_cluster_mask_for.get(desc.task_id, tuple(False for _ in desc.out_cells))
            placed_per_sm[sm_idx].append(
                TaskDescriptor(
                    task_id=desc.task_id,
                    device_func=desc.device_func,
                    coord=desc.coord,
                    in_cells=desc.in_cells,
                    out_cells=desc.out_cells,
                    cost_us=desc.cost_us,
                    in_cluster_mask=in_mask,
                    out_cluster_mask=out_mask,
                )
            )

    sm_queues = tuple(
        SMQueue(
            sm_id=i,
            tasks=tuple(placed_per_sm[i]),
            total_cost_us=per_sm_cost[i],
            cluster_id=sm_to_cluster[i],
        )
        for i in range(sm_count)
    )

    # --- 3. Event-tensor allocation specs. --------------------------------
    allocs: list[EventTensorAllocSpec] = []
    for name, et in graph.event_tensors.items():
        allocs.append(
            EventTensorAllocSpec(
                name=name,
                shape=tuple(et.shape),
                wait_count_default=int(et.wait_count_default),
                dtype=_canon_dtype(et),
                scope=et.scope,
            )
        )
    allocs.sort(key=lambda a: a.name)  # deterministic YAML output

    # --- 4. Launch config. ------------------------------------------------
    launch_config = LaunchConfig(
        grid_dim=(sm_count, 1, 1),
        block_dim=block_dim,
        cluster_dim=cluster_dim if (supports_clusters and cluster_dim is not None) else None,
        shared_mem_bytes=int(shared_mem_per_block_bytes),
        cooperative=True,
    )

    # --- 5. Build the final schedule. ------------------------------------
    cost_spread = max(per_sm_cost) - min(per_sm_cost) if per_sm_cost else 0.0
    total_cost = sum(per_sm_cost)
    metadata: dict[str, Any] = {
        "total_cost_us": total_cost,
        "cost_spread_us": cost_spread,
        "cost_balance_ratio": (min(per_sm_cost) / max(per_sm_cost)) if max(per_sm_cost) > 0 else 1.0,
        "num_events": len(graph.event_tensors),
        "policy": "static",
    }
    if cluster_aware:
        # Edge-locality audit. A high intra-cluster fraction means
        # Wave 1.6b's emitter has plenty of intra-cluster edges to
        # convert to cluster-DSM signals — the perf lever per #127.
        total_dep_edges = intra_cluster_edges + cross_cluster_edges
        metadata["cluster_size"] = cluster_size
        metadata["num_clusters"] = (sm_count + cluster_size - 1) // cluster_size
        metadata["intra_cluster_edges"] = intra_cluster_edges
        metadata["cross_cluster_edges"] = cross_cluster_edges
        metadata["intra_cluster_edge_fraction"] = intra_cluster_edges / total_dep_edges if total_dep_edges > 0 else 0.0

    schedule = StaticSchedule(
        graph_name=graph.name,
        sm_count=sm_count,
        sm_queues=sm_queues,
        event_tensor_allocs=tuple(allocs),
        launch_config=launch_config,
        total_tasks=len(graph._tasks),
        scheduling_metadata=metadata,
    )

    log.info(
        "event_static_schedule.done",
        graph=graph.name,
        sm_count=sm_count,
        total_tasks=schedule.total_tasks,
        total_cost_us=total_cost,
        cost_spread_us=cost_spread,
    )
    return schedule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _task_to_descriptor(t: _Task, *, cost_us: float) -> TaskDescriptor:
    """Convert a runtime task into a Phase-5-consumable descriptor."""
    in_cells = tuple((e.event_name, _resolve_cell(e, t.coord), e.decrement, e.peer_rank) for e in t.call.in_edges)
    out_cells = tuple((e.event_name, _resolve_cell(e, t.coord), e.decrement, e.peer_rank) for e in t.call.out_edges)
    return TaskDescriptor(
        task_id=t.task_id,
        device_func=t.call.name,
        coord=tuple(t.coord),
        in_cells=in_cells,
        out_cells=out_cells,
        cost_us=cost_us,
        # Default masks: all-False. ``compute_static_schedule`` rebuilds
        # the descriptor with the correct cluster masks once placement
        # is finalised (Wave 1.6b).
        in_cluster_mask=tuple(False for _ in in_cells),
        out_cluster_mask=tuple(False for _ in out_cells),
    )


def _resolve_cell(edge: EventEdge, coord: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve an ``EventEdge.index_fn`` into a concrete cell tuple.

    Data-dependent expressions (``topk[i]``) are still resolvable
    here because lowering has already baked the feeder tensors
    into the closure via ``index_env``.
    """
    return edge.resolve(coord)


def _canon_dtype(et: EventTensor) -> str:
    """Map :class:`EventTensor.dtype` to the dialect's string token.

    ``EventTensor.dtype`` is stored as a string key from
    ``_DTYPE_MAP`` ("i32" / "u32" / "i64" / "u64"). We canonicalize
    to the two supported values the alloc spec serializes.
    """
    raw = str(et.dtype)
    if raw in ("i32", "u32"):
        return "i32"
    if raw in ("i64", "u64"):
        return "i64"
    raise ValueError(f"event-tensor dtype {raw!r} not supported by the static scheduler")


__all__ = [
    "EventTensorAllocSpec",
    "LaunchConfig",
    "SMQueue",
    "StaticSchedule",
    "TaskDescriptor",
    "compute_static_schedule",
]

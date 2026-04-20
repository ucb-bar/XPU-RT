"""Helpers for constructing ExecutionPlans from Recipe IR modules.

The real ``Recipe IR → ExecutionPlan`` lowering lives in the W6 passes
(which we ship in Wave 6). What's here now is the small-but-real set
of builders that every W6 pass + every test that exercises Phase 5
needs: a declarative way to assemble a plan, walk its regions, merge
in buffer liveness + queue/stream assignments, and round-trip through
the schema dict.
"""

from __future__ import annotations

from collections.abc import Iterable

from compgen.runtime.execution_plan import (
    BufferDescriptor,
    CopyEdge,
    DependencyEdge,
    ExecutionPlan,
    Lifetime,
    Ownership,
    QueueAssignment,
    QueueEntry,
    RegionPlacement,
    Resource,
    StreamAnnotation,
    SyncEdge,
)


class ExecutionPlanBuilder:
    """Fluent builder over an :class:`ExecutionPlan`.

    Intent: let tests + Phase-5 passes compose plans without
    positional dataclass juggling. Every ``add_*`` method mutates the
    in-progress plan in place and returns ``self`` for chaining.
    """

    def __init__(self, workload: str, target: str) -> None:
        self.plan = ExecutionPlan(workload=workload, target=target)

    # --- resources ------------------------------------------------------

    def add_resource(
        self,
        id: str,
        kind: str,
        *,
        device: str = "",
        capacity: float = 0.0,
    ) -> ExecutionPlanBuilder:
        self.plan.resources.append(Resource(id=id, kind=kind, device=device, capacity=capacity))
        return self

    # --- region placement + queues -------------------------------------

    def add_region(
        self,
        region_id: str,
        device: str,
        queue: str,
        *,
        stream_id: int = 0,
        priority: int = 0,
    ) -> ExecutionPlanBuilder:
        self.plan.region_placement.append(
            RegionPlacement(
                region_id=region_id,
                device=device,
                queue=queue,
                stream_id=stream_id,
                priority=priority,
            )
        )
        return self

    def apply_queue_assignment(
        self,
        assignments: Iterable[QueueAssignment],
    ) -> ExecutionPlanBuilder:
        """Overwrite ``queue`` + ``priority`` for placements named in ``assignments``.

        Used by W6 ``assign_queue`` after the solver has decided which
        queue each region should run on.
        """
        by_region = {a.region_id: a for a in assignments}
        for rp in self.plan.region_placement:
            a = by_region.get(rp.region_id)
            if a is not None:
                rp.queue = a.queue
                rp.priority = a.priority
        return self

    def apply_stream_annotation(
        self,
        annotations: Iterable[StreamAnnotation],
    ) -> ExecutionPlanBuilder:
        """Write ``stream_id`` + optional async-wrap tag for regions.

        The ``kind`` tag lives in ``plan.summary['stream_kinds']`` so
        downstream IR-level passes can decide whether to wrap with an
        async op.
        """
        by_region = {a.region_id: a for a in annotations}
        kinds = dict(self.plan.summary.get("stream_kinds", {}))
        for rp in self.plan.region_placement:
            a = by_region.get(rp.region_id)
            if a is not None:
                rp.stream_id = a.stream_id
                kinds[rp.region_id] = a.kind
        if kinds:
            self.plan.summary["stream_kinds"] = kinds
        return self

    # --- dependency / sync / copy --------------------------------------

    def add_dependency(
        self,
        from_region: str,
        to_region: str,
        *,
        value_ref: str = "",
    ) -> ExecutionPlanBuilder:
        self.plan.dependency_edges.append(
            DependencyEdge(
                from_region=from_region,
                to_region=to_region,
                value_ref=value_ref,
            )
        )
        return self

    def add_copy(
        self,
        from_buffer: str,
        to_buffer: str,
        size_bytes: int,
        *,
        transfer_path: str,
        est_latency_ns: int = 0,
    ) -> ExecutionPlanBuilder:
        self.plan.copy_edges.append(
            CopyEdge(
                from_buffer=from_buffer,
                to_buffer=to_buffer,
                size_bytes=size_bytes,
                transfer_path=transfer_path,
                est_latency_ns=est_latency_ns,
            )
        )
        return self

    def add_sync(
        self,
        kind: str,
        producers: list[str],
        consumers: list[str],
        *,
        scope: str = "device",
    ) -> ExecutionPlanBuilder:
        self.plan.sync_edges.append(
            SyncEdge(
                kind=kind,
                producers=producers,
                consumers=consumers,
                scope=scope,
            )
        )
        return self

    # --- queue timeline ------------------------------------------------

    def add_queue_entry(
        self,
        queue: str,
        region_id: str,
        start_tick: int,
        *,
        est_duration_ns: int = 0,
    ) -> ExecutionPlanBuilder:
        self.plan.queue_timeline.append(
            QueueEntry(
                queue=queue,
                region_id=region_id,
                start_tick=start_tick,
                est_duration_ns=est_duration_ns,
            )
        )
        return self

    # --- buffers -------------------------------------------------------

    def add_buffer(
        self,
        buffer_id: str,
        size_bytes: int,
        memory_space: str,
        first_use_tick: int,
        last_use_tick: int,
        *,
        ownership: Ownership = "exclusive",
        alias_of: str = "",
        persistent: bool = False,
    ) -> ExecutionPlanBuilder:
        self.plan.buffers.append(
            BufferDescriptor(
                buffer_id=buffer_id,
                size_bytes=size_bytes,
                memory_space=memory_space,
                lifetime=Lifetime(
                    first_use_tick=first_use_tick,
                    last_use_tick=last_use_tick,
                    persistent=persistent,
                ),
                ownership=ownership,
                alias_of=alias_of,
            )
        )
        return self

    # --- finalize ------------------------------------------------------

    def build(self) -> ExecutionPlan:
        self.plan.validate()
        return self.plan


__all__ = ["ExecutionPlanBuilder"]

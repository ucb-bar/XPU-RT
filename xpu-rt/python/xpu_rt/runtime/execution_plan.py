"""ExecutionPlan dataclasses for the Phase 5 runtime contract.

Mirrors ``user_perspective/prototypes/schemas/execution_runtime.schema.yaml``
(schema v2.0). The dataclasses here are the Python-side carrier that
Phase 5 passes populate: ``assign_memory_space``, ``assign_queue``,
``assign_streams``, ``plan_buffers``, ``insert_copies``,
``alias_io_buffers``, ``insert_host_offload``,
``normalize_subbyte_post_layout``.

Design goals:

- **Schema-faithful**: every field name matches the YAML schema so the
  serialized dict passes external validators unchanged.
- **Mutable on construction, hashable at rest**: we use mutable
  dataclasses so passes can augment/rewrite plans in place, plus a
  ``to_dict()`` / ``from_dict()`` pair that round-trips stable output.
- **Self-validating**: ``validate()`` enforces the schema's cross-field
  invariants (alias ownership, lifetime ordering, resource refs).

The plan is the input to buffer liveness + interference graph
analysis (see ``compgen.runtime.liveness``) and the substrate that
every W6 runtime pass rewrites.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

ResourceKind = Literal["compute", "memory", "transfer", "synchronization"]
SyncKind = Literal["fence", "semaphore", "barrier"]
SyncScope = Literal["device", "system"]
Ownership = Literal["exclusive", "shared_readonly", "alias"]


@dataclass
class Resource:
    """A runtime resource: a compute unit, memory domain, or queue."""

    id: str
    kind: ResourceKind
    device: str = ""
    capacity: float = 0.0


@dataclass
class RegionPlacement:
    """Placement of one IR region on a concrete queue/stream."""

    region_id: str
    device: str
    queue: str
    stream_id: int = 0
    priority: int = 0


@dataclass
class DependencyEdge:
    """Value or control-flow dependency between two regions."""

    from_region: str
    to_region: str
    value_ref: str = ""


@dataclass
class CopyEdge:
    """A memory-to-memory copy between two buffers.

    ``transfer_path`` refs into
    ``target_resource.v2.transfer_paths[]`` so the plan carries
    hardware-specific cost + path metadata without re-encoding it
    here.
    """

    from_buffer: str
    to_buffer: str
    size_bytes: int
    transfer_path: str
    est_latency_ns: int = 0


@dataclass
class SyncEdge:
    """Cross-queue synchronization event."""

    kind: SyncKind
    producers: list[str]
    consumers: list[str]
    scope: SyncScope = "device"


@dataclass
class QueueEntry:
    """Occupancy entry on the queue timeline."""

    queue: str
    region_id: str
    start_tick: int
    est_duration_ns: int = 0


@dataclass
class Lifetime:
    """First / last use tick for a buffer."""

    first_use_tick: int
    last_use_tick: int
    persistent: bool = False

    def overlaps(self, other: Lifetime) -> bool:
        """Inclusive interval overlap. Persistent buffers overlap any live
        interval (they stay live across the whole program)."""
        if self.persistent or other.persistent:
            return True
        return not (
            self.last_use_tick < other.first_use_tick
            or other.last_use_tick < self.first_use_tick
        )


@dataclass
class BufferDescriptor:
    """One allocation in the buffer plan."""

    buffer_id: str
    size_bytes: int
    memory_space: str
    lifetime: Lifetime
    ownership: Ownership
    alias_of: str = ""


# --- Phase-5-pass-specific views -------------------------------------------


@dataclass
class QueueAssignment:
    """A region's queue_id + priority decision (XLA: queue_id annotation)."""

    region_id: str
    queue: str
    priority: int = 0


@dataclass
class StreamAnnotation:
    """A region's stream_id + async-wrap decision.

    ``kind`` ∈ {``sync``, ``async_wrap``, ``async_passthrough``} controls
    whether ``assign_streams`` should wrap the region in an async op.
    """

    region_id: str
    stream_id: int
    kind: str = "sync"


@dataclass
class FallbackTransition:
    """Runtime-guarded alternative plan ref."""

    condition: str
    alternative_plan_ref: str


@dataclass
class ExecutionPlan:
    """The full Phase 5 artifact.

    Mirrors ``execution_runtime.schema.yaml`` v2.0.
    """

    workload: str
    target: str
    target_resource_model_hash: str = ""
    schema_version: str = "2.0"

    resources: list[Resource] = field(default_factory=list)
    region_placement: list[RegionPlacement] = field(default_factory=list)
    dependency_edges: list[DependencyEdge] = field(default_factory=list)
    copy_edges: list[CopyEdge] = field(default_factory=list)
    sync_edges: list[SyncEdge] = field(default_factory=list)
    queue_timeline: list[QueueEntry] = field(default_factory=list)
    buffers: list[BufferDescriptor] = field(default_factory=list)
    fallback_transitions: list[FallbackTransition] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    # --- accessor helpers -----------------------------------------------

    @property
    def region_ids(self) -> list[str]:
        return [rp.region_id for rp in self.region_placement]

    @property
    def buffer_ids(self) -> list[str]:
        return [b.buffer_id for b in self.buffers]

    @property
    def queue_ids(self) -> list[str]:
        return sorted({rp.queue for rp in self.region_placement})

    def buffer(self, buffer_id: str) -> BufferDescriptor:
        for b in self.buffers:
            if b.buffer_id == buffer_id:
                return b
        raise KeyError(buffer_id)

    def placement_for(self, region_id: str) -> RegionPlacement:
        for rp in self.region_placement:
            if rp.region_id == region_id:
                return rp
        raise KeyError(region_id)

    # --- validation -----------------------------------------------------

    def validate(self) -> None:
        """Enforce cross-field invariants.

        Raises ``ValueError`` when the plan is inconsistent (duplicate
        region/buffer ids, dangling alias_of, negative lifetime
        interval, etc.).
        """
        seen_regions: set[str] = set()
        for rp in self.region_placement:
            if rp.region_id in seen_regions:
                raise ValueError(f"duplicate region_id {rp.region_id!r}")
            seen_regions.add(rp.region_id)

        seen_buffers: set[str] = set()
        for b in self.buffers:
            if b.buffer_id in seen_buffers:
                raise ValueError(f"duplicate buffer_id {b.buffer_id!r}")
            seen_buffers.add(b.buffer_id)
            if b.lifetime.first_use_tick > b.lifetime.last_use_tick:
                raise ValueError(
                    f"buffer {b.buffer_id!r}: first_use_tick "
                    f"({b.lifetime.first_use_tick}) > last_use_tick "
                    f"({b.lifetime.last_use_tick})"
                )
            if b.ownership == "alias":
                if not b.alias_of:
                    raise ValueError(
                        f"buffer {b.buffer_id!r}: ownership=alias requires "
                        f"alias_of to be set"
                    )
                if b.alias_of == b.buffer_id:
                    raise ValueError(
                        f"buffer {b.buffer_id!r}: alias_of must reference "
                        f"a different buffer"
                    )

        # Second-pass: resolve alias_of references against seen_buffers.
        for b in self.buffers:
            if b.alias_of and b.alias_of not in seen_buffers:
                raise ValueError(
                    f"buffer {b.buffer_id!r}: alias_of references unknown "
                    f"buffer {b.alias_of!r}"
                )

        for e in self.copy_edges:
            if e.from_buffer not in seen_buffers:
                raise ValueError(
                    f"copy_edge from_buffer {e.from_buffer!r} is unknown"
                )
            if e.to_buffer not in seen_buffers:
                raise ValueError(
                    f"copy_edge to_buffer {e.to_buffer!r} is unknown"
                )
            if e.size_bytes < 0:
                raise ValueError(
                    f"copy_edge {e.from_buffer}->{e.to_buffer}: "
                    f"size_bytes ({e.size_bytes}) must be non-negative"
                )

        for d in self.dependency_edges:
            if d.from_region not in seen_regions:
                raise ValueError(
                    f"dependency_edge from_region {d.from_region!r} is unknown"
                )
            if d.to_region not in seen_regions:
                raise ValueError(
                    f"dependency_edge to_region {d.to_region!r} is unknown"
                )

        for q in self.queue_timeline:
            if q.region_id not in seen_regions:
                raise ValueError(
                    f"queue_timeline entry refers to unknown region "
                    f"{q.region_id!r}"
                )

    # --- serialization --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a schema-faithful dict (stable for JSON/YAML dump)."""
        return {
            "schema_version": self.schema_version,
            "workload": self.workload,
            "target": self.target,
            "target_resource_model_hash": self.target_resource_model_hash,
            "resources": [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "device": r.device,
                    "capacity": r.capacity,
                }
                for r in self.resources
            ],
            "region_placement": [
                {
                    "region_id": rp.region_id,
                    "device": rp.device,
                    "queue": rp.queue,
                    "stream_id": rp.stream_id,
                    "priority": rp.priority,
                }
                for rp in self.region_placement
            ],
            "dependency_edges": [
                {
                    "from_region": d.from_region,
                    "to_region": d.to_region,
                    "value_ref": d.value_ref,
                }
                for d in self.dependency_edges
            ],
            "copy_edges": [
                {
                    "from_buffer": e.from_buffer,
                    "to_buffer": e.to_buffer,
                    "size_bytes": e.size_bytes,
                    "transfer_path": e.transfer_path,
                    "est_latency_ns": e.est_latency_ns,
                }
                for e in self.copy_edges
            ],
            "sync_edges": [
                {
                    "kind": s.kind,
                    "producers": list(s.producers),
                    "consumers": list(s.consumers),
                    "scope": s.scope,
                }
                for s in self.sync_edges
            ],
            "queue_timeline": [
                {
                    "queue": q.queue,
                    "region_id": q.region_id,
                    "start_tick": q.start_tick,
                    "est_duration_ns": q.est_duration_ns,
                }
                for q in self.queue_timeline
            ],
            "buffers": [
                {
                    "buffer_id": b.buffer_id,
                    "size_bytes": b.size_bytes,
                    "memory_space": b.memory_space,
                    "lifetime": {
                        "first_use_tick": b.lifetime.first_use_tick,
                        "last_use_tick": b.lifetime.last_use_tick,
                        "persistent": b.lifetime.persistent,
                    },
                    "ownership": b.ownership,
                    "alias_of": b.alias_of,
                }
                for b in self.buffers
            ],
            "fallback_transitions": [
                {
                    "condition": f.condition,
                    "alternative_plan_ref": f.alternative_plan_ref,
                }
                for f in self.fallback_transitions
            ],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        """Parse a schema-shaped dict into an ``ExecutionPlan``.

        Raises ``KeyError`` / ``ValueError`` on malformed input.
        """
        return cls(
            schema_version=data.get("schema_version", "2.0"),
            workload=data["workload"],
            target=data["target"],
            target_resource_model_hash=data.get("target_resource_model_hash", ""),
            resources=[
                Resource(
                    id=r["id"],
                    kind=r["kind"],
                    device=r.get("device", ""),
                    capacity=float(r.get("capacity", 0.0)),
                )
                for r in data.get("resources", [])
            ],
            region_placement=[
                RegionPlacement(
                    region_id=rp["region_id"],
                    device=rp["device"],
                    queue=rp["queue"],
                    stream_id=int(rp.get("stream_id", 0)),
                    priority=int(rp.get("priority", 0)),
                )
                for rp in data.get("region_placement", [])
            ],
            dependency_edges=[
                DependencyEdge(
                    from_region=d["from_region"],
                    to_region=d["to_region"],
                    value_ref=d.get("value_ref", ""),
                )
                for d in data.get("dependency_edges", [])
            ],
            copy_edges=[
                CopyEdge(
                    from_buffer=e["from_buffer"],
                    to_buffer=e["to_buffer"],
                    size_bytes=int(e["size_bytes"]),
                    transfer_path=e["transfer_path"],
                    est_latency_ns=int(e.get("est_latency_ns", 0)),
                )
                for e in data.get("copy_edges", [])
            ],
            sync_edges=[
                SyncEdge(
                    kind=s["kind"],
                    producers=list(s["producers"]),
                    consumers=list(s["consumers"]),
                    scope=s.get("scope", "device"),
                )
                for s in data.get("sync_edges", [])
            ],
            queue_timeline=[
                QueueEntry(
                    queue=q["queue"],
                    region_id=q["region_id"],
                    start_tick=int(q["start_tick"]),
                    est_duration_ns=int(q.get("est_duration_ns", 0)),
                )
                for q in data.get("queue_timeline", [])
            ],
            buffers=[
                BufferDescriptor(
                    buffer_id=b["buffer_id"],
                    size_bytes=int(b["size_bytes"]),
                    memory_space=b["memory_space"],
                    lifetime=Lifetime(
                        first_use_tick=int(b["lifetime"]["first_use_tick"]),
                        last_use_tick=int(b["lifetime"]["last_use_tick"]),
                        persistent=bool(b["lifetime"].get("persistent", False)),
                    ),
                    ownership=b["ownership"],
                    alias_of=b.get("alias_of", ""),
                )
                for b in data.get("buffers", [])
            ],
            fallback_transitions=[
                FallbackTransition(
                    condition=f["condition"],
                    alternative_plan_ref=f["alternative_plan_ref"],
                )
                for f in data.get("fallback_transitions", [])
            ],
            summary=dict(data.get("summary", {})),
        )


def ticks_spanned(plan: ExecutionPlan) -> int:
    """Max last_use_tick over all buffers + queue entries, clamped to 0.

    Used by W6 schedulers as the plan's "program length" in ticks.
    """
    max_tick = 0
    for b in plan.buffers:
        if b.lifetime.last_use_tick > max_tick:
            max_tick = b.lifetime.last_use_tick
    for q in plan.queue_timeline:
        if q.start_tick > max_tick:
            max_tick = q.start_tick
    return max_tick


__all__ = [
    "BufferDescriptor",
    "CopyEdge",
    "DependencyEdge",
    "ExecutionPlan",
    "FallbackTransition",
    "Lifetime",
    "Ownership",
    "QueueAssignment",
    "QueueEntry",
    "RegionPlacement",
    "Resource",
    "ResourceKind",
    "StreamAnnotation",
    "SyncEdge",
    "SyncKind",
    "SyncScope",
    "ticks_spanned",
]

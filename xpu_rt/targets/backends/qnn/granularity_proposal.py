"""Typed partition of a computation graph into named islands.

The "granularity" of a schedule is the partition the planner uses
to assign ops to backends. Each :class:`Island` is one element of
the partition; it can cover a single op (per-op), a fusion of
several (e.g. ``Conv+Relu``, ``MatMul+Relu+Conv``), a sharded slab
of one op (``MatMul`` split into K column shards), a logical block
(``yolov8n.backbone``), or the whole network.

A :class:`GranularityProposal` is one specific partition; the
agent compares proposals (each scored from real measurements +
contention factors) and picks the best one. The MOSEK MILP
schedules over the *islands* of the chosen proposal — never over
raw ops.

Hard rule: every island carries per-backend cost cells tagged
``provenance="measured"`` or ``"analytical_bound"``. The MILP
caller rejects the latter by default; the agent has to opt in
with ``bound_only=True`` per cell and the proof report flags it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from xpu_rt.targets.backends.qnn.target_spec import (
    BackendName, OpFootprint, analytical_bound_us,
)

IslandKind = Literal["op", "fused", "sharded", "block", "whole_net"]
Provenance = Literal["measured", "analytical_bound"]


@dataclasses.dataclass(frozen=True)
class CostCell:
    """One per-backend latency entry for an island."""

    mean_us: float
    provenance: Provenance
    iters: int | None = None           # how many on-board iters fed this cell
    source: str = ""                   # free-text trace (e.g., dlc path, tool)
    rationale: str | None = None       # roofline note ("compute-bound" / "memory-bound")
    bound_only: bool = False           # caller-side opt-in to schedule against this

    def is_measured(self) -> bool:
        return self.provenance == "measured"

    def is_schedulable(self) -> bool:
        """Cells that the MILP may consume."""
        return self.provenance == "measured" or self.bound_only

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ArtifactRef:
    """On-board file that realises an island for a backend.

    Tracks the source format so the executor knows which
    ``qnn-net-run`` flag to invoke (``--dlc_path`` vs
    ``--retrieve_context``).
    """

    remote_path: str
    kind: Literal["dlc", "context_binary"]
    backend: BackendName
    sha256: str = ""                   # set when we push from host

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Island:
    """One element of a granularity partition."""

    island_id: str
    op_ids: tuple[str, ...]
    kind: IslandKind
    workload_id: str
    # Predecessor island IDs in the producer-consumer DAG of the
    # parent proposal. Equivalent of IslandVariantGroup.upstream_*.
    predecessor_island_ids: tuple[str, ...] = ()
    cost: dict[BackendName, CostCell] = dataclasses.field(default_factory=dict)
    executor_artifact: dict[BackendName, ArtifactRef | None] = dataclasses.field(
        default_factory=dict,
    )
    # Backends where the island is theoretically legal (e.g. HTA only
    # supports k3 quantised conv). Empty = all backends legal until
    # measurement / build proves otherwise.
    backend_candidates: tuple[BackendName, ...] = ()

    def schedulable_backends(self) -> list[BackendName]:
        """Backends where (cost is real OR bound_only) AND artifact exists."""
        out: list[BackendName] = []
        for b, cell in self.cost.items():
            if not cell.is_schedulable():
                continue
            artifact = self.executor_artifact.get(b)
            if artifact is None:
                continue
            if self.backend_candidates and b not in self.backend_candidates:
                continue
            out.append(b)
        return out

    def planner_visible_backends(self) -> list[BackendName]:
        """Backends with a cell (real or bound) — even if no artifact yet."""
        out: list[BackendName] = []
        for b, cell in self.cost.items():
            if self.backend_candidates and b not in self.backend_candidates:
                continue
            out.append(b)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": self.island_id,
            "op_ids": list(self.op_ids),
            "kind": self.kind,
            "workload_id": self.workload_id,
            "predecessor_island_ids": list(self.predecessor_island_ids),
            "cost": {b: c.to_dict() for b, c in self.cost.items()},
            "executor_artifact": {
                b: (a.to_dict() if a else None)
                for b, a in self.executor_artifact.items()
            },
            "backend_candidates": list(self.backend_candidates),
        }


@dataclasses.dataclass(frozen=True)
class GranularityProposal:
    """One partition of the computation into islands.

    The proposal also carries a free-text label so the agent's
    rationale can refer to it (``"per_op"``, ``"whole_net"``,
    ``"conv_relu_fused"``, etc.).
    """

    label: str
    islands: tuple[Island, ...]
    # Free-form descriptor: agent-readable rationale for why this
    # partition was generated.
    description: str = ""

    def by_id(self, island_id: str) -> Island:
        for i in self.islands:
            if i.island_id == island_id:
                return i
        raise KeyError(island_id)

    def is_fully_schedulable(self) -> bool:
        """Every island has at least one schedulable backend."""
        return all(len(i.schedulable_backends()) > 0 for i in self.islands)

    def unschedulable_islands(self) -> list[Island]:
        return [i for i in self.islands if not i.schedulable_backends()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "islands": [i.to_dict() for i in self.islands],
        }


def _island_id(workload_id: str, op_ids: Sequence[str], kind: IslandKind) -> str:
    h = hashlib.sha256(("|".join(op_ids)).encode()).hexdigest()[:8]
    return f"{workload_id}.{kind}.{h}"


# --------------------------------------------------------------------------- #
# Helpers: enumerate common proposals over a logical graph
# --------------------------------------------------------------------------- #


def propose_whole_net(
    workload_id: str,
    op_ids: Sequence[str],
    *,
    backend_candidates: Sequence[BackendName] = (),
) -> GranularityProposal:
    """One island covering all ops; the simplest proposal."""
    island = Island(
        island_id=_island_id(workload_id, op_ids, "whole_net"),
        op_ids=tuple(op_ids),
        kind="whole_net",
        workload_id=workload_id,
        backend_candidates=tuple(backend_candidates),
    )
    return GranularityProposal(
        label=f"{workload_id}.whole_net",
        islands=(island,),
        description=(
            f"Whole-net island for {workload_id}; one DLC per backend."
        ),
    )


def propose_per_op(
    workload_id: str,
    op_ids: Sequence[str],
    *,
    backend_candidates: Sequence[BackendName] = (),
) -> GranularityProposal:
    """One island per op; sequential predecessors."""
    islands: list[Island] = []
    prev_id: str | None = None
    for oid in op_ids:
        iid = _island_id(workload_id, (oid,), "op")
        islands.append(Island(
            island_id=iid,
            op_ids=(oid,),
            kind="op",
            workload_id=workload_id,
            predecessor_island_ids=(prev_id,) if prev_id else (),
            backend_candidates=tuple(backend_candidates),
        ))
        prev_id = iid
    return GranularityProposal(
        label=f"{workload_id}.per_op",
        islands=tuple(islands),
        description=(
            f"Per-op partition for {workload_id} "
            f"({len(islands)} islands)."
        ),
    )


def propose_fusions(
    workload_id: str,
    op_ids: Sequence[str],
    fusion_groups: Sequence[Sequence[str]],
    *,
    backend_candidates: Sequence[BackendName] = (),
) -> GranularityProposal:
    """Custom partition: caller declares which op subsequences fuse.

    ``fusion_groups`` is a sequence of ordered op-id tuples. Each
    tuple becomes one fused island. Ops not in any group become
    standalone op islands. Order across groups defines the
    producer-consumer chain.
    """
    covered: set[str] = set()
    for grp in fusion_groups:
        for o in grp:
            if o in covered:
                raise ValueError(f"op {o!r} appears in multiple fusion groups")
            covered.add(o)
    extra = [o for o in op_ids if o not in covered]
    # Build the partition preserving the user-given op_ids order; each
    # op belongs either to a fusion group (use its group's island) or
    # is its own singleton.
    groups_by_first_op: dict[str, tuple[str, ...]] = {
        tuple(grp)[0]: tuple(grp) for grp in fusion_groups
    }
    in_group: dict[str, tuple[str, ...]] = {}
    for grp in fusion_groups:
        for o in grp:
            in_group[o] = tuple(grp)

    islands: list[Island] = []
    prev_id: str | None = None
    i = 0
    while i < len(op_ids):
        o = op_ids[i]
        if o in groups_by_first_op:
            grp = groups_by_first_op[o]
            iid = _island_id(workload_id, grp, "fused")
            islands.append(Island(
                island_id=iid,
                op_ids=grp,
                kind="fused",
                workload_id=workload_id,
                predecessor_island_ids=(prev_id,) if prev_id else (),
                backend_candidates=tuple(backend_candidates),
            ))
            prev_id = iid
            i += len(grp)
        elif o in in_group and o != groups_by_first_op[in_group[o][0]][0]:
            # We're not at the head of the group — skip; the head
            # already consumed this op.
            i += 1
        else:
            iid = _island_id(workload_id, (o,), "op")
            islands.append(Island(
                island_id=iid,
                op_ids=(o,),
                kind="op",
                workload_id=workload_id,
                predecessor_island_ids=(prev_id,) if prev_id else (),
                backend_candidates=tuple(backend_candidates),
            ))
            prev_id = iid
            i += 1
    return GranularityProposal(
        label=f"{workload_id}.fused",
        islands=tuple(islands),
        description=(
            f"Fused partition for {workload_id}: {len(fusion_groups)} "
            f"fusion groups + {len(extra)} singletons."
        ),
    )


def propose_shards(
    workload_id: str,
    op_id: str,
    n_shards: int,
    *,
    backend_candidates: Sequence[BackendName] = (),
) -> GranularityProposal:
    """Single op split into ``n_shards`` parallel slabs.

    Useful for matmul-style ops where the output dimension can be
    sliced and each slab runs on a different backend concurrently.
    The shards are siblings (no predecessor edges between them).
    """
    if n_shards < 2:
        raise ValueError(f"n_shards must be >= 2 (got {n_shards})")
    islands: list[Island] = []
    for k in range(n_shards):
        shard_op = f"{op_id}.shard{k}"
        islands.append(Island(
            island_id=_island_id(workload_id, (shard_op,), "sharded"),
            op_ids=(shard_op,),
            kind="sharded",
            workload_id=workload_id,
            predecessor_island_ids=(),
            backend_candidates=tuple(backend_candidates),
        ))
    return GranularityProposal(
        label=f"{workload_id}.shard{n_shards}",
        islands=tuple(islands),
        description=(
            f"Sharded partition: {op_id} split into {n_shards} parallel "
            f"slabs for {workload_id}."
        ),
    )


def annotate_with_analytical_bounds(
    proposal: GranularityProposal,
    op_footprints: dict[str, OpFootprint],
    backends: Iterable[BackendName],
) -> GranularityProposal:
    """Populate ``cost`` cells with roofline bounds where missing.

    Cells already carrying a measurement are left untouched. Newly
    computed cells are tagged ``provenance="analytical_bound"`` and
    ``bound_only=False`` — caller must opt in explicitly to schedule.
    """
    new_islands: list[Island] = []
    for island in proposal.islands:
        new_cost = dict(island.cost)
        # Sum the per-op footprints to get the island's footprint.
        flops = sum(op_footprints[o].flops
                    for o in island.op_ids if o in op_footprints)
        bread = sum(op_footprints[o].bytes_read
                    for o in island.op_ids if o in op_footprints)
        bwrite = sum(op_footprints[o].bytes_written
                     for o in island.op_ids if o in op_footprints)
        fp = OpFootprint(flops=flops, bytes_read=bread, bytes_written=bwrite)
        for b in backends:
            if b in new_cost and new_cost[b].is_measured():
                continue
            mean_us, rationale = analytical_bound_us(fp, b)
            new_cost[b] = CostCell(
                mean_us=mean_us,
                provenance="analytical_bound",
                source="target_spec.analytical_bound_us",
                rationale=rationale,
            )
        new_islands.append(dataclasses.replace(island, cost=new_cost))
    return dataclasses.replace(proposal, islands=tuple(new_islands))

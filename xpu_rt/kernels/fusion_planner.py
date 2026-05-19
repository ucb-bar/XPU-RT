"""Graph-level fusion planner.

Today the cost-modeled :func:`xpu_rt.kernels.fusion_oracle.should_fuse` and
:func:`xpu_rt.kernels.granularity_oracle.recommend_granularity` exist as
single-pair / single-region primitives but nothing actually walks a graph
through them. This module is the missing piece: it consumes a
:class:`xpu_rt.ir.payload.contract_graph.ContractGraph` and emits a
:class:`FusionPlan` partitioning every node into a :class:`FusionCluster`.

The vanilla KB path can't get this — it never sees a graph. The whole
pipeline-level study (see
``results/comparison/pipeline_level/report.md``) hinges on this planner
making honest fusion decisions and the agentic codegen path landing
them on the actual hardware.

Pipeline:

    ContractGraph  ──► FusionPlanner.plan(graph)  ──► FusionPlan
        │                       │                          │
        │           pairwise should_fuse                   │
        │           per-cluster recommend_granularity      │
        │                                                  ▼
        │                                          clusters: tuple[FusionCluster, …]
        │                                          estimated_speedup: float
        ▼
    nodes carry KernelContract (v1 from IR walk); the planner lifts each
    node to a :class:`KernelContractV3` for the oracles to consume, using
    the target's :class:`HardwareEnvelope` for the cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from xpu_rt.ir.payload.contract_graph import ContractGraph, ContractNode
from xpu_rt.ir.payload.contracts import KernelContract as KernelContractV1
from xpu_rt.ir.payload.contracts import LayoutKind as V1LayoutKind
from xpu_rt.kernels.contract_v3 import (
    DispatchModel,
    DispatchSpec,
    ExecutionEnvelope,
    FusionPolicy,
    HardwareEnvelope,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    LayoutKind as V3LayoutKind,
    MemorySpec,
    MemoryTier,
    OrchestrationSpec,
    ShapeClass,
    StaticAttr,
    TensorIO,
)
from xpu_rt.kernels.fusion_oracle import (
    FusionDecision,
    FusionVerdict,
    should_fuse,
)
from xpu_rt.kernels.granularity_oracle import (
    GranularityVerdict,
    recommend_granularity,
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionCluster:
    """A connected subset of nodes the planner has decided to fuse.

    Attributes:
        cluster_id: ``"cluster_<index>"``, stable within the
            containing plan.
        member_op_ids: ``ContractNode.op_id`` of every node in this
            cluster, in topological order (producer-first).
        rationale: Human-readable reason from the oracles. Carries
            the verbatim ``FusionVerdict.reason`` strings and the
            ``GranularityVerdict.reason`` so downstream report
            consumers can show the *why* alongside the *what*.
        estimated_speedup: Combined pairwise speedup estimate from the
            fusion oracle for chains of length ≥ 2; ``1.0`` for
            singletons.
    """

    cluster_id: str
    member_op_ids: tuple[str, ...]
    rationale: str
    estimated_speedup: float = 1.0


@dataclass(frozen=True)
class FusionPlan:
    """Partition of a :class:`ContractGraph` into fusion clusters.

    The planner guarantees that every node in :attr:`graph` belongs to
    exactly one cluster (``clusters`` is a partition of
    ``graph.nodes``).

    Attributes:
        graph: The input graph the plan was computed against.
        clusters: Tuple of clusters, in producer-first order.
        estimated_speedup: Geometric mean of per-cluster estimated
            speedups, weighted by cluster size. For a graph with no
            multi-node clusters this is ``1.0``.
        envelope_target: ``envelope.target_name`` of the envelope the
            planner ran against (recorded so reports show *which*
            target the verdicts apply to).
    """

    graph: ContractGraph
    clusters: tuple[FusionCluster, ...]
    estimated_speedup: float
    envelope_target: str
    per_pair_verdicts: tuple[tuple[str, str, FusionVerdict], ...] = field(default_factory=tuple)
    per_cluster_granularity: tuple[tuple[str, GranularityVerdict], ...] = field(default_factory=tuple)

    def cluster_for(self, op_id: str) -> FusionCluster:
        for c in self.clusters:
            if op_id in c.member_op_ids:
                return c
        raise KeyError(f"op_id {op_id!r} is not assigned to any cluster")


# ---------------------------------------------------------------------------
# v1 → v3 lifter (planner-internal)
# ---------------------------------------------------------------------------


_ARCHETYPE_BY_OP_FAMILY: dict[str, KernelArchetype] = {
    # Tiled compute
    "matmul": KernelArchetype.COMPUTE_TILED,
    "linalg.matmul": KernelArchetype.COMPUTE_TILED,
    "batch_matmul": KernelArchetype.COMPUTE_TILED,
    "linalg.batch_matmul": KernelArchetype.COMPUTE_TILED,
    "conv_2d_nhwc_hwcf": KernelArchetype.COMPUTE_TILED,
    "linalg.conv_2d_nhwc_hwcf": KernelArchetype.COMPUTE_TILED,
    "linear": KernelArchetype.COMPUTE_TILED,
    # Reductions
    "softmax": KernelArchetype.REDUCE,
    "layer_norm": KernelArchetype.REDUCE,
    "rms_norm": KernelArchetype.REDUCE,
    # Pointwise
    "add": KernelArchetype.POINTWISE,
    "mul": KernelArchetype.POINTWISE,
    "sub": KernelArchetype.POINTWISE,
    "div": KernelArchetype.POINTWISE,
    "arith.addf": KernelArchetype.POINTWISE,
    "arith.mulf": KernelArchetype.POINTWISE,
    # Activations
    "silu": KernelArchetype.ACTIVATION,
    "gelu": KernelArchetype.ACTIVATION,
    "relu": KernelArchetype.ACTIVATION,
    "tanh": KernelArchetype.ACTIVATION,
    "sigmoid": KernelArchetype.ACTIVATION,
    # Memory
    "where": KernelArchetype.MEMORY,
    "aten_where": KernelArchetype.MEMORY,
    "transpose": KernelArchetype.MEMORY,
}


def _classify_archetype(op_name: str) -> KernelArchetype:
    """Best-effort archetype lookup from a v1 contract's ``op_name``.

    Falls back to POINTWISE — the safest default for elementwise-ish
    ops the planner sees through ``func.call`` wrappers. The fusion
    oracle's eligibility gate will reject mismatches downstream.
    """
    name = op_name.lower()
    if name in _ARCHETYPE_BY_OP_FAMILY:
        return _ARCHETYPE_BY_OP_FAMILY[name]
    # strip "aten_" prefix + common torch suffixes
    base = name.removeprefix("aten_")
    for suffix in ("_default", "_tensor", "_scalar", "_self_int"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base in _ARCHETYPE_BY_OP_FAMILY:
        return _ARCHETYPE_BY_OP_FAMILY[base]
    # last-ditch: anything that looks reduce-y
    if "reduce" in base or "norm" in base:
        return KernelArchetype.REDUCE
    return KernelArchetype.POINTWISE


def _v1_layout_to_v3(layout: V1LayoutKind) -> V3LayoutKind:
    mapping = {
        V1LayoutKind.ROW_MAJOR: V3LayoutKind.ROW_MAJOR,
        V1LayoutKind.COLUMN_MAJOR: V3LayoutKind.COLUMN_MAJOR,
        V1LayoutKind.CUSTOM_STRIDES: V3LayoutKind.OPAQUE,
        V1LayoutKind.ANY: V3LayoutKind.ROW_MAJOR,
    }
    return mapping.get(layout, V3LayoutKind.ROW_MAJOR)


def _v1_shape_to_class(shape: tuple[int, ...]) -> ShapeClass:
    return ShapeClass(dims=tuple(d if d > 0 else None for d in shape))


def _ensure_unique_name(base: str, used: set[str]) -> str:
    if base and base not in used:
        used.add(base)
        return base
    i = 0
    while True:
        candidate = f"{base}_{i}" if base else f"t_{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def lift_v1_to_v3(
    contract: KernelContractV1,
    envelope: HardwareEnvelope,
    *,
    op_id_hint: str = "",
) -> KernelContractV3:
    """Lift a v1 ``KernelContract`` to a v3 ``KernelContractV3``.

    This is a **planner-internal** lift — it materialises the v3 shape
    just enough to feed :func:`should_fuse` and
    :func:`recommend_granularity`. The lift is not lossless; the
    resulting v3 contract is for cost-model consumption, not for
    codegen. The MegaContractEmitter (P1.4) does a richer lift.

    Args:
        contract: A v1 :class:`KernelContractV1` produced by
            :func:`xpu_rt.ir.payload.contracts._extract_op_contract`.
        envelope: The target :class:`HardwareEnvelope` to attach.
        op_id_hint: Optional disambiguator stamped into the IO operand
            names so adjacent contracts in a chain do not clash on
            ``IOContract`` name-uniqueness validation.

    Returns:
        A validated :class:`KernelContractV3`.
    """
    md = contract.metadata or {}
    input_shapes: list[tuple[int, ...]] = list(md.get("input_shapes", ()))
    output_shapes: list[tuple[int, ...]] = list(md.get("output_shapes", ()))

    # Concrete dtype string — pick a deterministic representative from
    # the v1 supported_dtypes set so dtype_class is stable.
    dtypes = sorted(contract.supported_dtypes) if contract.supported_dtypes else ["f32"]
    primary_dtype = dtypes[0]

    suffix = f"_{op_id_hint}" if op_id_hint else ""
    used: set[str] = set()
    inputs: list[TensorIO] = []
    for i, shape in enumerate(input_shapes):
        layout_v3 = _v1_layout_to_v3(
            contract.input_layouts[i].kind if i < len(contract.input_layouts) else V1LayoutKind.ROW_MAJOR
        )
        inputs.append(
            TensorIO(
                name=_ensure_unique_name(f"in_{i}{suffix}", used),
                shape=_v1_shape_to_class(shape),
                dtype_class=(primary_dtype,),
                layout=layout_v3,
            )
        )
    # COMPUTE_TILED requires ≥ 2 inputs — pad with a placeholder if the
    # IR only surfaced one. (Common when a matmul is wrapped behind
    # ``func.call`` and the second operand is implicit.)
    archetype = _classify_archetype(contract.op_name)
    if archetype is KernelArchetype.COMPUTE_TILED and len(inputs) < 2:
        inputs.append(
            TensorIO(
                name=_ensure_unique_name(f"in_pad{suffix}", used),
                shape=ShapeClass(dims=(None, None)),
                dtype_class=(primary_dtype,),
            )
        )

    outputs: list[TensorIO] = []
    for i, shape in enumerate(output_shapes):
        layout_v3 = _v1_layout_to_v3(
            contract.output_layouts[i].kind if i < len(contract.output_layouts) else V1LayoutKind.ROW_MAJOR
        )
        outputs.append(
            TensorIO(
                name=_ensure_unique_name(f"out_{i}{suffix}", used),
                shape=_v1_shape_to_class(shape),
                dtype_class=(primary_dtype,),
                layout=layout_v3,
            )
        )
    if not outputs:
        outputs.append(
            TensorIO(
                name=_ensure_unique_name(f"out{suffix}", used),
                shape=ShapeClass(dims=(None,)),
                dtype_class=(primary_dtype,),
            )
        )

    # Archetype-specific required attributes (validated by
    # KernelContractV3.__post_init__).
    attributes: list[StaticAttr] = []
    if archetype is KernelArchetype.REDUCE:
        attributes.append(StaticAttr(name="axis", value=-1))
    elif archetype is KernelArchetype.MEMORY:
        attributes.append(StaticAttr(name="kind", value=contract.op_name))

    io = IOContract(inputs=tuple(inputs), outputs=tuple(outputs), attributes=tuple(attributes))

    # The fusion oracle reads `producer.orchestration.fusion.is_boundary`
    # and `fusable_with`. Map v1's `fusable` flag onto these.
    if archetype is KernelArchetype.COMPUTE_TILED:
        # COMPUTE_TILED ops are typically fusion boundaries on their
        # *output* side only when v1 says fusable=False. Most matmuls
        # surface fusable=False in v1 (kernel boundary by convention),
        # which would block all fusion. The planner has to override
        # that: matmul→pointwise is the canonical Gemmini fusion
        # pattern, so we permit it.
        is_boundary = False
        fusable_with: tuple[str, ...] = ("pointwise", "activation", "reduce")
    else:
        is_boundary = not contract.fusable
        fusable_with = ("pointwise", "activation", "reduce", "compute_tiled")

    exec_env = ExecutionEnvelope(hardware=envelope)
    orchestration = OrchestrationSpec(
        execution=exec_env,
        memory=MemorySpec(
            input_tiers=tuple(MemoryTier.SCRATCHPAD for _ in inputs),
            output_tiers=tuple(MemoryTier.SCRATCHPAD for _ in outputs),
        ),
        fusion=FusionPolicy(is_boundary=is_boundary, fusable_with=fusable_with),
        dispatch=DispatchSpec(model=DispatchModel.ASYNC),
    )

    return KernelContractV3(
        op_name=contract.op_name,
        archetype=archetype,
        io=io,
        orchestration=orchestration,
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class FusionPlanner:
    """Drive :func:`should_fuse` + :func:`recommend_granularity` over a
    :class:`ContractGraph`.

    The planner is stateless across :meth:`plan` calls — it does not
    cache verdicts; the oracles are cheap and the graphs we run over
    are small. Re-running ``plan`` on the same graph with the same
    envelope returns the same partition (modulo oracle non-determinism,
    which is documented at ``shared_store.context_brief`` and does not
    affect the partition itself).

    Algorithm:

    1. Walk ``graph.topological_order``. For each consecutive pair
       (producer, consumer) connected by a direct edge, call
       :func:`should_fuse`. If the verdict is ``FUSE``, extend the
       current cluster candidate. Otherwise close the candidate and
       start a new one.
    2. After clustering, run :func:`recommend_granularity` over each
       cluster's contracts. If the granularity oracle returns anything
       other than ``MEGA`` for a multi-node cluster, split the cluster
       back to singletons (the granularity oracle has veto power).
    """

    def __init__(self, envelope: HardwareEnvelope) -> None:
        self.envelope = envelope

    def plan(self, graph: ContractGraph) -> FusionPlan:
        v3_by_id: dict[str, KernelContractV3] = {
            nid: lift_v1_to_v3(graph.nodes[nid].contract, self.envelope, op_id_hint=nid)
            for nid in graph.topological_order
        }
        adj_consumers: dict[str, set[str]] = {nid: set() for nid in graph.topological_order}
        for e in graph.edges:
            if e.producer_id in adj_consumers:
                adj_consumers[e.producer_id].add(e.consumer_id)

        pair_verdicts: list[tuple[str, str, FusionVerdict]] = []
        # Greedy cluster grow: only extend the current cluster when the
        # head's *direct* successor in topological order is also its
        # direct consumer via an edge AND the fusion oracle says FUSE.
        clusters_op_ids: list[list[str]] = []
        current: list[str] = []
        topo = graph.topological_order

        for i, nid in enumerate(topo):
            if not current:
                current.append(nid)
                continue
            # Try to extend the current cluster with nid: nid must be a
            # consumer of the cluster tail.
            tail = current[-1]
            edge_exists = nid in adj_consumers.get(tail, set())
            if not edge_exists:
                clusters_op_ids.append(current)
                current = [nid]
                continue
            verdict = should_fuse(v3_by_id[tail], v3_by_id[nid])
            pair_verdicts.append((tail, nid, verdict))
            if verdict.decision is FusionDecision.FUSE:
                current.append(nid)
            else:
                clusters_op_ids.append(current)
                current = [nid]
        if current:
            clusters_op_ids.append(current)

        # Granularity veto pass: for any multi-node cluster, ask the
        # granularity oracle. If it does not return MEGA, split the
        # cluster into singletons (preserving topo order).
        final_clusters: list[FusionCluster] = []
        granularity_verdicts: list[tuple[str, GranularityVerdict]] = []
        speedups: list[float] = []
        for idx, members in enumerate(clusters_op_ids):
            if len(members) == 1:
                cid = f"cluster_{idx:02d}"
                final_clusters.append(
                    FusionCluster(
                        cluster_id=cid,
                        member_op_ids=tuple(members),
                        rationale="singleton — no upstream FUSE verdict to extend",
                        estimated_speedup=1.0,
                    )
                )
                continue
            region = [v3_by_id[m] for m in members]
            gv = recommend_granularity(region, self.envelope)
            granularity_verdicts.append((f"cluster_{idx:02d}", gv))
            if gv.granularity.value == "mega":
                cid = f"cluster_{idx:02d}"
                # Speedup = chain_speedup_estimate; if zero (shouldn't be),
                # fall back to product of pairwise oracle ratios.
                if gv.chain_speedup_estimate > 0:
                    speedup = gv.chain_speedup_estimate
                else:
                    speedup = 1.0
                    for i in range(len(members) - 1):
                        pv = should_fuse(v3_by_id[members[i]], v3_by_id[members[i + 1]])
                        speedup *= pv.est_speedup_ratio
                final_clusters.append(
                    FusionCluster(
                        cluster_id=cid,
                        member_op_ids=tuple(members),
                        rationale=f"MEGA: {gv.reason}",
                        estimated_speedup=speedup,
                    )
                )
                speedups.append(speedup)
            else:
                # Veto — fall back to singletons. Keep granularity verdict
                # in the plan so the report can explain why.
                for k, m in enumerate(members):
                    final_clusters.append(
                        FusionCluster(
                            cluster_id=f"cluster_{idx:02d}_{k:02d}",
                            member_op_ids=(m,),
                            rationale=f"granularity oracle vetoed cluster: {gv.reason}",
                            estimated_speedup=1.0,
                        )
                    )

        # Geomean of per-cluster speedups (weighting by cluster size).
        if speedups:
            import math

            log_total = 0.0
            weight_total = 0
            for c in final_clusters:
                if c.estimated_speedup > 1.0:
                    log_total += math.log(c.estimated_speedup) * len(c.member_op_ids)
                    weight_total += len(c.member_op_ids)
            est = math.exp(log_total / weight_total) if weight_total else 1.0
        else:
            est = 1.0

        return FusionPlan(
            graph=graph,
            clusters=tuple(final_clusters),
            estimated_speedup=est,
            envelope_target=self.envelope.target_name,
            per_pair_verdicts=tuple(pair_verdicts),
            per_cluster_granularity=tuple(granularity_verdicts),
        )


def plan_fusion(graph: ContractGraph, envelope: HardwareEnvelope) -> FusionPlan:
    """Convenience: ``FusionPlanner(envelope).plan(graph)``."""
    return FusionPlanner(envelope).plan(graph)


__all__ = [
    "FusionCluster",
    "FusionPlan",
    "FusionPlanner",
    "lift_v1_to_v3",
    "plan_fusion",
]

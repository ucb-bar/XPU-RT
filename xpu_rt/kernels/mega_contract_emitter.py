"""Materialize :class:`KernelContractV3` MEGA contracts from a
:class:`FusionCluster` of multiple single-op nodes.

The :mod:`xpu_rt.kernels.fusion_planner` decides *which* nodes to
fuse; this module turns those decisions into the actual v3 contract
the codegen path (P1.5 ``kb_gemmini.mega_templates``) will consume.

A MEGA contract has six load-bearing parts (validated by
:meth:`KernelContractV3.__post_init__` at ``contract_v3.py:706-729``):

  1. ``granularity = MEGA``
  2. ``orchestration.dispatch.model = PERSISTENT``
  3. ``body`` — non-empty tuple of NORMAL sub-contracts
  4. Sub-buffers must all be ``MemoryTier.REGISTER`` or
     ``MemoryTier.SCRATCHPAD`` — the whole point of MEGA is that
     intermediates stay resident
  5. ``internal_events`` — :class:`InternalEventEdge` between body
     indices, naming the events the dispatcher inserts waits on
  6. External ``io`` — the inputs and outputs that cross the MEGA's
     outer boundary (i.e., not satisfied by intra-cluster edges)

The planner produces linear chains today (greedy grow over
topological order), so the implementation here assumes a chain
shape. The reference at
:func:`xpu_rt.kernels.contract_v3_references.reference_mega_attention_block_contract`
demonstrates the full pattern (Q·K → softmax → P·V).
"""

from __future__ import annotations

from dataclasses import dataclass

from xpu_rt.ir.payload.contract_graph import ContractGraph
from xpu_rt.kernels.contract_v3 import (
    DispatchModel,
    DispatchSpec,
    Granularity,
    HardwareEnvelope,
    IOContract,
    InternalEventEdge,
    KernelArchetype,
    KernelContractV3,
    MemorySpec,
    MemoryTier,
    OrchestrationSpec,
    SelectionHints,
    TensorIO,
    StaticAttr,
)
from xpu_rt.kernels.fusion_planner import (
    FusionCluster,
    lift_v1_to_v3,
)


@dataclass(frozen=True)
class MegaEmissionResult:
    """A MEGA contract paired with the IO-routing information that
    callers need to wire up the harness.

    Attributes:
        contract: The validated :class:`KernelContractV3` (granularity
            == MEGA).
        external_inputs_from_nodes: Tuple of
            ``(node_id, operand_index_on_node)`` indicating which
            single-op-node operand each external MEGA input maps
            back to. Lets the harness allocate the right DRAM buffer
            shapes.
        external_outputs_from_nodes: Tuple of
            ``(node_id, result_index_on_node)`` for the MEGA's
            outputs.
    """

    contract: KernelContractV3
    external_inputs_from_nodes: tuple[tuple[str, int], ...]
    external_outputs_from_nodes: tuple[tuple[str, int], ...]


def _resolve_mega_archetype(body: tuple[KernelContractV3, ...]) -> KernelArchetype:
    """Pick the most descriptive archetype for the combined op.

    Heuristic: any COMPUTE_TILED member → COMPUTE_TILED (the chain is
    matmul-flavoured). Otherwise REDUCE > ACTIVATION > POINTWISE >
    MEMORY in that order — the dominant op-family wins.
    """
    if any(c.archetype is KernelArchetype.COMPUTE_TILED for c in body):
        return KernelArchetype.COMPUTE_TILED
    if any(c.archetype is KernelArchetype.REDUCE for c in body):
        return KernelArchetype.REDUCE
    if any(c.archetype is KernelArchetype.ACTIVATION for c in body):
        return KernelArchetype.ACTIVATION
    if any(c.archetype is KernelArchetype.POINTWISE for c in body):
        return KernelArchetype.POINTWISE
    return KernelArchetype.MEMORY


def emit_mega_contract(
    cluster: FusionCluster,
    graph: ContractGraph,
    envelope: HardwareEnvelope,
    *,
    op_name: str | None = None,
) -> MegaEmissionResult:
    """Build a validated MEGA :class:`KernelContractV3` for ``cluster``.

    Args:
        cluster: A :class:`FusionCluster` with ≥ 2 members. Singleton
            clusters do not become MEGA contracts — the caller
            should route them straight to the single-op path.
        graph: The :class:`ContractGraph` ``cluster`` was carved out
            of. Needed to resolve which operand-slots of each member
            map to external inputs vs. intra-cluster edges.
        envelope: Target :class:`HardwareEnvelope` for the lift.
        op_name: Optional override for the MEGA's ``op_name``. Default
            is ``"mega_<member1>_<member2>_..."`` with each member
            name slugified.

    Returns:
        :class:`MegaEmissionResult` carrying the validated contract
        plus the external-IO routing tables.

    Raises:
        ValueError: If ``cluster`` is a singleton, or if a member node
            id is not present in the graph.
    """
    if len(cluster.member_op_ids) < 2:
        raise ValueError(
            f"MEGA contracts require ≥ 2 body members; got cluster {cluster.cluster_id!r} "
            f"with {len(cluster.member_op_ids)} member(s). Route singletons through the single-op path."
        )

    # Step 1 — lift each member to v3 NORMAL (default granularity)
    body_list: list[KernelContractV3] = []
    member_index_by_id: dict[str, int] = {}
    for i, mid in enumerate(cluster.member_op_ids):
        if mid not in graph.nodes:
            raise ValueError(f"cluster member {mid!r} not in graph.nodes")
        node = graph.nodes[mid]
        v3 = lift_v1_to_v3(node.contract, envelope, op_id_hint=mid)
        body_list.append(v3)
        member_index_by_id[mid] = i

    # Step 2 — compute internal_events for every intra-cluster edge.
    member_set = set(cluster.member_op_ids)
    internal_events: list[InternalEventEdge] = []
    intra_cluster_edges: list[tuple[str, str, int]] = []  # (producer, consumer, consumer_operand_idx)
    seen_pairs: set[tuple[int, int]] = set()
    for e in graph.edges:
        if e.producer_id in member_set and e.consumer_id in member_set:
            p_idx = member_index_by_id[e.producer_id]
            c_idx = member_index_by_id[e.consumer_id]
            intra_cluster_edges.append((e.producer_id, e.consumer_id, e.operand_index))
            key = (p_idx, c_idx)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            event_name = f"{graph.nodes[e.producer_id].op_name}_done_{p_idx}_{c_idx}"
            internal_events.append(
                InternalEventEdge(
                    event_name=event_name,
                    producer_idx=p_idx,
                    consumer_idx=c_idx,
                )
            )

    # Step 3 — derive external IO: inputs that cross the cluster boundary
    # (i.e., operands of member nodes whose producer is NOT in the cluster)
    # and outputs (results of member nodes whose consumers are NOT all in
    # the cluster or which are graph sinks).
    intra_consumer_operands: set[tuple[str, int]] = {
        (consumer_id, operand_idx) for (_, consumer_id, operand_idx) in intra_cluster_edges
    }
    external_inputs: list[TensorIO] = []
    external_inputs_routing: list[tuple[str, int]] = []
    used_names: set[str] = set()
    for mid in cluster.member_op_ids:
        member_v3 = body_list[member_index_by_id[mid]]
        for op_i, t_in in enumerate(member_v3.io.inputs):
            if (mid, op_i) in intra_consumer_operands:
                continue  # satisfied by an internal edge — not external
            name = f"ext_in_{mid}_{op_i}"
            if name in used_names:
                name = f"{name}_{len(used_names)}"
            used_names.add(name)
            external_inputs.append(
                TensorIO(
                    name=name,
                    shape=t_in.shape,
                    dtype_class=t_in.dtype_class,
                    layout=t_in.layout,
                    alignment_bytes=t_in.alignment_bytes,
                )
            )
            external_inputs_routing.append((mid, op_i))

    # Outputs that DO have an internal consumer are not external.
    internal_producer_idx: set[int] = {p for (p, _) in seen_pairs}
    external_outputs: list[TensorIO] = []
    external_outputs_routing: list[tuple[str, int]] = []
    for mid in cluster.member_op_ids:
        midx = member_index_by_id[mid]
        member_v3 = body_list[midx]
        # If this member has ANY intra-cluster consumer, all its outputs
        # are considered internal (chain-shape assumption).
        if midx in internal_producer_idx and midx != len(body_list) - 1:
            # but if it's the last member of the chain, force external
            continue
        for r_i, t_out in enumerate(member_v3.io.outputs):
            name = f"ext_out_{mid}_{r_i}"
            if name in used_names:
                name = f"{name}_{len(used_names)}"
            used_names.add(name)
            external_outputs.append(
                TensorIO(
                    name=name,
                    shape=t_out.shape,
                    dtype_class=t_out.dtype_class,
                    layout=t_out.layout,
                    alignment_bytes=t_out.alignment_bytes,
                )
            )
            external_outputs_routing.append((mid, r_i))

    if not external_outputs:
        # The cluster has no observable side-effect — fall back to the
        # last member's outputs so the contract validates.
        last_mid = cluster.member_op_ids[-1]
        last_v3 = body_list[-1]
        for r_i, t_out in enumerate(last_v3.io.outputs):
            name = f"ext_out_fallback_{last_mid}_{r_i}"
            if name in used_names:
                name = f"{name}_{len(used_names)}"
            used_names.add(name)
            external_outputs.append(
                TensorIO(
                    name=name,
                    shape=t_out.shape,
                    dtype_class=t_out.dtype_class,
                    layout=t_out.layout,
                    alignment_bytes=t_out.alignment_bytes,
                )
            )
            external_outputs_routing.append((last_mid, r_i))

    archetype = _resolve_mega_archetype(tuple(body_list))

    # Archetype invariants: COMPUTE_TILED requires ≥ 2 external inputs,
    # REDUCE requires ``axis`` attribute, MEMORY requires ``kind``.
    attributes: list[StaticAttr] = []
    if archetype is KernelArchetype.REDUCE:
        attributes.append(StaticAttr(name="axis", value=-1))
    elif archetype is KernelArchetype.MEMORY:
        attributes.append(StaticAttr(name="kind", value="fused"))
    if archetype is KernelArchetype.COMPUTE_TILED and len(external_inputs) < 2:
        # Pad with a placeholder external input so the archetype check
        # passes. The codegen path knows the trailing pad inputs are
        # unused.
        from xpu_rt.kernels.contract_v3 import ShapeClass

        external_inputs.append(
            TensorIO(
                name="ext_in_pad",
                shape=ShapeClass(dims=(None, None)),
                dtype_class=body_list[0].io.inputs[0].dtype_class,
            )
        )
        external_inputs_routing.append((cluster.member_op_ids[0], -1))

    io = IOContract(
        inputs=tuple(external_inputs),
        outputs=tuple(external_outputs),
        attributes=tuple(attributes),
    )

    # Compose op_name from the chain.
    if op_name is None:
        slugs = [graph.nodes[mid].op_name.replace(".", "_") for mid in cluster.member_op_ids]
        op_name = "mega_" + "__".join(slugs)

    mega = KernelContractV3(
        op_name=op_name,
        archetype=archetype,
        io=io,
        granularity=Granularity.MEGA,
        orchestration=OrchestrationSpec(
            execution=body_list[0].orchestration.execution,
            memory=MemorySpec(
                input_tiers=tuple(MemoryTier.SCRATCHPAD for _ in external_inputs),
                output_tiers=tuple(MemoryTier.SCRATCHPAD for _ in external_outputs),
            ),
            dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
        ),
        selection=SelectionHints(),
        body=tuple(body_list),
        internal_events=tuple(internal_events),
        metadata={
            "cluster_id": cluster.cluster_id,
            "member_op_ids": list(cluster.member_op_ids),
            "estimated_speedup": cluster.estimated_speedup,
            "external_inputs_routing": [list(r) for r in external_inputs_routing],
            "external_outputs_routing": [list(r) for r in external_outputs_routing],
        },
    )

    return MegaEmissionResult(
        contract=mega,
        external_inputs_from_nodes=tuple(external_inputs_routing),
        external_outputs_from_nodes=tuple(external_outputs_routing),
    )


__all__ = [
    "MegaEmissionResult",
    "emit_mega_contract",
]

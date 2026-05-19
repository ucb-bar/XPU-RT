"""Tests for :mod:`xpu_rt.kernels.mega_contract_emitter`.

The emitter turns a :class:`FusionCluster` into a validated v3
MEGA :class:`KernelContractV3`. These tests build synthetic clusters
and assert that the resulting contract:

  * passes :meth:`KernelContractV3.__post_init__` validation
    (which enforces granularity=MEGA, dispatch=PERSISTENT,
    SCRATCHPAD-only sub-buffers, in-range event indices);
  * has the right body[] length and ``internal_events`` chain;
  * routes external IO back to the originating single-op nodes.
"""

from __future__ import annotations

import pytest

from xpu_rt.ir.payload.contract_graph import (
    ContractEdge,
    ContractNode,
    build_contract_graph_from_nodes,
)
from xpu_rt.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement
from xpu_rt.kernels.contract_v3 import (
    DispatchModel,
    Granularity,
    HardwareEnvelope,
    KernelArchetype,
    MemoryTier,
)
from xpu_rt.kernels.fusion_planner import FusionCluster
from xpu_rt.kernels.mega_contract_emitter import (
    MegaEmissionResult,
    emit_mega_contract,
)


def _gemmini_envelope() -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name="gemmini_mx",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
    )


def _v1_activation_contract(op_name: str, shape: tuple[int, ...]) -> KernelContract:
    return KernelContract(
        op_name=op_name,
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"},
        cost=CostEstimate(flops=shape[0] * shape[1]),
        fusable=True,
        metadata={
            "input_shapes": [shape],
            "output_shapes": [shape],
            "region_id": op_name,
            "dispatch_id": op_name,
        },
    )


def _v1_matmul_contract(
    region_id: str,
    in_a: tuple[int, ...],
    in_b: tuple[int, ...],
    out: tuple[int, ...],
) -> KernelContract:
    # The op_name carries the canonical family ("matmul") — this is
    # what _classify_archetype matches on. region_id is the dispatch-
    # boundary tag stamped by the IR walker (REQ-026).
    return KernelContract(
        op_name="matmul",
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR), LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={"i8"},
        cost=CostEstimate(flops=out[0] * out[1] * in_a[1] * 2),
        fusable=False,
        metadata={
            "input_shapes": [in_a, in_b],
            "output_shapes": [out],
            "region_id": region_id,
            "dispatch_id": region_id,
        },
    )


def test_emit_mega_three_pointwise_chain_validates() -> None:
    env = _gemmini_envelope()
    add = _v1_activation_contract("add", (64, 32))
    silu = _v1_activation_contract("silu", (64, 32))
    mul = _v1_activation_contract("mul", (64, 32))
    nodes = [
        ContractNode(op_id="n0", contract=add, op_name="add", region_id="r0"),
        ContractNode(op_id="n1", contract=silu, op_name="silu", region_id="r1"),
        ContractNode(op_id="n2", contract=mul, op_name="mul", region_id="r2"),
    ]
    edges = [
        ContractEdge(producer_id="n0", consumer_id="n1", operand_index=0,
                     tensor_shape=(64, 32), dtype="i8", bytes_per_element=1),
        ContractEdge(producer_id="n1", consumer_id="n2", operand_index=0,
                     tensor_shape=(64, 32), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)

    cluster = FusionCluster(
        cluster_id="cluster_00",
        member_op_ids=("n0", "n1", "n2"),
        rationale="manual test cluster",
        estimated_speedup=2.5,
    )
    result = emit_mega_contract(cluster, graph, env)
    assert isinstance(result, MegaEmissionResult)

    mega = result.contract
    # Granularity / dispatch / body structure
    assert mega.granularity is Granularity.MEGA
    assert mega.orchestration.dispatch.model is DispatchModel.PERSISTENT
    assert len(mega.body) == 3
    for sub in mega.body:
        # No nested MEGA — and every sub-buffer must be in
        # REGISTER/SCRATCHPAD per contract_v3.py:706-729.
        assert sub.granularity is not Granularity.MEGA
        for tier in (*sub.orchestration.memory.input_tiers, *sub.orchestration.memory.output_tiers):
            assert tier in (MemoryTier.REGISTER, MemoryTier.SCRATCHPAD)

    # internal_events: 2 edges (n0→n1, n1→n2)
    assert len(mega.internal_events) == 2
    for edge in mega.internal_events:
        assert 0 <= edge.producer_idx < 3
        assert 0 <= edge.consumer_idx < 3
        assert edge.producer_idx < edge.consumer_idx
    indices = {(e.producer_idx, e.consumer_idx) for e in mega.internal_events}
    assert indices == {(0, 1), (1, 2)}

    # External IO routing: external input only on n0 (n1, n2 read from
    # internal scratchpad). External output only on n2 (n0, n1 outputs
    # are consumed internally).
    assert all(mid == "n0" for (mid, _) in result.external_inputs_from_nodes)
    assert all(mid == "n2" for (mid, _) in result.external_outputs_from_nodes)


def test_emit_mega_matmul_silu_matmul_chain_validates() -> None:
    """MLP-block-shaped chain: matmul → silu → matmul. Mirrors the
    SmolVLA MLP pattern. Validates COMPUTE_TILED archetype routing."""
    env = _gemmini_envelope()
    m1 = _v1_matmul_contract("m1", (64, 720), (720, 1440), (64, 1440))
    silu = _v1_activation_contract("silu", (64, 1440))
    m2 = _v1_matmul_contract("m2", (64, 1440), (1440, 720), (64, 720))
    nodes = [
        ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="r0"),
        ContractNode(op_id="n_silu", contract=silu, op_name="silu", region_id="r1"),
        ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="r2"),
    ]
    edges = [
        # m1 result → silu input
        ContractEdge(producer_id="n_m1", consumer_id="n_silu", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
        # silu result → m2 first input
        ContractEdge(producer_id="n_silu", consumer_id="n_m2", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)
    cluster = FusionCluster(
        cluster_id="cluster_mlp",
        member_op_ids=("n_m1", "n_silu", "n_m2"),
        rationale="mlp chain",
    )
    result = emit_mega_contract(cluster, graph, env)
    mega = result.contract
    assert mega.archetype is KernelArchetype.COMPUTE_TILED
    assert len(mega.body) == 3
    # Externals must include the m2 weight (it has no producer in the
    # cluster) and the m1 inputs (both block-arg or external).
    routing_nodes = {mid for (mid, _) in result.external_inputs_from_nodes}
    # n_m1 always external. n_m2 has the second matmul input that's not
    # an internal edge (its first operand is internal, second is the
    # weight matrix which must come from outside the cluster).
    assert "n_m1" in routing_nodes
    # External outputs only come from the chain tail (n_m2).
    assert all(mid == "n_m2" for (mid, _) in result.external_outputs_from_nodes)


def test_emit_mega_rejects_singleton_cluster() -> None:
    env = _gemmini_envelope()
    c = _v1_activation_contract("silu", (16, 16))
    node = ContractNode(op_id="solo", contract=c, op_name="silu", region_id="r0")
    graph = build_contract_graph_from_nodes([node], [])
    singleton = FusionCluster(
        cluster_id="cluster_singleton",
        member_op_ids=("solo",),
        rationale="singleton",
    )
    with pytest.raises(ValueError, match="≥ 2 body members"):
        emit_mega_contract(singleton, graph, env)


def test_emit_mega_records_cluster_provenance_in_metadata() -> None:
    """Downstream report consumers read cluster_id + estimated_speedup
    out of the MEGA's metadata. Don't let those fields silently drop."""
    env = _gemmini_envelope()
    add = _v1_activation_contract("add", (16, 16))
    relu = _v1_activation_contract("relu", (16, 16))
    nodes = [
        ContractNode(op_id="n0", contract=add, op_name="add", region_id="r0"),
        ContractNode(op_id="n1", contract=relu, op_name="relu", region_id="r1"),
    ]
    edges = [
        ContractEdge(producer_id="n0", consumer_id="n1", operand_index=0,
                     tensor_shape=(16, 16), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)
    cluster = FusionCluster(
        cluster_id="cluster_xx",
        member_op_ids=("n0", "n1"),
        rationale="test provenance",
        estimated_speedup=1.7,
    )
    mega = emit_mega_contract(cluster, graph, env).contract
    assert mega.metadata["cluster_id"] == "cluster_xx"
    assert mega.metadata["member_op_ids"] == ["n0", "n1"]
    assert mega.metadata["estimated_speedup"] == 1.7

"""Tests for :mod:`xpu_rt.kernels.fusion_planner`.

The planner is the only thing that walks a :class:`ContractGraph`
through ``should_fuse`` + ``recommend_granularity``. These tests build
synthetic graphs (no real IR walk) so the oracle verdicts are the
only thing exercised.
"""

from __future__ import annotations

from xpu_rt.ir.payload.contract_graph import (
    ContractEdge,
    ContractNode,
    build_contract_graph_from_nodes,
)
from xpu_rt.ir.payload.contracts import (
    CostEstimate,
    KernelContract,
    LayoutKind,
    LayoutRequirement,
)
from xpu_rt.kernels.contract_v3 import (
    Granularity,
    HardwareEnvelope,
    KernelArchetype,
)
from xpu_rt.kernels.fusion_planner import (
    FusionCluster,
    FusionPlan,
    FusionPlanner,
    lift_v1_to_v3,
    plan_fusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gemmini_envelope() -> HardwareEnvelope:
    """Gemmini-shaped envelope: 256 KiB scratchpad, narrow registers,
    int8 native — matches the existing kb_gemmini target card.
    """
    return HardwareEnvelope(
        target_name="gemmini_mx",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
        register_quota_per_thread=256,
    )


def _v1_matmul_contract(
    op_name: str,
    in_shape_a: tuple[int, ...],
    in_shape_b: tuple[int, ...],
    out_shape: tuple[int, ...],
    dtype: str = "i8",
) -> KernelContract:
    """Build a v1 KernelContract with the metadata an IR walk would
    have stamped on it."""
    return KernelContract(
        op_name=op_name,
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR), LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={dtype},
        cost=CostEstimate(flops=out_shape[0] * out_shape[1] * in_shape_a[1] * 2),
        fusable=False,
        metadata={
            "input_shapes": [in_shape_a, in_shape_b],
            "output_shapes": [out_shape],
            "region_id": op_name,
            "dispatch_id": op_name,
        },
    )


def _v1_activation_contract(op_name: str, shape: tuple[int, ...], dtype: str = "i8") -> KernelContract:
    return KernelContract(
        op_name=op_name,
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={dtype},
        cost=CostEstimate(flops=shape[0] * shape[1]),
        fusable=True,
        metadata={
            "input_shapes": [shape],
            "output_shapes": [shape],
            "region_id": op_name,
            "dispatch_id": op_name,
        },
    )


# ---------------------------------------------------------------------------
# Lifter
# ---------------------------------------------------------------------------


def test_lift_v1_matmul_to_v3_compute_tiled() -> None:
    env = _gemmini_envelope()
    v1 = _v1_matmul_contract("matmul", (64, 720), (720, 1440), (64, 1440))
    v3 = lift_v1_to_v3(v1, env, op_id_hint="n0")
    assert v3.archetype is KernelArchetype.COMPUTE_TILED
    assert len(v3.io.inputs) >= 2
    assert v3.io.outputs[0].shape.dims == (64, 1440)
    # The envelope must round-trip into the v3 contract so the oracle
    # picks the right target_name.
    env_v3 = v3.orchestration.execution.hardware
    assert env_v3.target_name == "gemmini_mx"
    assert env_v3.scratchpad_bytes == 256 * 1024


def test_lift_v1_activation_to_v3_pointwise_or_activation() -> None:
    env = _gemmini_envelope()
    v1 = _v1_activation_contract("silu", (64, 1440))
    v3 = lift_v1_to_v3(v1, env, op_id_hint="n1")
    assert v3.archetype is KernelArchetype.ACTIVATION


# ---------------------------------------------------------------------------
# Planner — fusion oracle says FUSE for compatible pointwise chain
# ---------------------------------------------------------------------------


def test_plan_pointwise_chain_fuses_into_mega() -> None:
    """Pointwise → activation chain on small shapes: the fusion
    oracle says FUSE on each pair, granularity oracle returns MEGA."""
    env = _gemmini_envelope()
    add = _v1_activation_contract("add", (64, 32))
    silu = _v1_activation_contract("silu", (64, 32))
    mul = _v1_activation_contract("mul", (64, 32))

    n0 = ContractNode(op_id="n0", contract=add, op_name="add", region_id="r0")
    n1 = ContractNode(op_id="n1", contract=silu, op_name="silu", region_id="r1")
    n2 = ContractNode(op_id="n2", contract=mul, op_name="mul", region_id="r2")
    e01 = ContractEdge(
        producer_id="n0", consumer_id="n1", operand_index=0,
        tensor_shape=(64, 32), dtype="i8", bytes_per_element=1,
    )
    e12 = ContractEdge(
        producer_id="n1", consumer_id="n2", operand_index=0,
        tensor_shape=(64, 32), dtype="i8", bytes_per_element=1,
    )
    graph = build_contract_graph_from_nodes([n0, n1, n2], [e01, e12])

    plan = plan_fusion(graph, env)
    assert isinstance(plan, FusionPlan)
    assert plan.envelope_target == "gemmini_mx"
    # Either the chain fuses into one MEGA cluster, OR the planner
    # honestly reports DONT_FUSE / granularity veto. The point of this
    # test is that the planner walks the chain through the oracles
    # and returns a coherent plan. The chain here is small enough that
    # the cost model favors fusion, so we assert MEGA.
    assert len(plan.clusters) == 1
    cluster = plan.clusters[0]
    assert cluster.member_op_ids == ("n0", "n1", "n2")
    assert cluster.estimated_speedup >= 1.0
    assert plan.per_pair_verdicts  # should have at least one pair verdict


def test_plan_singleton_when_no_edges() -> None:
    """A graph of isolated nodes produces N singleton clusters — no
    edges means no fusion can be proposed."""
    env = _gemmini_envelope()
    m1 = _v1_matmul_contract("m1", (64, 720), (720, 1440), (64, 1440))
    m2 = _v1_matmul_contract("m2", (64, 1440), (1440, 720), (64, 720))
    n0 = ContractNode(op_id="n0", contract=m1, op_name="matmul", region_id="r0")
    n1 = ContractNode(op_id="n1", contract=m2, op_name="matmul", region_id="r1")
    # NO edge between them.
    graph = build_contract_graph_from_nodes([n0, n1], [])

    plan = plan_fusion(graph, env)
    assert len(plan.clusters) == 2
    for c in plan.clusters:
        assert len(c.member_op_ids) == 1
    # No pair verdicts because there are no producer→consumer edges.
    assert plan.per_pair_verdicts == ()


def test_plan_covers_every_node_exactly_once() -> None:
    """The cluster set must partition graph.nodes (every node lives in
    exactly one cluster). Regression guard for partition correctness."""
    env = _gemmini_envelope()
    nodes_v1 = [
        ("n0", _v1_activation_contract("add", (32, 32))),
        ("n1", _v1_activation_contract("silu", (32, 32))),
        ("n2", _v1_activation_contract("mul", (32, 32))),
        ("n3", _v1_activation_contract("relu", (32, 32))),
    ]
    nodes = [ContractNode(op_id=nid, contract=c, op_name=c.op_name, region_id=nid) for nid, c in nodes_v1]
    edges = [
        ContractEdge(
            producer_id=nodes[i].op_id, consumer_id=nodes[i + 1].op_id, operand_index=0,
            tensor_shape=(32, 32), dtype="i8", bytes_per_element=1,
        )
        for i in range(len(nodes) - 1)
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)

    plan = plan_fusion(graph, env)
    members_seen = set()
    for c in plan.clusters:
        for m in c.member_op_ids:
            assert m not in members_seen, f"node {m!r} in multiple clusters"
            members_seen.add(m)
    assert members_seen == {n.op_id for n in nodes}


def test_plan_records_envelope_target_in_plan() -> None:
    """The plan must carry the envelope.target_name so reports can
    show which target the verdicts apply to."""
    env = _gemmini_envelope()
    c = _v1_activation_contract("silu", (16, 16))
    n = ContractNode(op_id="solo", contract=c, op_name="silu", region_id="r0")
    graph = build_contract_graph_from_nodes([n], [])
    plan = FusionPlanner(env).plan(graph)
    assert plan.envelope_target == "gemmini_mx"
    assert plan.cluster_for("solo").member_op_ids == ("solo",)


def test_plan_fuses_smolvla_mlp_chain_when_weight_tile_declared() -> None:
    """Regression guard for the weight-tiling cost-model fix.

    On real SmolVLA shapes (gate_proj 64×720→1440, silu, down_proj
    64×1440→720) the un-discounted scratchpad-budget check rejects
    the chain because the down_proj weight is 1.0 MB (4× Gemmini's
    256 KB scratchpad). When the envelope declares
    ``weight_tile_bytes`` (Gemmini's ``tiled_matmul_auto`` streams
    tiles of ~4 KiB), the planner must now see the chain as fusable
    and emit ONE MEGA cluster covering all three ops."""
    env = HardwareEnvelope(
        target_name="gemmini_mx",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
        register_quota_per_thread=256,
        weight_tile_bytes=4096,
    )
    m1 = _v1_matmul_contract("matmul", (64, 720), (720, 1440), (64, 1440))
    silu = _v1_activation_contract("silu", (64, 1440))
    m2 = _v1_matmul_contract("matmul", (64, 1440), (1440, 720), (64, 720))
    nodes = [
        ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="m1"),
        ContractNode(op_id="n_silu", contract=silu, op_name="silu", region_id="s"),
        ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="m2"),
    ]
    edges = [
        ContractEdge(producer_id="n_m1", consumer_id="n_silu", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
        ContractEdge(producer_id="n_silu", consumer_id="n_m2", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)

    plan = plan_fusion(graph, env)
    # The planner must emit ONE cluster covering all three nodes —
    # *without* the weight-tile discount this would be 3 singletons.
    assert len(plan.clusters) == 1
    cluster = plan.clusters[0]
    assert cluster.member_op_ids == ("n_m1", "n_silu", "n_m2")
    assert cluster.estimated_speedup > 1.0
    # And the per-cluster granularity oracle must agree it's MEGA.
    granularities = {cid: gv.granularity.value for (cid, gv) in plan.per_cluster_granularity}
    assert granularities.get(cluster.cluster_id) == "mega"


def test_plan_still_vetoes_smolvla_mlp_on_target_without_weight_tile() -> None:
    """Symmetric guard: a target that doesn't declare
    ``weight_tile_bytes`` still gets the conservative
    full-residency budget — we don't accidentally regress targets
    where the codegen doesn't stream weights."""
    env_no_tile = HardwareEnvelope(
        target_name="gemmini_mx",
        vector_lanes=16,
        scratchpad_bytes=256 * 1024,
        register_bytes=16,
        native_dtypes=("i8", "i32"),
        peak_bandwidth_gbps=8.0,
        register_quota_per_thread=256,
        # weight_tile_bytes deliberately omitted → 0 → full residency.
    )
    m1 = _v1_matmul_contract("matmul", (64, 720), (720, 1440), (64, 1440))
    silu = _v1_activation_contract("silu", (64, 1440))
    m2 = _v1_matmul_contract("matmul", (64, 1440), (1440, 720), (64, 720))
    nodes = [
        ContractNode(op_id="n_m1", contract=m1, op_name="matmul", region_id="m1"),
        ContractNode(op_id="n_silu", contract=silu, op_name="silu", region_id="s"),
        ContractNode(op_id="n_m2", contract=m2, op_name="matmul", region_id="m2"),
    ]
    edges = [
        ContractEdge(producer_id="n_m1", consumer_id="n_silu", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
        ContractEdge(producer_id="n_silu", consumer_id="n_m2", operand_index=0,
                     tensor_shape=(64, 1440), dtype="i8", bytes_per_element=1),
    ]
    graph = build_contract_graph_from_nodes(nodes, edges)
    plan = plan_fusion(graph, env_no_tile)
    # Without the tile discount, the chain still splits — at least
    # one cluster of size 1.
    assert any(len(c.member_op_ids) == 1 for c in plan.clusters)

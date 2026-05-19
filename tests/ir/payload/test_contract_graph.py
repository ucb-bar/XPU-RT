"""Tests for :mod:`xpu_rt.ir.payload.contract_graph`.

The harness builds small xDSL modules where every contracted op
carries an ``xpu_rt.region_id`` attribute (the invariant
:func:`_extract_op_contract` enforces) and verifies that
:func:`extract_contract_graph` walks them into a graph whose nodes
and edges match the hand-built dataflow.
"""

from __future__ import annotations

import pytest

xdsl_builtin = pytest.importorskip("xdsl.dialects.builtin")
xdsl_func = pytest.importorskip("xdsl.dialects.func")
xdsl_linalg = pytest.importorskip("xdsl.dialects.linalg")
xdsl_tensor = pytest.importorskip("xdsl.dialects.tensor")
xdsl_ir = pytest.importorskip("xdsl.ir")

from xdsl.dialects.builtin import Float32Type, ModuleOp, StringAttr, TensorType  # noqa: E402
from xdsl.dialects.func import FuncOp, ReturnOp  # noqa: E402
from xdsl.dialects.linalg import MatmulOp  # noqa: E402
from xdsl.dialects.tensor import EmptyOp  # noqa: E402
from xdsl.ir import Block, Region  # noqa: E402

from xpu_rt.ir.payload.contract_graph import (  # noqa: E402
    ContractEdge,
    ContractGraph,
    ContractNode,
    build_contract_graph_from_nodes,
    extract_contract_graph,
)
from xpu_rt.ir.payload.contracts import KernelContract  # noqa: E402


def _stamp(op, region_id: str) -> None:
    op.attributes["xpu_rt.region_id"] = StringAttr(region_id)


def test_extract_contract_graph_single_op() -> None:
    f32 = Float32Type()
    lhs_type = TensorType(f32, [64, 128])
    rhs_type = TensorType(f32, [128, 256])
    out_type = TensorType(f32, [64, 256])

    block = Block(arg_types=[lhs_type, rhs_type])
    empty = EmptyOp([], out_type)
    _stamp(empty, "r_init")
    matmul = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[empty.results[0]],
        res=[out_type],
    )
    _stamp(matmul, "r_matmul")
    ret = ReturnOp(matmul)
    block.add_ops([empty, matmul, ret])
    func_op = FuncOp("main", ([lhs_type, rhs_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    graph = extract_contract_graph(module)

    # Two contracted ops: the EmptyOp (a dispatch-boundary tensor
    # allocation in the canonical IR convention) and the MatmulOp.
    assert len(graph.nodes) == 2
    # The matmul consumes the empty's result as one of its operands —
    # that produces a single edge.
    matmul_node = next(n for n in graph.nodes.values() if "matmul" in n.op_name)
    empty_node = next(n for n in graph.nodes.values() if n.region_id == "r_init")
    edges_into_matmul = graph.in_edges(matmul_node.op_id)
    assert len(edges_into_matmul) == 1
    assert edges_into_matmul[0].producer_id == empty_node.op_id
    assert edges_into_matmul[0].consumer_id == matmul_node.op_id
    assert edges_into_matmul[0].dtype == "f32"
    assert edges_into_matmul[0].bytes_per_element == 4
    # Output shape of EmptyOp is 64×256 f32 → 65536 bytes.
    assert edges_into_matmul[0].bytes_total == 64 * 256 * 4


def test_extract_contract_graph_chains_two_matmuls() -> None:
    """A two-matmul chain: ``(A @ B) @ C`` should yield 2 ops + 1 edge
    between the matmuls (the intermediate ``empty`` allocations are
    contracted too and chain via their own edges)."""
    f32 = Float32Type()
    a_type = TensorType(f32, [32, 64])
    b_type = TensorType(f32, [64, 96])
    inter_type = TensorType(f32, [32, 96])
    c_type = TensorType(f32, [96, 16])
    out_type = TensorType(f32, [32, 16])

    block = Block(arg_types=[a_type, b_type, c_type])
    e1 = EmptyOp([], inter_type)
    _stamp(e1, "r_e1")
    m1 = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[e1.results[0]],
        res=[inter_type],
    )
    _stamp(m1, "r_m1")
    e2 = EmptyOp([], out_type)
    _stamp(e2, "r_e2")
    m2 = MatmulOp(
        inputs=[m1.results[0], block.args[2]],
        outputs=[e2.results[0]],
        res=[out_type],
    )
    _stamp(m2, "r_m2")
    ret = ReturnOp(m2)
    block.add_ops([e1, m1, e2, m2, ret])
    func_op = FuncOp("main", ([a_type, b_type, c_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    graph = extract_contract_graph(module)

    assert len(graph.nodes) == 4  # 2 empties + 2 matmuls
    # Topological order must satisfy SSA def-before-use.
    pos = {nid: i for i, nid in enumerate(graph.topological_order)}
    m1_id = next(n.op_id for n in graph.nodes.values() if n.region_id == "r_m1")
    m2_id = next(n.op_id for n in graph.nodes.values() if n.region_id == "r_m2")
    e1_id = next(n.op_id for n in graph.nodes.values() if n.region_id == "r_e1")
    e2_id = next(n.op_id for n in graph.nodes.values() if n.region_id == "r_e2")
    assert pos[e1_id] < pos[m1_id]
    assert pos[m1_id] < pos[m2_id]
    assert pos[e2_id] < pos[m2_id]

    # The m1→m2 dataflow edge must exist exactly once.
    m1_to_m2 = [e for e in graph.edges if e.producer_id == m1_id and e.consumer_id == m2_id]
    assert len(m1_to_m2) == 1
    edge = m1_to_m2[0]
    assert edge.operand_index == 0  # m2's first input is m1's result
    assert edge.tensor_shape == (32, 96)
    assert edge.bytes_total == 32 * 96 * 4


def test_extract_contract_graph_skips_block_args() -> None:
    """Function inputs (block args) must NOT show up as edges — they
    are external inputs to the graph, not producer→consumer flow."""
    f32 = Float32Type()
    a_type = TensorType(f32, [4, 4])
    b_type = TensorType(f32, [4, 4])
    out_type = TensorType(f32, [4, 4])

    block = Block(arg_types=[a_type, b_type])
    empty = EmptyOp([], out_type)
    _stamp(empty, "r_init")
    matmul = MatmulOp(
        inputs=[block.args[0], block.args[1]],
        outputs=[empty.results[0]],
        res=[out_type],
    )
    _stamp(matmul, "r_matmul")
    ret = ReturnOp(matmul)
    block.add_ops([empty, matmul, ret])
    func_op = FuncOp("main", ([a_type, b_type], [out_type]), Region(block))
    module = ModuleOp([func_op])

    graph = extract_contract_graph(module)

    matmul_node = next(n for n in graph.nodes.values() if "matmul" in n.op_name)
    in_edges = graph.in_edges(matmul_node.op_id)
    # Three operands on matmul (lhs, rhs, init) but only one is from a
    # contracted op (the EmptyOp init); lhs/rhs are block args.
    assert len(in_edges) == 1
    assert in_edges[0].operand_index == 2  # init is the third operand


def test_build_contract_graph_from_nodes_round_trips() -> None:
    """Programmatic construction works for synthetic graphs used by the
    fusion planner tests."""
    n1 = ContractNode(
        op_id="n1",
        contract=KernelContract(op_name="matmul"),
        op_name="matmul",
        region_id="r1",
    )
    n2 = ContractNode(
        op_id="n2",
        contract=KernelContract(op_name="silu"),
        op_name="silu",
        region_id="r2",
    )
    edge = ContractEdge(
        producer_id="n1",
        consumer_id="n2",
        operand_index=0,
        tensor_shape=(64, 1440),
        dtype="i8",
        bytes_per_element=1,
    )
    g = build_contract_graph_from_nodes([n1, n2], [edge])
    assert isinstance(g, ContractGraph)
    assert g.topological_order == ("n1", "n2")
    assert g.successors("n1") == ("n2",)
    assert g.predecessors("n2") == ("n1",)
    assert edge.bytes_total == 64 * 1440 * 1


def test_build_contract_graph_rejects_dangling_edges() -> None:
    n1 = ContractNode(
        op_id="n1",
        contract=KernelContract(op_name="matmul"),
        op_name="matmul",
        region_id="r1",
    )
    bad = ContractEdge(
        producer_id="n1",
        consumer_id="missing",
        operand_index=0,
        tensor_shape=(),
        dtype="f32",
        bytes_per_element=4,
    )
    with pytest.raises(ValueError, match="unknown consumer"):
        build_contract_graph_from_nodes([n1], [bad])

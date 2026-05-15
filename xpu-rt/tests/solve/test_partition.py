"""Tests for graph partitioning."""

from __future__ import annotations

from compgen.solve.partition import Partition, partition_graph
from xdsl.dialects import arith, func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region


def test_partition_construction() -> None:
    p = Partition(partition_id="p0", op_names=["matmul", "relu"], estimated_cost_us=100.0, memory_bytes=4096)
    assert p.partition_id == "p0"
    assert len(p.op_names) == 2


def test_partition_graph_simple() -> None:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

    partitions = partition_graph(module)
    assert len(partitions) >= 2
    # Each partition should have unique IDs
    ids = [p.partition_id for p in partitions]
    assert len(ids) == len(set(ids))


def test_partition_graph_dependencies() -> None:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    add = arith.AddiOp(a, b)
    block.add_op(add)
    mul = arith.MuliOp(add.result, b)
    block.add_op(mul)
    block.add_op(func.ReturnOp(mul.result))
    module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

    partitions = partition_graph(module)
    # The second partition (mul) should depend on the first (add)
    assert len(partitions) >= 2
    mul_partition = partitions[1]
    assert len(mul_partition.dependencies) >= 1


def test_partition_graph_max_partitions() -> None:
    idx = IndexType()
    block = Block(arg_types=[idx, idx])
    a, b = block.args
    prev = a
    for _ in range(20):
        add = arith.AddiOp(prev, b)
        block.add_op(add)
        prev = add.result
    block.add_op(func.ReturnOp(prev))
    module = ModuleOp([func.FuncOp("test", ([idx, idx], [idx]), Region([block]))])

    partitions = partition_graph(module, max_partitions=5)
    assert len(partitions) <= 5

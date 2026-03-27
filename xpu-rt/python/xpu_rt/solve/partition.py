"""Graph partitioning and region grouping.

Partitions the dispatch graph into groups suitable for device assignment.
This is the first step of the solver pipeline: compress the problem
into a tractable number of decision variables.

Invariants:
    - Partitions respect data dependencies.
    - Partitions do not split atomic operations.
    - Partition count is bounded by max_partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.linalg import MatmulOp
from xdsl.ir import Operation, OpResult


@dataclass(frozen=True)
class Partition:
    """A partition of the dispatch graph.

    Attributes:
        partition_id: Unique identifier.
        op_names: Ops in this partition.
        dependencies: Partition IDs this depends on.
        estimated_cost_us: Estimated execution cost in microseconds.
        memory_bytes: Estimated memory requirement.
    """

    partition_id: str
    op_names: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    estimated_cost_us: float = 0.0
    memory_bytes: int = 0


def _estimate_op_bytes(op: Operation) -> int:
    """Estimate memory bytes for an operation's operands and results."""
    total = 0
    for val in list(op.operands) + list(op.results):
        if isinstance(val.type, TensorType):
            shape = val.type.get_shape()
            elem_size = 4  # assume float32
            num_elems = 1
            for dim in shape:
                if dim > 0:
                    num_elems *= dim
            total += num_elems * elem_size
    return total


def partition_graph(module: ModuleOp, max_partitions: int = 64) -> list[Partition]:
    """Partition a module's ops into groups for device assignment.

    Creates one partition per significant op (matmul, generic, etc.)
    and groups small elementwise ops together.

    Args:
        module: xDSL ModuleOp.
        max_partitions: Maximum number of partitions.

    Returns:
        List of Partition objects.
    """
    partitions: list[Partition] = []
    op_to_partition: dict[Operation, str] = {}
    partition_count = 0

    for op in module.walk():
        if isinstance(op, (ModuleOp, FuncOp, ReturnOp)):
            continue
        if not op.results:
            continue

        pid = f"p_{partition_count}"
        partition_count += 1

        # Track which partition each op belongs to
        op_to_partition[op] = pid

        # Find dependencies: which partitions produce our operands?
        deps: list[str] = []
        for operand in op.operands:
            if isinstance(operand, OpResult):
                producer = operand.owner
                if producer in op_to_partition:
                    dep_pid = op_to_partition[producer]
                    if dep_pid != pid and dep_pid not in deps:
                        deps.append(dep_pid)

        partitions.append(Partition(
            partition_id=pid,
            op_names=[op.name],
            dependencies=deps,
            estimated_cost_us=1.0 if not isinstance(op, MatmulOp) else 10.0,
            memory_bytes=_estimate_op_bytes(op),
        ))

        if partition_count >= max_partitions:
            break

    return partitions


__all__ = ["Partition", "partition_graph"]

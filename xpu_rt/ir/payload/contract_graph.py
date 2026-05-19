"""Graph-with-edges view of an XPU-RT canonical IR module.

The existing ``extract_contracts(module)`` (in
:mod:`xpu_rt.ir.payload.contracts`) returns a flat list of
``KernelContract`` — one entry per op annotated with
``xpu_rt.region_id``. That is enough for kernel-at-a-time selection
but throws away the producer/consumer structure that the fusion
oracle and granularity oracle need to make graph-level decisions.

``extract_contract_graph`` walks the same set of contracted ops and
additionally records the SSA producer-of-value edges between them
into a :class:`ContractGraph`. Downstream code (the FusionPlanner,
the MegaContractEmitter, the pipeline-level benchmark) consumes the
graph instead of the flat list, and the graph trivially flattens
back when a caller only needs the nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType
from xdsl.ir import Operation, SSAValue

from xpu_rt.ir.payload.contracts import (
    KernelContract,
    LayoutKind,
    _dtype_bytes,
    _extract_op_contract,
)


@dataclass(frozen=True)
class ContractNode:
    """A single contracted op in the graph.

    Attributes:
        op_id: Unique-within-graph id (``"op_<index>_<region_id>"``).
            Stable for a given walk order, but not guaranteed equal
            across walks of structurally identical modules.
        contract: The single-op KernelContract for this node.
        op_name: Canonical op name (``contract.op_name``, mirrored
            here for ergonomic access without unwrapping the contract).
        region_id: The ``xpu_rt.region_id`` annotation that elected
            this op into the graph.
    """

    op_id: str
    contract: KernelContract
    op_name: str
    region_id: str


@dataclass(frozen=True)
class ContractEdge:
    """A producer→consumer dataflow edge between two contracted ops.

    The edge carries the bytes-per-element + shape needed by
    :func:`xpu_rt.kernels.fusion_oracle.should_fuse`'s DRAM-savings
    model — i.e. the byte traffic that would be eliminated if the
    consumer ran in-place against the producer's scratchpad output
    instead of round-tripping through DRAM.

    Attributes:
        producer_id: ``ContractNode.op_id`` of the producer.
        consumer_id: ``ContractNode.op_id`` of the consumer.
        operand_index: Which operand slot on the consumer this edge
            feeds (0-based).
        tensor_shape: Shape of the value flowing on this edge.
            ``-1`` denotes dynamic in any dimension.
        dtype: xDSL element-type string (e.g. ``"f32"``, ``"i8"``).
        bytes_per_element: Cached ``_dtype_bytes(dtype)`` for the cost
            model.
    """

    producer_id: str
    consumer_id: str
    operand_index: int
    tensor_shape: tuple[int, ...]
    dtype: str
    bytes_per_element: int

    @property
    def bytes_total(self) -> int:
        """Total bytes flowing on this edge for the static-shape case.

        Dynamic dims contribute a factor of 1 (consistent with
        :func:`_extract_op_contract`'s cost estimation). Callers that
        need a tighter bound should fold sample-input shapes into
        ``tensor_shape`` before reading this property.
        """
        n = 1
        for d in self.tensor_shape:
            if d > 0:
                n *= d
        return n * self.bytes_per_element


@dataclass(frozen=True)
class ContractGraph:
    """Producer/consumer graph over contracted ops.

    Construction is via :func:`extract_contract_graph`. Tests can
    also build one by hand from in-memory ``ContractNode`` /
    ``ContractEdge`` instances.

    The graph is acyclic (xDSL SSA is acyclic by construction).
    :attr:`topological_order` is the walk order in which the
    underlying ``ModuleOp.walk()`` visited the ops, which respects
    SSA def-before-use and therefore satisfies a topological order.

    Attributes:
        nodes: ``op_id`` → ``ContractNode``.
        edges: All producer/consumer dataflow edges between
            *contracted* ops. Edges to block-argument-rooted values
            or to non-contracted ops are not represented.
        topological_order: Tuple of ``op_id`` in def-before-use order.
    """

    nodes: dict[str, ContractNode]
    edges: tuple[ContractEdge, ...]
    topological_order: tuple[str, ...]

    def out_edges(self, op_id: str) -> tuple[ContractEdge, ...]:
        return tuple(e for e in self.edges if e.producer_id == op_id)

    def in_edges(self, op_id: str) -> tuple[ContractEdge, ...]:
        return tuple(e for e in self.edges if e.consumer_id == op_id)

    def predecessors(self, op_id: str) -> tuple[str, ...]:
        # preserve operand order on the consumer
        return tuple(e.producer_id for e in sorted(self.in_edges(op_id), key=lambda e: e.operand_index))

    def successors(self, op_id: str) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for e in self.edges:
            if e.producer_id == op_id and e.consumer_id not in seen:
                seen[e.consumer_id] = None
        return tuple(seen.keys())

    def contracts_in_topological_order(self) -> tuple[KernelContract, ...]:
        return tuple(self.nodes[op_id].contract for op_id in self.topological_order)


def _ssa_value_shape_dtype(value: SSAValue) -> tuple[tuple[int, ...], str, int] | None:
    """Return ``(shape, dtype_str, bytes_per_element)`` for a tensor
    SSA value, or ``None`` for non-tensor / unknown-typed values."""
    t = value.type
    if not isinstance(t, TensorType):
        return None
    shape = tuple(t.get_shape())
    dtype_str = str(t.element_type)
    return shape, dtype_str, _dtype_bytes(dtype_str)


def _region_id(op: Operation) -> str | None:
    attr = op.attributes.get("xpu_rt.region_id")
    if isinstance(attr, StringAttr):
        return attr.data
    return None


def extract_contract_graph(module: ModuleOp) -> ContractGraph:
    """Walk a canonical xDSL module into a ContractGraph.

    Algorithm:
        1. First pass: walk ``module``, materialise a ``ContractNode``
           for every op that has both an ``xpu_rt.region_id`` and a
           non-``None`` ``_extract_op_contract`` result. Build a side
           map ``op_to_node_id: dict[id(op), str]``.
        2. Second pass: for each contracted op, iterate its operands.
           For each operand SSA value, look at ``value.owner``. If the
           owner is also contracted, emit a :class:`ContractEdge`.
           Block-arg operands (function inputs) produce no edge —
           they are external graph inputs.

    The topological order is the order in which the first pass
    visited the contracted ops. xDSL's ``ModuleOp.walk()`` does a
    pre-order traversal that respects SSA def-before-use, so this
    is a valid topological order over the dataflow edges.

    Args:
        module: A canonicalized xDSL module annotated with
            ``xpu_rt.region_id`` on dispatch-boundary ops (the
            invariant guaranteed by the post-import annotation pass).

    Returns:
        A populated ContractGraph. Empty modules yield an empty graph.
    """
    nodes: dict[str, ContractNode] = {}
    topological: list[str] = []
    op_to_node_id: dict[int, str] = {}

    for i, op in enumerate(module.walk()):
        if _region_id(op) is None:
            continue
        contract = _extract_op_contract(op)
        if contract is None:
            continue
        rid = _region_id(op) or "unknown"
        node_id = f"op_{i:04d}_{rid}"
        node = ContractNode(
            op_id=node_id,
            contract=contract,
            op_name=contract.op_name,
            region_id=rid,
        )
        nodes[node_id] = node
        topological.append(node_id)
        op_to_node_id[id(op)] = node_id

    edges: list[ContractEdge] = []
    for op in module.walk():
        consumer_id = op_to_node_id.get(id(op))
        if consumer_id is None:
            continue
        for operand_index, operand in enumerate(op.operands):
            producer = operand.owner
            if not isinstance(producer, Operation):
                continue  # block argument — external input, no edge
            producer_id = op_to_node_id.get(id(producer))
            if producer_id is None:
                continue  # producer not contracted — external to the graph
            sd = _ssa_value_shape_dtype(operand)
            if sd is None:
                continue
            shape, dtype_str, bpe = sd
            edges.append(
                ContractEdge(
                    producer_id=producer_id,
                    consumer_id=consumer_id,
                    operand_index=operand_index,
                    tensor_shape=shape,
                    dtype=dtype_str,
                    bytes_per_element=bpe,
                )
            )

    return ContractGraph(
        nodes=nodes,
        edges=tuple(edges),
        topological_order=tuple(topological),
    )


def build_contract_graph_from_nodes(
    nodes: Iterable[ContractNode],
    edges: Iterable[ContractEdge],
) -> ContractGraph:
    """Programmatic constructor for unit tests and synthetic graphs.

    Node order in the iterable becomes :attr:`topological_order`.
    Edges referencing missing node ids raise :class:`ValueError` so
    callers can't silently build a malformed graph.
    """
    node_map: dict[str, ContractNode] = {}
    order: list[str] = []
    for n in nodes:
        if n.op_id in node_map:
            raise ValueError(f"duplicate ContractNode op_id: {n.op_id!r}")
        node_map[n.op_id] = n
        order.append(n.op_id)
    edge_tuple: list[ContractEdge] = []
    for e in edges:
        if e.producer_id not in node_map:
            raise ValueError(f"ContractEdge references unknown producer: {e.producer_id!r}")
        if e.consumer_id not in node_map:
            raise ValueError(f"ContractEdge references unknown consumer: {e.consumer_id!r}")
        edge_tuple.append(e)
    return ContractGraph(nodes=node_map, edges=tuple(edge_tuple), topological_order=tuple(order))


__all__ = [
    "ContractEdge",
    "ContractGraph",
    "ContractNode",
    "build_contract_graph_from_nodes",
    "extract_contract_graph",
]

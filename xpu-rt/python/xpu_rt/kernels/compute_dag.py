"""Per-kernel compute-graph contract.

Captures the *in-kernel* computation as a small typed DAG so the agent
can reason about kernel internals without re-traversing payload IR
each time. Three audiences:

  * **Codegen prompt** — ``to_prompt_text(dag)`` renders the DAG into a
    compact textual form that Claude Code reads when emitting kernel
    source. Replaces "guess the structure from a contract description"
    with "here are the named ops + edges".
  * **Diagnosis** — the bench-diagnose-refine loop reads dag.shape
    (linear chain vs reduction-then-broadcast vs gather-scatter) and
    tunes refinement hypotheses to the structure.
  * **Autocomp adapter** — translates the DAG into autocomp's per-target
    prompt format so an autocomp escalation gets the same problem
    description the Claude Code path saw.

Kept deliberately small: nodes carry only what the kernel codegen
needs (op kind, output dtype, output shape class, dim roles); edges
carry only producer→consumer SSA dataflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from xdsl.ir import Operation


class NodeKind(Enum):
    """High-level family for each node — determines how codegen lowers it."""

    INPUT = "input"  # entry tensor (kernel argument)
    OUTPUT = "output"  # exit tensor (kernel result)
    COMPUTE = "compute"  # arithmetic / matmul / dot
    REDUCE = "reduce"  # sum / max / mean over an axis
    BROADCAST = "broadcast"  # implicit broadcast
    POINTWISE = "pointwise"  # elementwise math (add/mul/silu/...)
    LOAD = "load"  # explicit load from a non-input buffer
    STORE = "store"  # explicit store to a non-output buffer


@dataclass(frozen=True)
class ComputeNode:
    """One node in the in-kernel DAG."""

    id: str  # unique within the DAG (e.g. "n_0")
    kind: NodeKind
    op_name: str  # original op name / pattern hint
    output_shape: tuple[int | None, ...] = ()  # symbolic dims as None
    output_dtype: str = "f32"
    dim_roles: tuple[str, ...] = ()  # parallel/reduce/broadcast/batch
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComputeEdge:
    """Directed dataflow edge: ``src`` → ``dst``."""

    src: str
    dst: str
    operand_idx: int = 0  # which operand on dst this feeds


@dataclass
class ComputeDAG:
    """Small typed DAG describing one kernel's computation."""

    nodes: list[ComputeNode] = field(default_factory=list)
    edges: list[ComputeEdge] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)  # node ids
    outputs: list[str] = field(default_factory=list)  # node ids

    def by_id(self, nid: str) -> ComputeNode | None:
        for n in self.nodes:
            if n.id == nid:
                return n
        return None

    def predecessors(self, nid: str) -> list[str]:
        return [e.src for e in self.edges if e.dst == nid]

    def successors(self, nid: str) -> list[str]:
        return [e.dst for e in self.edges if e.src == nid]

    def shape_summary(self) -> str:
        """Cheap classification used by diagnosis hypothesis selection.

        Returns one of:
          * ``"linear_chain"``        — N→1 edges, no fan-out
          * ``"reduce_then_broadcast"`` — has REDUCE node feeding > 1 consumers
          * ``"fan_out"``             — at least one node with > 1 successors
          * ``"single_op"``           — exactly one compute node
          * ``"empty"``
        """
        compute_nodes = [n for n in self.nodes if n.kind != NodeKind.INPUT and n.kind != NodeKind.OUTPUT]
        if not compute_nodes:
            return "empty"
        if len(compute_nodes) == 1:
            return "single_op"
        for n in self.nodes:
            if n.kind is NodeKind.REDUCE:
                if len(self.successors(n.id)) > 1:
                    return "reduce_then_broadcast"
        for n in self.nodes:
            if len(self.successors(n.id)) > 1:
                return "fan_out"
        return "linear_chain"


# ---------------------------------------------------------------------------
# Build from payload region
# ---------------------------------------------------------------------------


_KIND_BY_HINT: dict[str, NodeKind] = {
    "softmax": NodeKind.REDUCE,
    "rmsnorm": NodeKind.REDUCE,
    "reduce_mean": NodeKind.REDUCE,
    "rsqrt": NodeKind.POINTWISE,
    "silu": NodeKind.POINTWISE,
    "sigmoid": NodeKind.POINTWISE,
    "tanh": NodeKind.POINTWISE,
    "neg": NodeKind.POINTWISE,
    "where": NodeKind.POINTWISE,
}

_KIND_BY_OPNAME: dict[str, NodeKind] = {
    "linalg.matmul": NodeKind.COMPUTE,
    "linalg.batch_matmul": NodeKind.COMPUTE,
    "linalg.transpose": NodeKind.LOAD,
    "arith.addf": NodeKind.POINTWISE,
    "arith.mulf": NodeKind.POINTWISE,
    "arith.subf": NodeKind.POINTWISE,
    "arith.divf": NodeKind.POINTWISE,
    "tensor.empty": NodeKind.LOAD,
}


def _classify(op: Operation) -> NodeKind:
    attrs = getattr(op, "attributes", {})
    hint = attrs.get("compgen._pattern_hint") if attrs else None
    if hint is not None and hasattr(hint, "data"):
        if hint.data in _KIND_BY_HINT:
            return _KIND_BY_HINT[hint.data]
    return _KIND_BY_OPNAME.get(op.name, NodeKind.POINTWISE)


def _shape_of(op: Operation) -> tuple[int | None, ...]:
    if not op.results:
        return ()
    t = op.results[0].type
    if not hasattr(t, "get_shape"):
        return ()
    return tuple(d if d > 0 else None for d in t.get_shape())


def _dtype_of(op: Operation) -> str:
    if not op.results:
        return "?"
    t = op.results[0].type
    if hasattr(t, "element_type"):
        return str(t.element_type)
    return "?"


def from_payload_region(region: Operation, *, dag_id: str = "") -> ComputeDAG:
    """Build a ComputeDAG from a payload-IR region (a sequence of ops).

    ``region`` may be any iterable of ops (a func body, a fused subgraph,
    a single op). Inputs are detected as block arguments / SSA values
    that are used but not defined inside the region; outputs as values
    that escape the region.
    """
    from compgen.analysis.dim_semantics import analyze_op

    if hasattr(region, "walk"):
        ops = [op for op in region.walk() if op.results]
    elif isinstance(region, list | tuple):
        ops = list(region)
    else:
        ops = [region]

    dag = ComputeDAG()

    # First pass: build nodes + map SSA values → node id
    val_to_node: dict[Any, str] = {}
    for i, op in enumerate(ops):
        nid = f"n_{i}"
        ann = analyze_op(op)
        roles = tuple(r.value for r in (ann.output_roles if ann else ()))
        dag.nodes.append(
            ComputeNode(
                id=nid,
                kind=_classify(op),
                op_name=op.name,
                output_shape=_shape_of(op),
                output_dtype=_dtype_of(op),
                dim_roles=roles,
            )
        )
        if op.results:
            val_to_node[op.results[0]] = nid

    # Second pass: edges from operand SSA-values produced inside the region
    op_set = set(id(op) for op in ops)
    for i, op in enumerate(ops):
        for opidx, opnd in enumerate(op.operands):
            src = val_to_node.get(opnd)
            if src is not None:
                dag.edges.append(ComputeEdge(src=src, dst=f"n_{i}", operand_idx=opidx))

    # Inputs = nodes with no in-edges. Outputs = nodes whose result is used
    # outside the region (or, simpler, nodes with no consumer inside).
    in_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    out_degree: dict[str, int] = {n.id: 0 for n in dag.nodes}
    for e in dag.edges:
        in_degree[e.dst] += 1
        out_degree[e.src] += 1
    dag.inputs = [n.id for n in dag.nodes if in_degree[n.id] == 0 and n.kind != NodeKind.OUTPUT]
    dag.outputs = [n.id for n in dag.nodes if out_degree[n.id] == 0]

    return dag


# ---------------------------------------------------------------------------
# Render for prompt injection
# ---------------------------------------------------------------------------


def to_prompt_text(dag: ComputeDAG, *, max_nodes: int = 32) -> str:
    """Render the DAG as a compact textual form for codegen prompts.

    Format::

        ComputeDAG (shape=linear_chain, 5 nodes)
          n_0 [INPUT]                   shape=(M,K)   dtype=f16
          n_1 [COMPUTE]   linalg.matmul shape=(M,N)   dtype=f16   roles=(parallel, parallel)
          n_2 [POINTWISE] silu          shape=(M,N)   dtype=f16
          n_3 [REDUCE]    softmax       shape=(M,N)   dtype=f32   axis=-1
          n_4 [OUTPUT]                  shape=(M,N)   dtype=f16
        edges:
          n_0 → n_1[0]
          n_1 → n_2[0]
          n_2 → n_3[0]
          n_3 → n_4[0]
    """
    lines: list[str] = []
    shape = dag.shape_summary()
    lines.append(f"ComputeDAG (shape={shape}, {len(dag.nodes)} nodes)")
    for n in dag.nodes[:max_nodes]:
        roles_str = f" roles=({', '.join(n.dim_roles)})" if n.dim_roles else ""
        lines.append(
            f"  {n.id} [{n.kind.value:9s}] {n.op_name:20s} shape={n.output_shape}  dtype={n.output_dtype}{roles_str}"
        )
    if len(dag.nodes) > max_nodes:
        lines.append(f"  … +{len(dag.nodes) - max_nodes} more nodes elided")
    lines.append("edges:")
    for e in dag.edges[: max_nodes * 2]:
        lines.append(f"  {e.src} → {e.dst}[{e.operand_idx}]")
    if len(dag.edges) > max_nodes * 2:
        lines.append(f"  … +{len(dag.edges) - max_nodes * 2} more edges elided")
    return "\n".join(lines)


__all__ = [
    "ComputeDAG",
    "ComputeEdge",
    "ComputeNode",
    "NodeKind",
    "from_payload_region",
    "to_prompt_text",
]

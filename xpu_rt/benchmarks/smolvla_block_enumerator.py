"""Enumerate SmolVLA blocks as :class:`ContractGraph` subgraphs.

The flat extractor at :mod:`xpu_rt.benchmarks.smolvla_subset` returns
one contract per ``nn.Linear`` instance, deduplicated by region
signature. The pipeline-level study needs more: it needs to see the
*producer→consumer chain* a transformer block forms, so the
:class:`FusionPlanner` has something multi-node to walk.

We don't FX-capture SmolVLA — Dynamo's partition path leaves
``node.meta['val']`` empty for the action-expert subtree (the same
reason :mod:`smolvla_subset` walks modules instead of FX nodes).
Instead we **template-match** on the well-known transformer module
naming convention (PaliGemma/Gemma family):

  * MLP block at ``...layers.<N>.mlp.{gate_proj, up_proj, down_proj}``
    — Gemma fuses ``down_proj(silu(gate_proj(x)) * up_proj(x))``.
    We surface a *linearised* chain ``gate_proj → silu → down_proj``
    for the FusionPlanner (the up_proj branch is left as a separate
    single-op contract since the MEGA emitter assumes chains today).

  * Attention block at
    ``...layers.<N>.self_attn.{q_proj, k_proj, v_proj, o_proj}`` —
    we surface ``q_proj → attention_core → o_proj`` similarly, with
    ``attention_core`` as a synthesised REDUCE node standing in for
    the softmax + value-matmul.

  * Action head at ``...action_in_proj`` /
    ``...action_out_proj`` — emit as a 2-op chain
    ``action_in_proj → relu → action_out_proj``.

For Phase A of the pipeline-level study we run on the action-expert
layer 0 MLP block + the action head; the rest are out of scope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import nn

from xpu_rt.benchmarks.smolvla_subset import (
    _action_expert_layer_index_for_path,
    _iter_linear_modules,
)
from xpu_rt.ir.payload.contract_graph import (
    ContractEdge,
    ContractGraph,
    ContractNode,
    build_contract_graph_from_nodes,
)
from xpu_rt.ir.payload.contracts import (
    CostEstimate,
    KernelContract,
    LayoutKind,
    LayoutRequirement,
)


logger = logging.getLogger(__name__)


BlockKind = Literal["mlp", "attention", "head"]
Component = Literal["language_model", "action_expert", "action_head", "vision"]


@dataclass(frozen=True)
class BlockSpec:
    """One transformer block as a self-contained :class:`ContractGraph`.

    Attributes:
        block_id: ``"<component>.layer<N>.<kind>"`` or
            ``"<component>.<kind>"`` for non-layered blocks (e.g. the
            action head).
        block_kind: ``"mlp"``, ``"attention"``, or ``"head"``.
        component: ``"action_expert"`` / ``"action_head"`` / ...
        subgraph: A :class:`ContractGraph` with the block's chain
            (typically 3 nodes for MLP / head, 3 for attention).
        layer_index: Layer N if applicable, else ``None``.
        sample_module_paths: Originating module paths for the
            matmul nodes. Useful for report attribution.
    """

    block_id: str
    block_kind: BlockKind
    component: Component
    subgraph: ContractGraph
    layer_index: int | None
    sample_module_paths: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_matmul_node(*, op_id: str, region_id: str, M: int, K: int, N: int, dtype: str) -> ContractNode:
    contract = KernelContract(
        op_name="matmul",
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR), LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={dtype},
        cost=CostEstimate(flops=M * N * K * 2),
        fusable=False,
        metadata={
            "input_shapes": [(M, K), (K, N)],
            "output_shapes": [(M, N)],
            "region_id": region_id,
            "dispatch_id": region_id,
        },
    )
    return ContractNode(op_id=op_id, contract=contract, op_name="matmul", region_id=region_id)


def _make_activation_node(
    *,
    op_id: str,
    region_id: str,
    shape: tuple[int, int],
    op_name: str,
    dtype: str,
) -> ContractNode:
    contract = KernelContract(
        op_name=op_name,
        input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR)],
        supported_dtypes={dtype},
        cost=CostEstimate(flops=shape[0] * shape[1]),
        fusable=True,
        metadata={
            "input_shapes": [shape],
            "output_shapes": [shape],
            "region_id": region_id,
            "dispatch_id": region_id,
        },
    )
    return ContractNode(op_id=op_id, contract=contract, op_name=op_name, region_id=region_id)


def _matmul_edge(
    *, producer_id: str, consumer_id: str, shape: tuple[int, int], dtype: str
) -> ContractEdge:
    return ContractEdge(
        producer_id=producer_id,
        consumer_id=consumer_id,
        operand_index=0,
        tensor_shape=shape,
        dtype=dtype,
        bytes_per_element={"i8": 1, "i32": 4, "f32": 4, "f16": 2, "bf16": 2}.get(dtype, 4),
    )


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------


def _build_mlp_block(
    *,
    layer_index: int,
    gate_proj: nn.Linear,
    down_proj: nn.Linear,
    seq_len: int,
    dtype: str,
    component: Component,
    sample_paths: tuple[str, str],
) -> BlockSpec | None:
    """Build a linearised MLP block: matmul → silu → matmul.

    Returns None when the shapes don't compose as a chain (e.g.
    ``gate_proj.out != down_proj.in``).
    """
    K1 = int(gate_proj.in_features)
    K2_inter = int(gate_proj.out_features)
    K2 = int(down_proj.in_features)
    N = int(down_proj.out_features)
    if K2_inter != K2:
        logger.debug(
            "mlp block skipped: gate_proj.out=%d != down_proj.in=%d for layer %d",
            K2_inter, K2, layer_index,
        )
        return None
    block_id = f"{component}.layer{layer_index}.mlp"
    n_gate = _make_matmul_node(
        op_id=f"{block_id}.gate_proj", region_id=f"{block_id}.gate_proj",
        M=seq_len, K=K1, N=K2_inter, dtype=dtype,
    )
    n_silu = _make_activation_node(
        op_id=f"{block_id}.silu", region_id=f"{block_id}.silu",
        shape=(seq_len, K2_inter), op_name="silu", dtype=dtype,
    )
    n_down = _make_matmul_node(
        op_id=f"{block_id}.down_proj", region_id=f"{block_id}.down_proj",
        M=seq_len, K=K2, N=N, dtype=dtype,
    )
    edges = [
        _matmul_edge(producer_id=n_gate.op_id, consumer_id=n_silu.op_id,
                     shape=(seq_len, K2_inter), dtype=dtype),
        _matmul_edge(producer_id=n_silu.op_id, consumer_id=n_down.op_id,
                     shape=(seq_len, K2_inter), dtype=dtype),
    ]
    graph = build_contract_graph_from_nodes([n_gate, n_silu, n_down], edges)
    return BlockSpec(
        block_id=block_id,
        block_kind="mlp",
        component=component,
        subgraph=graph,
        layer_index=layer_index,
        sample_module_paths=sample_paths,
    )


def _build_attention_block(
    *,
    layer_index: int,
    q_proj: nn.Linear,
    o_proj: nn.Linear,
    seq_len: int,
    dtype: str,
    component: Component,
    sample_paths: tuple[str, str],
) -> BlockSpec | None:
    """Build a linearised attention chain: q_proj → attention_core → o_proj.

    ``attention_core`` is a synthesised REDUCE op stand-in for the
    softmax + value-matmul. The shape is ``(seq_len, q_proj.out)``
    cascading into o_proj.
    """
    K1 = int(q_proj.in_features)
    inter = int(q_proj.out_features)
    K2 = int(o_proj.in_features)
    N = int(o_proj.out_features)
    if inter != K2:
        logger.debug(
            "attn block skipped: q_proj.out=%d != o_proj.in=%d for layer %d",
            inter, K2, layer_index,
        )
        return None
    block_id = f"{component}.layer{layer_index}.attention"
    n_q = _make_matmul_node(
        op_id=f"{block_id}.q_proj", region_id=f"{block_id}.q_proj",
        M=seq_len, K=K1, N=inter, dtype=dtype,
    )
    n_core = _make_activation_node(
        op_id=f"{block_id}.attention_core", region_id=f"{block_id}.attention_core",
        shape=(seq_len, inter), op_name="softmax", dtype=dtype,
    )
    n_o = _make_matmul_node(
        op_id=f"{block_id}.o_proj", region_id=f"{block_id}.o_proj",
        M=seq_len, K=K2, N=N, dtype=dtype,
    )
    edges = [
        _matmul_edge(producer_id=n_q.op_id, consumer_id=n_core.op_id, shape=(seq_len, inter), dtype=dtype),
        _matmul_edge(producer_id=n_core.op_id, consumer_id=n_o.op_id, shape=(seq_len, inter), dtype=dtype),
    ]
    graph = build_contract_graph_from_nodes([n_q, n_core, n_o], edges)
    return BlockSpec(
        block_id=block_id,
        block_kind="attention",
        component=component,
        subgraph=graph,
        layer_index=layer_index,
        sample_module_paths=sample_paths,
    )


def _build_action_head_block(
    *,
    action_in: nn.Linear,
    action_out: nn.Linear,
    seq_len: int,
    dtype: str,
) -> BlockSpec | None:
    K1 = int(action_in.in_features)
    inter = int(action_in.out_features)
    K2 = int(action_out.in_features)
    N = int(action_out.out_features)
    if inter != K2:
        return None
    block_id = "action_head.head"
    n_in = _make_matmul_node(
        op_id=f"{block_id}.action_in_proj", region_id=f"{block_id}.action_in_proj",
        M=seq_len, K=K1, N=inter, dtype=dtype,
    )
    n_relu = _make_activation_node(
        op_id=f"{block_id}.relu", region_id=f"{block_id}.relu",
        shape=(seq_len, inter), op_name="relu", dtype=dtype,
    )
    n_out = _make_matmul_node(
        op_id=f"{block_id}.action_out_proj", region_id=f"{block_id}.action_out_proj",
        M=seq_len, K=K2, N=N, dtype=dtype,
    )
    edges = [
        _matmul_edge(producer_id=n_in.op_id, consumer_id=n_relu.op_id, shape=(seq_len, inter), dtype=dtype),
        _matmul_edge(producer_id=n_relu.op_id, consumer_id=n_out.op_id, shape=(seq_len, inter), dtype=dtype),
    ]
    graph = build_contract_graph_from_nodes([n_in, n_relu, n_out], edges)
    return BlockSpec(
        block_id=block_id,
        block_kind="head",
        component="action_head",
        subgraph=graph,
        layer_index=None,
        sample_module_paths=(action_in.__class__.__name__, action_out.__class__.__name__),
    )


# ---------------------------------------------------------------------------
# Enumerator
# ---------------------------------------------------------------------------


def _resolve_component_from_path(path: str) -> Component | None:
    low = path.lower()
    if "lm_expert" in low or "gemma_expert" in low:
        return "action_expert"
    if "action_in_proj" in low or "action_out_proj" in low or "action_head" in low:
        return "action_head"
    if "text_model" in low or "language_model" in low or "paligemma" in low:
        return "language_model"
    if "vision_tower" in low or "siglip" in low:
        return "vision"
    return None


@dataclass(frozen=True)
class BlockEnumeratorConfig:
    """Knobs for :func:`enumerate_blocks`.

    Attributes:
        seq_len: M dimension stamped into every matmul contract.
        dtype: dtype stamped on every contract (default ``"i8"`` —
            matches the Gemmini quantised flow).
        kinds: Which block kinds to enumerate. Default mlp + head;
            attention is opt-in because the linearisation drops more
            structural fidelity for attention than for MLP.
        components: Which components to enumerate. Default
            ``action_expert`` + ``action_head`` — Phase A scope.
        layer_indices: Restrict action_expert blocks to these layers.
            ``None`` = all layers.
    """

    seq_len: int = 64
    dtype: str = "i8"
    kinds: tuple[BlockKind, ...] = ("mlp", "head")
    components: tuple[Component, ...] = ("action_expert", "action_head")
    layer_indices: tuple[int, ...] | None = (0, 1, 2, 3)


def enumerate_blocks(
    model: nn.Module,
    config: BlockEnumeratorConfig | None = None,
) -> list[BlockSpec]:
    """Walk ``model`` for transformer blocks matching ``config``.

    Module-tree convention:
      * ``...lm_expert.layers.<N>.mlp.{gate_proj, up_proj, down_proj}``
      * ``...lm_expert.layers.<N>.self_attn.{q_proj, k_proj, v_proj, o_proj}``
      * ``...action_in_proj`` / ``...action_out_proj``

    Returns:
        List of :class:`BlockSpec`. Empty list when nothing matches —
        callers should treat that as "this model doesn't have any
        blocks I recognise" and skip rather than fail.
    """
    cfg = config or BlockEnumeratorConfig()

    # Group Linears by their owning block path: everything that shares
    # ``...layers.<N>.mlp.`` lives in the same block.
    linears = list(_iter_linear_modules(model))
    by_block: dict[str, dict[str, nn.Linear]] = {}
    block_components: dict[str, Component] = {}
    block_layer_idx: dict[str, int | None] = {}

    for name, mod in linears:
        component = _resolve_component_from_path(name)
        if component is None or component not in cfg.components:
            continue
        layer = _action_expert_layer_index_for_path(name) if component == "action_expert" else None
        if cfg.layer_indices is not None and layer is not None and layer not in cfg.layer_indices:
            continue
        # Block key: the portion of the path up to .mlp/.self_attn or
        # "action_head" for the head proj.
        low = name.lower()
        if ".mlp." in low and "mlp" in cfg.kinds:
            base = name.lower().split(".mlp.")[0] + ".mlp"
            tail = name.lower().split(".mlp.", 1)[1]
            by_block.setdefault(base, {})[tail] = mod
            block_components[base] = component
            block_layer_idx[base] = layer
        elif ".self_attn." in low and "attention" in cfg.kinds:
            base = name.lower().split(".self_attn.")[0] + ".self_attn"
            tail = name.lower().split(".self_attn.", 1)[1]
            by_block.setdefault(base, {})[tail] = mod
            block_components[base] = component
            block_layer_idx[base] = layer
        elif component == "action_head" and "head" in cfg.kinds:
            base = "action_head"
            tail = "in" if "action_in_proj" in low else "out"
            by_block.setdefault(base, {})[tail] = mod
            block_components[base] = "action_head"
            block_layer_idx[base] = None

    blocks: list[BlockSpec] = []
    for base, mods in sorted(by_block.items()):
        component = block_components[base]
        layer = block_layer_idx[base]
        if base.endswith(".mlp"):
            gate = mods.get("gate_proj")
            down = mods.get("down_proj")
            if gate is None or down is None:
                continue
            block = _build_mlp_block(
                layer_index=layer or 0,
                gate_proj=gate,
                down_proj=down,
                seq_len=cfg.seq_len,
                dtype=cfg.dtype,
                component=component,
                sample_paths=(f"{base}.gate_proj", f"{base}.down_proj"),
            )
        elif base.endswith(".self_attn"):
            q = mods.get("q_proj")
            o = mods.get("o_proj")
            if q is None or o is None:
                continue
            block = _build_attention_block(
                layer_index=layer or 0,
                q_proj=q,
                o_proj=o,
                seq_len=cfg.seq_len,
                dtype=cfg.dtype,
                component=component,
                sample_paths=(f"{base}.q_proj", f"{base}.o_proj"),
            )
        elif base == "action_head":
            ain = mods.get("in")
            aout = mods.get("out")
            if ain is None or aout is None:
                continue
            block = _build_action_head_block(
                action_in=ain, action_out=aout, seq_len=cfg.seq_len, dtype=cfg.dtype,
            )
        else:
            block = None
        if block is not None:
            blocks.append(block)
    return blocks


__all__ = [
    "BlockEnumeratorConfig",
    "BlockKind",
    "BlockSpec",
    "Component",
    "enumerate_blocks",
]

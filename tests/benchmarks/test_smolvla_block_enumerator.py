"""Tests for :mod:`xpu_rt.benchmarks.smolvla_block_enumerator`.

The enumerator templates-match transformer block conventions on a
real (or synthetic) ``nn.Module`` tree. These tests build a stub
``nn.Module`` whose named submodules mirror SmolVLA's PaliGemma
naming convention, then verify the enumerator finds the expected
blocks and produces ContractGraphs with the right edges.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from xpu_rt.benchmarks.smolvla_block_enumerator import (  # noqa: E402
    BlockEnumeratorConfig,
    BlockSpec,
    enumerate_blocks,
)


class _MLPStub(nn.Module):
    """SmolVLA-shaped MLP block: gate / up / down projections."""

    def __init__(self, hidden: int = 720, inter: int = 1440) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)


class _SelfAttnStub(nn.Module):
    """SmolVLA-shaped attention block."""

    def __init__(self, hidden: int = 720) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _TransformerLayerStub(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttnStub()
        self.mlp = _MLPStub()
        self.layer_idx = layer_idx


class _LMExpertStub(nn.Module):
    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_TransformerLayerStub(i) for i in range(n_layers))


class _SmolVLAStub(nn.Module):
    """Minimal SmolVLA-shaped module tree: an lm_expert with N layers
    + an action_in_proj / action_out_proj pair."""

    def __init__(self, n_layers: int = 4) -> None:
        super().__init__()
        # nest one level deep so the lookup matches '...lm_expert...'
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.lm_expert = _LMExpertStub(n_layers=n_layers)
        self.action_in_proj = nn.Linear(720, 320, bias=False)
        self.action_out_proj = nn.Linear(320, 7, bias=False)


def test_enumerate_mlp_blocks_from_lm_expert() -> None:
    model = _SmolVLAStub(n_layers=4)
    blocks = enumerate_blocks(
        model,
        BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",), layer_indices=(0, 1, 2, 3)),
    )
    assert len(blocks) == 4
    for b in blocks:
        assert isinstance(b, BlockSpec)
        assert b.block_kind == "mlp"
        assert b.component == "action_expert"
        # MLP chain: gate_proj → silu → down_proj
        assert len(b.subgraph.nodes) == 3
        assert len(b.subgraph.edges) == 2
        op_names = sorted(n.op_name for n in b.subgraph.nodes.values())
        assert op_names == ["matmul", "matmul", "silu"]


def test_enumerate_attention_blocks_when_opted_in() -> None:
    model = _SmolVLAStub(n_layers=2)
    blocks = enumerate_blocks(
        model,
        BlockEnumeratorConfig(kinds=("attention",), components=("action_expert",), layer_indices=(0, 1)),
    )
    assert len(blocks) == 2
    for b in blocks:
        assert b.block_kind == "attention"
        # Chain: q_proj → attention_core → o_proj
        assert len(b.subgraph.nodes) == 3
        assert len(b.subgraph.edges) == 2
        op_names = sorted(n.op_name for n in b.subgraph.nodes.values())
        assert op_names == ["matmul", "matmul", "softmax"]


def test_enumerate_action_head_block() -> None:
    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(
        model,
        BlockEnumeratorConfig(kinds=("head",), components=("action_head",), layer_indices=None),
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.block_kind == "head"
    assert b.component == "action_head"
    assert len(b.subgraph.nodes) == 3
    op_names = sorted(n.op_name for n in b.subgraph.nodes.values())
    assert op_names == ["matmul", "matmul", "relu"]


def test_layer_index_filter_restricts_blocks_to_named_layers() -> None:
    model = _SmolVLAStub(n_layers=6)
    # Only layers 0 and 1.
    blocks = enumerate_blocks(
        model,
        BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",), layer_indices=(0, 1)),
    )
    assert len(blocks) == 2
    # block_id encodes the layer index.
    ids = sorted(b.block_id for b in blocks)
    assert ids == ["action_expert.layer0.mlp", "action_expert.layer1.mlp"]


def test_enumerate_mlp_and_head_together_phase_a() -> None:
    """Phase A scope: MLP + head, no attention."""
    model = _SmolVLAStub(n_layers=4)
    blocks = enumerate_blocks(model)  # defaults = (mlp, head), (action_expert, action_head), (0,1,2,3)
    block_kinds = sorted(b.block_kind for b in blocks)
    # 4 mlp + 1 head
    assert block_kinds == ["head", "mlp", "mlp", "mlp", "mlp"]


def test_block_subgraph_has_topological_order() -> None:
    """The fusion planner depends on def-before-use topo order in
    the subgraph. Verify the enumerator stamps the right order."""
    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(model, BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",)))
    assert blocks
    sub = blocks[0].subgraph
    # MLP order: gate_proj before silu before down_proj.
    pos = {nid: i for i, nid in enumerate(sub.topological_order)}
    gate_id = next(nid for nid, n in sub.nodes.items() if "gate_proj" in n.region_id)
    silu_id = next(nid for nid, n in sub.nodes.items() if "silu" in n.region_id)
    down_id = next(nid for nid, n in sub.nodes.items() if "down_proj" in n.region_id)
    assert pos[gate_id] < pos[silu_id] < pos[down_id]

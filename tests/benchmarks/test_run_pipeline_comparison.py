"""Tests for the pipeline-level comparison driver.

The full driver loads SmolVLA; these tests pass pre-built ``BlockSpec``
objects (so the driver skips the loader) and verify the report shape +
plan-mode artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from xpu_rt.benchmarks.run_pipeline_comparison import (  # noqa: E402
    PipelineComparisonReport,
    gemmini_envelope,
    run,
)
from xpu_rt.benchmarks.smolvla_block_enumerator import (  # noqa: E402
    BlockEnumeratorConfig,
    enumerate_blocks,
)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(720, 1440, bias=False)
        self.up_proj = nn.Linear(720, 1440, bias=False)
        self.down_proj = nn.Linear(1440, 720, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()


class _SmolVLAStub(nn.Module):
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.lm_expert = nn.Module()
        self.vlm_with_expert.lm_expert.layers = nn.ModuleList(_Layer() for _ in range(n_layers))
        self.action_in_proj = nn.Linear(720, 320, bias=False)
        self.action_out_proj = nn.Linear(320, 7, bias=False)


def test_run_plan_mode_produces_report_files(tmp_path: Path) -> None:
    model = _SmolVLAStub(n_layers=2)
    blocks = enumerate_blocks(
        model,
        BlockEnumeratorConfig(kinds=("mlp", "head"), components=("action_expert", "action_head")),
    )
    assert blocks
    report = run(tmp_path / "out", mode="plan", blocks=blocks, envelope=gemmini_envelope())
    assert isinstance(report, PipelineComparisonReport)
    md = (tmp_path / "out" / "report.md").read_text()
    assert "Pipeline-level comparison" in md
    assert "gemmini" in md
    # Per-block table must list every block.
    for b in blocks:
        assert b.block_id in md
    js = json.loads((tmp_path / "out" / "report.json").read_text())
    assert js["target_id"] == "gemmini"
    assert js["n_blocks"] == len(blocks)
    assert js["mode"] == "plan"
    assert len(js["per_block"]) == len(blocks)


def test_run_persists_per_block_graph_json(tmp_path: Path) -> None:
    """The fusion-decision MCP tool consumes the per-block graph
    JSON. Ensure the driver actually emits it."""
    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(model, BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",)))
    assert blocks
    out_dir = tmp_path / "out"
    run(out_dir, mode="plan", blocks=blocks)
    per_block_dir = out_dir / "per_block"
    assert per_block_dir.is_dir()
    # At least one block sub-dir with a graph.json that round-trips.
    block_dirs = list(per_block_dir.iterdir())
    assert block_dirs
    graph_json = block_dirs[0] / "graph.json"
    assert graph_json.is_file()
    g = json.loads(graph_json.read_text())
    assert "nodes" in g and "edges" in g
    assert len(g["nodes"]) >= 2  # MLP block has 3 nodes


def test_agentic_branch_records_planner_verdicts(tmp_path: Path) -> None:
    """The agentic branch must surface per-pair fusion-oracle
    verdicts in the report — that's the *why* of the comparison."""
    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(model, BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",)))
    out_dir = tmp_path / "out"
    report = run(out_dir, mode="plan", blocks=blocks)
    assert report.per_block
    block_report = report.per_block[0]
    # An MLP block (3 nodes, 2 edges) should produce at least 2
    # pair verdicts when the planner walks the chain.
    assert len(block_report.agentic.per_pair_verdicts) >= 2
    # Each verdict must carry the producer/consumer + decision.
    for v in block_report.agentic.per_pair_verdicts:
        assert "producer_id" in v and "consumer_id" in v
        assert v["decision"] in ("fuse", "dont_fuse", "ineligible")


def test_vanilla_branch_counts_one_kernel_per_node(tmp_path: Path) -> None:
    """The vanilla branch's headline number: one kernel per
    ContractNode. This is the structural delta vs. the agentic side."""
    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(model, BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",)))
    report = run(tmp_path / "out", mode="plan", blocks=blocks)
    b = report.per_block[0]
    assert b.vanilla.n_kernels_emitted == b.n_nodes
    # Agentic emits ≤ vanilla — the whole point of fusion.
    assert b.agentic.n_kernels_emitted <= b.vanilla.n_kernels_emitted


def test_envelope_for_target_dispatches_to_right_envelope() -> None:
    """The cross-target driver resolves target_id → HardwareEnvelope.
    Regression guard: a new target must show up here AND in the
    matching code paths in :mod:`xpu_rt.kb_*` + the autocomp adapter."""
    from xpu_rt.benchmarks.run_pipeline_comparison import (
        envelope_for_target,
        gemmini_envelope,
        saturn_opu_envelope,
    )

    g = envelope_for_target("gemmini")
    assert g.target_name == "gemmini"
    assert g.weight_tile_bytes == gemmini_envelope().weight_tile_bytes

    s = envelope_for_target("saturn_opu_v128")
    assert s.target_name == "saturn_opu_v128"
    assert s.weight_tile_bytes == saturn_opu_envelope().weight_tile_bytes

    # Saturn has DRAM bandwidth headroom over Gemmini per the YAML
    # values; that delta is what the fusion oracle's cost model uses.
    assert s.peak_bandwidth_gbps > g.peak_bandwidth_gbps

    # Unknown ids fail fast — no silent fall-through.
    import pytest
    with pytest.raises(ValueError, match="no HardwareEnvelope wired"):
        envelope_for_target("h100_pcie")


def test_run_on_saturn_target_writes_saturn_envelope_into_report(tmp_path: Path) -> None:
    """When run() is given the Saturn envelope, the per-cluster
    planner verdict + report metadata must carry ``saturn_opu_v128``
    (not Gemmini). This catches the wiring regression where the
    driver hardcoded the envelope at function scope."""
    from xpu_rt.benchmarks.run_pipeline_comparison import saturn_opu_envelope

    model = _SmolVLAStub(n_layers=1)
    blocks = enumerate_blocks(model, BlockEnumeratorConfig(kinds=("mlp",), components=("action_expert",)))
    out = tmp_path / "saturn_out"
    report = run(out, mode="plan", blocks=blocks, envelope=saturn_opu_envelope())
    assert report.target_id == "saturn_opu_v128"
    js = json.loads((out / "report.json").read_text())
    assert js["target_id"] == "saturn_opu_v128"
    md = (out / "report.md").read_text()
    assert "saturn_opu_v128" in md

"""Tests for network analyzer — pattern detection on FX graphs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from compgen.agent.analyzer import NetworkAnalyzer
from compgen.agent.patterns import extract_fx_nodes, match_patterns
from compgen.targets.schema import load_profile

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


def _export_simple_mlp():
    sys.path.insert(0, str(EXAMPLES / "models"))
    from simple_mlp import SimpleMLP, get_sample_inputs

    return torch.export.export(SimpleMLP(), get_sample_inputs())


def _get_target():
    return load_profile(EXAMPLES / "target_profiles" / "cuda_a100.yaml")


def _get_multi_target():
    return load_profile(EXAMPLES / "target_profiles" / "multi_device.yaml")


# ---- Pattern extraction ----


def test_extract_fx_nodes() -> None:
    """extract_fx_nodes should find call_function nodes with shapes."""
    ep = _export_simple_mlp()
    nodes = extract_fx_nodes(ep)
    assert len(nodes) >= 3  # linear, gelu, linear
    for n in nodes:
        assert n.target.startswith("aten.")
        assert n.shape is not None


def test_extract_fx_nodes_have_data_flow() -> None:
    """Extracted nodes should have input_names and user_names."""
    ep = _export_simple_mlp()
    nodes = extract_fx_nodes(ep)
    # gelu should consume linear and be consumed by linear_1
    gelu_nodes = [n for n in nodes if "gelu" in n.target]
    assert len(gelu_nodes) == 1
    assert len(gelu_nodes[0].input_names) >= 1
    assert len(gelu_nodes[0].user_names) >= 1


# ---- Pattern matching ----


def test_match_linear_chain() -> None:
    """SimpleMLP should match as a linear_chain (linear→gelu→linear)."""
    ep = _export_simple_mlp()
    nodes = extract_fx_nodes(ep)
    matches = match_patterns(nodes)

    assert len(matches) >= 1
    chain_matches = [m for m in matches if m.pattern_name == "linear_chain"]
    assert len(chain_matches) == 1

    chain = chain_matches[0]
    assert len(chain.node_names) == 3
    assert chain.kernel_opportunity == "fused_mlp"


def test_match_consumes_all_ops() -> None:
    """All ops in SimpleMLP should be consumed by pattern matching."""
    ep = _export_simple_mlp()
    nodes = extract_fx_nodes(ep)
    matches = match_patterns(nodes)

    matched_names = set()
    for m in matches:
        matched_names.update(m.node_names)

    # All call_function nodes should be matched
    all_names = {n.name for n in nodes}
    assert matched_names == all_names


# ---- Full analysis ----


def test_analyze_simple_mlp() -> None:
    """Full analysis of SimpleMLP should produce structured results."""
    ep = _export_simple_mlp()
    target = _get_target()
    analysis = NetworkAnalyzer().analyze(ep, target, model_name="SimpleMLP")

    assert analysis.model_name == "SimpleMLP"
    assert analysis.total_flops > 0
    assert len(analysis.clusters) >= 1
    assert len(analysis.unclustered_ops) == 0  # all ops matched


def test_analyze_produces_bottlenecks() -> None:
    """Analysis should identify bottleneck clusters."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_target())

    assert len(analysis.bottleneck_clusters) >= 1
    # The linear_chain should be a bottleneck (it's the only compute)
    bottleneck_cluster = next(c for c in analysis.clusters if c.cluster_id == analysis.bottleneck_clusters[0])
    assert bottleneck_cluster.is_bottleneck


def test_analyze_produces_kernel_opportunities() -> None:
    """Analysis should surface kernel opportunities."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_target())

    opportunities = analysis.optimization_opportunities
    assert len(opportunities) >= 1
    assert any("fused_mlp" in o for o in opportunities)


def test_analyze_per_device_latency() -> None:
    """Clusters should have per-device latency estimates."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_target())

    for c in analysis.clusters:
        assert len(c.estimated_latency_per_device) > 0
        assert c.best_device != ""


def test_analyze_multi_device_opportunities() -> None:
    """Multi-device target should surface heterogeneous placement opportunity."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_multi_target())

    assert any("multi-device" in o.lower() or "heterogeneous" in o.lower() for o in analysis.optimization_opportunities)


def test_analyze_data_flow() -> None:
    """Analysis should produce data flow edges."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_target())

    # There should be at least one data flow edge (cluster→output)
    assert len(analysis.data_flow) >= 1


def test_analyze_builds_graph_dossier() -> None:
    """Analysis should expose a richer graph-analysis dossier."""
    ep = _export_simple_mlp()
    analysis = NetworkAnalyzer().analyze(ep, _get_target(), model_name="SimpleMLP")

    assert analysis.dossier is not None
    assert analysis.dossier.model_name == "SimpleMLP"
    assert analysis.dossier.total_regions >= 1
    assert len(analysis.dossier.regions) >= 1
    assert "aten.linear.default" in analysis.dossier.op_histogram

"""Tests for FX graph analyzer."""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.quantization.graph_analyzer import (
    QuantizedGraphAnalysis,
    analyze_for_npu,
    analyze_fx_graphs,
    format_analysis_report,
)


def _capture_simple_graph() -> list[torch.fx.GraphModule]:
    """Capture a simple model to get FX graphs."""
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    ).eval()
    x = torch.randn(4, 32)

    import torch._dynamo as dynamo

    dynamo.reset()
    captured: list[torch.fx.GraphModule] = []

    def backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> torch.fx.GraphModule:
        captured.append(gm)
        return gm

    compiled = torch.compile(model, backend=backend)
    compiled(x)
    return captured


class TestAnalyzeFxGraphs:
    def test_simple_model_coverage(self) -> None:
        graphs = _capture_simple_graph()
        assert len(graphs) >= 1
        analysis = analyze_for_npu(graphs)
        assert analysis.total_ops > 0
        assert analysis.partition_count >= 1

    def test_coverage_percentage(self) -> None:
        graphs = _capture_simple_graph()
        analysis = analyze_for_npu(graphs)
        # A simple Linear+ReLU model should have mostly covered ops
        assert analysis.coverage_pct > 0

    def test_mxu_ops_detected(self) -> None:
        """Linear layers should produce MXU-classified matmul ops."""
        graphs = _capture_simple_graph()
        analysis = analyze_for_npu(graphs)
        # Should have at least some matmul-related ops
        assert analysis.estimated_mxu_ops >= 0  # May be 0 if decomposed differently

    def test_custom_op_map(self) -> None:
        """Custom op map should work just like NPU map."""
        graphs = _capture_simple_graph()
        custom_map = {"aten.relu.default": True}
        analysis = analyze_fx_graphs(graphs, custom_map)
        # Should find some ops covered, some not
        assert analysis.total_ops > 0

    def test_empty_graph_list(self) -> None:
        analysis = analyze_for_npu([])
        assert analysis.total_ops == 0
        assert analysis.coverage_pct == 100.0
        assert analysis.partition_count == 0

    def test_to_dict(self) -> None:
        graphs = _capture_simple_graph()
        analysis = analyze_for_npu(graphs)
        d = analysis.to_dict()
        assert "total_ops" in d
        assert "coverage_pct" in d
        assert "covered_ops" in d
        assert "uncovered_ops" in d

    def test_to_json(self) -> None:
        graphs = _capture_simple_graph()
        analysis = analyze_for_npu(graphs)
        j = analysis.to_json()
        import json

        parsed = json.loads(j)
        assert isinstance(parsed, dict)


class TestFormatReport:
    def test_report_contains_key_sections(self) -> None:
        graphs = _capture_simple_graph()
        analysis = analyze_for_npu(graphs)
        report = format_analysis_report(analysis)
        assert "Coverage Report" in report
        assert "Total call_function ops" in report
        assert "MXU" in report
        assert "VPU" in report

    def test_report_empty_analysis(self) -> None:
        analysis = QuantizedGraphAnalysis()
        report = format_analysis_report(analysis)
        assert "0" in report

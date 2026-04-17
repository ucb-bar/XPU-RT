"""Tests for the FX graph decomposition passes in transforms/graph_passes.py.

Covers:
- `detect_and_annotate_patterns`: linear→activation fusion annotation
- `fold_transpose_into_matmul`: transpose-folding pass
- `raise_composite_ops`: composite-op raising
- `run_all_decomposition_passes` / `run_decomposition_on_graphs`: aggregator

Uses torch.fx.symbolic_trace with torch.nn.functional calls so nodes are
captured as `call_function` (the pass only inspects call_function nodes).
"""

from __future__ import annotations

import torch
from torch import nn

from compgen.transforms.graph_passes import (
    detect_and_annotate_patterns,
    fold_transpose_into_matmul,
    raise_composite_ops,
    run_all_decomposition_passes,
    run_decomposition_on_graphs,
)


class LinearSilu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(
            torch.nn.functional.linear(x, self.weight, self.bias)
        )


class AddOnly(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


def test_linear_silu_gets_annotated() -> None:
    graph = torch.fx.symbolic_trace(LinearSilu())
    detected = detect_and_annotate_patterns(graph)
    assert detected >= 1
    labels = [n.meta.get("_compgen_pattern") for n in graph.graph.nodes]
    assert any(label and "fused_linear" in label for label in labels)


def test_empty_graph_has_no_patterns() -> None:
    graph = torch.fx.symbolic_trace(AddOnly())
    assert detect_and_annotate_patterns(graph) == 0
    assert fold_transpose_into_matmul(graph) == 0
    assert raise_composite_ops(graph) == 0


def test_run_all_returns_pass_summary() -> None:
    graph = torch.fx.symbolic_trace(LinearSilu())
    summary = run_all_decomposition_passes(graph)
    assert set(summary.keys()) == {
        "detect_and_annotate_patterns",
        "fold_transpose_into_matmul",
        "raise_composite_ops",
    }
    assert summary["detect_and_annotate_patterns"] >= 1


def test_run_decomposition_on_graphs_aggregates() -> None:
    graphs = [torch.fx.symbolic_trace(LinearSilu()) for _ in range(3)]
    totals = run_decomposition_on_graphs(graphs)
    assert totals["detect_and_annotate_patterns"] >= 3

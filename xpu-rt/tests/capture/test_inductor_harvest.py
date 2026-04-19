"""Tests for the P21 inductor graph harvester."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from compgen.capture import InductorHarvestReport, harvest_inductor_graph


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def test_disabled_returns_skipped() -> None:
    m = _TinyMLP()
    r = harvest_inductor_graph(m, (torch.randn(2, 8),), enabled=False)
    assert r.status == "skipped"
    assert r.fx_graph_count == 0
    assert r.fx_node_count == 0


def test_enabled_captures_graphs() -> None:
    m = _TinyMLP()
    r = harvest_inductor_graph(m, (torch.randn(2, 8),), enabled=True)
    assert r.status == "ok", f"unexpected fallback: {r.fallback_reason}"
    assert r.fx_graph_count >= 1
    assert r.fx_node_count > 0
    assert r.elapsed_ms >= 0


def test_fallback_on_failure() -> None:
    class _Broken(nn.Module):
        def forward(self, x):  # type: ignore[override]
            raise RuntimeError("intentional")

    r = harvest_inductor_graph(_Broken(), (torch.randn(1),), enabled=True)
    assert r.status == "fallback"
    assert "intentional" in r.fallback_reason


def test_report_is_frozen_dataclass() -> None:
    r = InductorHarvestReport(status="ok", backend="test")
    with pytest.raises(Exception):
        r.status = "changed"  # type: ignore[misc]


def test_op_histogram_counts_opaque_names() -> None:
    """Even when targets are Python callables (pre-decomposition), we get stable names."""
    m = _TinyMLP()
    r = harvest_inductor_graph(m, (torch.randn(2, 8),), enabled=True)
    # Each captured name should be a non-empty string
    for op_name, count in r.fx_op_histogram.items():
        assert isinstance(op_name, str) and op_name
        assert count > 0

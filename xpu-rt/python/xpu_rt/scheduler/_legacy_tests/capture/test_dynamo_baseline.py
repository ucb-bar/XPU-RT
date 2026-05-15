"""Tests for dynamo baseline capture types."""

from __future__ import annotations

import torch
import torch.nn as nn
from compgen.capture.dynamo_baseline import (
    BaselineReport,
    DynamoReport,
    collect_diagnostics,
    compile_baseline,
)


def test_baseline_report_construction() -> None:
    """BaselineReport should be constructible with required fields."""
    report = BaselineReport(
        cold_compile_ms=1200.0,
        warm_run_ms=3.5,
        num_graph_breaks=2,
        compiled_op_fraction=0.95,
    )
    assert report.cold_compile_ms == 1200.0
    assert report.warm_run_ms == 3.5
    assert report.num_graph_breaks == 2
    assert report.compiled_op_fraction == 0.95
    assert report.backend == "inductor"


def test_dynamo_report_defaults() -> None:
    """DynamoReport should have sensible defaults."""
    report = DynamoReport()
    assert report.graph_breaks == []
    assert report.guard_failures == 0
    assert report.op_coverage == {}
    assert report.warnings == []


def test_compile_baseline_runs_torch_compile() -> None:
    """compile_baseline should run torch.compile and return a BaselineReport."""
    torch._dynamo.reset()

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    x = (torch.randn(2, 8),)
    report = compile_baseline(model, x, num_warmup=1, num_runs=2)

    assert isinstance(report, BaselineReport)
    assert report.cold_compile_ms > 0
    assert report.warm_run_ms > 0
    assert report.backend == "inductor"
    assert isinstance(report.num_graph_breaks, int)
    assert 0.0 <= report.compiled_op_fraction <= 1.0


def test_collect_diagnostics_returns_dynamo_report() -> None:
    """collect_diagnostics should return a DynamoReport with graph break info."""
    torch._dynamo.reset()

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
    x = (torch.randn(2, 8),)
    report = collect_diagnostics(model, x)

    assert isinstance(report, DynamoReport)
    assert isinstance(report.graph_breaks, list)
    assert isinstance(report.op_coverage, dict)
    assert isinstance(report.graph_count, int)

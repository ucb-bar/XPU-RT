"""Tests for dynamo baseline capture types."""

from __future__ import annotations

import pytest
from compgen.capture.dynamo_baseline import BaselineReport, DynamoReport


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


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_compile_baseline_runs_torch_compile() -> None:
    """compile_baseline should run torch.compile and return a BaselineReport."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_collect_diagnostics_returns_dynamo_report() -> None:
    """collect_diagnostics should return a DynamoReport with graph break info."""

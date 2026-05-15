"""Tests for the case/study runners."""

from __future__ import annotations

from pathlib import Path

from benchmarks.registry import build_default_registry
from benchmarks.runner import run_case
from benchmarks.spec import WorkspaceConfig


def test_run_case_with_expert_fixture_only(tmp_path) -> None:
    registry = build_default_registry()
    records = run_case(
        "pipeline_simple_mlp_cuda_a100",
        registry=registry,
        workspace=WorkspaceConfig.default(Path(__file__).resolve().parents[2]),
        output_dir=tmp_path,
        baseline_ids=["expert_fixture"],
    )
    assert len(records) == 1
    record = records[0]
    assert record.system_name == "expert_fixture"
    assert record.status == "pass"
    assert record.performance.latency_median_us > 0

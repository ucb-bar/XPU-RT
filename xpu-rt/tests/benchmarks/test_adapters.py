"""Tests for benchmark adapters."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.adapters import AdapterContext, ExpertFixtureAdapter, ExternalRepoAdapter, check_baseline_availability
from benchmarks.registry import build_default_registry
from benchmarks.spec import BaselineSpec, ExperimentCase, TargetSpec, WorkloadSpec, WorkspaceConfig


def _dummy_loader() -> tuple[object, tuple[object, ...]]:
    return object(), (object(),)


def test_expert_fixture_adapter_reads_fixture(tmp_path: Path) -> None:
    fixture_path = tmp_path / "expert.json"
    fixture_path.write_text(
        json.dumps(
            {
                "simple_mlp:cuda_a100": {
                    "latency_median_us": 123.0,
                    "latency_p99_us": 140.0,
                    "throughput_samples_per_sec": 10.0,
                    "peak_memory_bytes": 2048,
                }
            }
        )
    )

    registry = build_default_registry()
    workload = WorkloadSpec("simple_mlp", "tier_b", "dummy", _dummy_loader)
    target = TargetSpec("cuda_a100", tmp_path / "target.yaml", "target_profile", "dummy", "GPU")
    baseline = BaselineSpec("expert_fixture", "expert_fixture", "dummy", fixture_path=str(fixture_path))
    case = ExperimentCase("case", "study", "simple_mlp", "cuda_a100", ["expert_fixture"])
    ctx = AdapterContext(
        workspace=WorkspaceConfig.default(tmp_path),
        registry=registry,
        case=case,
        workload=workload,
        target=target,
        baseline=baseline,
        output_dir=tmp_path,
    )

    adapter = ExpertFixtureAdapter()
    available, reason = adapter.is_available(ctx)
    assert available, reason

    record = adapter.run(ctx)
    assert record.status == "pass"
    assert record.performance.latency_median_us == 123.0


def test_external_repo_adapter_skips_missing_repo(tmp_path: Path) -> None:
    registry = build_default_registry()
    workload = WorkloadSpec("simple_mlp", "tier_b", "dummy", _dummy_loader)
    target = TargetSpec("cuda_a100", tmp_path / "target.yaml", "target_profile", "dummy", "GPU")
    baseline = BaselineSpec("iree", "external_repo", "dummy", repo_name="iree")
    case = ExperimentCase("case", "study", "simple_mlp", "cuda_a100", ["iree"])
    ctx = AdapterContext(
        workspace=WorkspaceConfig.default(tmp_path / "CompGen"),
        registry=registry,
        case=case,
        workload=workload,
        target=target,
        baseline=baseline,
        output_dir=tmp_path,
    )

    record = ExternalRepoAdapter().run(ctx)
    assert record.status == "skip"
    assert "Sibling repo missing" in record.errors[0]


def test_check_baseline_availability_reports_states(tmp_path: Path) -> None:
    registry = build_default_registry()
    workspace = WorkspaceConfig(repo_root=tmp_path / "CompGen", external_roots={"iree": tmp_path / "iree"})
    (tmp_path / "iree").mkdir(parents=True)
    result = check_baseline_availability(registry, workspace, ["iree"])
    assert result["iree"] == "available"

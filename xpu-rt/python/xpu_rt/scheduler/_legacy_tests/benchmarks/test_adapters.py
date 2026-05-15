"""Tests for benchmark adapters."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.adapters import (
    AdapterContext,
    CompGenAdapter,
    ExpertFixtureAdapter,
    ExternalRepoAdapter,
    TorchEagerAdapter,
    check_baseline_availability,
)
from benchmarks.registry import build_default_registry
from benchmarks.spec import BaselineSpec, ExperimentCase, TargetSpec, WorkloadSpec, WorkspaceConfig


def _dummy_loader() -> tuple[object, tuple[object, ...]]:
    return object(), (object(),)


def _tiny_torch_loader() -> tuple[torch.nn.Module, tuple[torch.Tensor, ...]]:
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU()).eval()
    return model, (torch.randn(2, 8),)


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


def test_torch_eager_skips_analysis_only_workloads(tmp_path: Path) -> None:
    registry = build_default_registry()
    workload = WorkloadSpec("smolvla_one_step", "tier_frontier", "dummy", _dummy_loader, readiness="analysis_only")
    target = TargetSpec("cuda_a100", tmp_path / "target.yaml", "target_profile", "dummy", "GPU")
    baseline = BaselineSpec("torch_eager", "torch_eager", "dummy")
    case = ExperimentCase("case", "study", "smolvla_one_step", "cuda_a100", ["torch_eager"])
    ctx = AdapterContext(
        workspace=WorkspaceConfig.default(tmp_path),
        registry=registry,
        case=case,
        workload=workload,
        target=target,
        baseline=baseline,
        output_dir=tmp_path,
    )

    record = TorchEagerAdapter().run(ctx)
    assert record.status == "skip"


def test_compgen_adapter_composes_target_package_for_analysis_only_runs(tmp_path: Path) -> None:
    registry = build_default_registry()
    workload = WorkloadSpec(
        "tiny_suite_model",
        "tier_suite",
        "tiny torch model",
        _tiny_torch_loader,
        readiness="analysis_only",
    )
    target = TargetSpec(
        "cuda_a100",
        Path("examples/target_profiles/cuda_a100.yaml"),
        "target_profile",
        "gpu target",
        "TRITON_FRIENDLY",
    )
    baseline = BaselineSpec("compgen", "compgen", "dummy")
    case = ExperimentCase("case", "study", "tiny_suite_model", "cuda_a100", ["compgen"])
    ctx = AdapterContext(
        workspace=WorkspaceConfig(
            repo_root=tmp_path / "CompGen",
            suite_configs={"compgen": {"packs": ["cuda_tile", "iree_tracy"]}},
        ),
        registry=registry,
        case=case,
        workload=workload,
        target=target,
        baseline=baseline,
        output_dir=tmp_path,
    )

    record = CompGenAdapter().run(ctx)
    assert record.status == "pass"
    assert record.config["active_packs"] == ["cuda_tile", "iree_tracy"]
    assert "target_package" in record.artifacts.artifact_paths

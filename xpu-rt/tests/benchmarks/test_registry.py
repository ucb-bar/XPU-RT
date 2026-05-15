"""Tests for the benchmark registry and workspace config."""

from __future__ import annotations

from pathlib import Path

from benchmarks.collector import collect_synthesis_metrics
from benchmarks.record import RunRecord, SynthesisMetrics
from benchmarks.registry import build_default_registry
from benchmarks.spec import WorkspaceConfig


def test_workspace_default_external_resolution() -> None:
    ws = WorkspaceConfig.default("/tmp/compgen")
    resolved = ws.resolve_external("iree")
    assert resolved == Path("/tmp/iree")


def test_workspace_explicit_external_resolution(tmp_path: Path) -> None:
    ws = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        external_roots={"xla": tmp_path / "custom-xla"},
        suite_configs={"mlperf": {"dataset_root": str(tmp_path / "datasets")}},
    )
    assert ws.resolve_external("xla") == tmp_path / "custom-xla"
    assert ws.get_suite_config("mlperf")["dataset_root"] == str(tmp_path / "datasets")


def test_workspace_pack_and_llvm_resolution(tmp_path: Path) -> None:
    ws = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        pack_roots={"snax_mlir": tmp_path / "snax"},
        llvm_forks={"gemmini": tmp_path / "llvm-gemmini"},
    )
    assert ws.resolve_pack_root("snax_mlir") == tmp_path / "snax"
    assert ws.resolve_llvm_fork("gemmini") == tmp_path / "llvm-gemmini"


def test_default_registry_contains_paper_subset() -> None:
    registry = build_default_registry()
    assert "paper_subset" in registry.studies
    assert "guard_synthesis" in registry.studies
    assert "frontier_all" in registry.studies
    assert "frontier_robotics" in registry.studies
    assert "paper_frontier_ready" in registry.studies
    assert "cuda_a100" in registry.targets
    assert "riscv_soc" in registry.targets
    assert "multi_device" in registry.targets
    assert "BundleT" in registry.bundles
    assert "BundleM" in registry.bundles
    assert "simple_mlp" in registry.workloads
    assert "transformer_block" in registry.workloads
    assert "quantized_mlp" in registry.workloads
    assert "smolvla_one_step" in registry.workloads
    assert "groot_policy_step" in registry.workloads


def test_frontier_workload_metadata_is_registered() -> None:
    registry = build_default_registry()
    smolvla = registry.workloads["smolvla_one_step"]
    groot = registry.workloads["groot_policy_step"]

    assert smolvla.source_model_id == "lerobot/smolvla_base"
    assert smolvla.capture_mode == "torch_dynamo_partitioned"
    assert smolvla.readiness == "analysis_only"
    assert smolvla.expected_status == "pass"
    assert groot.expected_status == "xfail"


def test_verification_red_team_defects_registered() -> None:
    registry = build_default_registry()
    assert "wrong_tile_sizes" in registry.defects
    assert "numerically_wrong_kernel" in registry.defects


def test_collect_synthesis_metrics() -> None:
    metrics = collect_synthesis_metrics(
        {
            "fragments_proposed": 7,
            "promoted": 2,
            "average_guard_terms": 3.5,
            "families": {"fusion": {"promoted": 1}},
        }
    )
    assert metrics.fragments_proposed == 7
    assert metrics.promoted == 2
    assert metrics.families["fusion"]["promoted"] == 1


def test_run_record_has_synthesis_metrics() -> None:
    record = RunRecord()
    assert isinstance(record.synthesis, SynthesisMetrics)

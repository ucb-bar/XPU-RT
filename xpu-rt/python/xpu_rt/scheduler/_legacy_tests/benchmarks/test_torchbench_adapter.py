"""Tests for the TorchBench suite adapter."""

from __future__ import annotations

import json
from pathlib import Path

from compgen.benchmarks.common.manifest import SuiteManifestEntry
from compgen.benchmarks.torchbench.adapter import TorchBenchAdapter

from benchmarks.record import RunRecord


def test_suite_id_is_torchbench() -> None:
    adapter = TorchBenchAdapter()
    assert adapter.suite_id == "torchbench"


def test_enumerate_workloads_returns_builtin_manifest_without_workspace() -> None:
    adapter = TorchBenchAdapter()
    entries = adapter.enumerate_workloads()
    ids = {e.workload_id for e in entries}
    assert "hf_Bert" in ids
    assert "resnet50" in ids
    assert "timm_vision_transformer" in ids
    assert all(e.suite_id == "torchbench" for e in entries)


def test_enumerate_workloads_discovers_models_from_filesystem(tmp_path: Path) -> None:
    """Fake a torchbench models directory and verify discovery."""
    tb_root = tmp_path / "benchmark"
    models_dir = tb_root / "torchbenchmark" / "models"

    # Create fake model directories
    for name in ("resnet50", "hf_Bert", "custom_model", "another_model"):
        model_dir = models_dir / name
        model_dir.mkdir(parents=True)
        (model_dir / "__init__.py").write_text("")

    # Create a directory without __init__.py (should be skipped)
    (models_dir / "invalid_model").mkdir(parents=True)

    # Create a hidden directory (should be skipped)
    (models_dir / ".hidden").mkdir(parents=True)
    (models_dir / ".hidden" / "__init__.py").write_text("")

    # Use a minimal workspace config pointing to the fake root
    from benchmarks.spec import WorkspaceConfig

    workspace = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        external_roots={"torchbench": tb_root},
    )
    (tmp_path / "CompGen").mkdir(exist_ok=True)

    adapter = TorchBenchAdapter()
    entries = adapter.enumerate_workloads(workspace=workspace)
    ids = {e.workload_id for e in entries}

    assert "resnet50" in ids
    assert "hf_Bert" in ids
    assert "custom_model" in ids
    assert "another_model" in ids
    assert "invalid_model" not in ids
    assert ".hidden" not in ids
    # Builtin entry that was not on disk should still appear
    assert "timm_vision_transformer" in ids


def test_enumerate_workloads_blessed_only(tmp_path: Path) -> None:
    """blessed_only should filter to only blessed models."""
    tb_root = tmp_path / "benchmark"
    models_dir = tb_root / "torchbenchmark" / "models"
    for name in ("resnet50", "hf_Bert", "custom_model"):
        model_dir = models_dir / name
        model_dir.mkdir(parents=True)
        (model_dir / "__init__.py").write_text("")

    from benchmarks.spec import WorkspaceConfig

    workspace = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        external_roots={"torchbench": tb_root},
    )
    (tmp_path / "CompGen").mkdir(exist_ok=True)

    adapter = TorchBenchAdapter()
    blessed = adapter.enumerate_workloads(workspace=workspace, blessed_only=True)
    ids = {e.workload_id for e in blessed}

    # custom_model is not in builtin blessed set
    assert "custom_model" not in ids
    assert "resnet50" in ids
    assert "hf_Bert" in ids


def test_prepare_environment_not_installed() -> None:
    """Without workspace or importable package, environment should be unavailable."""
    adapter = TorchBenchAdapter()
    status = adapter.prepare_environment()
    # Without workspace=None and torchbenchmark not importable, should be unavailable
    assert status.suite_id == "torchbench"
    assert status.available is False


def test_prepare_environment_available_with_root(tmp_path: Path) -> None:
    """When a valid root exists, environment should be available."""
    tb_root = tmp_path / "benchmark"
    tb_root.mkdir()

    from benchmarks.spec import WorkspaceConfig

    workspace = WorkspaceConfig(
        repo_root=tmp_path / "CompGen",
        external_roots={"torchbench": tb_root},
    )
    (tmp_path / "CompGen").mkdir(exist_ok=True)

    adapter = TorchBenchAdapter()
    status = adapter.prepare_environment(workspace=workspace)
    assert status.suite_id == "torchbench"
    assert status.available is True
    assert status.source_root == str(tb_root)


def test_collect_metrics_produces_valid_normalized_results() -> None:
    """Given a mocked RunRecord, collect_metrics should produce NormalizedSuiteResult."""
    record = RunRecord(
        model_name="resnet50",
        target_name="cpu",
        system_name="torchbench_eager",
    )
    record.status = "pass"
    record.suite.suite_id = "torchbench"
    record.suite.upstream_workload_id = "resnet50"
    record.suite.mode = "inference"
    record.suite.device = "cpu"
    record.suite.dtype = "float32"
    record.performance.latency_median_us = 5000.0
    record.performance.latency_p90_us = 5500.0
    record.performance.throughput_samples_per_sec = 200.0
    record.performance.peak_memory_bytes = 64 * 1024 * 1024
    record.verification.overall_status = "pass"

    adapter = TorchBenchAdapter()
    results = adapter.collect_metrics([record])

    assert len(results) == 1
    result = results[0]
    assert result.suite == "torchbench"
    assert result.workload == "resnet50"
    assert result.device == "cpu"
    assert result.latency_ms_p50 == 5.0
    assert result.latency_ms_p90 == 5.5
    assert result.throughput == 200.0
    assert result.verification_ok is True
    assert result.correctness_ok is True


def test_collect_metrics_handles_failing_record() -> None:
    """A failing record should still produce a result with correctness_ok=False."""
    record = RunRecord(model_name="bad_model", system_name="torchbench_eager")
    record.status = "fail"
    record.errors.append("model_load_failed")
    record.suite.suite_id = "torchbench"
    record.suite.upstream_workload_id = "bad_model"
    record.suite.mode = "inference"
    record.suite.device = "cpu"
    record.suite.dtype = "float32"
    record.verification.overall_status = "fail"

    adapter = TorchBenchAdapter()
    results = adapter.collect_metrics([record])

    assert len(results) == 1
    assert results[0].correctness_ok is False
    assert results[0].verification_ok is False


def test_emit_artifacts_writes_json_files(tmp_path: Path) -> None:
    """emit_artifacts should write normalized JSON to output_dir."""
    record = RunRecord(model_name="resnet50", system_name="torchbench_eager")
    record.status = "pass"
    record.suite.suite_id = "torchbench"
    record.suite.upstream_workload_id = "resnet50"
    record.suite.mode = "inference"
    record.suite.device = "cpu"
    record.suite.dtype = "float32"
    record.verification.overall_status = "pass"

    adapter = TorchBenchAdapter()
    paths = adapter.emit_artifacts([record], output_dir=tmp_path / "artifacts")

    assert len(paths) == 1
    assert paths[0].exists()
    payload = json.loads(paths[0].read_text())
    assert payload["suite"] == "torchbench"
    assert payload["workload"] == "resnet50"


def test_prepare_inputs_returns_none_without_root() -> None:
    """Without a TorchBench root, prepare_inputs should return None."""
    entry = SuiteManifestEntry(
        suite_id="torchbench",
        workload_id="resnet50",
        description="test",
        upstream_workload_id="resnet50",
    )
    adapter = TorchBenchAdapter()
    result = adapter.prepare_inputs(entry)
    assert result is None


def test_run_reference_fails_without_model(tmp_path: Path) -> None:
    """run_reference should return a failing record when model can't load."""
    entry = SuiteManifestEntry(
        suite_id="torchbench",
        workload_id="resnet50",
        description="test",
        upstream_workload_id="resnet50",
    )
    adapter = TorchBenchAdapter()
    records = adapter.run_reference(entry, output_dir=tmp_path)

    assert len(records) == 1
    assert records[0].status == "fail"
    assert "model_load_failed" in records[0].errors


def test_run_compgen_fails_without_model(tmp_path: Path) -> None:
    """run_compgen should return a failing record when model can't load."""
    entry = SuiteManifestEntry(
        suite_id="torchbench",
        workload_id="resnet50",
        description="test",
        upstream_workload_id="resnet50",
    )
    adapter = TorchBenchAdapter()
    records = adapter.run_compgen(entry, output_dir=tmp_path)

    assert len(records) == 1
    assert records[0].status == "fail"
    assert "model_load_failed" in records[0].errors

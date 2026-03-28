"""Tests for recognized benchmark suite integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.record import RunRecord
from benchmarks.spec import WorkspaceConfig
from benchmarks.suite_runner import export_suite_results, list_suite_workloads, list_suites, run_suite_workload


def _fake_external_runner(path: Path) -> None:
    path.write_text(
        """
import json
import sys
from pathlib import Path

mode = sys.argv[1]
metrics_path = Path(sys.argv[2])
payload = {
    "status": "pass",
    "compile_time_ms": 250.0 if mode == "compgen" else 0.0,
    "latency_ms_p50": 4.0 if mode == "reference" else 5.5,
    "latency_ms_p90": 4.4 if mode == "reference" else 6.0,
    "throughput": 200.0 if mode == "reference" else 150.0,
    "peak_memory_mb": 64.0,
    "verification_ok": True,
    "correctness_ok": True,
    "official_metrics": [{"name": "offline_qps", "value": 2000, "unit": "qps"}],
}
metrics_path.write_text(json.dumps(payload))
""".strip()
    )


def test_list_suites_reports_all_registered_suites(tmp_path: Path) -> None:
    workspace = WorkspaceConfig.default(tmp_path / "CompGen")
    statuses = list_suites(workspace=workspace)
    assert "torchbench" in statuses
    assert "huggingface" in statuses
    assert "timm" in statuses
    assert "pack_integrations" in statuses
    assert "mlperf" in statuses
    assert "sol_execbench" in statuses
    assert "heterobench" in statuses


def test_list_suite_workloads_filters_blessed_subset() -> None:
    all_entries = list_suite_workloads("mlperf", blessed_only=False)
    blessed_entries = list_suite_workloads("mlperf", blessed_only=True)
    assert any(entry.workload_id == "rgat" for entry in all_entries)
    assert all(entry.workload_id != "rgat" for entry in blessed_entries)


def test_run_suite_workload_with_fake_external_commands(tmp_path: Path) -> None:
    repo_root = tmp_path / "CompGen"
    repo_root.mkdir()
    mlperf_root = tmp_path / "mlperf_inference"
    mlperf_root.mkdir()
    runner = mlperf_root / "runner.py"
    _fake_external_runner(runner)

    workspace = WorkspaceConfig(
        repo_root=repo_root,
        external_roots={"mlperf_inference": mlperf_root},
        suite_configs={
            "mlperf": {
                "reference_command": [sys.executable, "{suite_root}/runner.py", "reference", "{metrics_path}"],
                "compgen_command": [sys.executable, "{suite_root}/runner.py", "compgen", "{metrics_path}"],
            }
        },
    )

    records = run_suite_workload(
        "mlperf",
        "llama3.1-8b",
        workspace=workspace,
        output_dir=tmp_path / "results",
    )

    assert len(records) == 2
    assert all(record.status == "pass" for record in records)
    assert all(record.suite.suite_id == "mlperf" for record in records)
    assert any("normalized_result" in record.artifacts.artifact_paths for record in records)


def test_export_suite_results_writes_normalized_json(tmp_path: Path) -> None:
    record = RunRecord(model_name="mlperf_resnet50", system_name="mlperf_official")
    record.status = "pass"
    record.suite.suite_id = "mlperf"
    record.suite.upstream_workload_id = "resnet50-v1.5"
    record.suite.mode = "inference"
    record.suite.device = "cpu"
    record.suite.dtype = "float32"
    record.performance.latency_median_us = 5000.0
    record.performance.latency_p90_us = 5500.0
    record.performance.throughput_samples_per_sec = 250.0

    paths = export_suite_results([record], tmp_path / "normalized")
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text())
    assert payload["suite"] == "mlperf"
    assert payload["workload"] == "resnet50-v1.5"
    assert payload["latency_ms_p50"] == 5.0


def test_run_pack_integration_with_fake_commands(tmp_path: Path) -> None:
    repo_root = tmp_path / "CompGen"
    repo_root.mkdir()
    pack_root = tmp_path / "cuda-tile"
    pack_root.mkdir()
    (pack_root / "README.md").write_text("cuda tile")
    runner = pack_root / "runner.py"
    _fake_external_runner(runner)

    workspace = WorkspaceConfig(
        repo_root=repo_root,
        pack_roots={"cuda_tile": pack_root},
        integration_worktrees_root=tmp_path / "worktrees",
        pack_configs={
            "cuda_tile": {
                "reference_command": [sys.executable, "{suite_root}/runner.py", "reference", "{metrics_path}"],
                "compgen_command": [sys.executable, "{suite_root}/runner.py", "compgen", "{metrics_path}"],
            }
        },
    )

    records = run_suite_workload(
        "pack_integrations",
        "cuda_tile",
        workspace=workspace,
        output_dir=tmp_path / "results",
    )

    assert len(records) == 2
    assert all(record.suite.extra["pack_id"] == "cuda_tile" for record in records)
    assert all(record.suite.extra["probe_ok"] for record in records)
    assert all("branch_name" in record.suite.extra for record in records)

"""Recognized benchmark suite adapters."""

from __future__ import annotations

import csv
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from compgen.benchmarks import (
    NormalizedSuiteResult,
    SuiteEnvironmentStatus,
    SuiteManifestEntry,
    SuiteRunConfig,
    filter_manifest_entries,
    resolve_suite_root,
    write_normalized_suite_results,
)
from compgen.packs import default_pack_root, load_builtin_packs, load_pack

from benchmarks.adapters import AdapterContext, CompGenAdapter
from benchmarks.collector import collect_performance_metrics
from benchmarks.record import RunRecord
from benchmarks.registry import build_default_registry
from benchmarks.spec import BaselineSpec, ExperimentCase, WorkloadSpec, WorkspaceConfig


def _import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def _command_mapping(
    entry: SuiteManifestEntry,
    *,
    workspace: WorkspaceConfig | None,
    suite_root: Path | None,
    output_dir: Path,
    metrics_path: Path,
    config: SuiteRunConfig,
) -> dict[str, Any]:
    return {
        "python": sys.executable,
        "repo_root": str(workspace.repo_root) if workspace is not None else "",
        "suite_root": str(suite_root) if suite_root is not None else "",
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "workload_id": entry.workload_id,
        "upstream_workload_id": entry.upstream_workload_id or entry.workload_id,
        "mode": config.mode or entry.mode,
        "device": config.device,
        "dtype": config.dtype,
        "batch_size": config.batch_size or entry.batch_size,
    }


def _materialize_command(template: list[str] | str, mapping: dict[str, Any]) -> list[str]:
    if isinstance(template, str):
        return [template.format(**mapping)]
    return [str(part).format(**mapping) for part in template]


def _parse_metrics_file(path: Path, workload_id: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            if workload_id in payload and isinstance(payload[workload_id], dict):
                return dict(payload[workload_id])
            return dict(payload)
        return {}
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {}
        workload_keys = ("workload", "model", "name", "benchmark")
        for row in rows:
            if any((row.get(key) or "") in {workload_id, workload_id.replace("/", "_")} for key in workload_keys):
                return dict(row)
        return dict(rows[0])
    return {}


def _official_metrics_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("official_metrics"), list):
        return [dict(metric) for metric in payload["official_metrics"] if isinstance(metric, dict)]

    official_keys = {
        "offline_qps": "qps",
        "server_qps": "qps",
        "single_stream_latency_ms": "ms",
        "accuracy": "",
        "sol_score": "score",
        "geomean_sol_score": "score",
        "makespan_ms": "ms",
        "speedup": "x",
        "tokens_per_sec": "tokens/s",
        "prefill_tokens_per_sec": "tokens/s",
        "decode_tokens_per_sec": "tokens/s",
        "images_per_sec": "images/s",
    }
    metrics: list[dict[str, Any]] = []
    for key, unit in official_keys.items():
        if key in payload:
            metrics.append({"name": key, "value": payload[key], "unit": unit})
    return metrics


def _apply_metrics_payload(record: RunRecord, payload: dict[str, Any]) -> None:
    if not payload:
        return
    record.capture.export_success = bool(payload.get("export_success", record.capture.export_success))
    record.capture.analysis_success = bool(payload.get("capture_success", record.capture.analysis_success))
    record.capture.graph_break_count = int(
        payload.get("graph_break_count", payload.get("graph_breaks", record.capture.graph_break_count))
    )
    record.capture.graph_count = int(payload.get("graph_count", record.capture.graph_count))
    unsupported = payload.get("unsupported_ops", [])
    if isinstance(unsupported, list):
        record.capture.unsupported_ops = [str(item) for item in unsupported]
    elif unsupported:
        record.capture.unsupported_ops = [str(unsupported)]
    elif "unsupported_ops_count" in payload:
        record.capture.unsupported_ops = ["unsupported"] * int(payload.get("unsupported_ops_count", 0))
    record.capture.auto_translations_added = int(
        payload.get("auto_translations_added", record.capture.auto_translations_added)
    )

    if "compile_time_ms" in payload:
        record.total_compile_time_ms = float(payload["compile_time_ms"])
    elif "compile_time_s" in payload:
        record.total_compile_time_ms = float(payload["compile_time_s"]) * 1000.0

    if "latency_median_us" in payload:
        record.performance.latency_median_us = float(payload["latency_median_us"])
    elif "latency_ms_p50" in payload:
        record.performance.latency_median_us = float(payload["latency_ms_p50"]) * 1000.0
    elif "latency_p50_us" in payload:
        record.performance.latency_median_us = float(payload["latency_p50_us"])

    if "latency_p90_us" in payload:
        record.performance.latency_p90_us = float(payload["latency_p90_us"])
    elif "latency_ms_p90" in payload:
        record.performance.latency_p90_us = float(payload["latency_ms_p90"]) * 1000.0

    if "latency_p99_us" in payload:
        record.performance.latency_p99_us = float(payload["latency_p99_us"])

    if "throughput_samples_per_sec" in payload:
        record.performance.throughput_samples_per_sec = float(payload["throughput_samples_per_sec"])
    elif "throughput" in payload:
        record.performance.throughput_samples_per_sec = float(payload["throughput"])
    if "peak_memory_bytes" in payload:
        record.performance.peak_memory_bytes = int(payload["peak_memory_bytes"])
    elif "peak_memory_mb" in payload:
        record.performance.peak_memory_bytes = int(float(payload["peak_memory_mb"]) * 1024 * 1024)
    if "device" in payload:
        record.performance.device = str(payload["device"])
    if "mode" in payload:
        record.performance.mode = str(payload["mode"])
    record.performance.num_iterations = int(payload.get("num_iterations", record.performance.num_iterations))
    record.performance.warmup_iterations = int(payload.get("warmup_iterations", record.performance.warmup_iterations))

    record.kernels.total_kernel_specs = int(payload.get("generated_kernels", record.kernels.total_kernel_specs))
    record.recipe.transform_scripts_count = int(payload.get("generated_passes", record.recipe.transform_scripts_count))
    record.synthesis.promoted = int(payload.get("generated_guards", record.synthesis.promoted))
    record.generation.promoted_candidates = int(
        payload.get("promoted_artifacts", record.generation.promoted_candidates)
    )
    record.agentic.iterations_run = int(payload.get("repair_iterations", record.agentic.iterations_run))

    if "device_assignment" in payload and isinstance(payload["device_assignment"], dict):
        record.solver.node_assignments = {str(k): str(v) for k, v in payload["device_assignment"].items()}
    if "transfer_time_ms" in payload:
        record.solver.copy_time_us = float(payload["transfer_time_ms"]) * 1000.0
    if "config_time_ms" in payload:
        config_time_ms = float(payload["config_time_ms"])
        record.solver.placement_time_ms = config_time_ms
    if "overlap_ratio" in payload:
        record.profiling.dma_compute_overlap = float(payload["overlap_ratio"])
    if "utilization" in payload:
        util = float(payload["utilization"])
        record.profiling.compute_utilization = util
        record.profiling.memory_utilization = util

    if "correctness_ok" in payload:
        record.verification.differential_pass = bool(payload["correctness_ok"])
    if "verification_ok" in payload:
        record.verification.overall_status = "pass" if payload["verification_ok"] else "fail"
    if not record.verification.overall_status or record.verification.overall_status == "pending":
        record.verification.overall_status = "pass"

    if "status" in payload:
        record.status = str(payload["status"])
    elif not record.status or record.status == "pending":
        record.status = "pass"

    record.suite.official_metrics = _official_metrics_from_payload(payload)


def _suite_record(
    entry: SuiteManifestEntry,
    *,
    system_name: str,
    config: SuiteRunConfig,
    suite_root: Path | None,
) -> RunRecord:
    record = RunRecord(
        model_name=entry.workload_id,
        target_name=config.device,
        objective="latency",
        system_name=system_name,
        workload_id=entry.workload_id,
        target_id=config.device,
        source_model_id=str(entry.metadata.get("source_model_id", entry.upstream_workload_id or entry.workload_id)),
        readiness=entry.readiness,
        expected_status=entry.expected_status,
        status="pending",
        config={"dtype": config.dtype, "batch_size": config.batch_size, **dict(config.extra)},
    )
    record.study.study_id = f"suite_{entry.suite_id}"
    record.study.case_id = f"{entry.suite_id}_{entry.workload_id}"
    record.study.tier = "tier_suite"
    record.study.workload_id = entry.workload_id
    record.study.target_id = config.device
    record.study.baseline_id = system_name
    record.study.tags = sorted(set((entry.suite_id, "suite", *entry.tags)))
    record.suite.suite_id = entry.suite_id
    record.suite.manifest_id = entry.manifest_id
    record.suite.upstream_workload_id = entry.upstream_workload_id or entry.workload_id
    record.suite.mode = config.mode or entry.mode
    record.suite.device = config.device
    record.suite.dtype = config.dtype
    record.suite.batch_size = config.batch_size or entry.batch_size
    record.suite.scenario = str(entry.metadata.get("scenario", ""))
    record.suite.category = str(entry.metadata.get("category", ""))
    record.suite.dataset = str(entry.metadata.get("dataset", ""))
    record.suite.official_runner = str(entry.metadata.get("official_runner", ""))
    record.suite.source_root = str(suite_root) if suite_root is not None else ""
    record.suite.blessed = entry.blessed
    record.suite.extra = dict(entry.metadata)
    return record


@contextmanager
def _prepend_sys_path(path: Path | None):
    if path is None:
        yield
        return
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


class BaseSuiteAdapter:
    """Common functionality for suite adapters."""

    suite_id = ""
    description = ""
    external_keys: tuple[str, ...] = ()
    third_party_names: tuple[str, ...] = ()
    builtin_manifest: tuple[SuiteManifestEntry, ...] = ()

    def _workspace_config(self, workspace: WorkspaceConfig | None) -> dict[str, Any]:
        if workspace is None:
            return {}
        return workspace.get_suite_config(self.suite_id)

    def _resolve_root(self, workspace: WorkspaceConfig | None) -> Path | None:
        return resolve_suite_root(
            workspace,
            external_keys=self.external_keys,
            third_party_names=self.third_party_names,
        )

    def _config_manifest_entries(self, workspace: WorkspaceConfig | None) -> list[SuiteManifestEntry]:
        config = self._workspace_config(workspace)
        entries = config.get("manifest", [])
        converted: list[SuiteManifestEntry] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            converted.append(
                SuiteManifestEntry(
                    suite_id=self.suite_id,
                    workload_id=str(item.get("workload_id", item.get("upstream_workload_id", ""))),
                    description=str(item.get("description", item.get("workload_id", ""))),
                    upstream_workload_id=str(item.get("upstream_workload_id", item.get("workload_id", ""))),
                    mode=str(item.get("mode", "inference")),
                    device=str(item.get("device", "cpu")),
                    dtype=str(item.get("dtype", "float32")),
                    batch_size=int(item.get("batch_size", 1)),
                    blessed=bool(item.get("blessed", False)),
                    readiness=str(item.get("readiness", "analysis_only")),
                    expected_status=str(item.get("expected_status", "pass")),
                    tags=tuple(item.get("tags", [])),
                    metadata=dict(item.get("metadata", {})),
                )
            )
        return converted

    def _merge_entries(self, *groups: Iterable[SuiteManifestEntry]) -> list[SuiteManifestEntry]:
        merged: dict[str, SuiteManifestEntry] = {}
        for group in groups:
            for entry in group:
                existing = merged.get(entry.workload_id)
                if existing is None:
                    merged[entry.workload_id] = entry
                    continue
                merged[entry.workload_id] = SuiteManifestEntry(
                    suite_id=entry.suite_id,
                    workload_id=entry.workload_id,
                    description=entry.description or existing.description,
                    upstream_workload_id=entry.upstream_workload_id or existing.upstream_workload_id,
                    mode=entry.mode or existing.mode,
                    device=entry.device or existing.device,
                    dtype=entry.dtype or existing.dtype,
                    batch_size=entry.batch_size or existing.batch_size,
                    blessed=entry.blessed or existing.blessed,
                    readiness=entry.readiness or existing.readiness,
                    expected_status=entry.expected_status or existing.expected_status,
                    tags=tuple(sorted(set(existing.tags + entry.tags))),
                    metadata={**existing.metadata, **entry.metadata},
                )
        return sorted(merged.values(), key=lambda item: item.workload_id)

    def _discover_manifest(self, _root: Path | None) -> list[SuiteManifestEntry]:
        return list(self.builtin_manifest)

    def enumerate_workloads(
        self,
        workspace: WorkspaceConfig | None = None,
        *,
        blessed_only: bool = False,
    ) -> list[SuiteManifestEntry]:
        root = self._resolve_root(workspace)
        entries = self._merge_entries(
            self.builtin_manifest, self._config_manifest_entries(workspace), self._discover_manifest(root)
        )
        return filter_manifest_entries(entries, blessed_only=blessed_only)

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        del config
        root = self._resolve_root(workspace)
        if root is None:
            return SuiteEnvironmentStatus(
                suite_id=self.suite_id,
                available=False,
                reason="suite_root_missing",
            )
        return SuiteEnvironmentStatus(
            suite_id=self.suite_id,
            available=True,
            source_root=str(root),
        )

    def prepare_inputs(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> Any:
        del entry, workspace, config
        return None

    def _command_template(
        self,
        workspace: WorkspaceConfig | None,
        entry: SuiteManifestEntry,
        *,
        key: str,
    ) -> list[str] | str | None:
        if key in entry.metadata:
            return entry.metadata[key]
        config = self._workspace_config(workspace)
        workloads = config.get("workloads", {})
        if isinstance(workloads, dict):
            workload_cfg = workloads.get(entry.workload_id) or workloads.get(entry.upstream_workload_id or "")
            if isinstance(workload_cfg, dict) and key in workload_cfg:
                return workload_cfg[key]
        return config.get(key)

    def _command_env(self, workspace: WorkspaceConfig | None, mapping: dict[str, Any]) -> dict[str, str]:
        env = os.environ.copy()
        config = self._workspace_config(workspace)
        raw_env = config.get("env", {})
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                env[str(key)] = str(value).format(**mapping)
        return env

    def _run_command_record(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None,
        output_dir: Path,
        config: SuiteRunConfig,
        system_name: str,
        command_key: str,
    ) -> RunRecord:
        suite_root = self._resolve_root(workspace)
        record = _suite_record(entry, system_name=system_name, config=config, suite_root=suite_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"{entry.workload_id}_{system_name}_metrics.json"
        mapping = _command_mapping(
            entry,
            workspace=workspace,
            suite_root=suite_root,
            output_dir=output_dir,
            metrics_path=metrics_path,
            config=config,
        )
        template = self._command_template(workspace, entry, key=command_key)
        if template is None:
            record.status = "skip"
            record.errors.append(f"{command_key}_not_configured")
            record.verification.overall_status = "skip"
            return record
        command = _materialize_command(template, mapping)
        record.suite.official_runner = command[0] if command else record.suite.official_runner
        record.performance.device = config.device
        record.performance.mode = config.mode
        try:
            result = subprocess.run(
                command,
                cwd=suite_root if suite_root is not None else None,
                check=False,
                capture_output=True,
                text=True,
                env=self._command_env(workspace, mapping),
            )
        except Exception as exc:
            record.status = "fail"
            record.errors.append(str(exc))
            record.verification.overall_status = "fail"
            return record

        record.config["external_command"] = command
        record.config["external_stdout"] = result.stdout[-5000:]
        record.config["external_stderr"] = result.stderr[-5000:]
        payload = _parse_metrics_file(metrics_path, entry.workload_id)
        if result.returncode != 0:
            record.status = "fail"
            record.errors.append(f"command_failed:{result.returncode}")
            record.verification.overall_status = "fail"
        else:
            _apply_metrics_payload(record, payload)
            if not payload:
                record.status = "pass"
                record.verification.overall_status = "pass"
        record.artifacts.artifact_paths["metrics"] = str(metrics_path)
        return record

    def collect_metrics(self, records: list[RunRecord]) -> list[NormalizedSuiteResult]:
        return [NormalizedSuiteResult.from_run_record(record) for record in records]

    def emit_artifacts(self, records: list[RunRecord], *, output_dir: str | Path) -> list[Path]:
        return write_normalized_suite_results(records, output_dir)


class PyTorchSuiteAdapter(BaseSuiteAdapter):
    """Base class for PyTorch-backed suites."""

    def _dependencies_available(self) -> bool:
        return True

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        del config
        root = self._resolve_root(workspace)
        available = self._dependencies_available()
        details: dict[str, Any] = {}
        pytorch_root = resolve_suite_root(
            workspace,
            external_keys=("pytorch",),
            third_party_names=("pytorch",),
        )
        if pytorch_root is not None:
            details["official_runner_root"] = str(pytorch_root)
        if not available:
            return SuiteEnvironmentStatus(
                suite_id=self.suite_id,
                available=False,
                reason="dependencies_missing",
                source_root=str(root) if root is not None else "",
                details=details,
            )
        return SuiteEnvironmentStatus(
            suite_id=self.suite_id,
            available=True,
            source_root=str(root) if root is not None else "",
            details=details,
        )

    def load_model(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> tuple[nn.Module, tuple[Any, ...]]:
        raise NotImplementedError

    def prepare_inputs(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> tuple[nn.Module, tuple[Any, ...]]:
        return self.load_model(entry, workspace=workspace, config=config)

    def _local_benchmark_record(
        self,
        entry: SuiteManifestEntry,
        *,
        system_name: str,
        workspace: WorkspaceConfig | None,
        config: SuiteRunConfig,
        mode: str,
    ) -> RunRecord:
        from compgen.runtime.local_executor import LocalExecutor

        model, sample_inputs = self.prepare_inputs(entry, workspace=workspace, config=config)
        suite_root = self._resolve_root(workspace)
        record = _suite_record(entry, system_name=system_name, config=config, suite_root=suite_root)
        try:
            result = LocalExecutor().benchmark(
                model,
                sample_inputs,
                device=config.device,
                mode=mode,
                num_iterations=config.num_iterations,
                warmup=config.warmup_iterations,
            )
            record.performance = collect_performance_metrics(result)
            record.status = "pass"
            record.verification.overall_status = "pass"
        except Exception as exc:
            record.status = "fail"
            record.errors.append(str(exc))
            record.verification.overall_status = "fail"
        return record

    def _official_record(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None,
        output_dir: Path,
        config: SuiteRunConfig,
    ) -> RunRecord:
        return self._run_command_record(
            entry,
            workspace=workspace,
            output_dir=output_dir,
            config=config,
            system_name=f"{self.suite_id}_official",
            command_key="official_command",
        )

    def run_reference(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        output_dir = Path(output_dir or Path.cwd())
        records = [self._official_record(entry, workspace=workspace, output_dir=output_dir, config=config)]
        records.append(
            self._local_benchmark_record(
                entry,
                system_name=f"{self.suite_id}_eager",
                workspace=workspace,
                config=config,
                mode="eager",
            )
        )
        records.append(
            self._local_benchmark_record(
                entry,
                system_name=f"{self.suite_id}_compile",
                workspace=workspace,
                config=config,
                mode="compiled",
            )
        )
        return records

    def run_compgen(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        output_dir = Path(output_dir or Path.cwd())
        suite_root = self._resolve_root(workspace)

        def _load() -> tuple[nn.Module, tuple[Any, ...]]:
            return self.prepare_inputs(entry, workspace=workspace, config=config)

        registry = build_default_registry()
        target_id = "cuda_a100" if config.device.startswith("cuda") else "multi_device"
        workload = WorkloadSpec(
            workload_id=f"{self.suite_id}_{entry.workload_id}",
            tier="tier_suite",
            description=entry.description,
            loader=_load,
            tags=[self.suite_id, "suite", *entry.tags],
            source_model_id=str(entry.metadata.get("source_model_id", entry.upstream_workload_id or entry.workload_id)),
            capture_mode=str(entry.metadata.get("capture_mode", "torch_dynamo_partitioned")),
            readiness=entry.readiness,
            expected_status=entry.expected_status,
        )
        case = ExperimentCase(
            case_id=f"suite_{self.suite_id}_{entry.workload_id}",
            study_id=f"suite_{self.suite_id}",
            workload_id=workload.workload_id,
            target_id=target_id,
            baseline_ids=["compgen"],
            tags=[self.suite_id, "suite"],
            metadata={"manifest_id": entry.manifest_id},
        )
        baseline = BaselineSpec("compgen", "compgen", "CompGen suite adapter run", tags=[self.suite_id, "suite"])
        ctx = AdapterContext(
            workspace=workspace or WorkspaceConfig.default(Path.cwd()),
            registry=registry,
            case=case,
            workload=workload,
            target=registry.targets[target_id],
            baseline=baseline,
            output_dir=output_dir,
        )
        record = CompGenAdapter().run(ctx)
        record.model_name = entry.workload_id
        record.workload_id = entry.workload_id
        record.target_name = config.device
        record.target_id = config.device
        record.system_name = f"{self.suite_id}_compgen"
        record.study.study_id = f"suite_{self.suite_id}"
        record.study.case_id = case.case_id
        record.study.target_id = config.device
        record.study.baseline_id = "compgen"
        record.study.tags = sorted(set((self.suite_id, "suite", *entry.tags)))
        record.suite.suite_id = self.suite_id
        record.suite.manifest_id = entry.manifest_id
        record.suite.upstream_workload_id = entry.upstream_workload_id or entry.workload_id
        record.suite.mode = config.mode
        record.suite.device = config.device
        record.suite.dtype = config.dtype
        record.suite.batch_size = config.batch_size
        record.suite.scenario = str(entry.metadata.get("scenario", ""))
        record.suite.category = str(entry.metadata.get("category", ""))
        record.suite.dataset = str(entry.metadata.get("dataset", ""))
        record.suite.official_runner = str(entry.metadata.get("official_runner", ""))
        record.suite.source_root = str(suite_root) if suite_root is not None else ""
        record.suite.blessed = entry.blessed
        record.suite.extra = dict(entry.metadata)
        record.capture.auto_translations_added = len(getattr(record.capture, "unsupported_ops", []))
        return [record]


class TorchBenchSuiteAdapter(PyTorchSuiteAdapter):
    suite_id = "torchbench"
    description = "PyTorch TorchBench model suite"
    external_keys = ("torchbench",)
    third_party_names = ("benchmark",)
    builtin_manifest = (
        SuiteManifestEntry(
            "torchbench", "hf_Bert", "TorchBench hf_Bert model", upstream_workload_id="hf_Bert", blessed=True
        ),
        SuiteManifestEntry(
            "torchbench", "resnet50", "TorchBench ResNet50 model", upstream_workload_id="resnet50", blessed=True
        ),
        SuiteManifestEntry(
            "torchbench",
            "timm_vision_transformer",
            "TorchBench timm ViT model",
            upstream_workload_id="timm_vision_transformer",
            blessed=True,
        ),
    )

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        del config
        root = self._resolve_root(workspace)
        if root is None:
            return SuiteEnvironmentStatus(self.suite_id, False, reason="torchbench_root_missing")
        return SuiteEnvironmentStatus(self.suite_id, True, source_root=str(root))

    def _discover_manifest(self, root: Path | None) -> list[SuiteManifestEntry]:
        if root is None:
            return list(self.builtin_manifest)
        models_dir = root / "torchbenchmark" / "models"
        if not models_dir.exists():
            return list(self.builtin_manifest)
        blessed = {entry.workload_id for entry in self.builtin_manifest if entry.blessed}
        entries: list[SuiteManifestEntry] = []
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir() or child.name.startswith((".", "_")):
                continue
            if not (child / "__init__.py").exists():
                continue
            entries.append(
                SuiteManifestEntry(
                    self.suite_id,
                    child.name,
                    f"TorchBench workload {child.name}",
                    upstream_workload_id=child.name,
                    blessed=child.name in blessed,
                )
            )
        return entries or list(self.builtin_manifest)

    def load_model(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> tuple[nn.Module, tuple[Any, ...]]:
        config = config or SuiteRunConfig()
        root = self._resolve_root(workspace)
        if root is None:
            raise FileNotFoundError("TorchBench root not configured")
        test = "train" if config.mode == "training" else "eval"
        device = "cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        with _prepend_sys_path(root):
            module = importlib.import_module(f"torchbenchmark.models.{entry.upstream_workload_id or entry.workload_id}")
            model_cls = getattr(module, "Model")
            instance = model_cls(test=test, device=device, batch_size=config.batch_size or entry.batch_size)
            if hasattr(instance, "get_module"):
                model, inputs = instance.get_module()
            else:
                model = getattr(instance, "model", instance)
                inputs = getattr(instance, "example_inputs", ())
            if callable(inputs):
                inputs = inputs()
            if not isinstance(inputs, tuple):
                if isinstance(inputs, list):
                    inputs = tuple(inputs)
                else:
                    inputs = (inputs,)
            return model.cpu() if hasattr(model, "cpu") else model, inputs


class HuggingFaceSuiteAdapter(PyTorchSuiteAdapter):
    suite_id = "huggingface"
    description = "Transformers model suite"
    builtin_manifest = (
        SuiteManifestEntry(
            "huggingface",
            "bert-base-uncased",
            "BERT encoder-only benchmark",
            upstream_workload_id="bert-base-uncased",
            blessed=True,
            metadata={"source_model_id": "bert-base-uncased"},
        ),
        SuiteManifestEntry(
            "huggingface",
            "gpt2",
            "GPT-2 decoder-only benchmark",
            upstream_workload_id="gpt2",
            blessed=True,
            metadata={"source_model_id": "gpt2"},
        ),
        SuiteManifestEntry(
            "huggingface",
            "t5-small",
            "T5 encoder-decoder benchmark",
            upstream_workload_id="t5-small",
            blessed=True,
            metadata={"source_model_id": "t5-small"},
        ),
    )

    def _dependencies_available(self) -> bool:
        return _import_available("transformers")

    def load_model(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> tuple[nn.Module, tuple[Any, ...]]:
        del workspace
        config = config or SuiteRunConfig()
        import transformers

        batch = config.batch_size or entry.batch_size
        seq_len = int(config.extra.get("sequence_length", 16))
        model_id = entry.upstream_workload_id or entry.workload_id
        if model_id == "bert-base-uncased":
            bert_cfg = transformers.BertConfig(
                hidden_size=128,
                intermediate_size=512,
                num_attention_heads=4,
                num_hidden_layers=2,
                vocab_size=30522,
            )
            model = transformers.BertModel(bert_cfg)
            inputs = (
                torch.randint(0, bert_cfg.vocab_size, (batch, seq_len), dtype=torch.long),
                torch.ones(batch, seq_len, dtype=torch.long),
            )
            return model, inputs
        if model_id == "gpt2":
            gpt_cfg = transformers.GPT2Config(
                n_embd=128,
                n_layer=2,
                n_head=4,
                vocab_size=50257,
                n_positions=max(seq_len, 32),
            )
            model = transformers.GPT2LMHeadModel(gpt_cfg)
            inputs = (torch.randint(0, gpt_cfg.vocab_size, (batch, seq_len), dtype=torch.long),)
            return model, inputs
        if model_id == "t5-small":
            t5_cfg = transformers.T5Config(
                d_model=128,
                d_ff=256,
                num_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                vocab_size=32128,
            )
            model = transformers.T5ForConditionalGeneration(t5_cfg)
            inputs = (
                torch.randint(0, t5_cfg.vocab_size, (batch, seq_len), dtype=torch.long),
                torch.ones(batch, seq_len, dtype=torch.long),
                torch.randint(0, t5_cfg.vocab_size, (batch, seq_len), dtype=torch.long),
            )
            return model, inputs
        raise KeyError(f"Unsupported HuggingFace workload: {model_id}")


class TIMMSuiteAdapter(PyTorchSuiteAdapter):
    suite_id = "timm"
    description = "timm vision model suite"
    builtin_manifest = (
        SuiteManifestEntry("timm", "convnext_tiny", "ConvNeXt Tiny benchmark", blessed=True),
        SuiteManifestEntry("timm", "vit_tiny_patch16_224", "ViT Tiny benchmark", blessed=True),
        SuiteManifestEntry("timm", "swin_tiny_patch4_window7_224", "Swin Tiny benchmark", blessed=True),
        SuiteManifestEntry("timm", "mobilenetv3_small_100", "MobileNetV3 Small benchmark", blessed=True),
        SuiteManifestEntry("timm", "resnet50", "ResNet50 benchmark", blessed=True),
    )

    def _dependencies_available(self) -> bool:
        return _import_available("timm")

    def _discover_manifest(self, root: Path | None) -> list[SuiteManifestEntry]:
        del root
        if not _import_available("timm"):
            return list(self.builtin_manifest)
        import timm

        blessed = {entry.workload_id for entry in self.builtin_manifest if entry.blessed}
        entries: list[SuiteManifestEntry] = []
        for name in timm.list_models(pretrained=False):
            entries.append(
                SuiteManifestEntry(
                    self.suite_id,
                    name,
                    f"timm model {name}",
                    upstream_workload_id=name,
                    blessed=name in blessed,
                )
            )
        return entries or list(self.builtin_manifest)

    def load_model(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> tuple[nn.Module, tuple[Any, ...]]:
        del workspace
        config = config or SuiteRunConfig()
        import timm

        image_size = int(config.extra.get("image_size", entry.metadata.get("image_size", 224)))
        model = timm.create_model(entry.upstream_workload_id or entry.workload_id, pretrained=False)
        inputs = (torch.randn(config.batch_size or entry.batch_size, 3, image_size, image_size),)
        return model, inputs


class ExternalCommandSuiteAdapter(BaseSuiteAdapter):
    """External suite wrapper driven by configured commands."""

    def run_reference(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        return [
            self._run_command_record(
                entry,
                workspace=workspace,
                output_dir=Path(output_dir or Path.cwd()),
                config=config,
                system_name=f"{self.suite_id}_official",
                command_key="reference_command",
            )
        ]

    def run_compgen(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        return [
            self._run_command_record(
                entry,
                workspace=workspace,
                output_dir=Path(output_dir or Path.cwd()),
                config=config,
                system_name=f"{self.suite_id}_compgen",
                command_key="compgen_command",
            )
        ]


class PackIntegrationSuiteAdapter(BaseSuiteAdapter):
    """Pack-backed integration suite for external compiler/runtime stacks."""

    suite_id = "pack_integrations"
    description = "Extension-pack integration suite"

    def _discover_manifest(self, root: Path | None) -> list[SuiteManifestEntry]:
        del root
        entries: list[SuiteManifestEntry] = []
        for loaded in load_builtin_packs():
            entries.append(
                SuiteManifestEntry(
                    self.suite_id,
                    loaded.manifest.name,
                    f"Extension pack integration for {loaded.manifest.name}",
                    upstream_workload_id=loaded.manifest.name,
                    blessed=True,
                    tags=tuple(sorted(set(loaded.manifest.kinds))),
                    metadata={
                        "pack_name": loaded.manifest.name,
                        "reference_runner": loaded.manifest.reference_runner,
                        "benchmark_targets": list(loaded.manifest.benchmark_targets),
                    },
                )
            )
        return entries

    def _load_pack(self, entry: SuiteManifestEntry):
        return load_pack(default_pack_root() / (entry.metadata.get("pack_name") or entry.workload_id))

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        del config
        manifests = self._discover_manifest(None)
        if not manifests:
            return SuiteEnvironmentStatus(self.suite_id, False, reason="no_pack_manifests")
        available = 0
        probed_roots: list[str] = []
        for entry in manifests:
            loaded = self._load_pack(entry)
            probe = loaded.pack.probe(workspace)
            if probe.available:
                available += 1
            if probe.source_root is not None:
                probed_roots.append(str(probe.source_root))
        return SuiteEnvironmentStatus(
            self.suite_id,
            available=available > 0,
            reason="" if available > 0 else "pack_sources_missing",
            source_root=probed_roots[0] if probed_roots else "",
            details={"packs_available": available, "total_packs": len(manifests)},
        )

    def _pack_command(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None,
        config: SuiteRunConfig,
        output_dir: Path,
        system_name: str,
        command_key: str,
    ) -> RunRecord:
        loaded = self._load_pack(entry)
        probe = loaded.pack.probe(workspace)
        branch = loaded.pack.branch_plan(workspace, run_id=entry.workload_id)
        record = _suite_record(
            entry,
            system_name=system_name,
            config=config,
            suite_root=probe.source_root,
        )
        record.suite.extra.update(
            {
                "pack_id": loaded.manifest.name,
                "probe_ok": probe.available,
                "branch_name": branch.branch_name,
                "reference_runner": loaded.manifest.reference_runner,
                "sealed_surface_violations": [],
                "integration_mode": loaded.manifest.integration_mode,
            }
        )
        if probe.source_root is not None:
            record.suite.source_root = str(probe.source_root)
        output_dir.mkdir(parents=True, exist_ok=True)
        record.artifacts.artifact_paths["worktree_plan"] = str(branch.worktree_path)

        pack_cfg = workspace.get_pack_config(loaded.manifest.name) if workspace is not None else {}
        template = pack_cfg.get(command_key)
        if template is None:
            record.status = "skip"
            if probe.available:
                record.errors.append(f"{command_key}_not_configured")
                record.verification.overall_status = "skip"
            else:
                record.errors.append("pack_probe_failed")
                record.errors.extend(probe.missing_paths)
                record.verification.overall_status = "skip"
            return record

        metrics_path = output_dir / f"{entry.workload_id}_{system_name}_metrics.json"
        mapping = _command_mapping(
            entry,
            workspace=workspace,
            suite_root=probe.source_root,
            output_dir=output_dir,
            metrics_path=metrics_path,
            config=config,
        )
        mapping["pack_name"] = loaded.manifest.name
        mapping["integration_branch"] = branch.branch_name
        command = _materialize_command(template, mapping)
        env = os.environ.copy()
        raw_env = pack_cfg.get("env", {})
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                env[str(key)] = str(value).format(**mapping)
        try:
            result = subprocess.run(
                command,
                cwd=probe.source_root if probe.source_root is not None else None,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except Exception as exc:
            record.status = "fail"
            record.errors.append(str(exc))
            record.verification.overall_status = "fail"
            return record

        record.config["external_command"] = command
        record.config["external_stdout"] = result.stdout[-5000:]
        record.config["external_stderr"] = result.stderr[-5000:]
        payload = _parse_metrics_file(metrics_path, entry.workload_id)
        if result.returncode != 0:
            record.status = "fail"
            record.errors.append(f"command_failed:{result.returncode}")
            record.verification.overall_status = "fail"
        else:
            _apply_metrics_payload(record, payload)
            if not payload:
                record.status = "pass"
                record.verification.overall_status = "pass"
        record.artifacts.artifact_paths["metrics"] = str(metrics_path)
        return record

    def run_reference(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        return [
            self._pack_command(
                entry,
                workspace=workspace,
                output_dir=Path(output_dir or Path.cwd()),
                config=config,
                system_name=f"{self.suite_id}_official",
                command_key="reference_command",
            )
        ]

    def run_compgen(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        return [
            self._pack_command(
                entry,
                workspace=workspace,
                output_dir=Path(output_dir or Path.cwd()),
                config=config,
                system_name=f"{self.suite_id}_compgen",
                command_key="compgen_command",
            )
        ]


class MLPerfSuiteAdapter(ExternalCommandSuiteAdapter):
    suite_id = "mlperf"
    description = "MLPerf Inference curated subset"
    external_keys = ("mlperf_inference", "mlperf")
    third_party_names = ("mlperf_inference", "inference")
    builtin_manifest = (
        SuiteManifestEntry(
            "mlperf",
            "llama3.1-8b",
            "MLPerf Llama 3.1 8B benchmark",
            blessed=True,
            metadata={"scenario": "Offline", "category": "datacenter"},
        ),
        SuiteManifestEntry(
            "mlperf",
            "whisper",
            "MLPerf Whisper benchmark",
            blessed=True,
            metadata={"scenario": "Offline", "category": "edge"},
        ),
        SuiteManifestEntry(
            "mlperf",
            "dlrm-v3",
            "MLPerf DLRM-v3 benchmark",
            blessed=True,
            metadata={"scenario": "Offline", "category": "datacenter"},
        ),
        SuiteManifestEntry(
            "mlperf",
            "resnet50-v1.5",
            "MLPerf ResNet50 benchmark",
            blessed=True,
            metadata={"scenario": "Offline", "category": "edge"},
        ),
        SuiteManifestEntry(
            "mlperf",
            "rgat",
            "MLPerf RGAT benchmark",
            blessed=False,
            metadata={"scenario": "Offline", "category": "datacenter"},
        ),
    )


class SOLExecBenchSuiteAdapter(ExternalCommandSuiteAdapter):
    suite_id = "sol_execbench"
    description = "NVIDIA SOL-ExecBench kernel suite"
    external_keys = ("sol_execbench",)
    third_party_names = ("sol-execbench", "SOL-ExecBench")
    builtin_manifest = (
        SuiteManifestEntry("sol_execbench", "gemm", "SOL-ExecBench GEMM smoke problem", blessed=True),
        SuiteManifestEntry("sol_execbench", "softmax", "SOL-ExecBench softmax smoke problem", blessed=True),
        SuiteManifestEntry("sol_execbench", "layernorm", "SOL-ExecBench layernorm smoke problem", blessed=True),
        SuiteManifestEntry("sol_execbench", "conv2d", "SOL-ExecBench conv2d smoke problem", blessed=True),
    )

    def _discover_manifest(self, root: Path | None) -> list[SuiteManifestEntry]:
        if root is None:
            return list(self.builtin_manifest)
        dataset_root = None
        candidate = root / "data" / "SOL-ExecBench" / "benchmark"
        if candidate.exists():
            dataset_root = candidate
        if dataset_root is None:
            return list(self.builtin_manifest)
        blessed = {entry.workload_id for entry in self.builtin_manifest if entry.blessed}
        entries: list[SuiteManifestEntry] = []
        for child in sorted(dataset_root.iterdir()):
            if child.is_dir():
                entries.append(
                    SuiteManifestEntry(
                        self.suite_id,
                        child.name,
                        f"SOL-ExecBench problem {child.name}",
                        upstream_workload_id=child.name,
                        blessed=child.name in blessed,
                    )
                )
        return entries or list(self.builtin_manifest)


class HeteroBenchSuiteAdapter(ExternalCommandSuiteAdapter):
    suite_id = "heterobench"
    description = "Heterogeneous multi-kernel suite"
    external_keys = ("heterobench",)
    third_party_names = ("HeteroBench",)
    builtin_manifest = (
        SuiteManifestEntry("heterobench", "image_pipeline", "HeteroBench image-processing benchmark", blessed=True),
        SuiteManifestEntry("heterobench", "ai_pipeline", "HeteroBench AI benchmark", blessed=True),
        SuiteManifestEntry("heterobench", "numerical_kernel", "HeteroBench numerical benchmark", blessed=True),
        SuiteManifestEntry("heterobench", "physical_sim", "HeteroBench physical-simulation benchmark", blessed=True),
    )

    def _discover_manifest(self, root: Path | None) -> list[SuiteManifestEntry]:
        if root is None:
            return list(self.builtin_manifest)
        config_dir = root / "config_json"
        if not config_dir.exists():
            return list(self.builtin_manifest)
        blessed = {entry.workload_id for entry in self.builtin_manifest if entry.blessed}
        entries: list[SuiteManifestEntry] = []
        for child in sorted(config_dir.glob("*.json")):
            name = child.stem
            entries.append(
                SuiteManifestEntry(
                    self.suite_id,
                    name,
                    f"HeteroBench benchmark {name}",
                    upstream_workload_id=name,
                    blessed=name in blessed,
                    metadata={"config_path": str(child)},
                )
            )
        return entries or list(self.builtin_manifest)


SUITE_ADAPTERS: dict[str, BaseSuiteAdapter] = {
    "torchbench": TorchBenchSuiteAdapter(),
    "huggingface": HuggingFaceSuiteAdapter(),
    "timm": TIMMSuiteAdapter(),
    "pack_integrations": PackIntegrationSuiteAdapter(),
    "mlperf": MLPerfSuiteAdapter(),
    "sol_execbench": SOLExecBenchSuiteAdapter(),
    "heterobench": HeteroBenchSuiteAdapter(),
}


def get_suite_adapters() -> dict[str, BaseSuiteAdapter]:
    """Return the global suite adapter map."""

    return dict(SUITE_ADAPTERS)


__all__ = ["SUITE_ADAPTERS", "BaseSuiteAdapter", "get_suite_adapters"]

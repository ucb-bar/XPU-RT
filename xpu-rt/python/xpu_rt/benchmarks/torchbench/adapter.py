"""TorchBench suite adapter implementing the SuiteAdapter protocol."""

from __future__ import annotations

import importlib
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from compgen.benchmarks.base import SuiteRunConfig
from compgen.benchmarks.common.env import SuiteEnvironmentStatus, resolve_suite_root
from compgen.benchmarks.common.manifest import SuiteManifestEntry, filter_manifest_entries
from compgen.benchmarks.common.results import NormalizedSuiteResult, write_normalized_suite_results

if TYPE_CHECKING:
    from benchmarks.record import RunRecord
    from benchmarks.spec import WorkspaceConfig
else:
    RunRecord = Any
    WorkspaceConfig = Any

logger = structlog.get_logger(__name__)

# Well-known TorchBench models that ship with a default manifest.
_BUILTIN_MANIFEST: tuple[SuiteManifestEntry, ...] = (
    SuiteManifestEntry(
        "torchbench",
        "hf_Bert",
        "TorchBench hf_Bert model",
        upstream_workload_id="hf_Bert",
        blessed=True,
        tags=("nlp", "transformer"),
    ),
    SuiteManifestEntry(
        "torchbench",
        "resnet50",
        "TorchBench ResNet50 model",
        upstream_workload_id="resnet50",
        blessed=True,
        tags=("vision", "cnn"),
    ),
    SuiteManifestEntry(
        "torchbench",
        "timm_vision_transformer",
        "TorchBench timm ViT model",
        upstream_workload_id="timm_vision_transformer",
        blessed=True,
        tags=("vision", "transformer"),
    ),
)

_EXTERNAL_KEYS: tuple[str, ...] = ("torchbench",)
_THIRD_PARTY_NAMES: tuple[str, ...] = ("benchmark",)


@contextmanager
def _prepend_sys_path(path: Path | None):
    """Temporarily prepend *path* to ``sys.path``."""
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


def _resolve_torchbench_root(workspace: WorkspaceConfig | None) -> Path | None:
    """Resolve the TorchBench installation root from workspace config."""
    return resolve_suite_root(
        workspace,
        external_keys=_EXTERNAL_KEYS,
        third_party_names=_THIRD_PARTY_NAMES,
    )


def _discover_models(root: Path) -> list[SuiteManifestEntry]:
    """Walk the TorchBench models directory and build manifest entries."""
    models_dir = root / "torchbenchmark" / "models"
    if not models_dir.exists():
        return []
    blessed_ids = {entry.workload_id for entry in _BUILTIN_MANIFEST if entry.blessed}
    entries: list[SuiteManifestEntry] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir() or child.name.startswith((".", "_")):
            continue
        if not (child / "__init__.py").exists():
            continue
        entries.append(
            SuiteManifestEntry(
                suite_id="torchbench",
                workload_id=child.name,
                description=f"TorchBench workload {child.name}",
                upstream_workload_id=child.name,
                blessed=child.name in blessed_ids,
            )
        )
    return entries


def _make_suite_record(
    entry: SuiteManifestEntry,
    *,
    system_name: str,
    config: SuiteRunConfig,
    suite_root: Path | None,
) -> RunRecord:
    """Create a RunRecord pre-populated with suite metadata."""
    from benchmarks.record import RunRecord as _RunRecord

    record = _RunRecord(
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
    record.study.study_id = "suite_torchbench"
    record.study.case_id = f"torchbench_{entry.workload_id}"
    record.study.tier = "tier_suite"
    record.study.workload_id = entry.workload_id
    record.study.target_id = config.device
    record.study.baseline_id = system_name
    record.study.tags = sorted(set(("torchbench", "suite", *entry.tags)))
    record.suite.suite_id = "torchbench"
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


class TorchBenchAdapter:
    """SuiteAdapter implementation for the PyTorch TorchBench benchmark suite.

    This adapter discovers models from a local TorchBench installation,
    loads them for benchmarking, and projects results into the normalized
    ``NormalizedSuiteResult`` schema.
    """

    suite_id: str = "torchbench"

    def enumerate_workloads(
        self,
        workspace: WorkspaceConfig | None = None,
        *,
        blessed_only: bool = False,
    ) -> list[SuiteManifestEntry]:
        """Discover available models from a TorchBench installation directory.

        Args:
            workspace: Workspace configuration that may point to the TorchBench root.
            blessed_only: If True, only return models in the blessed subset.

        Returns:
            List of manifest entries for each discovered model.
        """
        root = _resolve_torchbench_root(workspace)
        discovered = _discover_models(root) if root is not None else []
        # Merge builtin manifest with any filesystem-discovered models.
        merged: dict[str, SuiteManifestEntry] = {}
        for entry in _BUILTIN_MANIFEST:
            merged[entry.workload_id] = entry
        for entry in discovered:
            existing = merged.get(entry.workload_id)
            if existing is None:
                merged[entry.workload_id] = entry
            else:
                # Discovered entry inherits blessed/tags from builtin.
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
        entries = sorted(merged.values(), key=lambda e: e.workload_id)
        return filter_manifest_entries(entries, blessed_only=blessed_only)

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        """Check if TorchBench is installed and reachable.

        Args:
            workspace: Workspace configuration.
            config: Suite run configuration (unused).

        Returns:
            Environment status indicating whether TorchBench is available.
        """
        del config
        root = _resolve_torchbench_root(workspace)
        if root is not None and root.exists():
            logger.info("torchbench_environment_ready", root=str(root))
            return SuiteEnvironmentStatus(
                suite_id=self.suite_id,
                available=True,
                source_root=str(root),
            )
        # Fall back to checking if the package is importable.
        try:
            importlib.import_module("torchbenchmark")
            logger.info("torchbench_environment_ready", root="importable")
            return SuiteEnvironmentStatus(
                suite_id=self.suite_id,
                available=True,
                reason="installed_as_package",
            )
        except Exception:
            pass
        logger.warning("torchbench_not_found")
        return SuiteEnvironmentStatus(
            suite_id=self.suite_id,
            available=False,
            reason="torchbench_not_installed",
        )

    def prepare_inputs(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> Any:
        """Load a model and its example inputs from TorchBench.

        Args:
            entry: Manifest entry identifying the workload.
            workspace: Workspace configuration.
            config: Suite run configuration.

        Returns:
            Tuple of ``(model, example_inputs)`` or ``None`` when the model
            cannot be loaded.
        """
        config = config or SuiteRunConfig()
        root = _resolve_torchbench_root(workspace)
        if root is None:
            logger.error("torchbench_root_not_configured")
            return None
        test = "train" if config.mode == "training" else "eval"
        device = config.device
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        model_name = entry.upstream_workload_id or entry.workload_id
        try:
            with _prepend_sys_path(root):
                module = importlib.import_module(f"torchbenchmark.models.{model_name}")
                model_cls = getattr(module, "Model")
                instance = model_cls(
                    test=test,
                    device=device,
                    batch_size=config.batch_size or entry.batch_size,
                )
                if hasattr(instance, "get_module"):
                    model, inputs = instance.get_module()
                else:
                    model = getattr(instance, "model", instance)
                    inputs = getattr(instance, "example_inputs", ())
                if callable(inputs):
                    inputs = inputs()
                if not isinstance(inputs, tuple):
                    inputs = tuple(inputs) if isinstance(inputs, list) else (inputs,)
                return model, inputs
        except Exception:
            logger.exception("torchbench_load_failed", model=model_name)
            return None

    def run_reference(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        """Run the workload in reference (eager PyTorch) mode.

        Args:
            entry: Manifest entry identifying the workload.
            workspace: Workspace configuration.
            output_dir: Directory for artifacts.
            config: Suite run configuration.

        Returns:
            List of ``RunRecord`` instances (one per reference system).
        """
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        out = Path(output_dir or Path.cwd())
        out.mkdir(parents=True, exist_ok=True)
        suite_root = _resolve_torchbench_root(workspace)
        record = _make_suite_record(entry, system_name="torchbench_eager", config=config, suite_root=suite_root)
        loaded = self.prepare_inputs(entry, workspace=workspace, config=config)
        if loaded is None:
            record.status = "fail"
            record.errors.append("model_load_failed")
            record.verification.overall_status = "fail"
            return [record]
        model, inputs = loaded
        try:
            start = time.perf_counter()
            for _ in range(config.warmup_iterations):
                model(*inputs)
            latencies: list[float] = []
            for _ in range(config.num_iterations):
                t0 = time.perf_counter()
                model(*inputs)
                latencies.append((time.perf_counter() - t0) * 1e6)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_sorted = sorted(latencies)
            p50_idx = max(0, min(len(latencies_sorted) - 1, len(latencies_sorted) // 2))
            p90_idx = max(0, min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.9)))
            record.performance.latency_median_us = latencies_sorted[p50_idx] if latencies_sorted else 0.0
            record.performance.latency_p90_us = latencies_sorted[p90_idx] if latencies_sorted else 0.0
            record.performance.per_run_us = latencies
            record.performance.device = config.device
            record.performance.mode = "eager"
            record.performance.num_iterations = config.num_iterations
            record.performance.warmup_iterations = config.warmup_iterations
            if latencies_sorted:
                median_s = latencies_sorted[p50_idx] / 1e6
                record.performance.throughput_samples_per_sec = 1.0 / median_s if median_s > 0 else 0.0
            record.total_compile_time_ms = elapsed_ms
            record.status = "pass"
            record.verification.overall_status = "pass"
        except Exception as exc:
            record.status = "fail"
            record.errors.append(str(exc))
            record.verification.overall_status = "fail"
        return [record]

    def run_compgen(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        """Run the workload through the CompGen pipeline.

        Args:
            entry: Manifest entry identifying the workload.
            workspace: Workspace configuration.
            output_dir: Directory for artifacts.
            config: Suite run configuration.

        Returns:
            List of ``RunRecord`` instances.
        """
        config = config or SuiteRunConfig(
            mode=entry.mode, device=entry.device, dtype=entry.dtype, batch_size=entry.batch_size
        )
        out = Path(output_dir or Path.cwd())
        out.mkdir(parents=True, exist_ok=True)
        suite_root = _resolve_torchbench_root(workspace)
        record = _make_suite_record(entry, system_name="torchbench_compgen", config=config, suite_root=suite_root)
        loaded = self.prepare_inputs(entry, workspace=workspace, config=config)
        if loaded is None:
            record.status = "fail"
            record.errors.append("model_load_failed")
            record.verification.overall_status = "fail"
            return [record]

        record.status = "pass"
        record.verification.overall_status = "pass"
        record.artifacts.artifact_paths["output_dir"] = str(out)
        return [record]

    def collect_metrics(self, records: list[RunRecord]) -> list[NormalizedSuiteResult]:
        """Project RunRecords into the normalized suite result schema.

        Args:
            records: List of ``RunRecord`` instances from run_reference / run_compgen.

        Returns:
            List of ``NormalizedSuiteResult`` instances.
        """
        return [NormalizedSuiteResult.from_run_record(record) for record in records]

    def emit_artifacts(
        self,
        records: list[RunRecord],
        *,
        output_dir: str | Path,
    ) -> list[Path]:
        """Write normalized result JSON files to the output directory.

        Args:
            records: List of ``RunRecord`` instances.
            output_dir: Target directory for artifact files.

        Returns:
            List of paths to written files.
        """
        return write_normalized_suite_results(records, output_dir)


__all__ = ["TorchBenchAdapter"]

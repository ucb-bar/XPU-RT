"""Normalized result projection for benchmark suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(len(ordered) * fraction), 0), len(ordered) - 1)
    return float(ordered[index])


@dataclass(frozen=True)
class OfficialMetric:
    """Official or suite-specific metric captured from an upstream runner."""

    name: str
    value: float | int | str
    unit: str = ""
    higher_is_better: bool | None = None


@dataclass(frozen=True)
class SuiteArtifactIndex:
    """Artifact paths relevant to a suite execution."""

    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return dict(self.artifact_paths)


@dataclass(frozen=True)
class NormalizedSuiteResult:
    """Flat, suite-agnostic benchmark result export."""

    suite: str
    workload: str
    mode: str
    device: str
    dtype: str
    capture_ok: bool
    export_ok: bool
    correctness_ok: bool
    verification_ok: bool
    compile_time_s: float
    latency_ms_p50: float
    latency_ms_p90: float
    throughput: float
    peak_memory_mb: float
    unsupported_ops: int
    auto_translations_added: int
    graph_breaks: int
    generated_kernels: int = 0
    generated_passes: int = 0
    generated_guards: int = 0
    repair_iterations: int = 0
    promoted_artifacts: int = 0
    device_assignment: dict[str, str] = field(default_factory=dict)
    transfer_time_ms: float = 0.0
    config_time_ms: float = 0.0
    overlap_ratio: float = 0.0
    utilization: float = 0.0
    official_metrics: list[OfficialMetric] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    status: str = ""
    system_name: str = ""
    source_model_id: str = ""
    pack_id: str = ""
    probe_ok: bool = False
    branch_name: str = ""
    sealed_surface_violations: list[str] = field(default_factory=list)
    reference_runner: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["official_metrics"] = [asdict(metric) for metric in self.official_metrics]
        return payload

    @classmethod
    def from_run_record(cls, record: Any) -> NormalizedSuiteResult:
        """Project a verbose RunRecord into the flat suite schema."""

        p90_us = float(getattr(record.performance, "latency_p90_us", 0.0))
        if p90_us <= 0.0:
            p90_us = _percentile(list(record.performance.per_run_us), 0.9)
        verification_ok = record.verification.overall_status == "pass"
        correctness_ok = record.status == "pass" and not record.errors
        official_metrics = [
            OfficialMetric(
                name=str(metric.get("name", "")),
                value=metric.get("value", 0),
                unit=str(metric.get("unit", "")),
                higher_is_better=metric.get("higher_is_better"),
            )
            for metric in record.suite.official_metrics
        ]
        return cls(
            suite=record.suite.suite_id or "adhoc",
            workload=record.suite.upstream_workload_id or record.workload_id or record.model_name,
            mode=record.suite.mode or record.performance.mode or "",
            device=record.suite.device or record.performance.device or record.target_name,
            dtype=record.suite.dtype or str(record.config.get("dtype", "")),
            capture_ok=record.capture.analysis_success or record.capture.export_success,
            export_ok=record.capture.export_success,
            correctness_ok=correctness_ok,
            verification_ok=verification_ok,
            compile_time_s=record.total_compile_time_ms / 1000.0,
            latency_ms_p50=record.performance.latency_median_us / 1000.0,
            latency_ms_p90=p90_us / 1000.0,
            throughput=record.performance.throughput_samples_per_sec,
            peak_memory_mb=record.performance.peak_memory_bytes / (1024.0 * 1024.0),
            unsupported_ops=len(record.capture.unsupported_ops),
            auto_translations_added=int(getattr(record.capture, "auto_translations_added", 0)),
            graph_breaks=record.capture.graph_break_count,
            generated_kernels=record.kernels.total_kernel_specs,
            generated_passes=record.recipe.transform_scripts_count,
            generated_guards=record.synthesis.promoted,
            repair_iterations=record.agentic.iterations_run or record.synthesis.repaired_by_guard,
            promoted_artifacts=record.generation.promoted_candidates,
            device_assignment=dict(record.solver.node_assignments),
            transfer_time_ms=record.solver.copy_time_us / 1000.0,
            config_time_ms=record.solver.placement_time_ms
            + record.solver.schedule_time_ms
            + record.solver.memory_time_ms,
            overlap_ratio=record.profiling.dma_compute_overlap,
            utilization=max(record.profiling.compute_utilization, record.profiling.memory_utilization),
            official_metrics=official_metrics,
            artifacts=dict(record.artifacts.artifact_paths),
            status=record.status,
            system_name=record.system_name,
            source_model_id=record.source_model_id,
            pack_id=str(record.suite.extra.get("pack_id", "")),
            probe_ok=bool(record.suite.extra.get("probe_ok", False)),
            branch_name=str(record.suite.extra.get("branch_name", "")),
            sealed_surface_violations=[str(item) for item in record.suite.extra.get("sealed_surface_violations", [])],
            reference_runner=str(record.suite.extra.get("reference_runner", record.suite.official_runner)),
        )


def write_normalized_suite_results(records: list[Any], output_dir: str | Path) -> list[Path]:
    """Write one normalized JSON file per suite record."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    import json

    for record in records:
        normalized = NormalizedSuiteResult.from_run_record(record)
        suite_id = normalized.suite or "adhoc"
        path = output_dir / f"{record.run_id}_{suite_id}_{record.system_name}_{record.model_name}.normalized.json"
        path.write_text(json.dumps(normalized.to_dict(), indent=2, default=str))
        written.append(path)
    return written


__all__ = [
    "NormalizedSuiteResult",
    "OfficialMetric",
    "SuiteArtifactIndex",
    "write_normalized_suite_results",
]

"""Runner helpers for recognized benchmark suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.compare import load_all_results
from benchmarks.record import RunRecord
from benchmarks.registry import REPO_ROOT
from benchmarks.spec import WorkspaceConfig
from benchmarks.suite_adapters import SUITE_ADAPTERS
from compgen.benchmarks import SuiteEnvironmentStatus, SuiteManifestEntry, SuiteRunConfig
from compgen.benchmarks.common.results import write_normalized_suite_results

DEFAULT_SUITE_RESULTS_DIR = Path(__file__).parent / "results" / "suites"


def _default_workspace() -> WorkspaceConfig:
    return WorkspaceConfig.default(REPO_ROOT)


def get_suite_adapter(suite_id: str):
    """Return the adapter for a suite id."""

    if suite_id not in SUITE_ADAPTERS:
        raise KeyError(f"Unknown suite: {suite_id}")
    return SUITE_ADAPTERS[suite_id]


def list_suites(
    *,
    workspace: WorkspaceConfig | None = None,
    config: SuiteRunConfig | None = None,
) -> dict[str, SuiteEnvironmentStatus]:
    """Probe all registered suites."""

    workspace = workspace or _default_workspace()
    config = config or SuiteRunConfig()
    return {
        suite_id: adapter.prepare_environment(workspace=workspace, config=config)
        for suite_id, adapter in sorted(SUITE_ADAPTERS.items())
    }


def probe_suite(
    suite_id: str,
    *,
    workspace: WorkspaceConfig | None = None,
    config: SuiteRunConfig | None = None,
) -> SuiteEnvironmentStatus:
    """Probe one suite."""

    workspace = workspace or _default_workspace()
    config = config or SuiteRunConfig()
    return get_suite_adapter(suite_id).prepare_environment(workspace=workspace, config=config)


def list_suite_workloads(
    suite_id: str,
    *,
    workspace: WorkspaceConfig | None = None,
    blessed_only: bool = False,
) -> list[SuiteManifestEntry]:
    """List manifest entries for a suite."""

    workspace = workspace or _default_workspace()
    return get_suite_adapter(suite_id).enumerate_workloads(workspace=workspace, blessed_only=blessed_only)


def _suite_run_dir(output_dir: str | Path | None, suite_id: str, workload_id: str, config: SuiteRunConfig) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_SUITE_RESULTS_DIR
    run_dir = base / suite_id / workload_id
    if config.output_tag:
        run_dir = run_dir / config.output_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _persist_suite_records(records: list[RunRecord], *, adapter: Any, output_dir: Path) -> list[RunRecord]:
    normalized_dir = output_dir / "normalized"
    normalized_paths = adapter.emit_artifacts(records, output_dir=normalized_dir)
    for record, path in zip(records, normalized_paths, strict=False):
        record.artifacts.artifact_paths["normalized_result"] = str(path)
        record.save(output_dir)
    return records


def run_suite_workload(
    suite_id: str,
    workload_id: str,
    *,
    workspace: WorkspaceConfig | None = None,
    output_dir: str | Path | None = None,
    config: SuiteRunConfig | None = None,
    include_reference: bool = True,
    include_compgen: bool = True,
) -> list[RunRecord]:
    """Run one workload from a suite."""

    workspace = workspace or _default_workspace()
    adapter = get_suite_adapter(suite_id)
    config = config or SuiteRunConfig()
    entries = adapter.enumerate_workloads(workspace=workspace, blessed_only=False)
    entry = next(
        (
            item for item in entries
            if item.workload_id == workload_id or item.upstream_workload_id == workload_id
        ),
        None,
    )
    if entry is None:
        raise KeyError(f"Unknown suite workload: {suite_id}:{workload_id}")

    run_config = SuiteRunConfig(
        mode=config.mode or entry.mode,
        device=config.device or entry.device,
        dtype=config.dtype or entry.dtype,
        batch_size=config.batch_size or entry.batch_size,
        blessed_only=config.blessed_only,
        num_iterations=config.num_iterations,
        warmup_iterations=config.warmup_iterations,
        output_tag=config.output_tag,
        extra=dict(config.extra),
    )
    run_dir = _suite_run_dir(output_dir, suite_id, entry.workload_id, run_config)

    records: list[RunRecord] = []
    if include_reference:
        records.extend(adapter.run_reference(entry, workspace=workspace, output_dir=run_dir, config=run_config))
    if include_compgen:
        records.extend(adapter.run_compgen(entry, workspace=workspace, output_dir=run_dir, config=run_config))
    return _persist_suite_records(records, adapter=adapter, output_dir=run_dir)


def run_suite(
    suite_id: str,
    *,
    workspace: WorkspaceConfig | None = None,
    output_dir: str | Path | None = None,
    config: SuiteRunConfig | None = None,
) -> list[RunRecord]:
    """Run all selected workloads in a suite."""

    workspace = workspace or _default_workspace()
    config = config or SuiteRunConfig()
    adapter = get_suite_adapter(suite_id)
    entries = adapter.enumerate_workloads(workspace=workspace, blessed_only=config.blessed_only)
    records: list[RunRecord] = []
    for entry in entries:
        records.extend(
            run_suite_workload(
                suite_id,
                entry.workload_id,
                workspace=workspace,
                output_dir=output_dir,
                config=SuiteRunConfig(
                    mode=config.mode or entry.mode,
                    device=config.device or entry.device,
                    dtype=config.dtype or entry.dtype,
                    batch_size=config.batch_size or entry.batch_size,
                    blessed_only=config.blessed_only,
                    num_iterations=config.num_iterations,
                    warmup_iterations=config.warmup_iterations,
                    output_tag=config.output_tag,
                    extra=dict(config.extra),
                ),
            )
        )
    return records


def export_suite_results(
    records_or_dir: list[RunRecord] | str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Export normalized suite JSON files for run records."""

    if isinstance(records_or_dir, (str, Path)):
        records = load_all_results(records_or_dir)
    else:
        records = list(records_or_dir)
    suite_records = [record for record in records if record.suite.suite_id]
    return write_normalized_suite_results(suite_records, output_dir)


__all__ = [
    "DEFAULT_SUITE_RESULTS_DIR",
    "export_suite_results",
    "get_suite_adapter",
    "list_suite_workloads",
    "list_suites",
    "probe_suite",
    "run_suite",
    "run_suite_workload",
]

"""Shared suite adapter API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from compgen.benchmarks.common.env import SuiteEnvironmentStatus
from compgen.benchmarks.common.manifest import SuiteManifestEntry
from compgen.benchmarks.common.results import NormalizedSuiteResult

if TYPE_CHECKING:
    from benchmarks.record import RunRecord
    from benchmarks.spec import WorkspaceConfig
else:
    RunRecord = Any
    WorkspaceConfig = Any


@dataclass(frozen=True)
class SuiteRunConfig:
    """Execution configuration for a suite run."""

    mode: str = "inference"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    blessed_only: bool = True
    num_iterations: int = 10
    warmup_iterations: int = 3
    output_tag: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class SuiteAdapter(Protocol):
    """Common interface implemented by all benchmark suites."""

    suite_id: str

    def enumerate_workloads(
        self,
        workspace: WorkspaceConfig | None = None,
        *,
        blessed_only: bool = False,
    ) -> list[SuiteManifestEntry]:
        ...

    def prepare_environment(
        self,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> SuiteEnvironmentStatus:
        ...

    def prepare_inputs(
        self,
        entry: SuiteManifestEntry,
        workspace: WorkspaceConfig | None = None,
        config: SuiteRunConfig | None = None,
    ) -> Any:
        ...

    def run_reference(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        ...

    def run_compgen(
        self,
        entry: SuiteManifestEntry,
        *,
        workspace: WorkspaceConfig | None = None,
        output_dir: str | Path | None = None,
        config: SuiteRunConfig | None = None,
    ) -> list[RunRecord]:
        ...

    def collect_metrics(self, records: list[RunRecord]) -> list[NormalizedSuiteResult]:
        ...

    def emit_artifacts(
        self,
        records: list[RunRecord],
        *,
        output_dir: str | Path,
    ) -> list[Path]:
        ...


__all__ = ["SuiteAdapter", "SuiteRunConfig"]

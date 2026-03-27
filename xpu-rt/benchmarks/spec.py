"""Study, workload, target, and workspace specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


ModelLoader = Callable[[], tuple[Any, tuple[Any, ...]]]


@dataclass(frozen=True)
class WorkspaceConfig:
    """Workspace layout for CompGen and sibling baseline repos."""

    repo_root: Path
    external_roots: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def default(cls, repo_root: str | Path) -> WorkspaceConfig:
        """Create a workspace config rooted at the current repo."""

        return cls(repo_root=Path(repo_root).resolve())

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkspaceConfig:
        """Load workspace config from YAML."""

        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        repo_root = Path(data.get("repo_root", path.parent)).resolve()
        external_roots = {
            name: Path(value).expanduser().resolve()
            if Path(value).is_absolute()
            else (path.parent / value).resolve()
            for name, value in (data.get("external_roots") or {}).items()
        }
        return cls(repo_root=repo_root, external_roots=external_roots)

    def resolve_external(self, repo_name: str, default_sibling: str | None = None) -> Path:
        """Resolve a sibling baseline repo path."""

        if repo_name in self.external_roots:
            return self.external_roots[repo_name]
        sibling = default_sibling or repo_name
        return (self.repo_root.parent / sibling).resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-friendly dict."""

        return {
            "repo_root": str(self.repo_root),
            "external_roots": {name: str(path) for name, path in self.external_roots.items()},
        }


@dataclass(frozen=True)
class WorkloadSpec:
    """A benchmarkable workload."""

    workload_id: str
    tier: str
    description: str
    loader: ModelLoader
    tags: list[str] = field(default_factory=list)
    shape_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkloadBundle:
    """Discovery/held-out workload bundle."""

    bundle_id: str
    description: str
    discovery_workloads: list[str]
    heldout_workloads: list[str]
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TargetSpec:
    """A benchmark target definition."""

    target_id: str
    path: Path
    kind: str  # "target_profile" or "hardware_spec"
    description: str
    target_class: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BaselineSpec:
    """A runnable or fixture-backed baseline."""

    baseline_id: str
    adapter: str
    description: str
    repo_name: str = ""
    repo_hint: str = ""
    runner_command: list[str] = field(default_factory=list)
    fixture_path: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExperimentCase:
    """A single study case."""

    case_id: str
    study_id: str
    workload_id: str
    target_id: str
    baseline_ids: list[str]
    objective: str = "latency"
    ablations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudySpec:
    """A named study composed of multiple cases."""

    study_id: str
    description: str
    case_ids: list[str]
    tier: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DefectSpec:
    """Verification red-team defect template."""

    defect_id: str
    defect_type: str
    description: str
    expected_stage: str
    severity: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunArtifactIndex:
    """Artifact index for a single run."""

    bundle_path: Path | None = None
    manifest_path: Path | None = None
    artifact_paths: dict[str, Path] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": str(self.bundle_path) if self.bundle_path else "",
            "manifest_path": str(self.manifest_path) if self.manifest_path else "",
            "artifact_paths": {name: str(path) for name, path in self.artifact_paths.items()},
        }


__all__ = [
    "BaselineSpec",
    "DefectSpec",
    "ExperimentCase",
    "RunArtifactIndex",
    "StudySpec",
    "TargetSpec",
    "WorkloadBundle",
    "WorkloadSpec",
    "WorkspaceConfig",
]

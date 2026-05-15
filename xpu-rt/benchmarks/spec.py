"""Study, workload, target, and workspace specifications."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ModelLoader = Callable[[], tuple[Any, tuple[Any, ...]]]


@dataclass(frozen=True)
class WorkspaceConfig:
    """Workspace layout for CompGen and sibling baseline repos."""

    repo_root: Path
    external_roots: dict[str, Path] = field(default_factory=dict)
    pack_roots: dict[str, Path] = field(default_factory=dict)
    llvm_forks: dict[str, Path] = field(default_factory=dict)
    integration_worktrees_root: Path | None = None
    suite_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    pack_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def default(cls, repo_root: str | Path) -> WorkspaceConfig:
        """Create a workspace config rooted at the current repo."""

        resolved = Path(repo_root).resolve()
        return cls(
            repo_root=resolved,
            integration_worktrees_root=resolved / ".compgen_external" / "worktrees",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkspaceConfig:
        """Load workspace config from YAML."""

        def _resolve_nested(value: Any, *, base_dir: Path) -> Any:
            if isinstance(value, dict):
                return {key: _resolve_nested(val, base_dir=base_dir) for key, val in value.items()}
            if isinstance(value, list):
                return [_resolve_nested(item, base_dir=base_dir) for item in value]
            if not isinstance(value, str):
                return value
            if value.startswith(("{", "$")):
                return value
            if any(token in value for token in ("/", "\\")) or value.endswith(
                ("_root", "_dir", ".json", ".yaml", ".csv", ".txt")
            ):
                candidate = Path(value).expanduser()
                if candidate.is_absolute():
                    return str(candidate.resolve())
                return str((base_dir / candidate).resolve())
            return value

        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        repo_root = Path(data.get("repo_root", path.parent)).resolve()
        external_roots = {
            name: Path(value).expanduser().resolve() if Path(value).is_absolute() else (path.parent / value).resolve()
            for name, value in (data.get("external_roots") or {}).items()
        }
        pack_roots = {
            name: Path(value).expanduser().resolve() if Path(value).is_absolute() else (path.parent / value).resolve()
            for name, value in (data.get("pack_roots") or {}).items()
        }
        llvm_forks = {
            name: Path(value).expanduser().resolve() if Path(value).is_absolute() else (path.parent / value).resolve()
            for name, value in (data.get("llvm_forks") or {}).items()
        }
        integration_worktrees_root = data.get("integration_worktrees_root")
        if integration_worktrees_root:
            candidate = Path(integration_worktrees_root).expanduser()
            if not candidate.is_absolute():
                candidate = (path.parent / candidate).resolve()
            integration_worktrees_root = candidate.resolve()
        else:
            integration_worktrees_root = repo_root / ".compgen_external" / "worktrees"
        suite_configs = {
            name: _resolve_nested(config or {}, base_dir=path.parent)
            for name, config in (data.get("suite_configs") or {}).items()
        }
        pack_configs = {
            name: _resolve_nested(config or {}, base_dir=path.parent)
            for name, config in (data.get("pack_configs") or {}).items()
        }
        return cls(
            repo_root=repo_root,
            external_roots=external_roots,
            pack_roots=pack_roots,
            llvm_forks=llvm_forks,
            integration_worktrees_root=integration_worktrees_root,
            suite_configs=suite_configs,
            pack_configs=pack_configs,
        )

    def resolve_external(self, repo_name: str, default_sibling: str | None = None) -> Path:
        """Resolve a sibling baseline repo path."""

        if repo_name in self.external_roots:
            return self.external_roots[repo_name]
        sibling = default_sibling or repo_name
        return (self.repo_root.parent / sibling).resolve()

    def resolve_pack_root(self, pack_name: str, default_sibling: str | None = None) -> Path:
        """Resolve a pack-owned external repo or source tree."""

        if pack_name in self.pack_roots:
            return self.pack_roots[pack_name]
        if pack_name in self.external_roots:
            return self.external_roots[pack_name]
        sibling = default_sibling or pack_name
        return (self.repo_root.parent / sibling).resolve()

    def resolve_llvm_fork(self, fork_name: str) -> Path:
        """Resolve a configured LLVM fork path for a pack."""

        if fork_name in self.llvm_forks:
            return self.llvm_forks[fork_name]
        return (self.repo_root.parent / fork_name).resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-friendly dict."""

        return {
            "repo_root": str(self.repo_root),
            "external_roots": {name: str(path) for name, path in self.external_roots.items()},
            "pack_roots": {name: str(path) for name, path in self.pack_roots.items()},
            "llvm_forks": {name: str(path) for name, path in self.llvm_forks.items()},
            "integration_worktrees_root": str(self.integration_worktrees_root)
            if self.integration_worktrees_root
            else "",
            "suite_configs": self.suite_configs,
            "pack_configs": self.pack_configs,
        }

    def get_suite_config(self, suite_id: str) -> dict[str, Any]:
        """Return configuration for a benchmark suite."""

        config = self.suite_configs.get(suite_id, {})
        return dict(config) if isinstance(config, dict) else {}

    def get_pack_config(self, pack_name: str) -> dict[str, Any]:
        """Return configuration for a pack-backed integration."""

        config = self.pack_configs.get(pack_name, {})
        return dict(config) if isinstance(config, dict) else {}


@dataclass(frozen=True)
class WorkloadSpec:
    """A benchmarkable workload."""

    workload_id: str
    tier: str
    description: str
    loader: ModelLoader
    tags: list[str] = field(default_factory=list)
    shape_config: dict[str, Any] = field(default_factory=dict)
    source_model_id: str = ""
    capture_mode: str = "torch_export"
    readiness: str = "full_pipeline"
    expected_status: str = "pass"
    model_spec: Any | None = None

    def load(self, workspace: WorkspaceConfig | None = None) -> tuple[Any, tuple[Any, ...]]:
        """Load the workload for execution or analysis."""

        if self.model_spec is not None:
            result: tuple[Any, tuple[Any, ...]] = self.model_spec.load(workspace)
            return result
        result = self.loader()
        return result


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

"""Declarative ExtensionPack implementation and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compgen.packs.schema import BranchPlan, ExtensionPackManifest, PackContribution, PackProbeResult

if TYPE_CHECKING:
    from benchmarks.spec import WorkspaceConfig
else:
    WorkspaceConfig = Any


def _as_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value)


@dataclass(frozen=True)
class ManifestExtensionPack:
    """Default pack implementation backed entirely by manifest metadata."""

    root: Path
    manifest: ExtensionPackManifest

    def resolve_source_root(self, workspace: WorkspaceConfig | None = None) -> Path | None:
        if self.manifest.source_root:
            declared = Path(self.manifest.source_root)
            if not declared.is_absolute():
                declared = (self.root / declared).resolve()
            return declared

        if workspace is not None:
            pack_roots = getattr(workspace, "pack_roots", {}) or {}
            external_roots = getattr(workspace, "external_roots", {}) or {}
            for key in (self.manifest.name, *self.manifest.workspace_keys):
                if key in pack_roots:
                    return Path(pack_roots[key]).resolve()
                if key in external_roots:
                    return Path(external_roots[key]).resolve()

            repo_root = Path(workspace.repo_root)
            for name in self.manifest.third_party_names:
                candidate = repo_root / "third_party" / name
                if candidate.exists():
                    return candidate.resolve()
        return None

    def probe(self, workspace: WorkspaceConfig | None = None) -> PackProbeResult:
        source_root = self.resolve_source_root(workspace)
        if source_root is None or not source_root.exists():
            return PackProbeResult(
                pack_name=self.manifest.name,
                available=False,
                source_root=source_root,
                missing_paths=("source_root",),
                details={"reason": "source_root_missing"},
            )

        missing = tuple(
            rel_path
            for rel_path in self.manifest.expected_files
            if not (source_root / rel_path).exists()
        )
        details = {
            "kinds": list(self.manifest.kinds),
            "reference_runner": self.manifest.reference_runner,
            "integration_mode": self.manifest.integration_mode,
        }
        return PackProbeResult(
            pack_name=self.manifest.name,
            available=not missing,
            source_root=source_root,
            missing_paths=missing,
            details=details,
        )

    def compose(self, workspace: WorkspaceConfig | None = None) -> PackContribution:
        probe = self.probe(workspace)
        runtime_artifacts = {
            "integration_mode": self.manifest.integration_mode,
            "reference_runner": self.manifest.reference_runner,
            "source_root": str(probe.source_root) if probe.source_root is not None else "",
        }
        if self.manifest.llvm_fork_key:
            runtime_artifacts["llvm_fork_key"] = self.manifest.llvm_fork_key
        return PackContribution(
            pack_name=self.manifest.name,
            owned_surfaces=self.manifest.owned_surfaces,
            sealed_surfaces=self.manifest.sealed_surfaces,
            generation_apertures=self.manifest.generation_apertures,
            runtime_artifacts=runtime_artifacts,
            benchmark_targets=self.manifest.benchmark_targets,
            available_profilers=self.manifest.available_profilers,
        )

    def branch_plan(self, workspace: WorkspaceConfig | None = None, *, run_id: str = "default") -> BranchPlan:
        if workspace is not None:
            worktrees_root = getattr(workspace, "integration_worktrees_root", None)
            if worktrees_root is None:
                repo_root = Path(workspace.repo_root)
                worktrees_root = repo_root / ".compgen_external" / "worktrees"
            else:
                worktrees_root = Path(worktrees_root)
            llvm_fork_path = None
            llvm_forks = getattr(workspace, "llvm_forks", {}) or {}
            if self.manifest.llvm_fork_key and self.manifest.llvm_fork_key in llvm_forks:
                llvm_fork_path = Path(llvm_forks[self.manifest.llvm_fork_key]).resolve()
        else:
            worktrees_root = self.root / ".compgen_external" / "worktrees"
            llvm_fork_path = None

        branch_name = f"compgen/integration/{self.manifest.name}/{run_id}"
        worktree_path = worktrees_root / self.manifest.name / run_id
        return BranchPlan(
            pack_name=self.manifest.name,
            integration_mode=self.manifest.integration_mode,
            branch_name=branch_name,
            worktree_path=worktree_path,
            llvm_fork_path=llvm_fork_path,
        )


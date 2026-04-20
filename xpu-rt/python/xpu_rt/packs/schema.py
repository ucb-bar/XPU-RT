"""Schema for ExtensionPack manifests and runtime integration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

PackKind = Literal[
    "TargetPack",
    "DialectPack",
    "LLVMForkPack",
    "KernelPack",
    "RuntimePack",
    "ProfilerPack",
    "PerfModelPack",
]

IntegrationMode = Literal["readonly", "overlay_branch", "fork_branch"]


@dataclass(frozen=True)
class ExtensionPackManifest:
    """Manifest describing one fixed external integration surface."""

    name: str
    version: str
    kinds: tuple[PackKind, ...]
    owned_surfaces: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    integration_mode: IntegrationMode = "readonly"
    benchmark_suite: str = "pack_integrations"
    benchmark_targets: tuple[str, ...] = ()
    reference_runner: str = ""
    source_root: str = ""
    workspace_keys: tuple[str, ...] = ()
    third_party_names: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    available_profilers: tuple[str, ...] = ()
    llvm_fork_key: str = ""
    entry_module: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackProbeResult:
    """Result of probing whether a pack is available in the current workspace."""

    pack_name: str
    available: bool
    source_root: Path | None = None
    missing_paths: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchPlan:
    """Integration branch/worktree plan for one pack run."""

    pack_name: str
    integration_mode: IntegrationMode
    branch_name: str
    worktree_path: Path
    llvm_fork_path: Path | None = None


@dataclass(frozen=True)
class PackContribution:
    """How a pack contributes to a composed target package."""

    pack_name: str
    owned_surfaces: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    runtime_artifacts: dict[str, str] = field(default_factory=dict)
    benchmark_targets: tuple[str, ...] = ()
    available_profilers: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackContextSummary:
    """Compact pack summary for target packages, agents, and benchmarks."""

    active_packs: tuple[str, ...] = ()
    sealed_surfaces: tuple[str, ...] = ()
    generation_apertures: tuple[str, ...] = ()
    available_profilers: tuple[str, ...] = ()
    benchmark_targets: tuple[str, ...] = ()
    integration_branch: str = ""

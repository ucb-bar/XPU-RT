"""Protocols and default implementations for ExtensionPacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from compgen.packs.schema import BranchPlan, ExtensionPackManifest, PackContribution, PackProbeResult

if TYPE_CHECKING:
    from benchmarks.spec import WorkspaceConfig
else:
    WorkspaceConfig = Any


@runtime_checkable
class ExtensionPack(Protocol):
    """Protocol implemented by declarative or custom extension packs."""

    manifest: ExtensionPackManifest
    root: Path

    def probe(self, workspace: WorkspaceConfig | None = None) -> PackProbeResult: ...

    def compose(self, workspace: WorkspaceConfig | None = None) -> PackContribution: ...

    def branch_plan(self, workspace: WorkspaceConfig | None = None, *, run_id: str = "default") -> BranchPlan: ...


@dataclass(frozen=True)
class LoadedPack:
    """Resolved pack root plus runtime object."""

    root: Path
    manifest: ExtensionPackManifest
    pack: ExtensionPack

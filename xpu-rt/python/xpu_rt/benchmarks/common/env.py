"""Environment helpers for benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from benchmarks.spec import WorkspaceConfig
else:
    WorkspaceConfig = Any


@dataclass(frozen=True)
class SuiteEnvironmentStatus:
    """Availability summary for a benchmark suite."""

    suite_id: str
    available: bool
    reason: str = ""
    source_root: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def resolve_suite_root(
    workspace: WorkspaceConfig | None,
    *,
    external_keys: tuple[str, ...],
    third_party_names: tuple[str, ...] = (),
) -> Path | None:
    """Resolve a suite root from workspace configuration or local third_party/."""

    if workspace is None:
        return None

    for key in external_keys:
        if key in getattr(workspace, "pack_roots", {}):
            root = Path(workspace.pack_roots[key])
            if root.exists():
                return root
        if key in getattr(workspace, "external_roots", {}):
            root = Path(workspace.external_roots[key])
            if root.exists():
                return root

    repo_root = Path(workspace.repo_root)
    for name in third_party_names:
        candidate = repo_root / "third_party" / name
        if candidate.exists():
            return candidate
    return None


__all__ = ["SuiteEnvironmentStatus", "resolve_suite_root"]

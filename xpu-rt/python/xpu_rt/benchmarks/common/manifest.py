"""Shared manifest types for benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class SuiteManifestEntry:
    """One workload entry in a benchmark suite manifest."""

    suite_id: str
    workload_id: str
    description: str
    upstream_workload_id: str = ""
    mode: str = "inference"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 1
    blessed: bool = False
    readiness: str = "analysis_only"
    expected_status: str = "pass"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_id(self) -> str:
        return f"{self.suite_id}:{self.workload_id}:{self.mode}:{self.dtype}:bs{self.batch_size}"


def filter_manifest_entries(
    entries: Iterable[SuiteManifestEntry],
    *,
    blessed_only: bool = False,
    workload_ids: set[str] | None = None,
) -> list[SuiteManifestEntry]:
    """Filter manifest entries for a suite run."""

    filtered: list[SuiteManifestEntry] = []
    allow = workload_ids or set()
    for entry in entries:
        if blessed_only and not entry.blessed:
            continue
        if allow and entry.workload_id not in allow and entry.upstream_workload_id not in allow:
            continue
        filtered.append(entry)
    return sorted(filtered, key=lambda item: item.workload_id)


__all__ = ["SuiteManifestEntry", "filter_manifest_entries"]

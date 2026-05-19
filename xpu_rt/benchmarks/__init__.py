"""Public benchmark-suite API facade."""

from __future__ import annotations

from xpu_rt.benchmarks.base import SuiteAdapter, SuiteRunConfig
from xpu_rt.benchmarks.common import (
    NormalizedSuiteResult,
    OfficialMetric,
    SuiteArtifactIndex,
    SuiteEnvironmentStatus,
    SuiteManifestEntry,
    filter_manifest_entries,
    resolve_suite_root,
    write_normalized_suite_results,
)
from xpu_rt.benchmarks.results import BenchmarkResult, compare_results, read_json

__all__ = [
    "BenchmarkResult",
    "NormalizedSuiteResult",
    "OfficialMetric",
    "SuiteAdapter",
    "SuiteArtifactIndex",
    "SuiteEnvironmentStatus",
    "SuiteManifestEntry",
    "SuiteRunConfig",
    "compare_results",
    "filter_manifest_entries",
    "read_json",
    "resolve_suite_root",
    "write_normalized_suite_results",
]

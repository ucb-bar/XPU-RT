"""Public benchmark-suite API facade."""

from __future__ import annotations

from compgen.benchmarks.base import SuiteAdapter, SuiteRunConfig
from compgen.benchmarks.common import (
    NormalizedSuiteResult,
    OfficialMetric,
    SuiteArtifactIndex,
    SuiteEnvironmentStatus,
    SuiteManifestEntry,
    filter_manifest_entries,
    resolve_suite_root,
    write_normalized_suite_results,
)
from compgen.benchmarks.results import BenchmarkResult, compare_results, read_json

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


def __getattr__(name: str):
    if name in {
        "export_suite_results",
        "get_suite_adapter",
        "list_suite_workloads",
        "list_suites",
        "probe_suite",
        "run_suite",
        "run_suite_workload",
    }:
        from benchmarks.suite_runner import (
            export_suite_results,
            get_suite_adapter,
            list_suite_workloads,
            list_suites,
            probe_suite,
            run_suite,
            run_suite_workload,
        )

        exports = {
            "export_suite_results": export_suite_results,
            "get_suite_adapter": get_suite_adapter,
            "list_suite_workloads": list_suite_workloads,
            "list_suites": list_suites,
            "probe_suite": probe_suite,
            "run_suite": run_suite,
            "run_suite_workload": run_suite_workload,
        }
        return exports[name]
    raise AttributeError(name)

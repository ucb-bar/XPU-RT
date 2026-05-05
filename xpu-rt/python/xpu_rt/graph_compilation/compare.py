"""Determinism comparator: diff stable fields between two graph compilation runs.

Two reruns of the same ``(model_config, target_config, seed)`` should
produce the same graph hash, the same FX node counts, the same
capture mode, and the same primary-capture status. Wall-clock fields
(timestamps, latencies) are explicitly ignored.

The comparator emits a ``determinism_report.json`` with a per-field
verdict so the user can see which field drifted if anything does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Stable fields we expect to match across two reruns.
_STABLE_REPORT_FIELDS: tuple[tuple[str, ...], ...] = (
    ("model_id",),
    ("target_id",),
    ("seed",),
    ("primary_capture",),
    ("canonical_capture",),
    ("torch_dynamo", "partition_count"),
    ("torch_dynamo", "graph_break_count"),
    ("torch_dynamo", "fullgraph"),
    ("torch_export", "status"),
    ("torch_export", "graph_hash"),
    ("torch_export", "num_ops"),
    ("torch_export", "round_trip_ok"),
    ("diagnostics", "graph_break_count"),
    ("diagnostics", "guard_failures"),
    ("llm_calls",),
)

# Fields ignored by design (wall-clock or environment-dependent).
_IGNORED_FIELDS = [
    "started_at_utc",
    "finished_at_utc",
    "compile_baseline.latency_ms_p50",
    "compile_baseline.latency_ms_p95",
    "compile_baseline.cold_compile_ms",
    "runtime_versions",
]


@dataclass
class DeterminismReport:
    schema_version: str
    a_run_dir: str
    b_run_dir: str
    overall: str  # "pass" | "fail"
    matches: list[dict[str, Any]]
    mismatches: list[dict[str, Any]]
    ignored: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "a_run_dir": self.a_run_dir,
            "b_run_dir": self.b_run_dir,
            "overall": self.overall,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "ignored": self.ignored,
        }


def _walk(obj: Any, key_path: tuple[str, ...]) -> Any:
    """Descend into a nested dict by tuple-of-keys, returning ``None`` if absent."""
    cur: Any = obj
    for k in key_path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def _load_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "00_graph_capture" / "capture_report.json"
    if not path.exists():
        raise FileNotFoundError(f"capture_report.json missing under {run_dir}")
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _load_dynamo_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "00_graph_capture" / "dynamo_summary.json"
    if not path.exists():
        return {}
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def compare_runs(a: Path, b: Path) -> DeterminismReport:
    report_a = _load_report(a)
    report_b = _load_report(b)
    summary_a = _load_dynamo_summary(a)
    summary_b = _load_dynamo_summary(b)

    matches: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for path_tuple in _STABLE_REPORT_FIELDS:
        va = _walk(report_a, path_tuple)
        vb = _walk(report_b, path_tuple)
        entry = {"field": ".".join(path_tuple), "a": va, "b": vb}
        if va == vb:
            matches.append(entry)
        else:
            mismatches.append(entry)

    # Per-partition graph hashes
    ph_a = [p.get("graph_hash") for p in summary_a.get("partitions", [])]
    ph_b = [p.get("graph_hash") for p in summary_b.get("partitions", [])]
    entry = {"field": "dynamo_partition_graph_hashes", "a": ph_a, "b": ph_b}
    if ph_a == ph_b:
        matches.append(entry)
    else:
        mismatches.append(entry)

    overall = "pass" if not mismatches else "fail"
    return DeterminismReport(
        schema_version="determinism_report_v1",
        a_run_dir=str(a),
        b_run_dir=str(b),
        overall=overall,
        matches=matches,
        mismatches=mismatches,
        ignored=list(_IGNORED_FIELDS),
    )

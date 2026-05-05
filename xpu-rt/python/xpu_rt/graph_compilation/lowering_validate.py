"""Consistency validation for ``01_payload_lowering/`` artifacts.

Catches summary-vs-detail drift the artifact contract validator
(``validate.py``) cannot see: it confirms file hashes per the
:class:`RunManifest`, but the manifest just lists artifacts — it has
no opinion on whether ``lowering_summary.totals`` actually equals
``sum(per-module totals)``.

Five checks (all required by the Payload Lowering spec):

1. payload_index_paths_exist
2. aggregate_counts_match_partition_reports
3. opaque_call_counts_match_reports
4. unsupported_counts_match_reports
5. payload_mlir_hashes_match
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compgen.graph_compilation.hashing import sha256_file


@dataclass(frozen=True)
class LoweringCheckResult:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class LoweringValidationReport:
    schema_version: str
    status: str  # "pass" | "fail"
    checks: tuple[LoweringCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
        }


def _read_json(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def validate_payload_lowering(run_dir: Path) -> LoweringValidationReport:
    """Run the 5 consistency checks; return a report.

    If ``01_payload_lowering/`` doesn't exist, every check is
    ``skipped`` and overall is ``pass`` (the validator is opt-in:
    runs that stop after Graph Capture aren't penalised here).
    """
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "01_payload_lowering"
    if not out_dir.is_dir():
        return LoweringValidationReport(
            schema_version="payload_lowering_validation_v1",
            status="pass",
            checks=(
                LoweringCheckResult(
                    name="payload_lowering_present",
                    status="skipped",
                    detail="01_payload_lowering/ not present; nothing to validate",
                ),
            ),
        )

    summary_path = out_dir / "lowering_summary.json"
    payload_index_path = out_dir / "payload_index.json"
    opaque_path = out_dir / "opaque_calls.json"
    unsupported_path = out_dir / "unsupported_ops.json"

    # If the lowering summary is absent, nothing was emitted by Payload
    # Lowering — this is a graph_capture-only run (or a contract-check
    # synthetic run that just happens to have a stub 01_payload_lowering/
    # directory). Skip cleanly rather than fail.
    if not summary_path.exists():
        return LoweringValidationReport(
            schema_version="payload_lowering_validation_v1",
            status="pass",
            checks=(
                LoweringCheckResult(
                    name="payload_lowering_emitted",
                    status="skipped",
                    detail=f"lowering_summary.json absent under {out_dir}",
                ),
            ),
        )

    if not payload_index_path.exists():
        return LoweringValidationReport(
            schema_version="payload_lowering_validation_v1",
            status="fail",
            checks=(
                LoweringCheckResult(
                    name="required_files_present",
                    status="fail",
                    detail=f"summary present but payload_index missing under {out_dir}",
                ),
            ),
        )

    summary = _read_json(summary_path)
    payload_index = _read_json(payload_index_path)
    opaque_top = _read_json(opaque_path) if opaque_path.exists() else {"opaque_calls": [], "summary": {"count": 0}}
    unsupported_top = (
        _read_json(unsupported_path)
        if unsupported_path.exists()
        else {"unsupported_ops": [], "summary": {"count": 0}}
    )

    checks: list[LoweringCheckResult] = []

    # ----- 1. payload_index_paths_exist -----
    missing = []
    for m in payload_index.get("modules", []):
        for key in ("input_graph", "payload_mlir", "lowering_report"):
            v = m.get(key)
            if not v:
                continue
            p = run_dir / v
            if not p.exists():
                missing.append(f"{m.get('module_id', '?')}::{key}={v}")
    checks.append(
        LoweringCheckResult(
            name="payload_index_paths_exist",
            status="pass" if not missing else "fail",
            detail="ok" if not missing else f"missing: {missing}",
        )
    )

    # ----- 2. aggregate_counts_match_partition_reports -----
    per_module_reports = []
    aggregate_drift: list[str] = []
    for m in payload_index.get("modules", []):
        if m.get("status") == "skipped":
            continue
        rp = run_dir / m["lowering_report"]
        if not rp.exists():
            aggregate_drift.append(f"missing report {m['lowering_report']}")
            continue
        per_module_reports.append(_read_json(rp))

    fields = {
        "payload_modules_total": ("len", per_module_reports),
        "fx_nodes_total": (
            "sum",
            [r.get("input", {}).get("num_fx_nodes", 0) for r in per_module_reports],
        ),
        "call_function_nodes_total": (
            "sum",
            [r.get("input", {}).get("num_call_function", 0) for r in per_module_reports],
        ),
        "payload_ops_total": (
            "sum",
            [r.get("output", {}).get("payload_ops_total", 0) for r in per_module_reports],
        ),
        "decomposed_ops_total": (
            "sum",
            [r.get("lowering", {}).get("decomposed_ops", 0) for r in per_module_reports],
        ),
        "opaque_ops_total": (
            "sum",
            [r.get("lowering", {}).get("opaque_ops", 0) for r in per_module_reports],
        ),
        "unsupported_ops_total": (
            "sum",
            [r.get("lowering", {}).get("unsupported_ops", 0) for r in per_module_reports],
        ),
    }
    summary_totals = summary.get("totals", {})
    for field, (op, src) in fields.items():
        expected = len(src) if op == "len" else sum(src)
        actual = summary_totals.get(field)
        if actual != expected:
            aggregate_drift.append(f"{field}: summary={actual} sum_of_modules={expected}")
    checks.append(
        LoweringCheckResult(
            name="aggregate_counts_match_partition_reports",
            status="pass" if not aggregate_drift else "fail",
            detail="ok" if not aggregate_drift else "; ".join(aggregate_drift),
        )
    )

    # ----- 3. opaque_call_counts_match_reports -----
    per_module_opaque = sum(
        r.get("lowering", {}).get("opaque_ops", 0) for r in per_module_reports
    )
    top_opaque = opaque_top.get("summary", {}).get("count", 0)
    list_opaque = len(opaque_top.get("opaque_calls", []))
    opaque_ok = top_opaque == list_opaque == per_module_opaque
    checks.append(
        LoweringCheckResult(
            name="opaque_call_counts_match_reports",
            status="pass" if opaque_ok else "fail",
            detail=(
                "ok"
                if opaque_ok
                else f"top_summary={top_opaque} list_len={list_opaque} per_module_sum={per_module_opaque}"
            ),
        )
    )

    # ----- 4. unsupported_counts_match_reports -----
    per_module_unsupp = sum(
        r.get("lowering", {}).get("unsupported_ops", 0) for r in per_module_reports
    )
    top_unsupp = unsupported_top.get("summary", {}).get("count", 0)
    list_unsupp = len(unsupported_top.get("unsupported_ops", []))
    unsupp_ok = top_unsupp == list_unsupp == per_module_unsupp
    checks.append(
        LoweringCheckResult(
            name="unsupported_counts_match_reports",
            status="pass" if unsupp_ok else "fail",
            detail=(
                "ok"
                if unsupp_ok
                else f"top_summary={top_unsupp} list_len={list_unsupp} per_module_sum={per_module_unsupp}"
            ),
        )
    )

    # ----- 5. payload_mlir_hashes_match -----
    hash_drift: list[str] = []
    for m in payload_index.get("modules", []):
        if m.get("status") == "skipped":
            continue
        declared = m.get("payload_mlir_sha256")
        path_rel = m.get("payload_mlir")
        if not declared or not path_rel:
            continue
        actual = sha256_file(run_dir / path_rel)
        if actual != declared:
            hash_drift.append(f"{m.get('module_id')}: declared={declared[:12]} actual={actual[:12]}")
    checks.append(
        LoweringCheckResult(
            name="payload_mlir_hashes_match",
            status="pass" if not hash_drift else "fail",
            detail="ok" if not hash_drift else "; ".join(hash_drift),
        )
    )

    overall = "pass" if all(c.status == "pass" for c in checks) else "fail"
    return LoweringValidationReport(
        schema_version="payload_lowering_validation_v1",
        status=overall,
        checks=tuple(checks),
    )


def write_lowering_validation_report(run_dir: Path, report: LoweringValidationReport) -> Path:
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lowering_validation.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

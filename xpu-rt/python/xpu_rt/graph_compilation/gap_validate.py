"""Consistency validation for ``03_gap_discovery/`` artifacts (legacy ``02_gap_discovery/`` accepted).

The artifact-contract validator (``validate.py``) re-checks file
existence and sha256 from the manifest. The lowering validator
(``lowering_validate.py``) checks that summary totals match per-module
detail. This validator does the same for Gap Discovery — it catches
contract violations the other two cannot see, like:

- a ``gap_id`` that doesn't match ``^gap_[0-9]{4}$``
- duplicate ``gap_id`` values
- ``allowed_actions == []``
- a ``critical_path`` gap whose only allowed action is ``keep_as_fallback``
- ``source_artifacts`` paths that don't resolve
- ``gap_action_queue.summary.count`` that lies about the list length

Skips cleanly (``status=pass``) when the gap_discovery dir is absent —
this validator is opt-in, just like ``lowering_validate``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_GAP_ID_RE = re.compile(r"^gap_\d{4}$")


@dataclass(frozen=True)
class GapCheckResult:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class GapValidationReport:
    schema_version: str
    status: str  # "pass" | "fail"
    checks: tuple[GapCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
        }


def _read_json(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def validate_gap_discovery(run_dir: Path) -> GapValidationReport:
    from compgen.graph_compilation.artifacts import stage_dir

    run_dir = Path(run_dir).resolve()
    out_dir = stage_dir(run_dir, "gap_discovery")
    assert isinstance(out_dir, Path)
    if not out_dir.is_dir():
        return GapValidationReport(
            schema_version="gap_validation_v1",
            status="pass",
            checks=(
                GapCheckResult(
                    name="gap_discovery_emitted",
                    status="skipped",
                    detail=f"{out_dir} absent — Gap Discovery not run",
                ),
            ),
        )

    queue_path = out_dir / "gap_action_queue.json"
    analysis_path = out_dir / "gap_analysis.json"
    dossier_path = out_dir / "dossier.json"
    report_path = out_dir / "gap_discovery_summary.json"

    # Skip cleanly when none of the Gap Discovery outputs are present —
    # this is a graph_capture-only / payload-lowering-only run, or a
    # contract-check synthetic run that just happens to have a stub
    # gap_discovery dir. Same policy as lowering_validate.
    if not queue_path.exists():
        return GapValidationReport(
            schema_version="gap_validation_v1",
            status="pass",
            checks=(
                GapCheckResult(
                    name="gap_discovery_emitted",
                    status="skipped",
                    detail=f"gap_action_queue.json absent under {out_dir}",
                ),
            ),
        )

    # If the queue is present but other outputs aren't, that IS a real broken run.
    missing_files = [p.name for p in (analysis_path, dossier_path, report_path) if not p.exists()]
    if missing_files:
        return GapValidationReport(
            schema_version="gap_validation_v1",
            status="fail",
            checks=(
                GapCheckResult(
                    name="required_files_present",
                    status="fail",
                    detail=f"queue present but missing: {missing_files}",
                ),
            ),
        )

    queue = _read_json(queue_path)
    analysis = _read_json(analysis_path)
    report = _read_json(report_path)
    gaps = queue.get("gaps", [])

    checks: list[GapCheckResult] = []

    # ----- 1. unique_gap_ids + format -----
    seen_ids: dict[str, int] = {}
    bad_format: list[str] = []
    duplicates: list[str] = []
    for g in gaps:
        gid = g.get("gap_id", "")
        if not _GAP_ID_RE.match(gid):
            bad_format.append(gid)
        seen_ids[gid] = seen_ids.get(gid, 0) + 1
    duplicates = [gid for gid, n in seen_ids.items() if n > 1]
    detail_parts: list[str] = []
    if bad_format:
        detail_parts.append(f"bad_format={bad_format}")
    if duplicates:
        detail_parts.append(f"duplicates={duplicates}")
    checks.append(
        GapCheckResult(
            name="gap_id_format_and_uniqueness",
            status="pass" if not detail_parts else "fail",
            detail="ok" if not detail_parts else "; ".join(detail_parts),
        )
    )

    # ----- 2. allowed_actions non-empty -----
    empty_actions = [g.get("gap_id") for g in gaps if not g.get("allowed_actions")]
    checks.append(
        GapCheckResult(
            name="allowed_actions_non_empty",
            status="pass" if not empty_actions else "fail",
            detail="ok" if not empty_actions else f"empty: {empty_actions}",
        )
    )

    # ----- 3. severity / action invariant -----
    bad_critical: list[str] = []
    for g in gaps:
        if g.get("severity") != "critical_path":
            continue
        actions = g.get("allowed_actions") or []
        if actions == ["keep_as_fallback"]:
            bad_critical.append(g.get("gap_id", "?"))
    checks.append(
        GapCheckResult(
            name="critical_gaps_have_real_actions",
            status="pass" if not bad_critical else "fail",
            detail="ok" if not bad_critical else f"critical-only-fallback: {bad_critical}",
        )
    )

    # ----- 4. source_artifacts paths exist -----
    missing_artifacts: list[str] = []
    for g in gaps:
        srcs = g.get("source_artifacts", {})
        for key, rel in srcs.items():
            if not rel:
                continue
            if not (run_dir / rel).exists():
                missing_artifacts.append(f"{g.get('gap_id', '?')}::{key}={rel}")
    checks.append(
        GapCheckResult(
            name="source_artifacts_paths_exist",
            status="pass" if not missing_artifacts else "fail",
            detail="ok" if not missing_artifacts else "; ".join(missing_artifacts),
        )
    )

    # ----- 5. summary counts match list lengths -----
    summary = queue.get("summary", {})
    drift: list[str] = []
    if summary.get("count", -1) != len(gaps):
        drift.append(f"summary.count={summary.get('count')} vs len(gaps)={len(gaps)}")
    by_kind = summary.get("by_kind", {})
    actual_by_kind: dict[str, int] = {}
    for g in gaps:
        k = g.get("gap_kind", "")
        actual_by_kind[k] = actual_by_kind.get(k, 0) + 1
    if by_kind != actual_by_kind:
        drift.append(f"by_kind summary={by_kind} actual={actual_by_kind}")
    checks.append(
        GapCheckResult(
            name="queue_summary_matches_list",
            status="pass" if not drift else "fail",
            detail="ok" if not drift else "; ".join(drift),
        )
    )

    # ----- 6a. evidence requirements per gap_kind -----
    # The GAP-00 validator: unsupported_op MUST require reference_semantics
    # and differential_tests; unsupported_quant_format MUST require
    # quant_format_spec, dequant_reference, rounding_policy, scale_layout,
    # and error_tolerance. The validator REJECTS gaps that violate this.
    evidence_required_by_kind = {
        "unsupported_op": {"reference_semantics", "differential_tests"},
        "unsupported_quant_format": {
            "quant_format_spec", "dequant_reference", "rounding_policy",
            "scale_layout", "error_tolerance",
        },
    }
    evidence_violations: list[str] = []
    for g in gaps:
        kind = g.get("gap_kind", "")
        required = evidence_required_by_kind.get(kind)
        if not required:
            continue
        actual = set(g.get("required_evidence") or [])
        missing = required - actual
        if missing:
            evidence_violations.append(
                f"{g.get('gap_id', '?')}({kind}): missing required evidence keys {sorted(missing)}"
            )
    checks.append(
        GapCheckResult(
            name="required_evidence_satisfies_kind_invariants",
            status="pass" if not evidence_violations else "fail",
            detail="ok" if not evidence_violations else "; ".join(evidence_violations),
        )
    )

    # ----- 6b. extension_id format + agreement with kind/target/slug -----
    from compgen.graph_compilation.gap_naming import extension_id as _expected_ext_id

    ext_id_violations: list[str] = []
    for g in gaps:
        gid = g.get("gap_id", "?")
        ext_id = g.get("extension_id", "")
        if not ext_id:
            ext_id_violations.append(f"{gid}: missing extension_id")
            continue
        # Format: <gap_kind>__<slug>__<target_id>__<sha8>
        parts = ext_id.split("__")
        if len(parts) != 4:
            ext_id_violations.append(f"{gid}: bad extension_id format {ext_id!r}")
            continue
        kind_part, slug_part, _target_part, sha_part = parts
        if kind_part != g.get("gap_kind"):
            ext_id_violations.append(f"{gid}: extension_id kind {kind_part} != gap_kind {g.get('gap_kind')}")
        if slug_part != g.get("slug"):
            ext_id_violations.append(f"{gid}: extension_id slug {slug_part} != slug field {g.get('slug')}")
        if len(sha_part) != 8 or any(c not in "0123456789abcdef" for c in sha_part):
            ext_id_violations.append(f"{gid}: extension_id sha {sha_part!r} is not 8-char lowercase hex")
        # Cross-check: the recomputed canonical id must match.
        try:
            expected = _expected_ext_id(
                gap_kind=g["gap_kind"],
                fx_target=g["fx_target"],
                target_id=g["target_id"],
                shape_signature=g.get("shape_signature", {}),
                dtype_signature=g.get("dtype_signature", {}),
            )
        except KeyError as exc:
            ext_id_violations.append(f"{gid}: missing field for canonical id: {exc}")
            continue
        if expected != ext_id:
            ext_id_violations.append(
                f"{gid}: extension_id {ext_id} != canonical {expected}"
            )
    checks.append(
        GapCheckResult(
            name="extension_id_canonical",
            status="pass" if not ext_id_violations else "fail",
            detail="ok" if not ext_id_violations else "; ".join(ext_id_violations),
        )
    )

    # ----- 7. report-vs-queue consistency + llm_calls + analysis-regions -----
    report_drift: list[str] = []
    rt = report.get("totals", {})
    if rt.get("gaps_total", -1) != len(gaps):
        report_drift.append(f"report.totals.gaps_total={rt.get('gaps_total')} vs {len(gaps)}")
    if report.get("llm_calls", -1) != 0:
        report_drift.append(f"report.llm_calls={report.get('llm_calls')} (must be 0)")
    if analysis.get("regions_total", -1) != len(analysis.get("ops", [])):
        report_drift.append(
            f"gap_analysis.regions_total={analysis.get('regions_total')} vs len(ops)={len(analysis.get('ops', []))}"
        )
    checks.append(
        GapCheckResult(
            name="report_and_analysis_consistency",
            status="pass" if not report_drift else "fail",
            detail="ok" if not report_drift else "; ".join(report_drift),
        )
    )

    overall = "pass" if all(c.status in ("pass", "skipped") for c in checks) else "fail"
    return GapValidationReport(
        schema_version="gap_validation_v1",
        status=overall,
        checks=tuple(checks),
    )


def write_gap_validation_report(run_dir: Path, report: GapValidationReport) -> Path:
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gap_validation.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

"""Consistency validation for ``04_gap_closure/`` artifacts (legacy ``03_gap_closure/`` accepted).

Six checks:

1. required files present (closure_summary, extensions_invoked, gap_delta)
2. counts match: summary.extensions_invoked_count == len(extensions_invoked.extensions)
3. registered count matches: summary.extensions_registered_count == #invocations with register_status==pass
4. registered extensions resolve on disk (extension_path exists, contract present)
5. registered extensions have ``verification.json`` and ``status==pass``
6. ``llm_calls == 0`` in closure_summary

Skips cleanly when the gap_closure dir is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClosureCheckResult:
    name: str
    status: str  # "pass" | "fail" | "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class ClosureValidationReport:
    schema_version: str
    status: str  # "pass" | "fail"
    checks: tuple[ClosureCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
        }


def _read_json(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def validate_gap_closure(run_dir: Path) -> ClosureValidationReport:
    from compgen.graph_compilation.artifacts import stage_dir

    run_dir = Path(run_dir).resolve()
    out_dir = stage_dir(run_dir, "gap_closure")
    assert isinstance(out_dir, Path)
    if not out_dir.is_dir():
        # Gap Closure has not been run for this run directory. Honest
        # report: ``not_applicable`` rather than ``pass`` — there is
        # nothing to attest to. The caller treats not_applicable as
        # non-failing (the run can still be valid through stage 02).
        return ClosureValidationReport(
            schema_version="closure_validation_v1",
            status="not_applicable",
            checks=(
                ClosureCheckResult(
                    name="gap_closure_emitted",
                    status="skipped",
                    detail=(
                        f"{out_dir} absent — extension closure was not "
                        "materialized, verified, or registered for this run"
                    ),
                ),
            ),
        )

    summary_path = out_dir / "closure_summary.json"
    invocations_path = out_dir / "extensions_invoked.json"
    delta_path = out_dir / "gap_delta.json"

    if not summary_path.exists():
        return ClosureValidationReport(
            schema_version="closure_validation_v1",
            status="not_applicable",
            checks=(
                ClosureCheckResult(
                    name="gap_closure_emitted",
                    status="skipped",
                    detail="closure_summary.json absent",
                ),
            ),
        )

    missing = [p.name for p in (invocations_path, delta_path) if not p.exists()]
    if missing:
        return ClosureValidationReport(
            schema_version="closure_validation_v1",
            status="fail",
            checks=(
                ClosureCheckResult(
                    name="required_files_present",
                    status="fail",
                    detail=f"missing: {missing}",
                ),
            ),
        )

    summary = _read_json(summary_path)
    invocations_obj = _read_json(invocations_path)
    invocations = invocations_obj.get("extensions", [])

    checks: list[ClosureCheckResult] = []

    checks.append(
        ClosureCheckResult(
            name="required_files_present",
            status="pass",
            detail="closure_summary, extensions_invoked, gap_delta all present",
        )
    )

    # ----- 2. invocation count matches list -----
    if summary.get("extensions_invoked_count") != len(invocations):
        checks.append(
            ClosureCheckResult(
                name="invocation_count_matches_list",
                status="fail",
                detail=(
                    f"summary={summary.get('extensions_invoked_count')} "
                    f"actual={len(invocations)}"
                ),
            )
        )
    else:
        checks.append(ClosureCheckResult(name="invocation_count_matches_list", status="pass"))

    # ----- 3. registered count matches -----
    actual_registered = sum(1 for inv in invocations if inv.get("register_status") == "pass")
    if summary.get("extensions_registered_count") != actual_registered:
        checks.append(
            ClosureCheckResult(
                name="registered_count_matches_list",
                status="fail",
                detail=(
                    f"summary={summary.get('extensions_registered_count')} "
                    f"actual={actual_registered}"
                ),
            )
        )
    else:
        checks.append(ClosureCheckResult(name="registered_count_matches_list", status="pass"))

    # ----- 4. registered extensions exist on disk -----
    missing_ext: list[str] = []
    bad_contract: list[str] = []
    for inv in invocations:
        if inv.get("register_status") != "pass":
            continue
        ext_dir = Path(inv.get("extension_path", ""))
        if not ext_dir.is_dir():
            missing_ext.append(inv.get("extension_id", "?"))
            continue
        contract = ext_dir / "extension_contract.json"
        if not contract.exists():
            bad_contract.append(inv.get("extension_id", "?"))
    detail = []
    if missing_ext:
        detail.append(f"missing_dir={missing_ext}")
    if bad_contract:
        detail.append(f"missing_contract={bad_contract}")
    checks.append(
        ClosureCheckResult(
            name="registered_extensions_resolve",
            status="pass" if not detail else "fail",
            detail="ok" if not detail else "; ".join(detail),
        )
    )

    # ----- 5. registered extensions have verification.json status==pass -----
    failing_verif: list[str] = []
    for inv in invocations:
        if inv.get("register_status") != "pass":
            continue
        ext_dir = Path(inv.get("extension_path", ""))
        verif_path = ext_dir / "results" / "verification.json"
        if not verif_path.exists():
            failing_verif.append(f"{inv.get('extension_id')}: verification.json missing")
            continue
        try:
            v = _read_json(verif_path)
        except Exception as exc:
            failing_verif.append(f"{inv.get('extension_id')}: bad json {exc}")
            continue
        if v.get("status") != "pass":
            failing_verif.append(f"{inv.get('extension_id')}: status={v.get('status')}")
    checks.append(
        ClosureCheckResult(
            name="registered_extensions_verified",
            status="pass" if not failing_verif else "fail",
            detail="ok" if not failing_verif else "; ".join(failing_verif),
        )
    )

    # ----- 6. llm_calls == 0 -----
    if summary.get("llm_calls", -1) != 0:
        checks.append(
            ClosureCheckResult(
                name="llm_calls_zero",
                status="fail",
                detail=f"summary.llm_calls={summary.get('llm_calls')}",
            )
        )
    else:
        checks.append(ClosureCheckResult(name="llm_calls_zero", status="pass"))

    overall = "pass" if all(c.status in ("pass", "skipped") for c in checks) else "fail"
    return ClosureValidationReport(
        schema_version="closure_validation_v1",
        status=overall,
        checks=tuple(checks),
    )


def write_closure_validation_report(run_dir: Path, report: ClosureValidationReport) -> Path:
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "closure_validation.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

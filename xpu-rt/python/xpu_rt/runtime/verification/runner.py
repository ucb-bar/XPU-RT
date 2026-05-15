"""Aggregator that runs the three D6 gates and emits a report.

Each individual gate raises its named typed error on failure when
``raise_on_fail=True`` (the default for in-pipeline use). The
aggregator runs all three with ``raise_on_fail=False`` so a single
invocation collects every failure for the report; only after writing
the report does it re-raise the first failure, so the JSON record on
disk is always complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from compgen.runtime.errors import (
    AbiConformanceError,
    ResourceBudgetError,
    RuntimeRefinementError,
)
from compgen.runtime.verification.abi_conformance import (
    AbiConformanceReport,
    check_abi_conformance,
)
from compgen.runtime.verification.plan_refinement import (
    PlanRefinementReport,
    check_plan_refinement,
)
from compgen.runtime.verification.resource_budget import (
    ResourceBudgetReport,
    check_resource_budget,
)


_RUNTIME_VERIFICATION_SCHEMA = "runtime_verification_report_v1"


@dataclass(frozen=True)
class RuntimeVerificationReport:
    overall: str  # "pass" | "fail"
    emit_dir: str
    refinement: PlanRefinementReport
    abi: AbiConformanceReport
    budget: ResourceBudgetReport
    report_path: Path | None = field(default=None)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _RUNTIME_VERIFICATION_SCHEMA,
            "generated_at_utc": datetime.now(tz=UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ",
            ),
            "overall": self.overall,
            "emit_dir": self.emit_dir,
            "refinement": self.refinement.to_dict(),
            "abi": self.abi.to_dict(),
            "budget": self.budget.to_dict(),
        }


def _pick_emit_path(emit_dir: Path) -> Path | None:
    for src_name in (
        "generated_plan_executor.c",
        "generated_plan_executor.cpp",
    ):
        p = emit_dir / src_name
        if p.exists():
            return p
    return None


def run_runtime_verification(
    emit_dir: Path,
    *,
    raise_on_fail: bool = True,
    write_report: bool = True,
) -> RuntimeVerificationReport:
    """Run the three D6 gates on a glue-emit directory.

    When ``write_report``, writes
    ``runtime_verification_report.json`` next to the emit. When
    ``raise_on_fail`` and any gate fails, raises the first typed
    error after the report is on disk.
    """
    emit_dir = Path(emit_dir).resolve()
    emit_path = _pick_emit_path(emit_dir)

    refinement = check_plan_refinement(emit_dir, raise_on_fail=False)
    if emit_path is not None:
        abi = check_abi_conformance(emit_path, raise_on_fail=False)
    else:
        abi = AbiConformanceReport(
            overall="fail", emit_path=str(emit_dir),
            forbidden_symbols=("__no_emit__",),
        )
    budget = check_resource_budget(emit_dir, raise_on_fail=False)

    overall = "pass" if all(
        r.overall == "pass" for r in (refinement, abi, budget)
    ) else "fail"

    report = RuntimeVerificationReport(
        overall=overall,
        emit_dir=str(emit_dir),
        refinement=refinement,
        abi=abi,
        budget=budget,
    )

    report_path: Path | None = None
    if write_report:
        report_path = emit_dir / "runtime_verification_report.json"
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    final = RuntimeVerificationReport(
        overall=overall,
        emit_dir=str(emit_dir),
        refinement=refinement,
        abi=abi,
        budget=budget,
        report_path=report_path,
    )

    if overall == "fail" and raise_on_fail:
        # Re-raise the first failing gate's named typed error so the
        # caller's `except` clauses fire with the right kind.
        if refinement.overall == "fail" and refinement.failures:
            kind, detail = refinement.failures[0]
            raise RuntimeRefinementError(kind, detail)
        if abi.overall == "fail":
            raise AbiConformanceError(
                abi.forbidden_symbols, emit_path=abi.emit_path,
            )
        if budget.overall == "fail" and budget.failures:
            res, declared, observed = budget.failures[0]
            raise ResourceBudgetError(
                res, declared, observed, emit_path=budget.emit_path,
            )

    return final

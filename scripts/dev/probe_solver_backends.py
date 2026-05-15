"""Probe all registered solver backends and emit a typed status report.

. Writes both JSON (machine-readable) and Markdown (human
summary) reports describing the availability of every solver
backend on this host. MOSEK license absence is reported as
``license_missing`` or ``license_token_unavailable`` and does NOT
fail the script — the open-source baseline (Z3 + OR-Tools + HiGHS)
is what's required.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from compgen.solve.backend_registry import default_registry
from compgen.solve.solver_types import (
    BackendAvailabilityStatus,
    SolverBackendName,
)

SCHEMA_VERSION = "solver_backend_status_v1"


def _build_report() -> dict:
    reg = default_registry()
    results = reg.probe_all(force=True)
    backends: dict[str, dict] = {}
    for name in (
        SolverBackendName.Z3,
        SolverBackendName.ORTOOLS_CP_SAT,
        SolverBackendName.MOSEK,
        SolverBackendName.HIGHS,
    ):
        result = results.get(name)
        if result is None:
            backends[name.value] = {
                "availability": BackendAvailabilityStatus.IMPORT_MISSING.value,
                "version": None,
                "supports": [],
                "detail": "backend not registered",
            }
        else:
            backends[name.value] = {
                "availability": result.availability.value,
                "version": result.version,
                "supports": list(result.supports),
                "detail": result.detail,
            }
    avail = {k: v["availability"] for k, v in backends.items()}
    required_baseline_ok = (
        avail["z3"] == "available"
        and avail["ortools_cp_sat"] == "available"
        and (avail["highs"] == "available" or avail["mosek"] == "available")
    )
    optional_accelerators_ok = avail["mosek"] == "available"
    notes: list[str] = []
    if not required_baseline_ok:
        notes.append("baseline NOT ok: need z3 + ortools_cp_sat + at least one of {highs, mosek}")
    if not optional_accelerators_ok:
        notes.append("mosek unavailable; HiGHS-only memory planning")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "backends": backends,
        "required_baseline_ok": required_baseline_ok,
        "optional_accelerators_ok": optional_accelerators_ok,
        "notes": notes,
    }


def _summarize_md(report: dict) -> str:
    lines = [
        "# Solver backend status",
        "",
        f"- **generated_at**: {report['generated_at']}",
        f"- **required_baseline_ok**: {report['required_baseline_ok']}",
        f"- **optional_accelerators_ok**: {report['optional_accelerators_ok']}",
        "",
        "| backend | availability | version | supports | detail |",
        "|---|---|---|---|---|",
    ]
    for name, body in report["backends"].items():
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                name,
                body["availability"],
                body["version"] or "-",
                ", ".join(body["supports"]) or "-",
                body["detail"] or "-",
            )
        )
    if report["notes"]:
        lines.append("")
        lines.append("## Notes")
        for note in report["notes"]:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe solver backends on this host.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/solver_probe"),
        help="Output directory for JSON+Markdown reports.",
    )
    args = parser.parse_args(argv)
    report = _build_report()
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "solver_backend_status.json"
    md_path = args.out / "solver_backend_status.md"
    json_path.write_text(json.dumps(report, sort_keys=True, indent=2))
    md_path.write_text(_summarize_md(report))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not report["required_baseline_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

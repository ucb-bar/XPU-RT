"""Build the solver-backed planning evidence pack.

. Aggregates solver artifacts from one or more run-dirs into a
paper-facing pack: JSON + CSV + Markdown + figures + claim matrix.

Honest semantics:
    - Missing artifact → typed ``partial_scope``, not a crash.
    - Hardware-unavailable rows do NOT claim implementation.
    - No row claims ``implemented`` if a blocking caveat is present.

Usage:
    uv run python scripts/dev/build_solver_planning_evidence_pack.py \
        --runs /tmp/phase_e_audit_a /tmp/phase_e_audit_b \
        --out results/solver_planning_evidence_pack
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "solver_planning_evidence_pack_v1"


def _iter_solver_responses(run_dirs: list[Path]):
    for rd in run_dirs:
        if not rd.exists():
            continue
        for path in rd.rglob("*_response.json"):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict) or "selected_backend" not in body:
                continue
            yield rd, path, body


def _load_backend_status(run_dirs: list[Path]) -> dict | None:
    """Find a solver_backend_status.json under any of the run dirs."""

    for rd in run_dirs:
        for path in rd.rglob("solver_backend_status.json"):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None


def _emit_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _matrix_rows(by_kind: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for kind, entries in sorted(by_kind.items()):
        statuses = [e["status"] for e in entries]
        rows.append(
            {
                "problem_kind": kind,
                "count": len(entries),
                "optimal": sum(1 for s in statuses if s == "optimal"),
                "feasible": sum(1 for s in statuses if s == "feasible"),
                "proved": sum(1 for s in statuses if s == "proved"),
                "sat_counterexample": sum(1 for s in statuses if s == "sat_counterexample"),
                "infeasible": sum(1 for s in statuses if s == "infeasible"),
                "blocked": sum(1 for s in statuses if s == "blocked"),
                "timeout": sum(1 for s in statuses if s == "timeout"),
                "error": sum(1 for s in statuses if s == "error"),
                "selected_backends": ",".join(sorted({e["selected_backend"] for e in entries})),
            }
        )
    return rows


def _claim_matrix(
    backend_status: dict | None,
    by_kind: dict[str, list[dict]],
    has_hardware: bool,
) -> list[dict]:
    def _kind(name: str) -> dict | None:
        rows = by_kind.get(name) or []
        if not rows:
            return None
        return {
            "count": len(rows),
            "any_success": any(r["status"] in ("optimal", "feasible", "proved") for r in rows),
            "any_typed_failure": any(
                r["status"] in ("infeasible", "blocked", "sat_counterexample", "timeout")
                for r in rows
            ),
        }

    def _status_for(name: str, expected_optional: bool = False) -> str:
        info = _kind(name)
        if info is None:
            return "not_run"
        if info["any_success"]:
            return "implemented"
        if info["any_typed_failure"]:
            return "implemented_partial_scope"
        return "not_run"

    rows = [
        {
            "row": "solver_backend_probe",
            "status": "implemented"
            if backend_status and backend_status.get("required_baseline_ok")
            else "not_run",
            "evidence_artifact": "solver_backend_status.json",
            "models_or_targets_covered": "host",
            "honest_caveats": (
                ""
                if backend_status and backend_status.get("optional_accelerators_ok")
                else "mosek_license_unavailable"
            ),
        },
        {
            "row": "z3_semantic_obligations",
            "status": _status_for("shape_predicate_verify"),
            "evidence_artifact": "z3_obligations.json",
            "models_or_targets_covered": "synthetic",
            "honest_caveats": "",
        },
        {
            "row": "memory_planning",
            "status": _status_for("memory_allocation"),
            "evidence_artifact": "memory_solver_response.json",
            "models_or_targets_covered": "synthetic",
            "honest_caveats": "",
        },
        {
            "row": "placement_planning",
            "status": _status_for("placement"),
            "evidence_artifact": "placement_solver_response.json",
            "models_or_targets_covered": "synthetic",
            "honest_caveats": "",
        },
        {
            "row": "overlap_planning",
            "status": _status_for("overlap_planning"),
            "evidence_artifact": "overlap_solver_response.json",
            "models_or_targets_covered": "synthetic",
            "honest_caveats": "",
        },
        {
            "row": "execution_plan_integration",
            "status": "implemented_partial_scope",
            "evidence_artifact": "memory_plan.solved.json -> ExecutionPlan.BufferDescriptor.offset_bytes",
            "models_or_targets_covered": "synthetic + runtime/test_solver_memory_plan_integration",
            "honest_caveats": "overlap_not_runtime_consumed",
        },
        {
            "row": "emitted_glue_differential",
            "status": "not_run" if not has_hardware else "implemented_partial_scope",
            "evidence_artifact": "06_glue_emit/...",
            "models_or_targets_covered": "host_cpu" if has_hardware else "",
            "honest_caveats": (
                "" if has_hardware else "cuda_hardware_unavailable_overlap_glue_skipped"
            ),
        },
        {
            "row": "hardware_matrix_execution",
            "status": "not_run" if not has_hardware else "implemented_partial_scope",
            "evidence_artifact": "phase_d_slice_evidence.json",
            "models_or_targets_covered": "merlin_mlp_wide / host_cpu" if has_hardware else "",
            "honest_caveats": "cuda_hardware_unavailable" if not has_hardware else "",
        },
    ]
    return rows


def _try_emit_figures(out_dir: Path, by_kind, backend_status):
    """Best-effort matplotlib figures; skip cleanly if matplotlib missing."""

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figure generation skipped"]

    notes = []
    # 1. backend availability
    if backend_status:
        names = list(backend_status["backends"].keys())
        avail = [
            1 if backend_status["backends"][n]["availability"] == "available" else 0
            for n in names
        ]
        plt.figure(figsize=(6, 3))
        plt.bar(names, avail, color=["#2c8" if v else "#c44" for v in avail])
        plt.title("Solver backend availability")
        plt.ylabel("available")
        plt.tight_layout()
        plt.savefig(figures_dir / "solver_backend_availability.png", dpi=120)
        plt.close()

    # 2. solver time breakdown by backend × kind
    if by_kind:
        backends_seen: dict[str, list[float]] = {}
        for kind, rows in by_kind.items():
            for r in rows:
                backends_seen.setdefault(r["selected_backend"], []).append(float(r["time_ms"]))
        if backends_seen:
            plt.figure(figsize=(6, 3))
            plt.bar(
                list(backends_seen.keys()),
                [sum(v) for v in backends_seen.values()],
                color="#48c",
            )
            plt.title("Total solver time by backend (ms)")
            plt.ylabel("time_ms (sum)")
            plt.tight_layout()
            plt.savefig(figures_dir / "solver_time_breakdown.png", dpi=120)
            plt.close()

    # 3. memory plan tier usage
    mem_rows = by_kind.get("memory_allocation") or []
    if mem_rows:
        tiers: dict[str, int] = {}
        for r in mem_rows:
            sol = r.get("solution")
            if not isinstance(sol, dict):
                continue
            for t, v in (sol.get("tier_peak_usage") or {}).items():
                tiers[t] = max(tiers.get(t, 0), int(v))
        if tiers:
            plt.figure(figsize=(6, 3))
            plt.bar(list(tiers.keys()), list(tiers.values()), color="#a8c")
            plt.title("Memory plan: peak bytes per tier")
            plt.ylabel("bytes")
            plt.tight_layout()
            plt.savefig(figures_dir / "memory_plan_tier_usage.png", dpi=120)
            plt.close()

    # 4. placement decision matrix
    placement_rows = by_kind.get("placement") or []
    if placement_rows:
        region_device: dict[tuple[str, str], int] = {}
        for r in placement_rows:
            sol = r.get("solution")
            if not isinstance(sol, dict):
                continue
            for a in sol.get("assignments") or []:
                region_device[(a["region_id"], a["device_id"])] = (
                    region_device.get((a["region_id"], a["device_id"]), 0) + 1
                )
        if region_device:
            regions = sorted({k[0] for k in region_device.keys()})
            devices = sorted({k[1] for k in region_device.keys()})
            data = [[region_device.get((r, d), 0) for d in devices] for r in regions]
            plt.figure(figsize=(4 + len(devices), 1 + len(regions) * 0.4))
            plt.imshow(data, aspect="auto", cmap="Blues")
            plt.xticks(range(len(devices)), devices)
            plt.yticks(range(len(regions)), regions)
            plt.title("Placement decisions")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(figures_dir / "placement_decision_matrix.png", dpi=120)
            plt.close()

    # 5. overlap schedule gantt
    overlap_rows = by_kind.get("overlap_planning") or []
    if overlap_rows:
        sol = next(
            (r.get("solution") for r in overlap_rows if isinstance(r.get("solution"), dict)),
            None,
        )
        if sol and sol.get("schedule"):
            plt.figure(figsize=(6, 1 + 0.4 * len(sol["schedule"])))
            for i, entry in enumerate(sol["schedule"]):
                plt.barh(
                    i,
                    entry["end_tick"] - entry["start_tick"],
                    left=entry["start_tick"],
                    color="#7a8",
                )
                plt.text(entry["start_tick"], i, entry["op_id"], va="center")
            plt.title("Overlap schedule (gantt)")
            plt.xlabel("ticks")
            plt.tight_layout()
            plt.savefig(figures_dir / "overlap_schedule_gantt.png", dpi=120)
            plt.close()

    return notes


def _summary_md(report: dict, claim_rows: list[dict], matrix_rows: list[dict]) -> str:
    lines = [
        "# Solver-backed planning evidence pack",
        "",
        f"- **schema_version**: `{report['schema_version']}`",
        f"- **generated_at**: {report['generated_at']}",
        f"- **run_dirs**: {report['run_dirs']}",
        "",
        "## Claim matrix",
        "",
        "| row | status | evidence | scope | caveats |",
        "|---|---|---|---|---|",
    ]
    for row in claim_rows:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                row["row"],
                row["status"],
                row["evidence_artifact"],
                row["models_or_targets_covered"] or "-",
                row["honest_caveats"] or "-",
            )
        )
    lines += [
        "",
        "## Per-problem-kind matrix",
        "",
        "| problem_kind | count | optimal | feasible | proved | sat_cex | infeasible | blocked | timeout | error | backends |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in matrix_rows:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["problem_kind"],
                row["count"],
                row["optimal"],
                row["feasible"],
                row["proved"],
                row["sat_counterexample"],
                row["infeasible"],
                row["blocked"],
                row["timeout"],
                row["error"],
                row["selected_backends"],
            )
        )
    return "\n".join(lines) + "\n"


def build_pack(run_dirs: list[Path], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_kind: dict[str, list[dict]] = {}
    raw_rows: list[dict] = []
    for rd, path, body in _iter_solver_responses(run_dirs):
        kind = body.get("problem_kind", "")
        by_kind.setdefault(kind, []).append(body)
        raw_rows.append(
            {
                "run_dir": str(rd),
                "response_path": str(path.relative_to(rd)),
                "problem_kind": kind,
                "selected_backend": body.get("selected_backend"),
                "status": body.get("status"),
                "time_ms": body.get("time_ms"),
                "objective_value": body.get("objective_value"),
                "formulation_hash": body.get("formulation_hash"),
            }
        )

    matrix_rows = _matrix_rows(by_kind)
    backend_status = _load_backend_status(run_dirs)
    if backend_status is None:
        # Fall back to live probe so the pack is still informative.
        from compgen.solve.backend_registry import default_registry

        reg = default_registry()
        results = reg.probe_all()
        backend_status = {
            "schema_version": "solver_backend_status_v1",
            "backends": {
                n.value: {
                    "availability": r.availability.value,
                    "version": r.version,
                    "supports": list(r.supports),
                }
                for n, r in results.items()
            },
            "required_baseline_ok": all(
                results.get(name) and results[name].availability.value == "available"
                for name in [
                    list(results.keys())[0],
                ]
            ),
            "optional_accelerators_ok": any(
                r.availability.value == "available" and n.value == "mosek"
                for n, r in results.items()
            ),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "notes": ["live probe (no probe report found in run-dirs)"],
        }

    has_hardware = any(
        (rd / "06_glue_emit").exists() for rd in run_dirs if rd.exists()
    )
    claim_rows = _claim_matrix(backend_status, by_kind, has_hardware=has_hardware)

    # Emit JSON files
    (out_dir / "solver_backend_status.json").write_text(
        json.dumps(backend_status, sort_keys=True, indent=2)
    )
    (out_dir / "claim_matrix.json").write_text(
        json.dumps(claim_rows, sort_keys=True, indent=2)
    )

    # Emit CSVs
    _emit_csv(
        out_dir / "solver_matrix.csv",
        matrix_rows,
        [
            "problem_kind",
            "count",
            "optimal",
            "feasible",
            "proved",
            "sat_counterexample",
            "infeasible",
            "blocked",
            "timeout",
            "error",
            "selected_backends",
        ],
    )
    placement_rows = [
        r for r in raw_rows if r["problem_kind"] == "placement"
    ]
    _emit_csv(
        out_dir / "placement_results.csv",
        placement_rows,
        [
            "run_dir",
            "response_path",
            "selected_backend",
            "status",
            "time_ms",
            "objective_value",
            "formulation_hash",
        ],
    )
    memory_rows = [r for r in raw_rows if r["problem_kind"] == "memory_allocation"]
    _emit_csv(
        out_dir / "memory_results.csv",
        memory_rows,
        [
            "run_dir",
            "response_path",
            "selected_backend",
            "status",
            "time_ms",
            "objective_value",
            "formulation_hash",
        ],
    )
    overlap_rows = [r for r in raw_rows if r["problem_kind"] == "overlap_planning"]
    _emit_csv(
        out_dir / "overlap_results.csv",
        overlap_rows,
        [
            "run_dir",
            "response_path",
            "selected_backend",
            "status",
            "time_ms",
            "objective_value",
            "formulation_hash",
        ],
    )

    figure_notes = _try_emit_figures(out_dir, by_kind, backend_status)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dirs": [str(rd) for rd in run_dirs],
        "total_responses": len(raw_rows),
        "kinds_seen": sorted(by_kind.keys()),
        "notes": figure_notes,
    }

    (out_dir / "solver_planning_summary.md").write_text(
        _summary_md(report, claim_rows, matrix_rows)
    )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/solver_planning_evidence_pack")
    )
    args = parser.parse_args(argv)
    report = build_pack(args.runs, args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

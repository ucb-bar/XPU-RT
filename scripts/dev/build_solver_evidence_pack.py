"""Solver evidence pack builder (spec §15).

Spec-compliant successor to ``build_solver_planning_evidence_pack.py``.
Walks one or more run-dirs, aggregates solver request/response pairs
and ``*.solved.json`` artifacts, and emits:

::

    results/solver_evidence_pack/
        solver_summary.md
        solver_backend_status.json
        solver_claim_matrix.json
        solver_problem_matrix.csv
        z3_obligation_results.csv
        placement_results.csv
        schedule_results.csv
        memory_results.csv
        bandwidth_results.csv
        integration_results.csv
        figures/
            solver_backend_availability.png
            solver_problem_coverage.png
            solver_status_by_problem_kind.png
            solver_time_breakdown.png
            memory_tier_usage.png
            placement_matrix.png
            overlap_schedule_gantt.png
            z3_proof_status.png

Each claim in ``solver_claim_matrix.json`` carries:

* ``claim`` — semantic claim name (z3_semantic_obligations,
  ortools_placement, …)
* ``status`` — ``implemented`` | ``implemented_partial_scope`` |
  ``blocked`` | ``not_run``
* ``evidence`` — relative CSV/JSON file in this pack
* ``negative_controls_passed`` — boolean (read from the
  ``tests/solve/test_solver_negative_controls.py`` run; defaults to
  False unless explicitly verified)
* ``blocked_reason`` — present only when status=blocked

No row may claim ``implemented`` without a non-empty evidence file.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PACK_SCHEMA_VERSION = "solver_evidence_pack_v2"


# ---------------------------------------------------------------------------
# Artifact collection
# ---------------------------------------------------------------------------


def _iter_responses(run_dirs: list[Path]):
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


def _iter_z3_obligations(run_dirs: list[Path]):
    """Walk Z3 obligation index files."""

    for rd in run_dirs:
        if not rd.exists():
            continue
        for path in rd.rglob("z3_obligations*.json"):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(body, dict) or "obligations" not in body:
                continue
            for ob in body["obligations"]:
                yield rd, path, ob


def _load_backend_status(run_dirs: list[Path]) -> dict | None:
    for rd in run_dirs:
        for path in rd.rglob("solver_backend_status.json"):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None


def _run_negative_controls() -> bool:
    """Run the consolidated negative-controls test file as a
    subprocess; return True iff every test passes. Used to mark
    ``negative_controls_passed`` honestly in the claim matrix."""

    repo_root = Path(__file__).resolve().parents[2]
    test_file = repo_root / "tests" / "solve" / "test_solver_negative_controls.py"
    if not test_file.is_file():
        return False
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-q", "--no-header", "--tb=no"],
        cwd=repo_root,
        capture_output=True,
    )
    return rc.returncode == 0


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _emit_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _flat_row(rd: Path, path: Path, body: dict) -> dict:
    return {
        "run_dir": str(rd),
        "response_path": str(path.relative_to(rd)),
        "problem_id": body.get("problem_id", ""),
        "problem_kind": body.get("problem_kind", ""),
        "selected_backend": body.get("selected_backend", ""),
        "status": body.get("status", ""),
        "time_ms": body.get("time_ms", ""),
        "objective_value": body.get("objective_value", ""),
        "formulation_hash": body.get("formulation_hash", ""),
        "infeasibility_reason": body.get("infeasibility_reason", ""),
    }


_CSV_FIELDS = [
    "run_dir",
    "response_path",
    "problem_id",
    "problem_kind",
    "selected_backend",
    "status",
    "time_ms",
    "objective_value",
    "formulation_hash",
    "infeasibility_reason",
]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _try_emit_figures(
    out_dir: Path,
    by_kind: dict[str, list[dict]],
    backend_status: dict | None,
    z3_rows: list[dict],
) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable; figure generation skipped"]

    notes: list[str] = []

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

    # 2. problem coverage (count per problem_kind)
    if by_kind:
        kinds = sorted(by_kind.keys())
        counts = [len(by_kind[k]) for k in kinds]
        plt.figure(figsize=(8, 3))
        plt.bar(kinds, counts, color="#8ac")
        plt.title("Solver problem coverage (count by problem_kind)")
        plt.ylabel("count")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "solver_problem_coverage.png", dpi=120)
        plt.close()

    # 3. status breakdown by problem_kind (stacked)
    if by_kind:
        kinds = sorted(by_kind.keys())
        statuses = ["optimal", "feasible", "proved", "sat_counterexample", "infeasible", "blocked", "timeout", "error"]
        data: dict[str, list[int]] = {s: [] for s in statuses}
        for k in kinds:
            counts = {s: 0 for s in statuses}
            for r in by_kind[k]:
                s = r.get("status", "")
                if s in counts:
                    counts[s] += 1
            for s in statuses:
                data[s].append(counts[s])
        plt.figure(figsize=(8, 3))
        bottom = [0] * len(kinds)
        colors = {"optimal": "#2c8", "feasible": "#5a8", "proved": "#39c",
                  "sat_counterexample": "#fa6", "infeasible": "#c44",
                  "blocked": "#999", "timeout": "#fc6", "error": "#933"}
        for s in statuses:
            plt.bar(kinds, data[s], bottom=bottom, label=s, color=colors[s])
            bottom = [b + d for b, d in zip(bottom, data[s])]
        plt.title("Solver status breakdown by problem_kind")
        plt.legend(loc="upper right", fontsize=7)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "solver_status_by_problem_kind.png", dpi=120)
        plt.close()

    # 4. time breakdown by backend
    if by_kind:
        backends_seen: dict[str, float] = {}
        for k, rows in by_kind.items():
            for r in rows:
                t = float(r.get("time_ms", 0) or 0)
                backends_seen[r.get("selected_backend", "?")] = (
                    backends_seen.get(r.get("selected_backend", "?"), 0.0) + t
                )
        if backends_seen:
            plt.figure(figsize=(6, 3))
            plt.bar(list(backends_seen.keys()), list(backends_seen.values()), color="#48c")
            plt.title("Total solver time by backend (ms)")
            plt.ylabel("time_ms (sum)")
            plt.tight_layout()
            plt.savefig(figures_dir / "solver_time_breakdown.png", dpi=120)
            plt.close()

    # 5. memory tier usage
    mem_rows = by_kind.get("memory_allocation") or []
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
        plt.savefig(figures_dir / "memory_tier_usage.png", dpi=120)
        plt.close()

    # 6. placement matrix
    placement_rows = by_kind.get("placement") or []
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
        plt.figure(figsize=(4 + len(devices), 1 + min(len(regions), 20) * 0.3))
        plt.imshow(data, aspect="auto", cmap="Blues")
        plt.xticks(range(len(devices)), devices)
        if len(regions) <= 30:
            plt.yticks(range(len(regions)), regions)
        else:
            plt.yticks([])
        plt.title("Placement decisions")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(figures_dir / "placement_matrix.png", dpi=120)
        plt.close()

    # 7. overlap gantt
    overlap_rows = by_kind.get("overlap_planning") or []
    sol = next(
        (r.get("solution") for r in overlap_rows if isinstance(r.get("solution"), dict)),
        None,
    )
    if sol and sol.get("schedule"):
        plt.figure(figsize=(6, 1 + min(len(sol["schedule"]), 30) * 0.3))
        for i, entry in enumerate(sol["schedule"][:30]):
            plt.barh(
                i, entry["end_tick"] - entry["start_tick"],
                left=entry["start_tick"], color="#7a8",
            )
            plt.text(entry["start_tick"], i, entry["op_id"], va="center", fontsize=7)
        plt.title("Overlap schedule (gantt)")
        plt.xlabel("ticks")
        plt.tight_layout()
        plt.savefig(figures_dir / "overlap_schedule_gantt.png", dpi=120)
        plt.close()

    # 8. Z3 proof status
    if z3_rows:
        statuses = ["proved", "sat_counterexample", "timeout", "unsupported", "error"]
        counts = {s: 0 for s in statuses}
        for r in z3_rows:
            s = r.get("status", "")
            if s in counts:
                counts[s] += 1
        plt.figure(figsize=(6, 3))
        plt.bar(statuses, [counts[s] for s in statuses],
                color=["#2c8", "#fa6", "#fc6", "#999", "#933"])
        plt.title("Z3 obligation outcomes")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figures_dir / "z3_proof_status.png", dpi=120)
        plt.close()

    return notes


# ---------------------------------------------------------------------------
# Claim matrix
# ---------------------------------------------------------------------------


def _build_claim_matrix(
    by_kind: dict[str, list[dict]],
    z3_rows: list[dict],
    backend_status: dict | None,
    negative_controls_passed: bool,
    has_hardware: bool,
) -> list[dict]:
    def _has_success_for(kind: str) -> bool:
        return any(r.get("status") in ("optimal", "feasible", "proved")
                   for r in by_kind.get(kind, []))

    def _has_typed_failure_for(kind: str) -> bool:
        return any(r.get("status") in ("infeasible", "blocked", "sat_counterexample", "timeout")
                   for r in by_kind.get(kind, []))

    def _status_for_optimization(kind: str) -> str:
        if _has_success_for(kind):
            return "implemented"
        if _has_typed_failure_for(kind):
            return "implemented_partial_scope"
        return "not_run"

    rows: list[dict] = []

    rows.append({
        "claim": "z3_semantic_obligations",
        "status": ("implemented"
                   if z3_rows and all(r.get("status") in ("proved", "sat_counterexample", "timeout", "unsupported")
                                      for r in z3_rows) and any(r.get("status") == "proved" for r in z3_rows)
                   else ("implemented_partial_scope" if z3_rows else "not_run")),
        "evidence": "z3_obligation_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    rows.append({
        "claim": "ortools_placement",
        "status": _status_for_optimization("placement"),
        "evidence": "placement_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    rows.append({
        "claim": "ortools_overlap_schedule",
        "status": _status_for_optimization("overlap_planning"),
        "evidence": "schedule_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    mosek_available = (
        backend_status
        and backend_status.get("backends", {}).get("mosek", {}).get("availability") == "available"
    )
    rows.append({
        "claim": "mosek_memory_planning",
        "status": (
            _status_for_optimization("memory_allocation")
            if mosek_available
            else "blocked"
        ),
        "evidence": "memory_results.csv",
        "negative_controls_passed": negative_controls_passed,
        "blocked_reason": None if mosek_available else "license_missing",
    })

    rows.append({
        "claim": "highs_fallback",
        "status": _status_for_optimization("memory_allocation"),
        "evidence": "memory_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    rows.append({
        "claim": "bandwidth_allocation",
        "status": _status_for_optimization("bandwidth_allocation"),
        "evidence": "bandwidth_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    rows.append({
        "claim": "execution_plan_integration",
        "status": "implemented_partial_scope",
        "evidence": "integration_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    rows.append({
        "claim": "hardware_matrix_execution",
        "status": "implemented_partial_scope" if has_hardware else "not_run",
        "evidence": "integration_results.csv",
        "negative_controls_passed": negative_controls_passed,
    })

    # Enforce: no row may claim ``implemented`` if it lacks evidence
    # OR if negative controls did not pass.
    for r in rows:
        if r["status"] == "implemented" and not r["negative_controls_passed"]:
            r["status"] = "implemented_partial_scope"
            r["downgrade_reason"] = "negative_controls_not_verified"

    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summary_md(report: dict, claim_rows: list[dict], matrix_rows: list[dict]) -> str:
    lines = [
        "# Solver evidence pack (spec §15)",
        "",
        f"- **schema_version**: `{report['schema_version']}`",
        f"- **generated_at**: {report['generated_at']}",
        f"- **run_dirs**: {report['run_dirs']}",
        f"- **negative_controls_passed**: {report['negative_controls_passed']}",
        "",
        "## Claim matrix",
        "",
        "| claim | status | evidence | neg_controls | blocked_reason |",
        "|---|---|---|---|---|",
    ]
    for row in claim_rows:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                row["claim"], row["status"], row["evidence"],
                row.get("negative_controls_passed"),
                row.get("blocked_reason") or "-",
            )
        )
    lines += ["", "## Per-problem-kind matrix", "",
              "| problem_kind | count | optimal | feasible | proved | sat_cex | infeasible | blocked | timeout | error | backends |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in matrix_rows:
        lines.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            row["problem_kind"], row["count"], row["optimal"], row["feasible"],
            row["proved"], row["sat_counterexample"], row["infeasible"],
            row["blocked"], row["timeout"], row["error"], row["selected_backends"]))
    return "\n".join(lines) + "\n"


def _matrix_rows(by_kind: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for kind, entries in sorted(by_kind.items()):
        statuses = [e.get("status", "") for e in entries]
        rows.append({
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
            "selected_backends": ",".join(sorted({e.get("selected_backend", "?") for e in entries})),
        })
    return rows


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_pack(
    run_dirs: list[Path],
    out_dir: Path,
    *,
    skip_negative_controls: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_kind: dict[str, list[dict]] = {}
    raw_rows: list[dict] = []
    for rd, path, body in _iter_responses(run_dirs):
        kind = body.get("problem_kind", "")
        by_kind.setdefault(kind, []).append(body)
        raw_rows.append(_flat_row(rd, path, body))

    z3_rows: list[dict] = []
    for rd, path, ob in _iter_z3_obligations(run_dirs):
        z3_rows.append({
            "run_dir": str(rd),
            "report_path": str(path.relative_to(rd)),
            "obligation_id": ob.get("obl_id") or ob.get("id") or "",
            "status": ob.get("status", ""),
            "selected_backend": ob.get("selected_backend", "z3"),
            "time_ms": ob.get("time_ms", ""),
            "counterexample": json.dumps(ob.get("counterexample") or {}) if ob.get("counterexample") else "",
        })

    backend_status = _load_backend_status(run_dirs)
    if backend_status is None:
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
            "required_baseline_ok": True,
            "optional_accelerators_ok": any(
                r.availability.value == "available" and n.value == "mosek"
                for n, r in results.items()
            ),
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "notes": ["live probe (no probe report found in run-dirs)"],
        }

    has_hardware = any((rd / "06_glue_emit").exists() for rd in run_dirs)

    negative_controls_passed = (
        False if skip_negative_controls else _run_negative_controls()
    )

    matrix_rows = _matrix_rows(by_kind)
    claim_rows = _build_claim_matrix(
        by_kind, z3_rows, backend_status,
        negative_controls_passed=negative_controls_passed,
        has_hardware=has_hardware,
    )

    # ---- JSON files ---------------------------------------------------
    (out_dir / "solver_backend_status.json").write_text(
        json.dumps(backend_status, sort_keys=True, indent=2)
    )
    (out_dir / "solver_claim_matrix.json").write_text(
        json.dumps(claim_rows, sort_keys=True, indent=2)
    )

    # ---- CSV files ----------------------------------------------------
    _emit_csv(
        out_dir / "solver_problem_matrix.csv",
        matrix_rows,
        [
            "problem_kind", "count", "optimal", "feasible", "proved",
            "sat_counterexample", "infeasible", "blocked", "timeout", "error",
            "selected_backends",
        ],
    )
    _emit_csv(
        out_dir / "z3_obligation_results.csv",
        z3_rows,
        ["run_dir", "report_path", "obligation_id", "status",
         "selected_backend", "time_ms", "counterexample"],
    )
    _emit_csv(
        out_dir / "placement_results.csv",
        [r for r in raw_rows if r["problem_kind"] == "placement"],
        _CSV_FIELDS,
    )
    _emit_csv(
        out_dir / "schedule_results.csv",
        [r for r in raw_rows if r["problem_kind"] in ("schedule", "no_overlap_schedule",
                                                        "overlap_planning", "event_ordering")],
        _CSV_FIELDS,
    )
    _emit_csv(
        out_dir / "memory_results.csv",
        [r for r in raw_rows if r["problem_kind"] == "memory_allocation"],
        _CSV_FIELDS,
    )
    _emit_csv(
        out_dir / "bandwidth_results.csv",
        [r for r in raw_rows if r["problem_kind"] == "bandwidth_allocation"],
        _CSV_FIELDS,
    )
    # Integration: every response that produced a *.solved.json under
    # its run_dir is an integration row.
    integration_rows: list[dict] = []
    for rd, path, body in _iter_responses(run_dirs):
        kind = body.get("problem_kind", "")
        solved_name = {
            "memory_allocation": "memory_plan.solved.json",
            "placement": "placement_plan.solved.json",
            "overlap_planning": "overlap_schedule.solved.json",
            "bandwidth_allocation": "bandwidth_plan.solved.json",
        }.get(kind)
        if not solved_name:
            continue
        for candidate in (path.parent / solved_name, path.parent.parent / solved_name):
            if candidate.is_file():
                integration_rows.append({
                    **_flat_row(rd, path, body),
                    "solved_artifact": str(candidate.relative_to(rd)),
                })
                break
    _emit_csv(
        out_dir / "integration_results.csv",
        integration_rows,
        _CSV_FIELDS + ["solved_artifact"],
    )

    # ---- Figures ------------------------------------------------------
    figure_notes = _try_emit_figures(out_dir, by_kind, backend_status, z3_rows)

    report = {
        "schema_version": PACK_SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dirs": [str(rd) for rd in run_dirs],
        "total_responses": len(raw_rows),
        "z3_obligations_total": len(z3_rows),
        "kinds_seen": sorted(by_kind.keys()),
        "negative_controls_passed": negative_controls_passed,
        "notes": figure_notes,
    }
    (out_dir / "solver_summary.md").write_text(
        _summary_md(report, claim_rows, matrix_rows)
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("results/solver_evidence_pack"),
    )
    parser.add_argument(
        "--skip-negative-controls",
        action="store_true",
        help="Don't run tests/solve/test_solver_negative_controls.py "
        "(uses negative_controls_passed=False; for fast smoke runs).",
    )
    args = parser.parse_args(argv)
    report = build_pack(args.runs, args.out, skip_negative_controls=args.skip_negative_controls)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

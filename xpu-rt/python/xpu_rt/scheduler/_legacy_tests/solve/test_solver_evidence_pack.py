"""Spec §15: solver evidence pack.

The pack must emit JSON + CSV + Markdown + figures matching the
spec layout, and the claim matrix may NOT mark a row
``implemented`` unless the underlying evidence file is non-empty
AND negative controls have been verified.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _emit_real_run(run_dir: Path) -> None:
    """Run the demo to produce real solver artifacts under ``run_dir``."""

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dev" / "run_solver_planning_demo.py"),
         "--out", str(run_dir)],
        check=True,
    )


def _build_pack(run_dirs: list[Path], out: Path, *, skip_neg: bool = True) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dev" / "build_solver_evidence_pack.py"),
        "--out", str(out),
        "--runs",
    ]
    cmd.extend(str(r) for r in run_dirs)
    if skip_neg:
        cmd.append("--skip-negative-controls")
    subprocess.run(cmd, check=True)


def test_evidence_pack_emits_spec_layout(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out)

    # Required JSON / Markdown / CSV files (spec §15).
    expected = [
        "solver_summary.md",
        "solver_backend_status.json",
        "solver_claim_matrix.json",
        "solver_problem_matrix.csv",
        "z3_obligation_results.csv",
        "placement_results.csv",
        "schedule_results.csv",
        "memory_results.csv",
        "bandwidth_results.csv",
        "integration_results.csv",
    ]
    for name in expected:
        assert (out / name).is_file(), f"missing pack artifact: {name}"

    # Required figures (spec §15, at least these 8).
    figures = {p.name for p in (out / "figures").glob("*.png")}
    must_have = {
        "solver_backend_availability.png",
        "solver_problem_coverage.png",
        "solver_status_by_problem_kind.png",
        "solver_time_breakdown.png",
        "memory_tier_usage.png",
        "placement_matrix.png",
        "overlap_schedule_gantt.png",
        "z3_proof_status.png",
    }
    missing = must_have - figures
    assert not missing, f"missing figures: {missing}"


def test_claim_matrix_carries_required_fields(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out)

    claim = json.loads((out / "solver_claim_matrix.json").read_text())
    assert isinstance(claim, list) and claim
    for row in claim:
        assert "claim" in row
        assert "status" in row
        assert "evidence" in row
        assert "negative_controls_passed" in row
        assert row["status"] in {
            "implemented",
            "implemented_partial_scope",
            "blocked",
            "not_run",
        }


def test_claim_matrix_downgrades_implemented_without_neg_controls(tmp_path: Path):
    """The pack must NOT mark any row ``implemented`` when negative
    controls were skipped (or failed)."""

    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out, skip_neg=True)
    claim = json.loads((out / "solver_claim_matrix.json").read_text())
    for row in claim:
        if row["status"] == "implemented":
            # implemented is only allowed when negative_controls_passed is True
            assert row["negative_controls_passed"], (
                f"row {row['claim']} is implemented without neg-controls: {row}"
            )


def test_problem_matrix_csv_has_required_columns(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out)

    with (out / "solver_problem_matrix.csv").open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = set(reader.fieldnames or [])
    for c in (
        "problem_kind", "count", "optimal", "feasible", "proved",
        "sat_counterexample", "infeasible", "blocked", "timeout", "error",
        "selected_backends",
    ):
        assert c in cols, f"problem_matrix missing column {c!r}"
    assert rows, "problem_matrix has no rows"


def test_z3_obligation_results_includes_real_z3_runs(tmp_path: Path):
    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out)

    with (out / "z3_obligation_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "z3_obligation_results.csv is empty; spec requires real Z3 evidence"
    for row in rows:
        assert row["selected_backend"] == "z3"
        assert row["status"] in {"proved", "sat_counterexample", "timeout", "unsupported", "error"}


def test_missing_run_dir_produces_partial_scope(tmp_path: Path):
    """When a run-dir has no solver artifacts, the pack still builds
    but every per-kind row degrades to ``not_run``."""

    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "pack"
    _build_pack([empty], out)
    claim = json.loads((out / "solver_claim_matrix.json").read_text())
    # All per-solver claim rows must be not_run.
    for row in claim:
        if row["claim"] in ("z3_semantic_obligations", "ortools_placement",
                            "ortools_overlap_schedule", "highs_fallback",
                            "bandwidth_allocation"):
            assert row["status"] in ("not_run", "blocked")


def test_integration_results_csv_traces_solved_artifacts(tmp_path: Path):
    """Every integration row points at a real ``*.solved.json`` next
    to its response."""

    run = tmp_path / "run"
    out = tmp_path / "pack"
    _emit_real_run(run)
    _build_pack([run], out)

    with (out / "integration_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r.get("solved_artifact"), f"integration row missing solved_artifact: {r}"
        solved = run / r["solved_artifact"]
        assert solved.is_file(), f"solved artifact does not exist: {solved}"

"""solver-planning evidence pack tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_evidence_pack_builds_from_run_dirs(tmp_path):
    """End-to-end: run the demo, build the pack, check all artifacts."""

    import subprocess
    import sys

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "pack"

    subprocess.run(
        [sys.executable, "scripts/dev/run_solver_planning_demo.py", "--out", str(run_dir)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/dev/build_solver_planning_evidence_pack.py",
            "--runs",
            str(run_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
    )

    assert (out_dir / "solver_planning_summary.md").is_file()
    assert (out_dir / "solver_backend_status.json").is_file()
    assert (out_dir / "claim_matrix.json").is_file()
    assert (out_dir / "solver_matrix.csv").is_file()
    assert (out_dir / "memory_results.csv").is_file()
    assert (out_dir / "placement_results.csv").is_file()
    assert (out_dir / "overlap_results.csv").is_file()


def test_evidence_pack_figures_emitted(tmp_path):
    pytest.importorskip("matplotlib")
    import subprocess
    import sys

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "pack"
    subprocess.run(
        [sys.executable, "scripts/dev/run_solver_planning_demo.py", "--out", str(run_dir)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/dev/build_solver_planning_evidence_pack.py",
            "--runs",
            str(run_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
    )
    figures = list((out_dir / "figures").glob("*.png"))
    assert len(figures) >= 4, f"expected ≥4 figures, found {len(figures)}: {figures}"
    for f in figures:
        assert f.stat().st_size > 0, f"empty figure: {f}"


def test_claim_matrix_has_all_required_rows(tmp_path):
    import subprocess
    import sys

    run_dir = tmp_path / "run"
    out_dir = tmp_path / "pack"
    subprocess.run(
        [sys.executable, "scripts/dev/run_solver_planning_demo.py", "--out", str(run_dir)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/dev/build_solver_planning_evidence_pack.py",
            "--runs",
            str(run_dir),
            "--out",
            str(out_dir),
        ],
        check=True,
    )
    claim = json.loads((out_dir / "claim_matrix.json").read_text())
    rows = {r["row"] for r in claim}
    assert rows == {
        "solver_backend_probe",
        "z3_semantic_obligations",
        "memory_planning",
        "placement_planning",
        "overlap_planning",
        "execution_plan_integration",
        "emitted_glue_differential",
        "hardware_matrix_execution",
    }


def test_missing_artifact_produces_typed_partial_scope(tmp_path):
    """When the input run-dir contains no solver responses, the pack
    still builds but rows are honest (``not_run``)."""

    import subprocess
    import sys

    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    out_dir = tmp_path / "pack"
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/dev/build_solver_planning_evidence_pack.py",
            "--runs",
            str(empty_run),
            "--out",
            str(out_dir),
        ],
    )
    assert rc.returncode == 0
    claim = json.loads((out_dir / "claim_matrix.json").read_text())
    statuses = {r["row"]: r["status"] for r in claim}
    # Backend probe falls back to live probe; can be implemented.
    # All consumer rows must be not_run.
    for row in (
        "z3_semantic_obligations",
        "memory_planning",
        "placement_planning",
        "overlap_planning",
    ):
        assert statuses[row] == "not_run", f"{row} status: {statuses[row]}"


def test_solver_gates_audit_real_run(tmp_path):
    """All 5 solver gates pass on a fresh demo run-dir."""

    import subprocess
    import sys

    run_dir = tmp_path / "run"
    subprocess.run(
        [sys.executable, "scripts/dev/run_solver_planning_demo.py", "--out", str(run_dir)],
        check=True,
    )

    from compgen.audit.solver_gates import all_solver_gates

    gates = all_solver_gates(run_dir=run_dir)
    failed = [g for g in gates if g.status == "fail"]
    assert not failed, f"failing gates: {failed}"
    assert len(gates) == 5

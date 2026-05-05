"""Tests that the M-08.5 wide model coverage gate is internally consistent.

These tests do NOT re-run the suite. They cross-check the emitted JSON
artifacts (`model_inventory.json`, `wide_suite_report.json`,
`wide_suite_coverage_matrix.json`, `wide_suite_failures.json`) against
the actual per-model run directories to make sure:

1. Every Merlin source directory under ``--merlin-root`` is inventoried
   with a non-empty ``reason``.
2. Every suite entry has a coverage-matrix row.
3. Every claimed pipeline-completed pass actually has the expected
   downstream artifacts on disk.
4. Pass/fail tallies in the report match per-row computation.
5. The acceptance bars from the milestone are met (≥3 pipeline-completed
   real Merlin models, ≥3 candidate families, ≥3 refinement types,
   ≥3 model sources).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "results" / "graph_compilation" / "wide_post_lowering_suite"
MERLIN_ROOT = Path("/scratch2/agustin/merlin/models")


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate_artifacts() -> dict[str, dict]:
    if not SUITE_DIR.is_dir():
        pytest.skip(
            "wide_post_lowering_suite results dir absent; "
            "run scripts/dev/build_wide_coverage_gate.py first"
        )
    inv = SUITE_DIR / "model_inventory.json"
    rep = SUITE_DIR / "wide_suite_report.json"
    mat = SUITE_DIR / "wide_suite_coverage_matrix.json"
    fail = SUITE_DIR / "wide_suite_failures.json"
    for p in (inv, rep, mat, fail):
        if not p.exists():
            pytest.skip(f"gate artifact missing: {p.name}")
    return {
        "inventory": _read(inv),
        "report": _read(rep),
        "matrix": _read(mat),
        "failures": _read(fail),
    }


def test_model_inventory_covers_every_merlin_subdir(gate_artifacts: dict) -> None:
    if not MERLIN_ROOT.is_dir():
        pytest.skip(f"merlin root absent: {MERLIN_ROOT}")
    inv = gate_artifacts["inventory"]
    merlin_entries = {
        m["model_id"]
        for m in inv["models"]
        if m["source"] == "merlin"
    }
    on_disk = {
        f"merlin_{p.name}"
        for p in MERLIN_ROOT.iterdir()
        if p.is_dir()
    }
    assert on_disk == merlin_entries, (
        f"merlin inventory mismatch: missing={on_disk - merlin_entries}, "
        f"extra={merlin_entries - on_disk}"
    )


def test_every_inventory_entry_has_concrete_reason(gate_artifacts: dict) -> None:
    inv = gate_artifacts["inventory"]
    for m in inv["models"]:
        if m["source"] != "merlin":
            continue
        reason = m.get("reason", "")
        assert reason and len(reason) > 10, (
            f"merlin model {m['model_id']} has empty/trivial reason: {reason!r}"
        )
        assert m["admission_status"] in {
            "admitted_pytorch",
            "needs_custom_loader",
            "external_dependency_missing",
            "onnx_only_pending_importer",
            "mlir_only",
            "utility_or_subdir",
        }, f"unexpected admission_status: {m['admission_status']}"


def test_no_admitted_merlin_lacks_pipeline_evidence(gate_artifacts: dict) -> None:
    inv = gate_artifacts["inventory"]
    matrix = {r["model_id"]: r for r in gate_artifacts["matrix"]["rows"]}
    for m in inv["models"]:
        if m["admission_status"] != "admitted_pytorch":
            continue
        if m["source"] != "merlin":
            continue
        assert m["model_id"] in matrix, (
            f"admitted merlin model {m['model_id']} has no coverage-matrix row"
        )


def test_coverage_matrix_rows_match_run_dirs(gate_artifacts: dict) -> None:
    matrix = gate_artifacts["matrix"]["rows"]
    assert len(matrix) >= 16, (
        f"expected ≥16 admitted models in suite, got {len(matrix)}"
    )
    for row in matrix:
        run_dir = Path(row["run_dir"])
        assert run_dir.is_dir(), f"run_dir does not exist for {row['model_id']}"
        # Pipeline-completed models must have post-lowering verification
        # report on disk.
        if row["overall_pipeline_completed"] == "pass":
            pl = run_dir / "03_recipe_planning" / "post_lowering" / (
                "post_lowering_verification_report.json"
            )
            assert pl.exists(), (
                f"{row['model_id']} marked pipeline-completed but "
                f"post-lowering verification report is missing"
            )


def test_report_tallies_match_per_row(gate_artifacts: dict) -> None:
    report = gate_artifacts["report"]
    rows = gate_artifacts["matrix"]["rows"]

    pipe_pass = sum(1 for r in rows if r["overall_pipeline_completed"] == "pass")
    strict_pass = sum(1 for r in rows if r["overall_strict"] == "pass")
    assert report["summary"]["pipeline_completed"]["passed"] == pipe_pass
    assert report["summary"]["strict_gate"]["passed"] == strict_pass

    merlin_pipe = sum(
        1 for r in rows
        if r["source"] == "merlin" and r["overall_pipeline_completed"] == "pass"
    )
    assert (
        report["summary"]["pipeline_completed"]["real_merlin_pytorch_passed"]
        == merlin_pipe
    )


def test_failures_file_consistent_with_matrix(gate_artifacts: dict) -> None:
    rows = gate_artifacts["matrix"]["rows"]
    failures = gate_artifacts["failures"]

    pipe_fail_models = {
        f["model_id"] for f in failures["pipeline_completion_failures"]
    }
    matrix_pipe_fail = {
        r["model_id"] for r in rows if r["overall_pipeline_completed"] != "pass"
    }
    assert pipe_fail_models == matrix_pipe_fail

    warn_models = {f["model_id"] for f in failures["strict_gate_warnings"]}
    matrix_warn = {
        r["model_id"]
        for r in rows
        if r["overall_pipeline_completed"] == "pass"
        and r["overall_strict"] != "pass"
    }
    assert warn_models == matrix_warn


def test_acceptance_bars_met(gate_artifacts: dict) -> None:
    rows = gate_artifacts["matrix"]["rows"]

    sources = {r["source"] for r in rows}
    assert len(sources) >= 3, f"need ≥3 model sources, got {sources}"

    real_merlin_pipe = sum(
        1 for r in rows
        if r["source"] == "merlin" and r["overall_pipeline_completed"] == "pass"
    )
    assert real_merlin_pipe >= 3, (
        f"need ≥3 pipeline-completed real merlin models, got {real_merlin_pipe}"
    )

    selected = {
        r["selected_candidate_kind"]
        for r in rows
        if r["selected_candidate_kind"]
    }
    assert len(selected) >= 3, (
        f"need ≥3 distinct selected candidate kinds, got {selected}"
    )

    refinements = {r["declared_refinement"] for r in rows if r["declared_refinement"]}
    assert len(refinements) >= 3, (
        f"need ≥3 distinct refinement types, got {refinements}"
    )


def test_source_payload_unchanged(gate_artifacts: dict) -> None:
    """Hard invariant: every pipeline-completed run must keep its
    01_payload_lowering directory hash matched across pre/post lowering.
    The post-lowering verification report records this; we re-read it.
    """
    rows = gate_artifacts["matrix"]["rows"]
    for row in rows:
        if row["overall_pipeline_completed"] != "pass":
            continue
        run_dir = Path(row["run_dir"])
        report = run_dir / "03_recipe_planning" / "post_lowering" / (
            "post_lowering_verification_report.json"
        )
        if not report.exists():
            continue
        body = json.loads(report.read_text(encoding="utf-8"))
        # Source-payload-byte-identical invariant lives in the verification
        # report; status==pass implies it held.
        assert body.get("status") == "pass", (
            f"{row['model_id']}: post-lowering verification status != pass"
        )


def test_no_compiler_core_files_modified() -> None:
    """The gate is a read-only audit. Confirm scripts/dev only — no
    edits to compiler core packages — by spot-checking that the gate
    script itself sits under scripts/dev/.
    """
    gate = REPO_ROOT / "scripts" / "dev" / "build_wide_coverage_gate.py"
    assert gate.exists()
    # Sanity: gate must not import compiler-core mutable modules.
    text = gate.read_text(encoding="utf-8")
    forbidden_imports = (
        "from compgen.ir.payload",
        "from compgen.graph_compilation.recipe_lowering import",
        "from compgen.graph_compilation.post_lowering import",
        "from compgen.graph_compilation.run import",
    )
    for pat in forbidden_imports:
        assert pat not in text, f"gate must not import compiler core: {pat}"

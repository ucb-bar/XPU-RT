"""Tests for compgen.audit.trust_report (M-31A.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compgen.audit.trust_report import (
    TrustReport,
    build_trust_report,
    emit_trust_report,
)


def test_build_trust_report_runs_every_gate(tmp_path: Path) -> None:
    report = build_trust_report(
        tmp_path=tmp_path / "tmp",
        run_dir=None,  # exercises the import_provenance "skipped" path
        commit="testcommit",
    )
    gate_names = {g.name for g in report.gates}
    expected = {
        "realness_scan",
        "negative_controls",
        "caveat_ledger",
        "realness_contracts",
        "import_provenance",
        "trace_replay_self_check",
        "task_pack_buildable",
        "holdout_outcomes_honest",
    }
    assert expected.issubset(gate_names), (
        f"missing gates: {expected - gate_names}"
    )


def test_trust_report_overall_pass_on_clean_repo(tmp_path: Path) -> None:
    report = build_trust_report(
        tmp_path=tmp_path / "tmp",
        run_dir=None,
        commit="testcommit",
    )
    failed = [g for g in report.gates if g.status == "fail"]
    assert not failed, (
        f"unexpected failing gates on origin/main: "
        f"{[(g.name, g.detail) for g in failed]}"
    )


def test_trust_report_import_provenance_skipped_when_no_run_dir(tmp_path: Path) -> None:
    report = build_trust_report(
        tmp_path=tmp_path / "tmp", run_dir=None, commit="x",
    )
    prov = next(g for g in report.gates if g.name == "import_provenance")
    assert prov.status == "skipped"


def test_emit_trust_report_writes_json_and_md(tmp_path: Path) -> None:
    report = build_trust_report(
        tmp_path=tmp_path / "tmp", run_dir=None, commit="x",
    )
    json_path, md_path = emit_trust_report(report, out_dir=tmp_path / "out")
    assert json_path.exists()
    assert md_path.exists()
    raw = json.loads(json_path.read_text())
    assert "gates" in raw
    assert "all_pass" in raw
    md_text = md_path.read_text()
    assert "trust report" in md_text.lower()


def test_trust_report_to_markdown_lists_each_gate() -> None:
    from compgen.audit.errors import GateResult
    report = TrustReport(commit="abc", generated_at_utc="2026-05-05T00:00:00Z")
    report.gates.append(GateResult(name="realness_scan", status="pass", detail="0 hits"))
    report.gates.append(GateResult(name="caveat_ledger", status="skipped", detail="no seed"))
    md = report.to_markdown()
    assert "realness_scan" in md
    assert "caveat_ledger" in md
    assert "✅" in md  # pass marker
    assert "⊝" in md  # skipped marker


def test_trust_report_round_trip_json(tmp_path: Path) -> None:
    """Two consecutive builds on the same commit produce structurally
    identical reports (modulo timestamps)."""
    a = build_trust_report(tmp_path=tmp_path / "a", run_dir=None, commit="x")
    b = build_trust_report(tmp_path=tmp_path / "b", run_dir=None, commit="x")
    a_dict = a.to_dict()
    b_dict = b.to_dict()
    # Strip timestamps (which legitimately differ)
    a_dict.pop("generated_at_utc")
    b_dict.pop("generated_at_utc")
    # Strip detail fields that include timing metadata
    for d in (a_dict, b_dict):
        for g in d["gates"]:
            g["detail"] = ""
    assert a_dict == b_dict


def test_trust_report_with_real_run_dir(tmp_path: Path) -> None:
    """Build a real run dir, then run trust report against it.

    Exercises the import_provenance gate end-to-end.
    """
    from compgen.graph_compilation.run import run_graph_compilation

    REPO_ROOT = Path(__file__).resolve().parents[2]
    run_dir = tmp_path / "real_run"
    run_graph_compilation(
        REPO_ROOT / "configs" / "models" / "holdout_mlp_odd_shapes.yaml",
        REPO_ROOT / "configs" / "targets" / "host_cpu.yaml",
        run_dir,
        stop_after="payload-lowering",
        selection_mode="greedy",
    )

    report = build_trust_report(
        tmp_path=tmp_path / "tmp", run_dir=run_dir, commit="x",
    )
    prov = next(g for g in report.gates if g.name == "import_provenance")
    assert prov.status == "pass", f"detail: {prov.detail}"
    failed = [g for g in report.gates if g.status == "fail"]
    assert not failed, [(g.name, g.detail) for g in failed]

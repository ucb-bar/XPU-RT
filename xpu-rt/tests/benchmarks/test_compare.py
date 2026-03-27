"""Tests for benchmark comparison and export."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmarks.compare import ablation_table, export_csv, load_all_results, summary_table
from benchmarks.record import RunRecord


def test_summary_table() -> None:
    r = RunRecord(model_name="mlp", target_name="cuda")
    r.total_compile_time_ms = 100.0
    r.eqsat.ops_reduction_pct = 15.0
    r.eqsat.changed = True
    r.recipe.total_recipe_ops = 20
    r.verification.overall_status = "pass"
    table = summary_table([r])
    assert "mlp" in table
    assert "cuda" in table
    assert "pass" in table


def test_summary_table_empty() -> None:
    table = summary_table([])
    assert "No records" in table


def test_ablation_table() -> None:
    r = RunRecord(model_name="mlp", target_name="cuda", config={"ablation": "no_eqsat"})
    r.total_compile_time_ms = 80.0
    table = ablation_table([r])
    assert "no_eqsat" in table


def test_export_csv() -> None:
    r = RunRecord(model_name="mlp", target_name="cuda")
    r.eqsat.ops_before = 42
    r.eqsat.ops_after = 30
    r.llm.total_tokens = 5000
    with tempfile.TemporaryDirectory() as tmpdir:
        path = export_csv([r], Path(tmpdir) / "results.csv")
        assert path.exists()
        content = path.read_text()
        assert "mlp" in content
        assert "42" in content


def test_load_all_results() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        r1 = RunRecord(model_name="a")
        r2 = RunRecord(model_name="b")
        r1.save(tmpdir)
        r2.save(tmpdir)
        loaded = load_all_results(tmpdir)
        assert len(loaded) == 2

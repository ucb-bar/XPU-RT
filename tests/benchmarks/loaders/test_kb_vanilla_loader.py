"""Tests for the KB-vanilla loader.

Verifies the loader parses the actual on-disk
``results/comparison/vanilla_kb_gemmini/report.md`` and emits 14
canonical rows (7 correct with cycles, 7 incorrect with cycles=None).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xpu_rt.benchmarks.loaders.kb_vanilla_loader import (
    DEFAULT_REPORT_PATH,
    load_kb_vanilla_rows,
)


def test_load_kb_vanilla_rows_from_actual_report() -> None:
    """Against the real on-disk report: 14 rows, 7 correct with cycles."""
    if not DEFAULT_REPORT_PATH.is_file():
        pytest.skip(f"prior report not at {DEFAULT_REPORT_PATH}")
    rows = load_kb_vanilla_rows()
    assert len(rows) == 14
    correct = [r for r in rows if r.correctness]
    incorrect = [r for r in rows if not r.correctness]
    assert len(correct) == 7
    assert len(incorrect) == 7
    for r in correct:
        assert r.cycles is not None and r.cycles > 0
        assert r.cycle_source == "cached"
        assert r.backend == "kb-vanilla"
        assert r.target == "gemmini_mx"
        assert r.workload == "smolvla_matmuls"
        assert r.repeat == 0
    for r in incorrect:
        assert r.cycles is None


def test_load_kb_vanilla_rows_includes_expected_winners() -> None:
    """Spot-check three known winners against the report's numbers."""
    if not DEFAULT_REPORT_PATH.is_file():
        pytest.skip(f"prior report not at {DEFAULT_REPORT_PATH}")
    rows = load_kb_vanilla_rows()
    by_shape = {r.shape_id: r for r in rows}
    # From report.md:
    #   [64, 720]×[720, 320] → ✓ round 0 composite 12 251
    #   [64, 32]×[32, 720]    → ✓ round 0 composite 21 830
    #   [64, 720]×[720, 32]   → ✓ round 1 tile_ws_dataflow 3 637
    r1 = by_shape["[64, 720]×[720, 320]"]
    assert r1.correctness and r1.cycles == 12251 and r1.rounds_used == 0
    assert "composite" in r1.notes
    r2 = by_shape["[64, 32]×[32, 720]"]
    assert r2.correctness and r2.cycles == 21830
    r3 = by_shape["[64, 720]×[720, 32]"]
    assert r3.correctness and r3.cycles == 3637 and r3.rounds_used == 1
    assert "tile_ws_dataflow" in r3.notes


def test_load_kb_vanilla_rows_missing_report_returns_empty(tmp_path: Path) -> None:
    assert load_kb_vanilla_rows(tmp_path / "nope.md") == []


def test_load_kb_vanilla_rows_handles_hand_authored_table(tmp_path: Path) -> None:
    """Loader robustness — a hand-trimmed table with just the header
    and one row must round-trip cleanly."""
    md = (
        "# heading\n\n"
        "| # | shape | result | round | strategy | cycles |\n"
        "|---|---|---|---|---|---|\n"
        "| 1 | [16, 32]×[32, 64] | **✓** | 2 | tile_ws | 1 234 |\n"
        "| 2 | [16, 32]×[32, 64] | ✗ | — | — | — |\n"
    )
    path = tmp_path / "report.md"
    path.write_text(md)
    rows = load_kb_vanilla_rows(path)
    assert len(rows) == 2
    assert rows[0].correctness and rows[0].cycles == 1234 and rows[0].rounds_used == 2
    assert "tile_ws" in rows[0].notes
    assert not rows[1].correctness and rows[1].cycles is None

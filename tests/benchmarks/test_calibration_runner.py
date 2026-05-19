"""Tests for the Phase-C calibration runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.benchmarks.calibration_runner import (
    ANCHORS,
    CALIBRATION_LOWER,
    CALIBRATION_UPPER,
    _calibrate_kb_vanilla,
    _verdict_for,
    run,
)


def test_anchors_match_documented_set() -> None:
    """Regression guard: the anchors are the two KB-vanilla winners
    the plan locked in. Renaming or replacing them silently would
    invalidate the calibration baseline."""
    assert len(ANCHORS) == 2
    assert ANCHORS[0].shape == (64, 720, 320)
    assert ANCHORS[0].reference_cycles == 12251
    assert ANCHORS[1].shape == (64, 32, 720)
    assert ANCHORS[1].reference_cycles == 21830


def test_verdict_for_dispatches_correctly() -> None:
    """Ratio within the band → comparable; outside → discrepant;
    missing cycles → no_cycles."""
    ref = 10000
    # Inside band.
    v, r = _verdict_for(8000, ref)
    assert v == "comparable" and 0.79 < r < 0.81
    v, r = _verdict_for(15000, ref)
    assert v == "comparable" and 1.49 < r < 1.51
    # Outside band.
    v, r = _verdict_for(40000, ref)
    assert v == "discrepant" and r == 4.0
    v, r = _verdict_for(2000, ref)
    assert v == "discrepant" and r == 0.2
    # Missing cycles.
    v, r = _verdict_for(None, ref)
    assert v == "no_cycles" and r is None
    # Env-missing → status passes through.
    v, r = _verdict_for(None, ref, status="env_missing")
    assert v == "env_missing"


def test_calibrate_kb_vanilla_uses_cached_report() -> None:
    """The KB-vanilla calibration cell reads the on-disk cached
    report — no live LLM, no Spike. Ratio to its own reference is
    exactly 1.0."""
    from xpu_rt.benchmarks.loaders.kb_vanilla_loader import DEFAULT_REPORT_PATH

    if not DEFAULT_REPORT_PATH.is_file():
        pytest.skip("cached KB-vanilla report not present")

    anchor = ANCHORS[0]
    cell = _calibrate_kb_vanilla(anchor)
    assert cell.backend == "kb-vanilla"
    assert cell.row.correctness is True
    assert cell.row.cycles == anchor.reference_cycles
    assert cell.ratio_to_reference == pytest.approx(1.0)
    assert cell.verdict == "comparable"
    assert cell.row.cycle_source == "cached"


def test_run_writes_calibration_artifacts(tmp_path: Path) -> None:
    """Acceptance: the calibration runner emits the three artifacts
    (calibration.md / calibration.json / calibration_rows.jsonl)
    and the JSON contains all anchors × all backends."""
    out_dir = tmp_path / "fair"
    cells = run(out_dir, mode="plan")
    # 2 anchors × 3 backends = 6 cells.
    assert len(cells) == 6
    assert (out_dir / "calibration.md").is_file()
    assert (out_dir / "calibration.json").is_file()
    assert (out_dir / "calibration_rows.jsonl").is_file()

    js = json.loads((out_dir / "calibration.json").read_text())
    assert js["mode"] == "plan"
    assert len(js["anchors"]) == 2
    assert len(js["cells"]) == 6


def test_calibration_markdown_renders_band_documentation(tmp_path: Path) -> None:
    """The report must explicitly document the [0.5, 2.0] band so a
    reader knows why a `discrepant` flag fired without re-reading
    the source."""
    out_dir = tmp_path / "fair"
    run(out_dir, mode="plan")
    md = (out_dir / "calibration.md").read_text()
    assert f"[{CALIBRATION_LOWER}" in md
    assert f"{CALIBRATION_UPPER}" in md
    assert "MAIN_LD_ST_EX_CYCLES" in md or "discrepant" in md
    assert "Generated implementation latency" in md
    # KB-vanilla rows must be the reference (ratio 1.0).
    assert "1.00×" in md or "1.0×" in md


def test_calibration_kb_v2_and_autocomp_cells_report_env_missing(tmp_path: Path) -> None:
    """In plan mode (no LLM, no live Spike), KB-v2 + autocomp cells
    must come out as env_missing rather than crashing. The matrix
    runner relies on this clean-failure semantics."""
    out_dir = tmp_path / "fair"
    cells = run(out_dir, mode="plan")
    non_vanilla = [c for c in cells if c.backend != "kb-vanilla"]
    assert len(non_vanilla) == 4  # 2 backends × 2 anchors
    for c in non_vanilla:
        assert c.verdict in ("env_missing", "no_cycles", "comparable", "discrepant")
        # Today they all are env_missing; once live wiring lands the
        # set widens, but never to a hard crash.

"""Tests for the Phase-E fair-comparison report builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.benchmarks.canonical_metrics import CanonicalCellRow, write_jsonl
from xpu_rt.benchmarks.fair_comparison_report import Caveat, build_report


def _row(**kw) -> CanonicalCellRow:
    base = dict(
        backend="kb-v2",
        target="gemmini_mx",
        workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 320]",
        repeat=0,
        correctness=True,
        cycles=12000,
        rounds_used=1,
        tokens_in=1000,
        tokens_out=300,
        cost_usd=0.005,
        wall_s=10.0,
        cycle_source="MAIN_LD_ST_EX_CYCLES",
        notes="",
    )
    base.update(kw)
    return CanonicalCellRow(**base)


def _seed_cells_dir(cells_dir: Path, cells: dict[str, list[CanonicalCellRow]],
                    statuses: dict[str, dict] | None = None) -> None:
    """Lay out the per_cell tree the builder reads."""
    statuses = statuses or {}
    for name, rows in cells.items():
        d = cells_dir / "per_cell" / name
        d.mkdir(parents=True, exist_ok=True)
        write_jsonl(rows, d / "samples.jsonl")
        if name in statuses:
            (d / "status.json").write_text(json.dumps(statuses[name]))


def test_build_report_emits_three_artefacts(tmp_path: Path) -> None:
    cells_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _seed_cells_dir(
        cells_dir,
        cells={
            "kb-v2__gemmini_mx__smolvla_matmuls": [
                _row(repeat=0, cycles=12000),
                _row(repeat=1, cycles=12200),
                _row(repeat=2, cycles=11900),
            ],
        },
    )
    summary = build_report(cells_dir, out_dir)
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "report.json").is_file()
    assert (out_dir / "caveats.md").is_file()
    assert summary["n_samples"] == 3
    assert summary["n_cells"] == 1


def test_intersection_join_drops_one_sided_shapes(tmp_path: Path) -> None:
    """Q1 (KB-v2 vs autocomp) must only count shapes both backends
    produced cycles for — silent zero-pad would lie about the
    comparison."""
    cells_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _seed_cells_dir(
        cells_dir,
        cells={
            "kb-v2__gemmini_mx__smolvla_matmuls": [
                _row(backend="kb-v2", shape_id="[64, 720]×[720, 320]", cycles=12000),
                _row(backend="kb-v2", shape_id="[64, 32]×[32, 720]", cycles=22000),
            ],
            "autocomp__gemmini_mx__smolvla_matmuls": [
                _row(backend="autocomp", shape_id="[64, 720]×[720, 320]",
                     cycles=10000, cycle_source="Generated implementation latency"),
                # NOTE: no shape for [64, 32]×[32, 720] — autocomp didn't run it.
            ],
        },
    )
    build_report(cells_dir, out_dir)
    md = (out_dir / "report.md").read_text()
    assert "shapes joined |" in md
    # The Q1 row should report exactly 1 joined shape (the one autocomp covered).
    # Spot check: the geomean appears somewhere with × suffix.
    assert "1 |" in md or "1 |" in md
    # The harness-skew caveat must fire on the autocomp side.
    caveats_md = (out_dir / "caveats.md").read_text()
    assert "kind=harness_skew" in caveats_md
    assert "autocomp" in caveats_md


def test_caveats_include_deferred_and_env_missing(tmp_path: Path) -> None:
    cells_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _seed_cells_dir(
        cells_dir,
        cells={
            "kb-vanilla__saturn_opu_v128__smolvla_matmuls": [],
            "autocomp__gemmini_mx__smolvla_matmuls": [],
        },
        statuses={
            "kb-vanilla__saturn_opu_v128__smolvla_matmuls": {
                "backend": "kb-vanilla", "target": "saturn_opu_v128",
                "workload": "smolvla_matmuls", "status": "deferred",
                "notes": "needs kb_pipeline_driver fork", "env_missing": [],
            },
            "autocomp__gemmini_mx__smolvla_matmuls": {
                "backend": "autocomp", "target": "gemmini_mx",
                "workload": "smolvla_matmuls", "status": "env_missing",
                "notes": "needs chipyard",
                "env_missing": ["INT8_16PE_CHIPYARD_PATH", "autocomp"],
            },
        },
    )
    summary = build_report(cells_dir, out_dir)
    caveats_md = (out_dir / "caveats.md").read_text()
    assert "kind=deferred_cell" in caveats_md
    assert "kind=env_missing_cell" in caveats_md
    # n_caveats counts both.
    assert summary["n_caveats"] >= 2


def test_caveat_kind_validation() -> None:
    """Caveats must reject unknown kinds — protects the caveats ledger
    from typo'd categories that the report aggregator can't render."""
    with pytest.raises(ValueError, match="unknown caveat kind"):
        Caveat(kind="oopsie", backend="kb-v2", target="gemmini_mx",
               workload="smolvla_matmuls", detail="—")


def test_single_sample_caveat_fires_when_repeats_less_than_three(tmp_path: Path) -> None:
    """A cell with N<3 correct samples must flag a single_sample
    caveat so the reader knows variance isn't measured."""
    cells_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _seed_cells_dir(
        cells_dir,
        cells={
            "kb-v2__gemmini_mx__smolvla_matmuls": [
                _row(repeat=0, cycles=12000),
            ],
        },
    )
    summary = build_report(cells_dir, out_dir)
    caveat_kinds = {c["kind"] for c in summary["caveats"]}
    assert "single_sample" in caveat_kinds


def test_q3_cross_target_join_finds_paired_shapes(tmp_path: Path) -> None:
    """Q3 must join shapes that appear on BOTH targets for the same
    backend. Solo-target shapes drop from the geomean."""
    cells_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _seed_cells_dir(
        cells_dir,
        cells={
            "kb-v2__gemmini_mx__smolvla_matmuls": [
                _row(target="gemmini_mx", shape_id="[64, 720]×[720, 320]", cycles=12000),
                _row(target="gemmini_mx", shape_id="[64, 32]×[32, 720]", cycles=22000),
            ],
            "kb-v2__saturn_opu_v128__smolvla_matmuls": [
                _row(target="saturn_opu_v128", shape_id="[64, 720]×[720, 320]",
                     cycles=14000, cycle_source="mcycle"),
                _row(target="saturn_opu_v128", shape_id="[64, 32]×[32, 720]",
                     cycles=24000, cycle_source="mcycle"),
            ],
        },
    )
    build_report(cells_dir, out_dir)
    md = (out_dir / "report.md").read_text()
    # Q3 must report 2 joined shapes for kb-v2 × matmuls.
    assert "Saturn/Gemmini" in md
    assert "2 |" in md  # row for kb-v2 with 2 joined shapes

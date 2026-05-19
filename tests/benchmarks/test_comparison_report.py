"""Tests for the SmolVLA comparison-report aggregator.

Drives :func:`comparison_report.main` against a synthetic study dir
containing a few JSONL rows. Verifies the markdown report shape, the
join-by-region behaviour, and the per-backend stat computation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.benchmarks import comparison_report as cr


def _kb_row(region: str, *, compile: bool, intr: float, shape: bool, cost: float) -> dict:
    return {
        "backend": "kb_vanilla",
        "contract": {
            "region_id": region,
            "op_family": "matmul",
            "input_shapes": [[64, 768], [768, 768]],
            "output_shapes": [[64, 768]],
            "dtypes": ["i8", "i8", "i32"],
            "target_name": "gemmini_mx",
        },
        "rounds": 3,
        "compile": compile,
        "intrinsic_use_rate": intr,
        "intrinsic_matched": int(intr * 8),
        "intrinsic_total": 8,
        "shape_consistency": shape,
        "shape_missing": [],
        "final_strategy": "tile_and_dma_overlap",
        "tokens_in": 4000,
        "tokens_out": 1500,
        "cost_usd": cost,
        "wall_s": 12.0,
        "attempts": [],
    }


def _xr_row(region: str, *, correct: bool, cycles: int | None, speedup: float, cost: float) -> dict:
    return {
        "backend": "xpu_rt_kb_v2",
        "contract": {
            "region_id": region,
            "op_family": "matmul",
            "input_shapes": [[64, 768], [768, 768]],
            "output_shapes": [[64, 768]],
            "dtypes": ["i8", "i8", "i32"],
            "target_name": "gemmini_mx",
        },
        "rounds": 2,
        "found": correct,
        "correct": correct,
        "speedup": speedup,
        "cycles": cycles,
        "plan": "tile-WS",
        "state_hash": "abc",
        "aborted": False,
        "abort_reason": "",
        "tokens_in": 5000,
        "tokens_out": 2000,
        "cost_usd": cost,
        "wall_s": 30.0,
        "attempts": [],
    }


def _seed_study_dir(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    kb_path = tmp / "kb_vanilla.jsonl"
    xr_path = tmp / "xpu_rt.jsonl"
    with kb_path.open("w") as f:
        for row in (
            _kb_row("layers.0.q_proj", compile=True, intr=0.8, shape=True, cost=0.005),
            _kb_row("layers.0.k_proj", compile=False, intr=0.4, shape=True, cost=0.006),
            _kb_row("layers.0.v_proj", compile=False, intr=0.0, shape=False, cost=0.004),
        ):
            f.write(json.dumps(row) + "\n")
    with xr_path.open("w") as f:
        for row in (
            _xr_row("layers.0.q_proj", correct=True, cycles=12345, speedup=4.0, cost=0.012),
            _xr_row("layers.0.k_proj", correct=True, cycles=20000, speedup=2.5, cost=0.015),
            _xr_row("layers.0.v_proj", correct=False, cycles=None, speedup=0.0, cost=0.018),
        ):
            f.write(json.dumps(row) + "\n")
    (tmp / "run_summary.json").write_text(
        json.dumps(
            {
                "pre_spend": {"cumulative_usd": 0.5, "calls": 10},
                "post_spend": {"cumulative_usd": 0.6, "calls": 22},
            }
        )
    )
    return tmp


def test_kb_vanilla_stats(tmp_path: Path) -> None:
    study = _seed_study_dir(tmp_path)
    rows = cr._read_jsonl(study / "kb_vanilla.jsonl")
    s = cr._stats_kb_vanilla(rows)
    assert s.rows == 3
    assert s.compile_rate == pytest.approx(1 / 3)
    assert s.intrinsic_use_rate_mean == pytest.approx((0.8 + 0.4 + 0.0) / 3)
    assert s.intrinsic_use_rate_max == pytest.approx(0.8)
    assert s.shape_consistency_rate == pytest.approx(2 / 3)
    assert s.total_cost_usd == pytest.approx(0.015)


def test_xpu_rt_stats(tmp_path: Path) -> None:
    study = _seed_study_dir(tmp_path)
    rows = cr._read_jsonl(study / "xpu_rt.jsonl")
    s = cr._stats_xpu_rt(rows)
    assert s.rows == 3
    assert s.correctness_rate == pytest.approx(2 / 3)
    assert s.cycles_seen == 2
    # Geometric mean of (12345, 20000) ~ 15715
    assert s.cycles_geomean is not None
    assert 15000 < s.cycles_geomean < 16500
    assert s.speedup_max == pytest.approx(4.0)
    assert s.total_cost_usd == pytest.approx(0.045)


def test_main_generates_report(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    study = _seed_study_dir(tmp_path)
    rc = cr.main([str(study)])
    assert rc == 0
    md_path = study / "report.md"
    json_path = study / "report.json"
    assert md_path.is_file()
    assert json_path.is_file()
    md = md_path.read_text()
    assert "SmolVLA-on-Gemmini comparison report" in md
    # Formatter truncates region_id to its last two dot-parts: "0.q_proj".
    assert "0.q_proj" in md
    # Spend summary is rendered.
    assert "$0.5000" in md or "$0.5" in md
    # Aggregate row counts and rates.
    j = json.loads(json_path.read_text())
    assert j["kb_vanilla"]["rows"] == 3
    assert j["xpu_rt"]["rows"] == 3
    assert j["pairs"] == 3


def test_join_handles_unmatched_regions(tmp_path: Path) -> None:
    kb_only = [_kb_row("solo_region", compile=True, intr=0.5, shape=True, cost=0.001)]
    xr_only = [_xr_row("other_region", correct=False, cycles=None, speedup=0.0, cost=0.002)]
    pairs = cr._join_by_region(kb_only, xr_only)
    assert len(pairs) == 2
    # One pair has only KB, one only XR.
    has_kb_only = [p for p in pairs if p[0] is not None and p[1] is None]
    has_xr_only = [p for p in pairs if p[0] is None and p[1] is not None]
    assert len(has_kb_only) == 1
    assert len(has_xr_only) == 1

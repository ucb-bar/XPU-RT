"""Tests for the autocomp loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.benchmarks.loaders.autocomp_loader import load_autocomp_row


def _write_eval_result(out_dir: Path, iter_n: int, cand_n: int, body: dict) -> None:
    d = out_dir / f"eval-results-iter-{iter_n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"code_{cand_n}_result.txt").write_text(json.dumps(body))


def _write_iter_metrics(out_dir: Path, iter_n: int, *, in_tok: int, out_tok: int, wall: float) -> None:
    body = {
        "iteration": iter_n,
        "iteration_total_s": wall,
        "plan_generation": {
            "gemini-2.5-flash": {"input_tokens": in_tok, "output_tokens": out_tok}
        },
    }
    (out_dir / f"metrics-iter-{iter_n}.json").write_text(json.dumps(body))


def test_load_autocomp_row_picks_lowest_latency_correct(tmp_path: Path) -> None:
    """When several iterations land correct candidates, the loader
    must pick the lowest-cycle one (best result is the winner)."""
    out_dir = tmp_path / "autocomp_out"
    _write_eval_result(out_dir, 0, 0, {"correct": True, "latency": 50000, "compiled": True})
    _write_eval_result(out_dir, 1, 0, {"correct": True, "latency": 18000, "compiled": True})
    _write_eval_result(out_dir, 2, 0, {"correct": False, "latency": None, "compiled": True})
    _write_iter_metrics(out_dir, 0, in_tok=1000, out_tok=300, wall=12.0)
    _write_iter_metrics(out_dir, 1, in_tok=1200, out_tok=400, wall=14.5)
    _write_iter_metrics(out_dir, 2, in_tok=900, out_tok=200, wall=11.0)

    row = load_autocomp_row(
        out_dir, target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[64, 720]×[720, 320]", repeat=0,
    )
    assert row.correctness is True
    assert row.cycles == 18000
    assert row.rounds_used == 2  # iter 1, 1-indexed
    assert row.tokens_in == 3100
    assert row.tokens_out == 900
    assert row.cost_usd > 0
    assert row.cycle_source == "Generated implementation latency"
    assert row.wall_s == pytest.approx(37.5)


def test_load_autocomp_row_no_correct_returns_failure(tmp_path: Path) -> None:
    """No correct candidate → correctness=False, cycle_source=none,
    but tokens / rounds still recorded."""
    out_dir = tmp_path / "autocomp_out"
    _write_eval_result(out_dir, 0, 0, {"correct": False, "compiled": True})
    _write_eval_result(out_dir, 1, 0, {"correct": False, "compiled": False})
    _write_iter_metrics(out_dir, 0, in_tok=1500, out_tok=200, wall=10.0)
    _write_iter_metrics(out_dir, 1, in_tok=1200, out_tok=150, wall=9.0)

    row = load_autocomp_row(
        out_dir, target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[64, 960]×[960, 960]", repeat=1,
    )
    assert row.correctness is False
    assert row.cycles is None
    assert row.tokens_in == 2700
    assert row.cycle_source == "none"
    assert "no correct candidate" in row.notes
    assert row.rounds_used >= 2


def test_load_autocomp_row_missing_dir(tmp_path: Path) -> None:
    row = load_autocomp_row(
        tmp_path / "missing", target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[16, 16]×[16, 16]", repeat=0,
    )
    assert row.correctness is False
    assert row.cycles is None
    assert "not found" in row.notes


def test_load_autocomp_row_aggregates_tokens_across_phases(tmp_path: Path) -> None:
    """Tokens live under separate phase keys in metrics-iter-N.json
    (plan_generation, code_generation, etc.). The loader must sum
    them all into one tokens_in / tokens_out per cell."""
    out_dir = tmp_path / "autocomp_out"
    _write_eval_result(out_dir, 0, 0, {"correct": True, "latency": 30000, "compiled": True})
    body = {
        "iteration": 0,
        "iteration_total_s": 7.5,
        "plan_generation": {
            "gemini-2.5-flash": {"input_tokens": 1000, "output_tokens": 200}
        },
        "code_generation": {
            "gemini-2.5-flash": {"input_tokens": 1500, "output_tokens": 400}
        },
    }
    (out_dir / "metrics-iter-0.json").write_text(json.dumps(body))

    row = load_autocomp_row(
        out_dir, target="gemmini_mx", workload="smolvla_matmuls",
        shape_id="[64, 32]×[32, 720]", repeat=0,
    )
    assert row.tokens_in == 2500
    assert row.tokens_out == 600

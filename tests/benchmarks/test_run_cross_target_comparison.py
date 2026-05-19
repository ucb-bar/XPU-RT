"""Tests for the cross-target matrix driver + report aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from xpu_rt.benchmarks.run_cross_target_comparison import (
    CellResult,
    _run_autocomp_cell,
    _run_kb_v2_cell,
    _run_kb_vanilla_cell,
    run,
)


def test_run_plan_mode_produces_report_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Block the live SmolVLA loader so the test uses the stub.
    monkeypatch.setattr(
        "xpu_rt.benchmarks.smolvla_subset.SubsetSelector.load",
        lambda self: (_ for _ in ()).throw(RuntimeError("stub")),
    )
    out_dir = tmp_path / "cross_target"
    results = run(
        out_dir,
        backends=("kb-v2",),
        targets=("gemmini_mx", "saturn_opu_v128"),
        workloads=("smolvla_mlp_block",),
        mode="plan",
    )
    # 1 backend × 2 targets × 1 workload = 2 cells.
    assert len(results) == 2
    # Both should be ok in plan mode (KB-v2 always wires).
    statuses = {c.status for c in results}
    assert statuses == {"ok"}
    md = (out_dir / "report.md").read_text()
    assert "Cross-target × cross-backend comparison" in md
    assert "gemmini_mx" in md
    assert "saturn_opu_v128" in md
    js = json.loads((out_dir / "report.json").read_text())
    assert js["n_cells"] == 2
    assert js["mode"] == "plan"


def test_kb_vanilla_on_saturn_is_deferred(tmp_path: Path) -> None:
    """KB-vanilla on Saturn must emit `deferred` with a clear note —
    don't silently drop the cell."""
    cell = _run_kb_vanilla_cell(
        target="saturn_opu_v128", workload="smolvla_matmuls", mode="plan", out_dir=tmp_path,
    )
    assert cell.status == "deferred"
    assert "Saturn" in cell.notes
    assert "kb_pipeline_driver" in cell.notes.lower() or "kb_pipeline_driver" in cell.notes


def test_kb_vanilla_on_gemmini_matmuls_reuses_prior_report(tmp_path: Path) -> None:
    """KB-vanilla on Gemmini × matmuls reuses the prior batch's
    7/14 result + $0.16 spend rather than re-paying."""
    cell = _run_kb_vanilla_cell(
        target="gemmini_mx", workload="smolvla_matmuls", mode="plan", out_dir=tmp_path,
    )
    # If the prior report file is present locally, status is ok.
    # Otherwise the cell explicitly reports error (not silent skip).
    assert cell.status in ("ok", "error")
    if cell.status == "ok":
        assert cell.correctness_rate == pytest.approx(7.0 / 14.0)
        assert cell.cost_usd == pytest.approx(0.16)


def test_autocomp_cell_reports_env_missing_when_chipyard_path_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autocomp cells must emit `env_missing` (with the missing var
    name) when the chipyard env var isn't set — not crash inside
    autocomp's eval backend."""
    monkeypatch.delenv("INT8_16PE_CHIPYARD_PATH", raising=False)
    cell = _run_autocomp_cell(
        target="gemmini_mx", workload="smolvla_matmuls", mode="plan", out_dir=tmp_path,
    )
    # Either env_missing (autocomp installed but chipyard not set) OR
    # env_missing with autocomp-import in the list (autocomp not in venv).
    assert cell.status == "env_missing"
    missing_str = " ".join(cell.env_missing)
    assert ("INT8_16PE_CHIPYARD_PATH" in missing_str
            or "autocomp" in missing_str.lower())


def test_kb_v2_full_mode_reports_env_missing_when_toolchain_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In full mode, KB-v2 cells emit env_missing when the riscv-tools
    conda env / Gemini key isn't present — never crash inside the
    evaluator."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMMINI_API", raising=False)
    monkeypatch.setenv("XPU_RT_RISCV_CONDA_ROOT", "/nope/not/here")
    monkeypatch.setattr(
        "xpu_rt.benchmarks.smolvla_subset.SubsetSelector.load",
        lambda self: (_ for _ in ()).throw(RuntimeError("stub")),
    )
    cell = _run_kb_v2_cell(
        target="gemmini_mx", workload="smolvla_mlp_block", mode="full", out_dir=tmp_path,
    )
    assert cell.status == "env_missing"
    missing_str = " ".join(cell.env_missing)
    assert "GOOGLE_API_KEY" in missing_str
    assert "riscv-tools" in missing_str


def test_cross_target_report_renders_all_three_questions(tmp_path: Path) -> None:
    """The aggregator must surface Q1 / Q2 / Q3 sections in the markdown."""
    from xpu_rt.benchmarks.cross_target_report import write_reports

    results = [
        CellResult(backend="kb-v2", target="gemmini_mx", workload="smolvla_mlp_block",
                   status="ok", n_kernels_vanilla=3, n_kernels_agentic=1,
                   planner_estimated_speedup=6.97),
        CellResult(backend="kb-v2", target="saturn_opu_v128", workload="smolvla_mlp_block",
                   status="ok", n_kernels_vanilla=3, n_kernels_agentic=1,
                   planner_estimated_speedup=6.25),
        CellResult(backend="autocomp", target="gemmini_mx", workload="smolvla_matmuls",
                   status="env_missing", env_missing=("INT8_16PE_CHIPYARD_PATH",),
                   notes="needs chipyard"),
    ]
    write_reports(tmp_path, results, mode="plan")
    md = (tmp_path / "report.md").read_text()
    assert "## Q1" in md
    assert "## Q2" in md
    assert "## Q3" in md
    assert "Env-readiness" in md
    assert "INT8_16PE_CHIPYARD_PATH" in md

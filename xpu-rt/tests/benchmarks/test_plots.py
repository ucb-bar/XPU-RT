"""Tests for benchmark plot generation."""

from __future__ import annotations

import tempfile

import pytest

from benchmarks.record import RunRecord

try:
    import matplotlib  # noqa: F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _make_sample_records() -> list[RunRecord]:
    r1 = RunRecord(model_name="mlp", target_name="cuda_a100")
    r1.agentic.iteration_costs = [100.0, 80.0, 70.0, 65.0, 63.0]
    r1.agentic.iteration_improvements = [0.0, 20.0, 12.5, 7.1, 3.1]
    r1.agentic.iterations_run = 5
    r1.eqsat.ops_before = 42
    r1.eqsat.ops_after = 30
    r1.eqsat.ops_reduction_pct = 28.6
    r1.eqsat.changed = True
    r1.recipe.total_recipe_ops = 25
    r1.recipe.scope_ops = 4
    r1.recipe.fact_ops = 8
    r1.recipe.candidate_ops = 6
    r1.recipe.choice_ops = 2
    r1.recipe.verify_ops = 3
    r1.recipe.provenance_ops = 2
    r1.solver.placement_time_ms = 15.0
    r1.solver.schedule_time_ms = 8.0
    r1.solver.memory_time_ms = 2.0
    r1.solver.placement_gap = 0.0
    r1.baselines.eager_cpu_latency_us = 500.0
    r1.baselines.eager_gpu_latency_us = 200.0
    r1.baselines.compiled_gpu_latency_us = 100.0
    r1.baselines.compgen_latency_us = 90.0
    r1.verification.structural_pass = True
    r1.verification.check_assertions_pass = True
    r1.verification.differential_pass = True
    r1.verification.overall_status = "pass"
    r1.llm.total_prompt_tokens = 5000
    r1.llm.total_completion_tokens = 2000
    r1.llm.total_cost_usd = 0.10
    r1.capture.export_time_ms = 50.0
    r1.eqsat.eqsat_time_ms = 120.0
    r1.recipe.seed_generation_time_ms = 10.0
    r1.total_compile_time_ms = 250.0
    r1.performance.per_run_us = [88.0, 90.0, 92.0, 89.0, 91.0]
    r1.kernels.strategy_histogram = {"native": 3, "autocomp": 2, "library": 1}
    return [r1]


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_all_plots_generate() -> None:
    from benchmarks.plots import generate_all_plots

    records = _make_sample_records()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_all_plots(records, tmpdir)
        assert len(paths) >= 8  # at least 8 of 10 should succeed
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_convergence_plot() -> None:
    from benchmarks.plots import plot_convergence

    records = _make_sample_records()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_convergence(records, tmpdir)
        assert path.exists()
        assert "convergence" in path.name


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
def test_baseline_comparison_plot() -> None:
    from benchmarks.plots import plot_baseline_comparison

    records = _make_sample_records()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = plot_baseline_comparison(records, tmpdir)
        assert path.exists()

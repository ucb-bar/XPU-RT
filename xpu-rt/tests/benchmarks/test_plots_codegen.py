"""Tests for codegen-specific benchmark plot generation."""

from __future__ import annotations

import pytest
from pathlib import Path

from benchmarks.record import RunRecord

pytest.importorskip("matplotlib")


def _make_sample_records() -> list[RunRecord]:
    """Create sample records with all codegen fields populated for plotting."""
    r1 = RunRecord(model_name="simple_mlp", target_name="cuda-a100")
    r1.capture.export_success = True
    r1.ir.total_ops = 20
    r1.kernels.total_kernel_specs = 10
    r1.kernels.strategy_histogram = {"native": 5, "library": 3, "autocomp": 2}
    r1.kernels.pct_native = 50.0
    r1.kernels.pct_library = 30.0
    r1.kernels.pct_generated = 20.0
    r1.kernels.roofline_gap = 1.3
    r1.kernels.region_details = [
        {"region_id": "r1", "selected_strategy": "autocomp", "search_iterations_used": 5, "speedup_vs_reference": 1.3},
    ]
    r1.verification.overall_status = "pass"
    r1.artifacts.completeness_score = 0.9
    r1.solver.placement_feasible = True
    r1.solver.schedule_makespan_us = 500.0
    r1.solver.placement_time_ms = 10.0
    r1.solver.schedule_time_ms = 20.0
    r1.solver.memory_time_ms = 5.0
    r1.recipe.total_recipe_ops = 15
    r1.baselines.speedup_vs_compiled = 1.2
    r1.agentic.iteration_improvements = [5.0, 3.0, 1.0]
    r1.agentic.total_improvement_pct = 9.0
    r1.agentic.iteration_costs = [1000.0, 950.0, 920.0]
    r1.config = {"ablation": "full"}
    r1.total_compile_time_ms = 500.0
    r1.fallback_pressure.fallback_region_count = 1
    r1.layout_friction.materialized_transposes = 2
    r1.codegen_funnel.eligible = 10
    r1.codegen_funnel.attempted = 8
    r1.codegen_funnel.faster = 6

    r2 = RunRecord(model_name="transformer_block", target_name="cuda-a100")
    r2.capture.export_success = True
    r2.ir.total_ops = 20
    r2.kernels.total_kernel_specs = 10
    r2.kernels.strategy_histogram = {"native": 3, "fallback": 4, "ukernel": 1}
    r2.kernels.pct_native = 37.5
    r2.kernels.pct_fallback = 50.0
    r2.kernels.pct_generated = 12.5
    r2.kernels.roofline_gap = 2.1
    r2.kernels.region_details = [
        {"region_id": "r2", "selected_strategy": "native", "search_iterations_used": 3, "speedup_vs_reference": 1.1},
    ]
    r2.verification.overall_status = "pass"
    r2.artifacts.completeness_score = 0.85
    r2.solver.placement_feasible = True
    r2.solver.schedule_makespan_us = 800.0
    r2.solver.placement_time_ms = 15.0
    r2.solver.schedule_time_ms = 30.0
    r2.solver.memory_time_ms = 8.0
    r2.recipe.total_recipe_ops = 22
    r2.baselines.speedup_vs_compiled = 1.05
    r2.agentic.iteration_improvements = [3.0, 2.0]
    r2.agentic.total_improvement_pct = 5.0
    r2.agentic.iteration_costs = [1100.0, 1050.0]
    r2.config = {"ablation": "no_eqsat"}
    r2.total_compile_time_ms = 750.0
    r2.fallback_pressure.fallback_region_count = 4
    r2.layout_friction.materialized_transposes = 5
    r2.codegen_funnel.eligible = 8
    r2.codegen_funnel.attempted = 6
    r2.codegen_funnel.faster = 3

    r3 = RunRecord(model_name="llama31_decoder_block", target_name="cuda-a100")
    r3.capture.export_success = True
    r3.ir.total_ops = 20
    r3.kernels.total_kernel_specs = 10
    r3.kernels.strategy_histogram = {"native": 8, "library": 5, "fallback": 2}
    r3.kernels.pct_native = 53.3
    r3.kernels.pct_library = 33.3
    r3.kernels.pct_fallback = 13.3
    r3.kernels.roofline_gap = 1.8
    r3.kernels.region_details = [
        {"region_id": "r3", "selected_strategy": "library", "search_iterations_used": 2, "speedup_vs_reference": 1.5},
    ]
    r3.verification.overall_status = "pass"
    r3.artifacts.completeness_score = 0.95
    r3.solver.placement_feasible = True
    r3.solver.schedule_makespan_us = 600.0
    r3.solver.placement_time_ms = 12.0
    r3.solver.schedule_time_ms = 25.0
    r3.solver.memory_time_ms = 6.0
    r3.recipe.total_recipe_ops = 30
    r3.baselines.speedup_vs_compiled = 1.4
    r3.agentic.iteration_improvements = [4.0, 2.5, 1.5]
    r3.agentic.total_improvement_pct = 8.0
    r3.agentic.iteration_costs = [900.0, 870.0, 850.0]
    r3.config = {"ablation": "no_codegen"}
    r3.total_compile_time_ms = 620.0
    r3.fallback_pressure.fallback_region_count = 2
    r3.layout_friction.materialized_transposes = 3
    r3.codegen_funnel.eligible = 15
    r3.codegen_funnel.attempted = 12
    r3.codegen_funnel.faster = 10

    return [r1, r2, r3]


class TestCodegenPlots:
    """Tests for each codegen plot function."""

    def test_plot_coverage_waterfall(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_coverage_waterfall
        path = plot_coverage_waterfall(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_strategy_mix(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_strategy_mix
        path = plot_strategy_mix(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_roofline_gap(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_roofline_gap
        path = plot_roofline_gap(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_speedup_vs_search_cost(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_speedup_vs_search_cost
        path = plot_speedup_vs_search_cost(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_multidevice_planning(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_multidevice_planning
        path = plot_multidevice_planning(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_recipe_scale_payoff(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_recipe_scale_payoff
        path = plot_recipe_scale_payoff(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_agentic_outcome(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_agentic_outcome
        path = plot_agentic_outcome(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_plot_ablation_heatmap(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import plot_ablation_heatmap
        path = plot_ablation_heatmap(_make_sample_records(), tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_generate_all_codegen_plots(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import generate_all_codegen_plots
        paths = generate_all_codegen_plots(_make_sample_records(), tmp_path)
        assert len(paths) >= 5
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0


class TestEdgeCases:
    """Edge case tests for codegen plot functions."""

    def test_empty_records_dont_crash(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import (
            plot_coverage_waterfall,
            plot_strategy_mix,
            plot_roofline_gap,
            plot_speedup_vs_search_cost,
            plot_multidevice_planning,
            plot_recipe_scale_payoff,
            plot_agentic_outcome,
            plot_ablation_heatmap,
        )
        for fn in [
            plot_coverage_waterfall,
            plot_strategy_mix,
            plot_roofline_gap,
            plot_speedup_vs_search_cost,
            plot_multidevice_planning,
            plot_recipe_scale_payoff,
            plot_agentic_outcome,
            plot_ablation_heatmap,
        ]:
            path = fn([], tmp_path)
            assert path.exists()
            assert path.stat().st_size > 0

    def test_single_record(self, tmp_path: Path) -> None:
        from benchmarks.plots_codegen import generate_all_codegen_plots
        paths = generate_all_codegen_plots(_make_sample_records()[:1], tmp_path)
        assert len(paths) >= 5
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
            assert p.stat().st_size > 0

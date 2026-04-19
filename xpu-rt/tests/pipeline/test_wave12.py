"""Tests for Wave 12: stacked fixtures + autotuner + bench + disk cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.bench import BenchmarkReport, measure_pipeline, measure_pipeline_suite
from compgen.options import CompGenOptions, cuda_a100_defaults
from compgen.pipeline import PipelineCache, compile_and_diff
from compgen.search import Autotuner, AutotuneResult, OptionsAxis

from tests._fixtures.real_workloads import (
    attention_mlp_tiny,
    gemma_stack_3,
    smolvla_stack_2,
    tinyllama_stack_3,
)


# =========================================================================
# Stacked full-model fixtures (W12.1)
# =========================================================================


class TestStackedFixtures:
    @pytest.mark.parametrize(
        "fn",
        [tinyllama_stack_3, gemma_stack_3, smolvla_stack_2],
        ids=lambda f: f.__name__,
    )
    def test_stacked_fixture_builds_and_eager_is_deterministic(self, fn):
        a = fn()
        b = fn()
        import torch
        assert torch.allclose(a.eager_output, b.eager_output)

    def test_tinyllama_stack_has_embed_and_lm_head(self):
        fx = tinyllama_stack_3()
        assert hasattr(fx.model, "embed")
        assert hasattr(fx.model, "lm_head")
        # Tied weights: embed.weight and lm_head.weight share storage.
        assert fx.model.embed.weight.data_ptr() == fx.model.lm_head.weight.data_ptr()

    def test_gemma_stack_has_3_blocks(self):
        fx = gemma_stack_3()
        assert len(fx.model.blocks) == 3

    def test_smolvla_stack_has_7dof_head(self):
        fx = smolvla_stack_2()
        assert fx.eager_output.shape[-1] == 7


class TestStackedFixturesE2E:
    @pytest.mark.parametrize(
        "fn",
        [tinyllama_stack_3, gemma_stack_3, smolvla_stack_2],
        ids=lambda f: f.__name__,
    )
    def test_stacked_fixture_compiles_end_to_end(self, fn):
        fx = fn()
        report = compile_and_diff(
            fx.model, fx.example_inputs,
            options=cuda_a100_defaults(),
            fixture_name=fx.name,
            eager_reference=fx.eager_output,
            exported_program=fx.exported,
            opaque_rate_threshold=0.8,  # stacks have more reshaping
        )
        assert report.bridge_path in {"torch_mlir", "fx_importer"}
        assert report.module_verified
        assert report.plan_validated


# =========================================================================
# Autotuner (W12.2)
# =========================================================================


class TestAutotuner:
    def test_baseline_strategy_runs_one_trial(self):
        fx = attention_mlp_tiny()
        tuner = Autotuner(
            base=cuda_a100_defaults(),
            axes=[],
            strategy="baseline",
        )
        result = tuner.search(fx.model, fx.example_inputs, workload_name=fx.name)
        assert len(result.trials) == 1
        assert result.best_trial is not None

    def test_grid_strategy_explores_product(self):
        fx = attention_mlp_tiny()
        tuner = Autotuner(
            base=cuda_a100_defaults(),
            axes=[
                OptionsAxis("demote_target_type", ("bf16", "f16")),
                OptionsAxis("enable_dma_overlap", (True, False)),
            ],
            strategy="grid",
        )
        result = tuner.search(fx.model, fx.example_inputs, workload_name=fx.name)
        assert len(result.trials) == 4
        assert result.best_index >= 0

    def test_random_strategy_with_n_trials(self):
        fx = attention_mlp_tiny()
        tuner = Autotuner(
            base=cuda_a100_defaults(),
            axes=[
                OptionsAxis("enable_dma_overlap", (True, False)),
                OptionsAxis("enable_plan_reduction", (True, False)),
            ],
            strategy="random",
            n_trials=5,
            seed=42,
        )
        result = tuner.search(fx.model, fx.example_inputs, workload_name=fx.name)
        assert len(result.trials) == 5

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="is not a CompGenOptions field"):
            OptionsAxis("not_a_field", (1, 2))

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            Autotuner(
                base=cuda_a100_defaults(),
                axes=[OptionsAxis("enable_dma_overlap", (True, False))],
                strategy="bogus",
            )

    def test_best_trial_minimizes_metric(self):
        fx = attention_mlp_tiny()
        # Custom metric: trial index (lower is better), so trial 0 wins.
        tuner = Autotuner(
            base=cuda_a100_defaults(),
            axes=[OptionsAxis("enable_dma_overlap", (True, False))],
            strategy="grid",
            metric_fn=lambda pr: float(pr.stages_run),  # any monotone-ish
        )
        result = tuner.search(fx.model, fx.example_inputs, workload_name=fx.name)
        assert result.best_trial is not None
        assert result.best_trial.metric == min(t.metric for t in result.trials)

    def test_summary_is_printable(self):
        fx = attention_mlp_tiny()
        tuner = Autotuner(
            base=cuda_a100_defaults(), axes=[], strategy="baseline",
        )
        result = tuner.search(fx.model, fx.example_inputs, workload_name=fx.name)
        text = result.summary()
        assert "best metric" in text


# =========================================================================
# Benchmark harness (W12.3)
# =========================================================================


class TestBenchmarkHarness:
    def test_measure_pipeline_returns_report(self):
        fx = attention_mlp_tiny()
        report = measure_pipeline(
            fx.model, fx.example_inputs,
            options=cuda_a100_defaults(),
            fixture_name=fx.name,
            n_iter=2,
            exported_program=fx.exported,
        )
        assert isinstance(report, BenchmarkReport)
        assert report.compile_time_ms > 0
        assert report.compile_time_cached_ms >= 0
        assert report.pipeline_stages_run >= 10

    def test_cache_speedup_is_recorded(self):
        fx = attention_mlp_tiny()
        report = measure_pipeline(
            fx.model, fx.example_inputs,
            options=cuda_a100_defaults(),
            n_iter=2, exported_program=fx.exported,
        )
        assert report.compile_speedup > 1.0

    def test_suite_runs_multiple_fixtures(self):
        reports = measure_pipeline_suite(
            [attention_mlp_tiny, smolvla_stack_2],
            options=cuda_a100_defaults(),
            n_iter=2,
        )
        assert len(reports) == 2
        for r in reports:
            assert r.fixture_name


# =========================================================================
# Disk-backed cache (W12.4)
# =========================================================================


class TestDiskCache:
    def test_save_load_manifest_round_trip(self, tmp_path: Path):
        cache = PipelineCache(max_entries=4)
        fx = attention_mlp_tiny()
        cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
        cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())  # hit

        manifest_path = tmp_path / "cache.json"
        cache.save_manifest(manifest_path)
        assert manifest_path.exists()

        loaded = PipelineCache.load_manifest(manifest_path)
        assert loaded.max_entries == 4
        assert loaded.stats.hits == 1
        assert loaded.stats.misses == 1

    def test_manifest_records_every_cached_entry(self, tmp_path: Path):
        import json
        cache = PipelineCache(max_entries=8)
        fx = attention_mlp_tiny()
        cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
        manifest_path = tmp_path / "cache.json"
        cache.save_manifest(manifest_path)
        data = json.loads(manifest_path.read_text())
        assert data["version"] == 1
        assert len(data["entries"]) == 1


# =========================================================================
# Executor fidelity improvements (W12.6)
# =========================================================================


class TestExecutorFidelity:
    def test_linalg_generic_elementwise_with_constant_recovered(self):
        """The executor now recovers arith.constant values embedded in
        linalg.generic bodies, so `mul by 2.0` produces the right
        output instead of identity."""
        from compgen.runtime.cpu_executor import _find_constant_in_body

        import torch
        from xdsl.dialects.arith import ConstantOp, MulfOp
        from xdsl.dialects.builtin import FloatAttr, Float32Type
        from xdsl.dialects.linalg import YieldOp
        from xdsl.ir import Block

        block = Block(arg_types=[Float32Type(), Float32Type()])
        c = ConstantOp(FloatAttr(2.0, Float32Type()))
        mul = MulfOp(block.args[0], c.result)
        block.add_op(c)
        block.add_op(mul)
        block.add_op(YieldOp(mul.result))

        val = _find_constant_in_body(block)
        assert val == 2.0

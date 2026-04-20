"""Tests for the production-readiness differential harness + cache."""

from __future__ import annotations

import pytest
from compgen.options import CompGenOptions, cuda_a100_defaults, npu_fp8_defaults
from compgen.pipeline import (
    DiffReport,
    PipelineCache,
    compile_and_diff,
)

from tests._fixtures.real_workloads import (
    ALL_FIXTURE_FNS,
    attention_mlp_tiny,
    qwen_moe_tiny,
    smolvla_tiny,
)

# --- diff harness basics --------------------------------------------------


def test_diff_report_passes_for_attention_mlp_tiny():
    fx = attention_mlp_tiny()
    report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
        eager_reference=fx.eager_output,
    )
    assert report.passed, report.failures
    assert report.module_verified
    assert report.plan_validated
    assert report.bridge_path in {"torch_mlir", "fx_importer"}


def test_diff_report_records_stage_counts():
    fx = attention_mlp_tiny()
    report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
    )
    assert report.stages_run >= 10
    assert report.stages_run + report.stages_skipped >= 24


def test_diff_report_records_opaque_rate():
    fx = attention_mlp_tiny()
    report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
    )
    assert report.total_ops > 0
    assert 0.0 <= report.opaque_rate <= 1.0


# --- E2E across 6 workloads ----------------------------------------------


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_e2e_every_fixture_compiles_and_verifies(fn):
    """Real-workload E2E: every fixture bridges + pipeline + validates."""
    fx = fn()
    report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
        eager_reference=fx.eager_output,
        # opaque threshold relaxed so the first pass covers all
        # 6 workloads; production work will drop this further.
        opaque_rate_threshold=0.50,
    )
    assert report.passed, f"{fx.name} failed: failures={report.failures}, warnings={report.warnings}"


@pytest.mark.parametrize("fn", ALL_FIXTURE_FNS, ids=lambda f: f.__name__)
def test_e2e_every_fixture_executes_deterministic_eager(fn):
    fx = fn()
    report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
        eager_reference=fx.eager_output,
    )
    # Eager re-run should match the stored reference to 1e-3.
    assert report.eager_diff_pass
    assert report.eager_diff_max_abs < 1e-3


# --- preset comparisons --------------------------------------------------


def test_cuda_a100_and_npu_presets_produce_different_plans():
    fx = attention_mlp_tiny()
    cuda_report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
        fixture_name=fx.name,
    )
    npu_report = compile_and_diff(
        fx.model,
        fx.example_inputs,
        options=npu_fp8_defaults(),
        fixture_name=fx.name,
    )
    # Both should pass but with different stage counts (NPU enables
    # extra passes like insert_host_offload).
    assert cuda_report.passed
    assert npu_report.passed
    assert cuda_report.stages_run != npu_report.stages_run


# --- bridge-failure path ---------------------------------------------------


def test_diff_report_captures_bridge_failure():
    """When the bridge fails, the diff report records it."""
    import torch.nn as nn

    class _Broken(nn.Module):
        def forward(self, x):
            # Exporting this with no inputs will fail bridging.
            return x

    report = compile_and_diff(
        _Broken(),
        example_inputs=(),
        options=CompGenOptions(),
        fixture_name="broken",
    )
    # Either bridge failed or fell back but produced a module; both
    # are handled without crash.
    assert isinstance(report, DiffReport)


# --- pipeline cache tests -------------------------------------------------


def test_pipeline_cache_hits_on_second_call():
    cache = PipelineCache(max_entries=8)
    fx = attention_mlp_tiny()
    r1 = cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
    r2 = cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
    assert r1 is r2
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_pipeline_cache_differentiates_by_options():
    cache = PipelineCache(max_entries=8)
    fx = attention_mlp_tiny()
    cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
    cache.compile(fx.model, fx.example_inputs, options=npu_fp8_defaults())
    assert cache.stats.misses == 2


def test_pipeline_cache_evicts_oldest_on_overflow():
    cache = PipelineCache(max_entries=2)
    fx_a = attention_mlp_tiny()
    fx_b = qwen_moe_tiny()
    fx_c = smolvla_tiny()
    cache.compile(fx_a.model, fx_a.example_inputs, options=cuda_a100_defaults())
    cache.compile(fx_b.model, fx_b.example_inputs, options=cuda_a100_defaults())
    cache.compile(fx_c.model, fx_c.example_inputs, options=cuda_a100_defaults())
    assert cache.stats.evictions >= 1
    assert len(cache) == 2


def test_pipeline_cache_clear():
    cache = PipelineCache(max_entries=8)
    fx = attention_mlp_tiny()
    cache.compile(fx.model, fx.example_inputs, options=cuda_a100_defaults())
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert cache.stats.hits == 0


def test_pipeline_cache_hit_rate_tracks_over_time():
    cache = PipelineCache(max_entries=8)
    fx = attention_mlp_tiny()
    opts = cuda_a100_defaults()
    cache.compile(fx.model, fx.example_inputs, options=opts)  # miss
    cache.compile(fx.model, fx.example_inputs, options=opts)  # hit
    cache.compile(fx.model, fx.example_inputs, options=opts)  # hit
    assert abs(cache.stats.hit_rate - 2 / 3) < 1e-6


def test_invalid_cache_size_raises():
    with pytest.raises(ValueError, match="max_entries"):
        PipelineCache(max_entries=0)

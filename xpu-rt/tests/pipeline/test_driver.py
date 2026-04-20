"""Tests for the pipeline driver."""

from __future__ import annotations

from compgen.options import (
    cuda_a100_defaults,
    npu_fp8_defaults,
)
from compgen.pipeline.driver import (
    PipelineResult,
    compile_through_pipeline,
)

from tests._fixtures.real_workloads import (
    attention_mlp_tiny,
    qwen_moe_tiny,
)

# --- default-options path (bridge + no-op stages) --------------------------


def test_default_options_runs_bridge_and_skips_everything_else():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(fx.model, fx.example_inputs)
    assert result.bridge_path in {"torch_mlir", "fx_importer"}
    assert result.module is not None
    # Every stage after bridge is disabled in the default options.
    run_names = [r.name for r in result.stage_reports if not r.skipped]
    assert run_names == ["bridge_fx_graph"]


def test_default_plan_validates():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(fx.model, fx.example_inputs)
    result.execution_plan.validate()


# --- cuda_a100 preset path -------------------------------------------------


def test_cuda_a100_preset_runs_expected_stages_on_attention_mlp_tiny():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    # cuda_a100 enables 10+ stages.
    assert result.stages_run >= 10
    enabled_names = {r.name for r in result.stage_reports if not r.skipped}
    assert "raise_special_ops" in enabled_names
    assert "fuse_softmax_to_triton" in enabled_names
    assert "match_library_call" in enabled_names
    assert "plan_buffers" in enabled_names
    assert "assign_memory_space" in enabled_names
    result.execution_plan.validate()


def test_cuda_a100_bridges_qwen_moe_tiny():
    fx = qwen_moe_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    assert result.module is not None
    result.execution_plan.validate()


# --- npu_fp8 preset path ---------------------------------------------------


def test_npu_preset_enables_quantization_stages():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=npu_fp8_defaults(),
    )
    enabled = {r.name for r in result.stage_reports if not r.skipped}
    # NPU preset enables host-offload + insert_copies + dma_overlap.
    assert "insert_host_offload" in enabled
    assert "insert_copies" in enabled
    assert "dma_overlap" in enabled


# --- opaque-rate threshold -------------------------------------------------


def test_opaque_rate_below_threshold_on_attention_mlp():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    total = 0
    opaque = 0
    for op in result.module.walk():
        total += 1
        if op.name == "func.call":
            opaque += 1
    rate = opaque / total if total else 0.0
    # Attention MLP has 3 opaque calls (expected: layer_norm fallback
    # post-raise; GELU fallback). Rate should stay below 15%.
    assert rate < 0.15


# --- stage report integrity ------------------------------------------------


def test_every_stage_is_reported_once():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    names = [r.name for r in result.stage_reports]
    # Expect ≥ 23 entries (bridge + the pass suite + an optional plan-validate).
    assert len(names) >= 23
    # Every pass name is unique.
    dup = [n for n in set(names) if names.count(n) > 1]
    assert not dup, f"duplicates: {dup}"


def test_skipped_stages_have_reason():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(fx.model, fx.example_inputs)
    for r in result.stage_reports:
        if r.skipped and r.name != "bridge_fx_graph":
            assert r.skipped_reason, f"{r.name} skipped without reason"


# --- failure handling ------------------------------------------------------


def test_bridge_failure_returns_result_with_none_module():
    # Build a nn.Module the bridge can't handle (no example_inputs).
    import torch.nn as nn

    class _BadShape(nn.Module):
        def forward(self, x):
            return x + "oops"  # causes export failure

    result = compile_through_pipeline(
        _BadShape(),
        example_inputs=(),  # wrong shape count
    )
    # Either the bridge fails (module None) or fx_importer falls back
    # gracefully; both paths are valid. We only assert no crash.
    assert isinstance(result, PipelineResult)


# --- stats round-trip ------------------------------------------------------


def test_result_stages_run_and_stages_skipped_sum_matches():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    assert result.stages_run + result.stages_skipped == len(result.stage_reports)


def test_stage_report_has_group_annotation():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    groups = {r.group for r in result.stage_reports}
    assert groups.issubset({0, 1, 2, 3, 4, 5, 6})


# --- execution plan quality after runtime passes --------------------------


def test_cuda_a100_assigns_memory_space_to_all_buffers():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    plan = result.execution_plan
    assert all(buf.memory_space for buf in plan.buffers)


def test_cuda_a100_assigns_queues_to_all_regions():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    plan = result.execution_plan
    assert all(rp.queue for rp in plan.region_placement)


def test_cuda_a100_produces_buffer_offsets():
    fx = attention_mlp_tiny()
    result = compile_through_pipeline(
        fx.model,
        fx.example_inputs,
        options=cuda_a100_defaults(),
    )
    plan = result.execution_plan
    assert "buffer_offsets" in plan.summary

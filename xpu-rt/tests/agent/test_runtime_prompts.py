"""Tests for agent runtime prompt modules."""

from __future__ import annotations

import json

from compgen.agent.prompts.runtime_dispatch import (
    DispatchConfig,
    DispatchContext,
    format_prompt as dispatch_prompt,
    parse_response as parse_dispatch,
)
from compgen.agent.prompts.runtime_profile import (
    ProfileHookConfig,
    ProfileHookContext,
    format_prompt as profile_prompt,
    parse_response as parse_profile,
)


# ---- runtime_profile.py ----


class TestProfilePrompt:
    def test_format(self) -> None:
        ctx = ProfileHookContext(
            target_name="cuda-a100",
            available_backends=["nsight_systems", "perf"],
            available_counters=["cycles", "sm_active"],
            current_bottlenecks=[
                {"region": "matmul_0", "kind": "compute_bound",
                 "severity": 0.8, "suggestion": "tile"},
            ],
            tile_profiling_available=True,
            runtime_env="linux_userspace",
        )
        prompt = profile_prompt(ctx)
        assert "cuda-a100" in prompt
        assert "nsight_systems" in prompt
        assert "matmul_0" in prompt
        assert "tile" in prompt

    def test_parse_valid(self) -> None:
        response = json.dumps({
            "instrumentation_level": "TILE_LEVEL",
            "counters_to_enable": ["cycles", "instructions"],
            "custom_hooks": {"pre_dispatch": "my_hook();"},
            "analysis_focus": "memory",
            "reasoning": "memory-bound workload",
        })
        config = parse_profile(response)
        assert config.instrumentation_level == "TILE_LEVEL"
        assert len(config.counters_to_enable) == 2
        assert "pre_dispatch" in config.custom_hooks
        assert config.analysis_focus == "memory"

    def test_parse_markdown(self) -> None:
        response = '```json\n{"instrumentation_level": "OP_LEVEL"}\n```'
        config = parse_profile(response)
        assert config.instrumentation_level == "OP_LEVEL"

    def test_parse_invalid(self) -> None:
        config = parse_profile("not json at all")
        assert "Failed to parse" in config.reasoning

    def test_empty_context(self) -> None:
        ctx = ProfileHookContext(target_name="test")
        prompt = profile_prompt(ctx)
        assert "test" in prompt
        assert "(none detected)" in prompt


# ---- runtime_dispatch.py ----


class TestDispatchPrompt:
    def test_format(self) -> None:
        ctx = DispatchContext(
            target_name="test-soc",
            topology_summary={
                "deployment": "multi_domain_soc",
                "num_nodes": 2,
            },
            device_utilization={"cpu": 30.0, "npu": 85.0},
            num_cross_device_copies=5,
            total_transfer_bytes=1024 * 1024,
            current_strategy="bulk_sync",
            runtime_env="zephyr_rtos",
        )
        prompt = dispatch_prompt(ctx)
        assert "test-soc" in prompt
        assert "zephyr_rtos" in prompt
        assert "multi_domain_soc" in prompt
        assert "bulk_sync" in prompt

    def test_parse_valid(self) -> None:
        response = json.dumps({
            "strategy": "pipeline",
            "transport_overrides": {"host->npu": "zephyr_ipc"},
            "thread_config": {"dispatch": 3, "dma_handler": 2},
            "double_buffer": True,
            "dma_tile_size": 65536,
            "reasoning": "pipeline overlaps compute and DMA",
        })
        config = parse_dispatch(response)
        assert config.strategy == "pipeline"
        assert config.transport_overrides["host->npu"] == "zephyr_ipc"
        assert config.thread_config["dispatch"] == 3
        assert config.double_buffer is True
        assert config.dma_tile_size == 65536

    def test_parse_invalid(self) -> None:
        config = parse_dispatch("garbage")
        assert "Failed to parse" in config.reasoning

    def test_empty_context(self) -> None:
        ctx = DispatchContext(target_name="simple")
        prompt = dispatch_prompt(ctx)
        assert "simple" in prompt
        assert "(single device)" in prompt

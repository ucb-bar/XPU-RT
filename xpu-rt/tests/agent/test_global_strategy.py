"""Tests for cross-module global strategy (Unit 16)."""

from __future__ import annotations

import json

from compgen.agent.prompts.global_strategy import GlobalStrategyContext, format_prompt, parse_response


class TestGlobalStrategy:
    def test_format_includes_modules(self):
        ctx = GlobalStrategyContext(
            module_count=3,
            per_module_summaries=[
                {"name": "encoder", "op_count": 100, "flops": 1000000, "bottleneck": "matmul"},
                {"name": "decoder", "op_count": 200, "flops": 5000000, "bottleneck": "attention"},
                {"name": "head", "op_count": 20, "flops": 50000, "bottleneck": "none"},
            ],
            target_name="gpu_a100",
        )
        prompt = format_prompt(ctx)
        assert "3 total" in prompt
        assert "encoder" in prompt
        assert "decoder" in prompt

    def test_parse_sequential(self):
        text = json.dumps(
            {
                "strategy": "sequential",
                "priority_order": ["decoder", "encoder", "head"],
                "shared_optimizations": ["fuse attention blocks"],
                "reasoning": "decoder is largest",
            }
        )
        result = parse_response(text)
        assert result is not None
        assert result["strategy"] == "sequential"
        assert result["priority_order"] == ["decoder", "encoder", "head"]

    def test_parse_parallel(self):
        text = json.dumps({"strategy": "parallel", "priority_order": ["a", "b"]})
        result = parse_response(text)
        assert result["strategy"] == "parallel"

    def test_parse_invalid(self):
        assert parse_response("garbage") is None

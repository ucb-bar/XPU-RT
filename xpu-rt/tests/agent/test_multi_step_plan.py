"""Tests for multi-step planning prompt (Unit 2)."""
from __future__ import annotations
import json
import pytest
from compgen.agent.prompts.plan_multi_step import PlanContext, format_prompt, parse_response


class TestPlanContext:
    def test_format_prompt_includes_budget(self):
        ctx = PlanContext(
            observation_summary="model has 10 ops",
            history_summary="Step 1: tile matmul_0 +2.1%",
            legal_actions_summary="1. tile matmul_0 -12us [safe]",
            budget_remaining=5,
        )
        prompt = format_prompt(ctx)
        assert "5 iterations" in prompt
        assert "10 ops" in prompt

    def test_format_prompt_with_error_patterns(self):
        ctx = PlanContext(
            observation_summary="model",
            history_summary="",
            legal_actions_summary="",
            budget_remaining=3,
            error_patterns=[{"action_type": "tile", "failure_reason": "dim too small"}],
        )
        prompt = format_prompt(ctx)
        assert "tile" in prompt
        assert "dim too small" in prompt


class TestParsePlan:
    def test_parse_valid_plan(self):
        text = json.dumps([
            {"action_type": "tile", "target": "matmul_0", "reason": "compute bound"},
            {"action_type": "fuse", "target": "gelu_0", "reason": "adjacent"},
            {"action_type": "eqsat", "target": "all", "reason": "cleanup"},
        ])
        result = parse_response(text)
        assert result is not None
        assert len(result) == 3
        assert result[0]["action_type"] == "tile"

    def test_parse_json_in_markdown(self):
        text = "```json\n" + json.dumps([{"action_type": "noop", "target": "", "reason": "done"}]) + "\n```"
        result = parse_response(text)
        assert result is not None
        assert len(result) == 1

    def test_parse_invalid_returns_none(self):
        assert parse_response("not json") is None
        assert parse_response('{"key": "value"}') is None  # not an array

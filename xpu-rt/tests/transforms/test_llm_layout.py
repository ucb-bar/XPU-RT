"""Tests for LLM-guided layout planning (Unit 7)."""

from __future__ import annotations

import json

from compgen.agent.prompts.layout_plan import LayoutPlanContext, format_prompt, parse_response


class TestLayoutPlan:
    def test_format_includes_encoding(self):
        ctx = LayoutPlanContext(
            op_name="linalg.matmul",
            encoding_str="tiled_128x64",
            target_name="gpu_a100",
            capabilities_summary="tensor_core: 312 TFLOPS",
            tile_family_hint="matmul_16x16",
        )
        prompt = format_prompt(ctx)
        assert "tiled_128x64" in prompt
        assert "gpu_a100" in prompt
        assert "matmul_16x16" in prompt

    def test_parse_valid_response(self):
        text = json.dumps(
            {
                "inner_tiles": [16, 16],
                "outer_perm": [0, 1],
                "padding_value": "zero",
                "reasoning": "tensor core alignment",
            }
        )
        result = parse_response(text)
        assert result is not None
        assert result["inner_tiles"] == [16, 16]
        assert result["padding_value"] == "zero"

    def test_parse_minimal_response(self):
        text = json.dumps({"inner_tiles": [32, 32], "reasoning": "ok"})
        result = parse_response(text)
        assert result["inner_tiles"] == [32, 32]

    def test_parse_invalid(self):
        assert parse_response("not json") is None
        assert parse_response('{"key": "value"}') is None


class TestSolverConfig:
    """Also test solver_config prompt (Unit 8)."""

    def test_format_and_parse(self):
        from compgen.agent.prompts.solver_config import SolverConfigContext
        from compgen.agent.prompts.solver_config import format_prompt as fmt_sc
        from compgen.agent.prompts.solver_config import parse_response as parse_sc

        ctx = SolverConfigContext(
            num_regions=50,
            num_devices=2,
            problem_type="placement",
            estimated_complexity="moderate",
            current_timeout_ms=10000,
        )
        prompt = fmt_sc(ctx)
        assert "50" in prompt
        assert "placement" in prompt

        text = json.dumps(
            {
                "timeout_ms": 5000,
                "predicted_hardness": "medium",
                "solver_hints": {"symmetry_breaking": True, "objective_gap_pct": 0.05},
                "reasoning": "moderate problem",
            }
        )
        result = parse_sc(text)
        assert result["timeout_ms"] == 5000
        assert result["predicted_hardness"] == "medium"


class TestRecipeSeed:
    """Also test recipe_seed prompt (Unit 9)."""

    def test_format_and_parse(self):
        from compgen.agent.prompts.recipe_seed import RecipeSeedContext
        from compgen.agent.prompts.recipe_seed import format_prompt as fmt_rs
        from compgen.agent.prompts.recipe_seed import parse_response as parse_rs

        ctx = RecipeSeedContext(
            op_histogram={"matmul": 5, "relu": 10, "add": 20},
            target_name="gpu_a100",
            objective="latency",
            total_flops=1000000,
            total_bytes=500000,
            num_devices=1,
        )
        prompt = fmt_rs(ctx)
        assert "matmul" in prompt
        assert "1,000,000" in prompt

        text = json.dumps(
            {
                "prioritize_ops": ["matmul"],
                "skip_ops": ["add"],
                "default_tile_sizes": {"matmul": [128, 64, 32]},
                "aggressive_fusion": True,
                "reasoning": "matmul dominant",
            }
        )
        result = parse_rs(text)
        assert result["aggressive_fusion"] is True
        assert result["prioritize_ops"] == ["matmul"]


class TestGuardPropose:
    """Also test guard_propose prompt (Unit 10)."""

    def test_format_and_parse(self):
        from compgen.agent.prompts.guard_propose import GuardProposeContext
        from compgen.agent.prompts.guard_propose import format_prompt as fmt_gp
        from compgen.agent.prompts.guard_propose import parse_response as parse_gp

        ctx = GuardProposeContext(
            variable_names=["M", "N", "K"],
            variable_types={"M": "int", "N": "int", "K": "int"},
            positive_examples_summary="  {M=128, N=64, K=32}",
            negative_examples_summary="  {M=3, N=2, K=1}",
            num_positives=10,
            num_negatives=5,
        )
        prompt = fmt_gp(ctx)
        assert "M" in prompt
        assert "10 total" in prompt

        text = json.dumps(
            {
                "fragments": [
                    {"var": "M", "op": ">=", "value": 16},
                    {"var": "N", "op": "%", "divisor": 8, "remainder": 0},
                ],
                "reasoning": "alignment",
            }
        )
        result = parse_gp(text)
        assert len(result) == 2
        assert result[0]["var"] == "M"

"""Tests for verification-related prompts."""

from __future__ import annotations

from compgen.agent.prompts.verify_strategy import (
    VerificationAssignment,
    VerifyStrategyContext,
    format_prompt,
    parse_response,
)
from compgen.agent.prompts.counterexample_repair import (
    CounterexampleRepairContext,
    RepairProposal,
)
from compgen.agent.prompts.counterexample_repair import (
    format_prompt as fmt_repair,
    parse_response as parse_repair,
)


class TestVerifyStrategyPrompt:
    """Test verify strategy prompt formatting and parsing."""

    def test_format_prompt(self) -> None:
        ctx = VerifyStrategyContext(
            regions=[
                {"region_id": "matmul_0", "op_type": "matmul", "transform_applied": "tile"},
                {"region_id": "relu_0", "op_type": "relu", "transform_applied": "fuse"},
            ],
            verification_budget_ms=30000,
            verifiable_ops=["arith.addi", "arith.muli"],
            past_failures=[],
        )
        prompt = format_prompt(ctx)
        assert "matmul_0" in prompt
        assert "30000" in prompt
        assert "arith.addi" in prompt

    def test_parse_response(self) -> None:
        text = '[{"region_id": "matmul_0", "level": "tv"}, {"region_id": "relu_0", "level": "differential"}]'
        assignments = parse_response(text)
        assert len(assignments) == 2
        assert assignments[0].region_id == "matmul_0"
        assert assignments[0].level == "tv"
        assert assignments[1].level == "differential"

    def test_parse_empty_response(self) -> None:
        assignments = parse_response("I don't understand")
        assert assignments == []


class TestCounterexampleRepairPrompt:
    """Test counterexample repair prompt formatting and parsing."""

    def test_format_prompt(self) -> None:
        ctx = CounterexampleRepairContext(
            region_id="matmul_0",
            transform_applied="tile [32, 32]",
            counterexample={
                "inputs": {"arg0": "0xFFFFFFFF", "arg1": "0x00000001"},
                "expected": {"ret0": "0x00000000"},
                "actual": {"ret0": "0x00000002"},
                "summary": "overflow at boundary",
            },
            verification_error="invalid — outputs differ",
            available_alternatives=["tile [64, 64]", "fuse", "noop"],
        )
        prompt = fmt_repair(ctx)
        assert "matmul_0" in prompt
        assert "0xFFFFFFFF" in prompt
        assert "tile [64, 64]" in prompt

    def test_parse_response(self) -> None:
        text = '{"diagnosis": "overflow", "action_type": "tile", "params": {"sizes": [64, 64]}, "reasoning": "larger tiles"}'
        proposal = parse_repair(text)
        assert proposal is not None
        assert proposal.action_type == "tile"
        assert proposal.diagnosis == "overflow"

    def test_parse_noop(self) -> None:
        text = '{"diagnosis": "unfixable", "action_type": "noop", "params": {}, "reasoning": "no safe alternative"}'
        proposal = parse_repair(text)
        assert proposal is not None
        assert proposal.action_type == "noop"

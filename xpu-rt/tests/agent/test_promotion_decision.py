"""Tests for LLM-guided promotion decisions (Unit 16)."""
from __future__ import annotations
import json
import pytest
from compgen.agent.prompts.promotion_decision import PromotionContext, format_prompt, parse_response


class TestPromotionDecision:
    def test_format_includes_improvement(self):
        ctx = PromotionContext(
            improvement_pct=5.2,
            verification_summary="all passed",
            target_name="gpu_a100",
            similar_promoted_count=3,
            iterations_run=10,
            best_latency_us=100.0,
            initial_latency_us=200.0,
        )
        prompt = format_prompt(ctx)
        assert "+5.2%" in prompt
        assert "gpu_a100" in prompt
        assert "200.0" in prompt

    def test_parse_promote_true(self):
        text = json.dumps({"promote": True, "confidence": 0.9, "reason": "significant improvement"})
        result = parse_response(text)
        assert result is not None
        assert result["promote"] is True
        assert result["confidence"] == 0.9

    def test_parse_promote_false(self):
        text = json.dumps({"promote": False, "confidence": 0.3, "reason": "marginal improvement"})
        result = parse_response(text)
        assert result["promote"] is False

    def test_parse_invalid(self):
        assert parse_response("not json") is None

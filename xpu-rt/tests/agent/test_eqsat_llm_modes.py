"""Tests for EqSat LLM modes (Units 4, 5)."""
from __future__ import annotations
import json
import pytest

from compgen.agent.prompts.eqsat_search_state import SearchStateContext, format_prompt as fmt_ss, parse_response as parse_ss
from compgen.agent.prompts.eqsat_extraction_weights import WeightsContext, format_prompt as fmt_ew, parse_response as parse_ew
from compgen.agent.prompts.eqsat_blackbox import BlackboxContext, format_prompt as fmt_bb, parse_response as parse_bb
from compgen.agent.prompts.eqsat_segment import SegmentContext, format_prompt as fmt_seg, parse_response as parse_seg


class TestSearchState:
    def test_format_includes_egraph_summary(self):
        ctx = SearchStateContext(
            egraph_summary="10 eclasses, 50 enodes",
            rule_stats={"commute_add": 5, "zero_add": 2},
            best_cost=100.0,
            iteration=3,
        )
        prompt = fmt_ss(ctx)
        assert "10 eclasses" in prompt
        assert "commute_add: 5" in prompt

    def test_parse_change_weights(self):
        text = json.dumps({"action": "CHANGE_WEIGHTS", "parameters": {}, "reasoning": "cost model suboptimal"})
        result = parse_ss(text)
        assert result is not None
        assert result["action"] == "CHANGE_WEIGHTS"

    def test_parse_propose_rule(self):
        text = json.dumps({"action": "PROPOSE_RULE", "parameters": {}, "reasoning": "stale rules"})
        result = parse_ss(text)
        assert result["action"] == "PROPOSE_RULE"


class TestExtractionWeights:
    def test_format_includes_current_weights(self):
        ctx = WeightsContext(
            egraph_summary="graph",
            target_description="gpu_target",
            current_fusion_weight=1.0,
            current_transfer_weight=0.5,
            current_backend_match_weight=2.0,
        )
        prompt = fmt_ew(ctx)
        assert "1.000" in prompt
        assert "0.500" in prompt

    def test_parse_weights(self):
        text = json.dumps({"fusion_weight": 1.5, "transfer_weight": 0.8, "backend_match_weight": 2.0, "reasoning": "ok"})
        result = parse_ew(text)
        assert result is not None
        assert result["fusion_weight"] == 1.5
        assert result["transfer_weight"] == 0.8


class TestBlackbox:
    def test_format_includes_ops(self):
        ctx = BlackboxContext(
            op_types_counts={"arith.addi": 10, "linalg.matmul": 3},
            current_open=["arith.addi"],
            current_closed=["linalg.matmul"],
            target_name="gpu",
        )
        prompt = fmt_bb(ctx)
        assert "arith.addi: 10" in prompt

    def test_parse_open_close(self):
        text = json.dumps({"open": ["arith.addi"], "close": ["func.call"], "reasoning": "ok"})
        result = parse_bb(text)
        assert result["open"] == ["arith.addi"]


class TestSegment:
    def test_format_includes_threshold(self):
        ctx = SegmentContext(
            op_count=100,
            op_types_summary="matmul: 10, add: 50",
            dataflow_depth=12,
            current_threshold=200,
        )
        prompt = fmt_seg(ctx)
        assert "200" in prompt

    def test_parse_threshold(self):
        text = json.dumps({"threshold": 150, "forced_boundaries": ["region_5"], "reasoning": "ok"})
        result = parse_seg(text)
        assert result["threshold"] == 150

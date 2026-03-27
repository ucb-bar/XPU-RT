"""Tests for LLM protocol types and dataclasses."""

from __future__ import annotations

import pytest
from compgen.llm.base import LLMConfig, Objective, PromptContext


def test_llm_config_defaults() -> None:
    """LLMConfig should have sensible defaults."""
    config = LLMConfig(model="test-model")
    assert config.model == "test-model"
    assert config.temperature == 0.7
    assert config.max_tokens == 8192


def test_objective_enum() -> None:
    """Objective enum should have all expected values."""
    assert Objective.LATENCY.value == "latency"
    assert Objective.THROUGHPUT.value == "throughput"
    assert Objective.MEMORY.value == "memory"
    assert Objective.ENERGY.value == "energy"


def test_prompt_context_construction() -> None:
    """PromptContext should be constructible with required fields."""
    ctx = PromptContext(
        model_ir_summary="test IR",
        target_profile_summary="test target",
        available_transforms=["tile", "vectorize"],
        kernel_contracts=["contract1"],
        objective=Objective.LATENCY,
    )
    assert ctx.objective == Objective.LATENCY
    assert len(ctx.available_transforms) == 2


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_generation_response_token_tracking() -> None:
    """GenerationResponse should track token usage."""

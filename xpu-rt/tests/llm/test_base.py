"""Tests for LLM protocol types and dataclasses."""

from __future__ import annotations

from compgen.llm.base import (
    GenerationRequest,
    GenerationResponse,
    LLMConfig,
    Objective,
    PromptContext,
)
from compgen.llm.mock_client import MockLLMClient


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


def test_generation_response_token_tracking() -> None:
    """GenerationResponse should track token usage."""
    mock = MockLLMClient(strict=False)
    mock.add_response("test IR", "generated transform script content")

    request = GenerationRequest(
        prompt_template="Generate a transform for this model.",
        context=PromptContext(
            model_ir_summary="test IR",
            target_profile_summary="target A100",
            available_transforms=["tile"],
            kernel_contracts=[],
            objective=Objective.LATENCY,
        ),
        config=LLMConfig(model="mock"),
    )

    response = mock.generate(request)

    assert isinstance(response, GenerationResponse)
    assert response.raw_text == "generated transform script content"
    assert response.parsed_artifacts == ["generated transform script content"]
    assert response.model_id == "mock"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0

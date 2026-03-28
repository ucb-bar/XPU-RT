"""Opt-in real-provider smoke tests for CompGen LLM clients."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from compgen.llm import LLMRecorder, create_llm_client
from compgen.llm.base import GenerationRequest, LLMConfig, Objective, PromptContext


def _enabled_backend() -> str:
    if os.environ.get("COMPGEN_RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("Set COMPGEN_RUN_REAL_LLM_TESTS=1 to enable real LLM smoke tests.")
    backend = os.environ.get("COMPGEN_REAL_LLM_BACKEND", "").strip()
    if not backend:
        pytest.skip("Set COMPGEN_REAL_LLM_BACKEND to gemini, openai, claude-cli, codex-cli, or anthropic.")
    return backend


@pytest.mark.slow
def test_real_llm_client_smoke(tmp_path: Path) -> None:
    backend = _enabled_backend()
    model = os.environ.get("COMPGEN_REAL_LLM_MODEL")
    client = create_llm_client(backend, model=model, working_dir=tmp_path)
    recorder = LLMRecorder(client, log_dir=tmp_path / "logs")

    request = GenerationRequest(
        prompt_template="Say ready in one word.",
        context=PromptContext(
            model_ir_summary="region r0",
            target_profile_summary="smoke-target",
            available_transforms=["tile"],
            kernel_contracts=[],
            objective=Objective.LATENCY,
            frontend_diagnostics_summary="graph_breaks=0",
            analysis_dossier_summary="regions=1",
        ),
        config=LLMConfig(model=model or getattr(client, "model", "default"), temperature=0.0, max_tokens=64),
    )

    response = recorder.generate(request)
    assert response.raw_text.strip()
    assert recorder.total_calls == 1

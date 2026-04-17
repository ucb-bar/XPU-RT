"""Opt-in end-to-end smoke test for the real agentic LLM loop."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from compgen.agent.loop import AgenticCompilationLoop
from compgen.agent.env import CompilerEnv
from compgen.capture.torch_export import capture_frontend_artifact
from compgen.ir.payload.import_fx import fx_to_xdsl
from compgen.llm import LLMRecorder, create_llm_client
from compgen.targets.schema import load_profile


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.linear(x))


def _enabled_backend() -> tuple[str, str | None]:
    if os.environ.get("COMPGEN_RUN_REAL_LLM_TESTS") != "1":
        pytest.skip("Set COMPGEN_RUN_REAL_LLM_TESTS=1 to enable real LLM smoke tests.")
    backend = os.environ.get("COMPGEN_REAL_LLM_BACKEND", "").strip()
    if not backend:
        pytest.skip("Set COMPGEN_REAL_LLM_BACKEND to select a real backend.")
    return backend, os.environ.get("COMPGEN_REAL_LLM_MODEL")


@pytest.mark.slow
def test_agentic_loop_runs_with_real_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, model = _enabled_backend()
    target = load_profile("examples/target_profiles/cuda_a100.yaml")

    pytorch_model = _TinyMLP().eval()
    sample_inputs = (torch.randn(2, 8),)
    artifact = capture_frontend_artifact(pytorch_model, sample_inputs)
    module = fx_to_xdsl(artifact.exported_program, **artifact.strict_import_options())

    env = CompilerEnv()
    env.reset(
        module,
        target,
        budget=2,
        exported_program=artifact.exported_program,
        capture_artifact=artifact,
        pytorch_model=pytorch_model,
        sample_inputs=sample_inputs,
    )

    client = create_llm_client(backend, model=model, working_dir=Path.cwd())
    recorder = LLMRecorder(client, log_dir=tmp_path / "llm_logs")
    loop = AgenticCompilationLoop(llm_client=recorder, env=env, budget=1)

    monkeypatch.setattr(loop, "_orchestrate_runtime", lambda obs, tgt: {})
    result = loop.run(target)

    assert recorder.total_calls >= 1
    assert result.best_observation is not None

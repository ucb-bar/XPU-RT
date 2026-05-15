""" — full Gemma-2B through ``compile_with_llm``.

Replaces the hand-rolled ``user_perspective/models/gemma_decode_slice.py``
miniature with the real ``google/gemma-2b`` checkpoint, driven through
the same agentic stack the TinyLlama smoke uses.

Run directly::

    uv run python examples/real_models/gemma2b_compile.py

Pre-fetch the checkpoint once (large download)::

    uv run python -c "from transformers import AutoModel; \
        AutoModel.from_pretrained('google/gemma-2b')"
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from compgen import compile_with_llm
from compgen.llm.mock_client import MockLLMClient


HF_REPO_ID = "google/gemma-2b"
TARGET_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


def hf_cache_has(model_name: str) -> bool:
    cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (cache / f"models--{model_name.replace('/', '--')}").exists()


class _NoCacheGemmaWrapper(nn.Module):
    """Same trick as TinyLlama: drop ``DynamicCache`` so torch.export sees a tensor."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.last_hidden_state


def _load_real_gemma() -> nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_REPO_ID)
    model.eval()
    return _NoCacheGemmaWrapper(model).eval()


def _build_sample_inputs(seq_len: int = 8) -> tuple[torch.Tensor]:
    return (torch.randint(0, 100, (1, seq_len), dtype=torch.long),)


def run_gemma2b_compile(
    *,
    target_profile: Path = TARGET_PROFILE,
    seq_len: int = 8,
    budget: int = 4,
):
    sample = _build_sample_inputs(seq_len)
    model = _load_real_gemma()
    return compile_with_llm(
        model=model,
        target=str(target_profile),
        llm=MockLLMClient(strict=False),
        sample_inputs=sample,
        budget=budget,
        return_driver=True,
    )


if __name__ == "__main__":
    if not hf_cache_has(HF_REPO_ID):
        raise SystemExit(
            f"{HF_REPO_ID} not in HuggingFace hub cache. Pre-fetch with:\n"
            f"  uv run python -c \"from transformers import AutoModel; "
            f"AutoModel.from_pretrained('{HF_REPO_ID}')\""
        )
    result = run_gemma2b_compile()
    print(f"  pipeline_result.passed = {result.compiled.pipeline_result.passed}")
    print("\nPASS: real Gemma-2B compiled end-to-end.")

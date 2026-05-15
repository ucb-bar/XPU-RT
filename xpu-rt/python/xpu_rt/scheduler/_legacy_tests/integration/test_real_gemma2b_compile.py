"""acceptance — real Gemma-2B through ``compile_with_llm``.

Skip-gated by ``transformers`` import + HF hub cache presence.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from examples.real_models.gemma2b_compile import (
    HF_REPO_ID,
    hf_cache_has,
    run_gemma2b_compile,
)

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_RUN_REAL_MODEL_TESTS = os.environ.get("COMPGEN_RUN_REAL_MODEL_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(
        not _RUN_REAL_MODEL_TESTS,
        reason="Set COMPGEN_RUN_REAL_MODEL_TESTS=1 to enable real-model acceptance tests.",
    ),
    pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed"),
    pytest.mark.skipif(
        not hf_cache_has(HF_REPO_ID),
        reason=f"HF hub cache lacks {HF_REPO_ID}; pre-fetch to enable.",
    ),
    pytest.mark.slow,
]


def test_real_gemma2b_compiles_end_to_end() -> None:
    result = run_gemma2b_compile(seq_len=4, budget=2)
    assert result.compiled is not None
    assert result.compiled.pipeline_result.passed, "pipeline gate did not pass on real Gemma-2B"


def test_real_gemma2b_preserves_model_identity() -> None:
    import torch

    result = run_gemma2b_compile(seq_len=4, budget=2)
    sample = (torch.randint(0, 100, (1, 4), dtype=torch.long),)
    with torch.no_grad():
        out = result.compiled.model(*sample)
    assert out.shape[:2] == (1, 4)

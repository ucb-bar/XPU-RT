"""acceptance — real SmolVLA through ``compile_with_llm``.

Skip-gated by ``transformers`` + ``lerobot`` imports + HF hub cache.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from examples.real_models.smolvla_compile import (
    SMOLVLA_REPO_ID,
    hf_cache_has,
    run_smolvla_compile,
)

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
_RUN_REAL_MODEL_TESTS = os.environ.get("COMPGEN_RUN_REAL_MODEL_TESTS") == "1"

pytestmark = [
    pytest.mark.skipif(
        not _RUN_REAL_MODEL_TESTS,
        reason="Set COMPGEN_RUN_REAL_MODEL_TESTS=1 to enable real-model acceptance tests.",
    ),
    pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers not installed"),
    pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed"),
    pytest.mark.skipif(
        not hf_cache_has(SMOLVLA_REPO_ID),
        reason=f"HF hub cache lacks {SMOLVLA_REPO_ID}; pre-fetch to enable.",
    ),
    pytest.mark.slow,
]


def test_real_smolvla_compiles_end_to_end() -> None:
    result = run_smolvla_compile(budget=2)
    assert result.compiled is not None
    assert result.compiled.pipeline_result.passed, "pipeline gate did not pass on real SmolVLA"

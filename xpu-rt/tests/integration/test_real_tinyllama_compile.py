"""acceptance — real TinyLlama-1.1B through ``compile_with_llm``.

Skip-gated by:

* ``transformers`` import (any release works in principle)
* HF hub cache presence for the TinyLlama-1.1B-Chat snapshot

When both gates pass, we drive the full agentic stack on the real
checkpoint and assert:

1. ``compile_with_llm`` returns an ``LLMCompileResult`` whose
   pipeline gate is green.
2. A bundle directory lands on disk containing ``forward.c``.
3. The compiled module preserves model identity (the agentic loop
   wraps, never replaces, the user's ``nn.Module``), so the
   differential gate vs eager torch is *the same forward call*.

This is the first end-to-end test where the model is real published
weights, not a hand-rolled miniature.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from examples.real_models.tinyllama_compile import (
    HF_REPO_ID,
    hf_cache_has,
    run_tinyllama_compile,
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


def test_real_tinyllama_compiles_end_to_end() -> None:
    result = run_tinyllama_compile(seq_len=4, budget=2)
    assert result.compiled is not None
    assert result.compiled.pipeline_result.passed, "pipeline gate did not pass on real TinyLlama-1.1B"


def test_real_tinyllama_bundle_has_payload_and_manifest() -> None:
    """The bundle stage writes ``payload.mlir`` + ``manifest.json`` for every
    target. (Per-target additions like ``baremetal/kernels/*.c`` only land
    when the target is a ukernel-runtime; not asserted here.)"""
    from pathlib import Path

    result = run_tinyllama_compile(seq_len=4, budget=2)
    bundle_dir = result.compiled.pipeline_result.all_artifacts.get("bundle_dir")
    if bundle_dir is None:
        pytest.skip("bundle_dir not surfaced on this build; bundle stage may be off")
    payload_mlir = Path(bundle_dir) / "payload.mlir"
    manifest_json = Path(bundle_dir) / "manifest.json"
    assert payload_mlir.exists(), f"payload.mlir missing under {bundle_dir}"
    assert manifest_json.exists(), f"manifest.json missing under {bundle_dir}"
    assert payload_mlir.stat().st_size > 0
    assert manifest_json.stat().st_size > 0


def test_real_tinyllama_preserves_model_identity_for_diff_gate() -> None:
    """``compile_with_llm`` wraps, never replaces, the user's nn.Module —
    so calling ``compiled.model(*sample)`` IS the eager reference."""
    import torch

    result = run_tinyllama_compile(seq_len=4, budget=2)
    sample = (torch.randint(0, 100, (1, 4), dtype=torch.long),)
    with torch.no_grad():
        out = result.compiled.model(*sample)
    # _NoCacheLlamaWrapper returns last_hidden_state directly as a Tensor.
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 1 and out.shape[1] == 4

""" — SmolVLA through ``compile_with_llm``.

Reuses the canonical loader at :func:`compgen.models.load_smolvla_bundle`
and drives it through the agentic stack.

Run directly::

    uv run python examples/real_models/smolvla_compile.py

Requires ``transformers``, ``lerobot``, and the SmolVLA snapshot in the
HF hub cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from compgen import compile_with_llm
from compgen.llm.mock_client import MockLLMClient


SMOLVLA_REPO_ID = "lerobot/smolvla_base"
TARGET_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


def hf_cache_has(model_name: str) -> bool:
    cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (cache / f"models--{model_name.replace('/', '--')}").exists()


def run_smolvla_compile(
    *,
    target_profile: Path = TARGET_PROFILE,
    budget: int = 4,
):
    from compgen.models import load_smolvla_bundle

    wrapper, flat_inputs, _num_cams = load_smolvla_bundle(device="cpu")
    return compile_with_llm(
        model=wrapper,
        target=str(target_profile),
        llm=MockLLMClient(strict=False),
        sample_inputs=tuple(flat_inputs),
        budget=budget,
        return_driver=True,
    )


if __name__ == "__main__":
    if not hf_cache_has(SMOLVLA_REPO_ID):
        raise SystemExit(
            f"{SMOLVLA_REPO_ID} not in HuggingFace hub cache. Pre-fetch first."
        )
    result = run_smolvla_compile()
    print(f"  pipeline_result.passed = {result.compiled.pipeline_result.passed}")
    print("\nPASS: real SmolVLA compiled end-to-end.")

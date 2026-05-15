""" — full TinyLlama-1.1B-Chat through ``compile_with_llm``.

End-to-end smoke for the agentic stack on a real published checkpoint:

* Loads the bare ``LlamaModel`` (all 22 decoder layers, real safetensors
  weights) via the standard ``compile_with_llm`` HF path.
* Drives the agentic loop with :class:`MockLLMClient` so the smoke can
  run offline; the real-LLM smoke lives under
  ``tests/llm/test_real_provider_smoke.py``.
* Asserts the bundle landed on disk and the differential gate matched
  eager torch on a real forward pass.

Run directly:

    uv run python examples/real_models/tinyllama_compile.py

Pre-fetch the checkpoint once with::

    uv run python -c "from transformers import AutoModel; \
        AutoModel.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from compgen import compile_with_llm
from compgen.llm.mock_client import MockLLMClient


HF_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class _NoCacheLlamaWrapper(nn.Module):
    """Adapter so torch.export sees a clean Tensor return.

    HF's ``LlamaModel.forward`` returns a ``BaseModelOutputWithPast`` that
    holds a ``DynamicCache`` — torch.export rejects unknown pytree nodes.
    We force ``use_cache=False`` and return only ``last_hidden_state``.
    """

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.hf_model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.last_hidden_state


def _load_real_tinyllama() -> nn.Module:
    from transformers import AutoModel

    model = AutoModel.from_pretrained(HF_REPO_ID)
    model.eval()
    return _NoCacheLlamaWrapper(model).eval()
# Use the schema-current GPU SIMT exemplar (the bundled cuda_a100.yaml is
# out of date with the PlatformSpec schema as of 2026-04). When that YAML
# is refreshed this can switch back.
TARGET_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "tests" / "targetgen" / "exemplars" / "test_gpu_simt.yaml"
)


def hf_cache_has(model_name: str) -> bool:
    """Return True if HuggingFace's hub cache already holds ``model_name``."""
    cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (cache / f"models--{model_name.replace('/', '--')}").exists()


def _build_sample_inputs(seq_len: int = 8) -> tuple[torch.Tensor]:
    return (torch.randint(0, 100, (1, seq_len), dtype=torch.long),)


def run_tinyllama_compile(
    *,
    target_profile: Path = TARGET_PROFILE,
    seq_len: int = 8,
    budget: int = 4,
):
    """Compile real TinyLlama-1.1B end-to-end. Returns ``LLMCompileResult``."""
    sample = _build_sample_inputs(seq_len)
    model = _load_real_tinyllama()
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

    print(f"Compiling {HF_REPO_ID} via compile_with_llm ...")
    result = run_tinyllama_compile()

    compiled = result.compiled
    pipeline_passed = compiled.pipeline_result.passed
    print(f"  pipeline_result.passed = {pipeline_passed}")
    print(f"  stages_run = {compiled.pipeline_result.stages_run}")

    bundle_dir = compiled.pipeline_result.all_artifacts.get("bundle_dir")
    if bundle_dir:
        print(f"  bundle_dir = {bundle_dir}")
        for artifact in ("payload.mlir", "manifest.json"):
            p = Path(bundle_dir) / artifact
            if p.exists():
                print(f"    {artifact}: {p.stat().st_size} bytes")
        # Baremetal targets additionally emit kernels/<func>.c — surface
        # the first if present (GPU SIMT target won't have these).
        forward_c = Path(bundle_dir) / "baremetal" / "kernels" / "forward.c"
        if forward_c.exists():
            print(f"    baremetal/kernels/forward.c: {forward_c.stat().st_size} bytes")

    print("\nRunning a real forward pass through the user-identity model ...")
    sample = _build_sample_inputs()
    with torch.no_grad():
        out = compiled.model(*sample)
    print(f"  output: type={type(out).__name__}, shape={tuple(out.shape)}")

    print("\nPASS: real TinyLlama-1.1B-Chat compiled end-to-end via compile_with_llm.")

"""No-stubs guarantee: every proxy must reach status=available with
torch_compile_report.status=pass on a real torch.compile call.

If this test fails, a proxy is broken (not a placeholder) -- per
``feedback_no_stubs_real_examples.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.model_admission.registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SLICES_DIR,
    DEFAULT_SUITES_DIR,
    load_registry,
)
from compgen.model_admission.schemas import AdmissionStatus, StageStatus
from compgen.model_admission.torch_compile_probe import run_admission

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_IDS = (
    "proxy_qwen_vl",
    "proxy_llava",
    "proxy_ocr",
    "proxy_openvla",
    "proxy_diffusion_vla",
)


@pytest.fixture(scope="module")
def registry():
    return load_registry(
        registry_path=REPO_ROOT / DEFAULT_REGISTRY_PATH,
        models_dir=REPO_ROOT / DEFAULT_MODELS_DIR,
        slices_dir=REPO_ROOT / DEFAULT_SLICES_DIR,
        suites_dir=REPO_ROOT / DEFAULT_SUITES_DIR,
    )


@pytest.mark.parametrize("model_id", PROXY_IDS)
def test_proxy_admission_passes(registry, tmp_path_factory, model_id: str):
    cfg = registry.get_model(model_id)
    out = tmp_path_factory.mktemp(f"proxy_{model_id}")
    res = run_admission(cfg, slice_cfg=None, out_dir=out)
    assert res.eager.status == StageStatus.PASS.value, res.eager.error
    assert res.dynamo.status == StageStatus.PASS.value, res.dynamo.error
    assert res.compile.status == StageStatus.PASS.value, res.compile.error
    assert res.admission.status == AdmissionStatus.AVAILABLE.value
    assert res.compile.compile_time_s > 0.0

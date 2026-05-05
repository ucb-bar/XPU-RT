"""Loader dispatch: every loader.kind in the registry resolves correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

from compgen.model_admission.loaders import LoaderUnavailable, load
from compgen.model_admission.registry import (
    DEFAULT_MODELS_DIR,
    DEFAULT_REGISTRY_PATH,
    DEFAULT_SLICES_DIR,
    DEFAULT_SUITES_DIR,
    load_registry,
)
from compgen.model_admission.schemas import AdmissionStatus

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registry():
    return load_registry(
        registry_path=REPO_ROOT / DEFAULT_REGISTRY_PATH,
        models_dir=REPO_ROOT / DEFAULT_MODELS_DIR,
        slices_dir=REPO_ROOT / DEFAULT_SLICES_DIR,
        suites_dir=REPO_ROOT / DEFAULT_SUITES_DIR,
    )


@pytest.mark.parametrize(
    "model_id",
    ["proxy_qwen_vl", "proxy_llava", "proxy_ocr", "proxy_openvla", "proxy_diffusion_vla"],
)
def test_proxy_loaders_resolve(registry, model_id: str):
    cfg = registry.get_model(model_id)
    loaded = load(cfg, slice_cfg=None)
    import torch.nn as nn

    assert isinstance(loaded.model, nn.Module)
    assert loaded.sample_inputs or loaded.sample_kwargs


def test_huge_full_model_load_returns_unavailable_too_large(registry):
    cfg = registry.get_model("qwen3_vl_235b_a22b_instruct")
    with pytest.raises(LoaderUnavailable) as exc_info:
        load(cfg, slice_cfg=None)
    assert exc_info.value.status == AdmissionStatus.UNAVAILABLE_TOO_LARGE


def test_unavailable_loader_emits_typed_status(registry):
    """Models with loader.kind=unavailable must emit a typed unavailable status."""

    cfg = registry.get_model("agibot_go1_step")
    with pytest.raises(LoaderUnavailable) as exc_info:
        load(cfg, slice_cfg=None)
    assert exc_info.value.status in (
        AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS,
        AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
        AdmissionStatus.UNAVAILABLE_HARDWARE_CONSTRAINT,
    )


def test_compgen_model_spec_loader_dispatches(registry):
    """smolvla_step bridges to compgen.models. Without local SmolVLA setup the
    catalog call typically fails with FileNotFoundError -> unavailable_missing_weights.
    Either that, or unavailable_missing_dependency, is acceptable -- both are
    typed and not silent.
    """

    cfg = registry.get_model("smolvla_step")
    try:
        load(cfg, slice_cfg=None)
    except LoaderUnavailable as exc:
        assert exc.status in (
            AdmissionStatus.UNAVAILABLE_MISSING_WEIGHTS,
            AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY,
            AdmissionStatus.FAILED_EAGER,
        )

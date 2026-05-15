"""End-to-end probe wiring on a trivial nn.Module: every report file emitted,
every status field populated, no exception escapes the probe boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from compgen.model_admission.proxies.proxy_ocr import build_proxy
from compgen.model_admission.schemas import (
    AdmissionStatus,
    CompileConfig,
    ExpectedOutcomes,
    InputsSpec,
    ModelConfig,
    ModelLoaderConfig,
    ModelSource,
    StageStatus,
    SupportPolicy,
)
from compgen.model_admission.torch_compile_probe import run_admission


def _proxy_model_config() -> ModelConfig:
    return ModelConfig(
        schema_version="model_config_v1",
        model_id="proxy_ocr",
        family="proxy",
        source=ModelSource(),
        loader=ModelLoaderConfig(kind="proxy", proxy_module="compgen.model_admission.proxies.proxy_ocr"),
        inputs=InputsSpec(kind="page_crop"),
        compile=CompileConfig(),
        support=SupportPolicy(mode="full_or_slice_smoke"),
        expected=ExpectedOutcomes(),
        notes=(),
        raw_path=Path("/tmp/proxy_ocr.yaml"),
    )


def test_probe_emits_all_reports(tmp_path: Path):
    cfg = _proxy_model_config()
    out = tmp_path / "run"
    res = run_admission(cfg, slice_cfg=None, out_dir=out)
    for name in (
        "admission_report.json",
        "eager_report.json",
        "dynamo_report.json",
        "torch_compile_report.json",
        "environment.json",
        "input_summary.json",
    ):
        assert (out / name).exists(), name

    admission = json.loads((out / "admission_report.json").read_text())
    assert admission["schema_version"] == "admission_report_v1"
    assert admission["model_id"] == "proxy_ocr"

    assert res.eager.status == StageStatus.PASS.value
    assert res.dynamo.status == StageStatus.PASS.value
    assert res.compile.status == StageStatus.PASS.value
    assert res.compile.attempted is True
    assert res.compile.compile_time_s >= 0.0
    assert res.admission.status == AdmissionStatus.AVAILABLE.value


def test_probe_handles_unavailable_loader(tmp_path: Path):
    """A misconfigured proxy module name must produce a typed unavailable, not raise."""

    cfg = ModelConfig(
        schema_version="model_config_v1",
        model_id="proxy_missing",
        family="proxy",
        source=ModelSource(),
        loader=ModelLoaderConfig(kind="proxy", proxy_module="nonexistent.module.path"),
        inputs=InputsSpec(),
        compile=CompileConfig(),
        support=SupportPolicy(),
        expected=ExpectedOutcomes(),
        notes=(),
        raw_path=Path("/tmp/missing.yaml"),
    )
    out = tmp_path / "run"
    res = run_admission(cfg, slice_cfg=None, out_dir=out)
    assert res.admission.status == AdmissionStatus.UNAVAILABLE_MISSING_DEPENDENCY.value
    assert (out / "error.txt").exists()


def test_proxy_inputs_serialise_to_input_summary(tmp_path: Path):
    cfg = _proxy_model_config()
    out = tmp_path / "run"
    run_admission(cfg, slice_cfg=None, out_dir=out)
    payload = json.loads((out / "input_summary.json").read_text())
    assert payload["model_id"] == "proxy_ocr"
    # proxy_ocr has one positional tensor input.
    assert len(payload["positional_inputs"]) == 1
    assert payload["positional_inputs"][0]["kind"] == "tensor"


def test_proxy_module_smoke():
    """Direct factory call still works -- guards against regressions in build_proxy."""
    import torch

    model, inputs = build_proxy()
    with torch.no_grad():
        out = model(*inputs)
    assert out.shape[0] == 1

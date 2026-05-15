"""Tests for TorchAO quantization pipeline types."""

from __future__ import annotations

import pytest
from compgen.capture.torchao_pipeline import AccuracyReport, QuantizationConfig


def test_quantization_config_construction() -> None:
    """QuantizationConfig should be constructible with required fields."""
    config = QuantizationConfig(scheme="int8_weight_only")
    assert config.scheme == "int8_weight_only"
    assert config.calibration_samples == 100
    assert config.group_size is None
    assert config.extra_args == {}


def test_accuracy_report_construction() -> None:
    """AccuracyReport should be constructible with all required fields."""
    report = AccuracyReport(
        l2_error=0.001,
        max_abs_error=0.005,
        cosine_similarity=0.9999,
        within_tolerance=True,
        tolerance=0.01,
    )
    assert report.l2_error == 0.001
    assert report.max_abs_error == 0.005
    assert report.cosine_similarity == 0.9999
    assert report.within_tolerance is True
    assert report.tolerance == 0.01


def test_apply_quantization_with_int8() -> None:
    """apply_quantization should quantize a model with int8_weight_only."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")

    from compgen.capture.torchao_pipeline import apply_quantization

    class SimpleLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(64, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = SimpleLinear()
    model.eval()

    config = QuantizationConfig(scheme="int8_weight_only")
    quantized = apply_quantization(model, config)

    # The model should still be callable and produce output of correct shape
    x = torch.randn(2, 64)
    with torch.no_grad():
        out = quantized(x)
    assert out.shape == (2, 32)


def test_verify_quant_accuracy_reports_errors() -> None:
    """verify_quant_accuracy should report L2 and cosine similarity metrics."""
    torch = pytest.importorskip("torch")

    from compgen.capture.torchao_pipeline import verify_quant_accuracy

    class IdentityModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class NoisyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 0.001

    original = IdentityModel()
    noisy = NoisyModel()
    test_inputs = (torch.randn(4, 8),)

    report = verify_quant_accuracy(original, noisy, test_inputs, tolerance=0.01)

    assert isinstance(report, AccuracyReport)
    assert report.l2_error > 0.0
    assert report.max_abs_error > 0.0
    assert report.cosine_similarity > 0.0
    assert report.tolerance == 0.01
    # The noise is 0.001 which is within tolerance 0.01
    assert report.within_tolerance is True

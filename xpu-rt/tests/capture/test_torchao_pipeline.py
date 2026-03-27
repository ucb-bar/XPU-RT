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


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_apply_quantization_with_int8() -> None:
    """apply_quantization should quantize a model with int8_weight_only."""


@pytest.mark.skip(reason="scaffold only -- implementation pending")
def test_verify_quant_accuracy_reports_errors() -> None:
    """verify_quant_accuracy should report L2 and cosine similarity metrics."""

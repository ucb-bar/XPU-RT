"""Tests for wave-6 C.4 — FORMAT_PRESETS + tolerance_for_format."""

from __future__ import annotations

import pytest
from compgen.semantic.verify.compare import (
    DTYPE_PRESETS,
    FORMAT_PRESETS,
    ComparisonConfig,
    tolerance_for_format,
)


def test_dtype_presets_unchanged():
    """Backward compatibility — we must not silently mutate the torch-keyed dict."""
    assert len(DTYPE_PRESETS) == 4  # float32, float16, bfloat16, int8


def test_format_presets_covers_all_quantization_families():
    required_prefixes = (
        "float8_e4m3fn",
        "float8_e5m2",
        "int4",
        "uint4",
        "intx",
        "mx4",
        "mx6",
        "mx9",
        "nvfp4",
    )
    for prefix in required_prefixes:
        assert any(k == prefix or k.startswith(f"{prefix}_") for k in FORMAT_PRESETS), (
            f"no FORMAT_PRESETS entry for {prefix}"
        )


def test_tolerance_for_format_exact_key():
    cfg = tolerance_for_format("int4_per_channel")
    assert cfg.atol == 0.5
    assert cfg.rtol == 2e-2


def test_tolerance_for_format_suffix_strip_fallback():
    # "int4_per_tensor" explicitly exists; test an untouched suffix form.
    cfg = tolerance_for_format("int4_per_block")  # not explicitly in table
    # Suffix-stripping falls back to the bare ``int4`` entry.
    assert cfg.atol == 0.5
    assert cfg.rtol == 0.0


def test_tolerance_for_format_unknown_tag_default():
    cfg = tolerance_for_format("totally_unknown_format")
    # Permissive FP16-like default
    assert cfg.atol == 1e-2
    assert cfg.rtol == 1e-2


def test_tolerance_for_format_explicit_default():
    override = ComparisonConfig(atol=42.0, rtol=0.1)
    cfg = tolerance_for_format("unknown", default=override)
    assert cfg is override


@pytest.mark.parametrize(
    "tag,expected_atol",
    [
        ("float8_e4m3fn", 1e-2),
        ("float8_e5m2", 1.5e-2),
        ("int4_per_group", 0.5),
        ("uint4_per_channel", 0.5),
        ("intx_per_group", 1.0),
        ("mx4", 2e-2),
        ("nvfp4_block", 1.5e-2),
    ],
)
def test_presets_have_expected_atol(tag, expected_atol):
    cfg = tolerance_for_format(tag)
    assert cfg.atol == expected_atol

"""Tests for ``compgen.options.CompGenOptions`` + presets."""

from __future__ import annotations

import pytest
from compgen.options import (
    CompGenOptions,
    cuda_a100_defaults,
    cuda_h100_defaults,
    npu_fp8_defaults,
)

# --- defaults ---------------------------------------------------------------


def test_all_passes_off_by_default():
    opts = CompGenOptions()
    true_flags = [
        name
        for name, value in opts.to_dict().items()
        if value is True and name != "enable_differential_test" and name != "fuse_dequant_reassoc_safe"
    ]
    assert true_flags == [], f"unexpected True flags by default: {true_flags}"


def test_default_numerics_policy_preserves_f32():
    assert CompGenOptions().numerics_policy == "preserve_f32"


def test_default_scheduling_policy_is_static():
    assert CompGenOptions().scheduling_policy == "static"


# --- validation -------------------------------------------------------------


def test_invalid_numerics_policy_raises():
    with pytest.raises(ValueError, match="numerics_policy"):
        CompGenOptions(numerics_policy="quantum")


def test_invalid_scheduling_policy_raises():
    with pytest.raises(ValueError, match="scheduling_policy"):
        CompGenOptions(scheduling_policy="omniscient")


def test_invalid_reduction_policy_raises():
    with pytest.raises(ValueError, match="reduction_policy"):
        CompGenOptions(reduction_policy="genius")


def test_invalid_quantized_matmul_policy_raises():
    with pytest.raises(ValueError, match="quantized_matmul_policy"):
        CompGenOptions(quantized_matmul_policy="yolo")


def test_invalid_demote_target_type_raises():
    with pytest.raises(ValueError, match="demote_target_type"):
        CompGenOptions(demote_target_type="i4")


def test_negative_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance_atol"):
        CompGenOptions(regression_tolerance_atol=-0.1)


# --- hashability + stable_key ----------------------------------------------


def test_options_are_frozen_and_hashable():
    o = CompGenOptions()
    # frozen=True -> hashable
    hash(o)


def test_stable_key_is_hashable():
    hash(cuda_a100_defaults().stable_key())


def test_stable_key_is_deterministic():
    k1 = cuda_a100_defaults().stable_key()
    k2 = cuda_a100_defaults().stable_key()
    assert k1 == k2


# --- serialization ---------------------------------------------------------


def test_to_dict_round_trip_preserves_equality():
    orig = npu_fp8_defaults()
    round = CompGenOptions.from_dict(orig.to_dict())
    assert orig == round


def test_to_dict_converts_frozenset_to_sorted_list():
    opts = CompGenOptions(kernel_family_allowlist=frozenset({"triton", "cublas"}))
    out = opts.to_dict()
    assert out["kernel_family_allowlist"] == ["cublas", "triton"]


# --- replace ---------------------------------------------------------------


def test_replace_returns_new_instance():
    a = CompGenOptions()
    b = a.replace(enable_raise_special_ops=True)
    assert a.enable_raise_special_ops is False
    assert b.enable_raise_special_ops is True
    assert a != b


def test_replace_preserves_frozenset_fields():
    a = CompGenOptions(library_allowlist=frozenset({"triton", "cublas"}))
    b = a.replace(enable_match_library_call=True)
    assert b.library_allowlist == frozenset({"triton", "cublas"})


# --- presets ---------------------------------------------------------------


def test_cuda_a100_preset_enables_triton_kernels():
    opts = cuda_a100_defaults()
    assert opts.enable_fuse_softmax_to_triton
    assert "triton" in opts.kernel_family_allowlist


def test_cuda_h100_preset_enables_dma_overlap_and_fp16():
    opts = cuda_h100_defaults()
    assert opts.enable_dma_overlap
    assert opts.demote_target_type == "f16"


def test_npu_preset_enables_quant_passes():
    opts = npu_fp8_defaults()
    assert opts.enable_lower_quantized_matmul
    assert opts.enable_fuse_dequant_matmul
    assert opts.enable_normalize_subbyte
    assert opts.enable_insert_host_offload

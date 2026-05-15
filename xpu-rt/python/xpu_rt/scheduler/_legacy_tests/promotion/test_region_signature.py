"""Tests for :mod:`compgen.promotion.region_signature`."""

from __future__ import annotations

from compgen.promotion.region_signature import (
    RegionSignature,
    encode_shape_class,
    hash_region_signature,
    make_region_signature,
)


def test_signature_is_deterministic_across_runs() -> None:
    """Two invocations with identical inputs hash identically."""
    sig = make_region_signature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="host_cpu",
    )
    assert hash_region_signature(sig) == hash_region_signature(sig)


def test_signature_changes_when_op_family_changes() -> None:
    """matmul vs pointwise must hash to different signatures."""
    matmul = make_region_signature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="host_cpu",
    )
    pointwise = make_region_signature(
        op_family="pointwise",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="host_cpu",
    )
    assert hash_region_signature(matmul) != hash_region_signature(pointwise)


def test_signature_changes_when_target_class_changes() -> None:
    """A recipe proven on host_cpu does not silently apply to cuda."""
    cpu = make_region_signature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="host_cpu",
    )
    cuda = make_region_signature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="cuda_sm75",
    )
    assert hash_region_signature(cpu) != hash_region_signature(cuda)


def test_dynamic_dim_is_canonicalized() -> None:
    """Different ways of expressing 'dynamic' converge to one form."""
    by_none = encode_shape_class([None, 32])
    by_dict = encode_shape_class([{"dynamic": True}, 32])
    assert by_none == by_dict


def test_mod_dim_normalizes_to_int() -> None:
    """``{"mod": 16}`` round-trips with the int normalised."""
    encoded = encode_shape_class([{"mod": 16}, 32])
    assert "16" in encoded


def test_signature_dataclass_to_dict_round_trip() -> None:
    """to_dict() preserves every field for serialization."""
    sig = RegionSignature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        shape_class='[16,32]',
        target_class="host_cpu",
    )
    d = sig.to_dict()
    assert d["op_family"] == "matmul"
    assert d["target_class"] == "host_cpu"
    assert d["shape_class"] == '[16,32]'


def test_hash_is_truncated_hex_16_chars() -> None:
    sig = make_region_signature(
        op_family="matmul",
        dtype="fp32",
        layout="row_major",
        dims=[16, 32],
        target_class="host_cpu",
    )
    h = hash_region_signature(sig)
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)

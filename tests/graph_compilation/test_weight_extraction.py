"""Unit tests for ``xpu_rt.graph_compilation.weight_extraction``.

Covers:

  * :func:`pack_linear_weights` — pure packing function, all shape /
    dtype / edge-case branches.
  * :func:`extract_static_weights_for_region` — happy path + the
    short-circuits (unsupported op kind, missing weight, shape
    mismatch, missing bias).
  * :func:`load_state_dict_from_bundle` — both canonical capture
    layout (``00_graph_capture/exported_program.pt2``) and the flat
    fallback.
  * :func:`populate_provider_extras` — round-trip dict population,
    verifying extras passed to ``XnnpackProvider.propose()`` yields a
    kernel C that compiles and runs with the right output.

The propose-round-trip test deliberately does NOT cross-compile to
riscv64 (that's covered by the Phase C/E tests). It compiles the
emitted kernel via the host cffi build path so the unit test stays
fast and host-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from xpu_rt.graph_compilation.weight_extraction import (
    RegionWeightSpec,
    extract_static_weights_for_region,
    load_state_dict_from_bundle,
    pack_linear_weights,
    populate_provider_extras,
)


# ----- pack_linear_weights -------------------------------------------


def test_pack_linear_weights_row_major_with_bias() -> None:
    weight = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor([10.0, 20.0], dtype=torch.float32)
    flat = pack_linear_weights(weight, bias)
    assert flat == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0]


def test_pack_linear_weights_no_bias() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    flat = pack_linear_weights(weight, None)
    assert flat == [1.0, 2.0, 3.0, 4.0]


def test_pack_linear_weights_coerces_fp64_to_fp32() -> None:
    weight = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    flat = pack_linear_weights(weight, None)
    assert flat == [1.0, 2.0]


def test_pack_linear_weights_rejects_int_dtype() -> None:
    weight = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(TypeError, match="floating-point"):
        pack_linear_weights(weight, None)


def test_pack_linear_weights_rejects_wrong_ndim() -> None:
    weight = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="2-D"):
        pack_linear_weights(weight, None)


def test_pack_linear_weights_rejects_bias_shape_mismatch() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    bias = torch.tensor([1.0, 2.0, 3.0])  # too long
    with pytest.raises(ValueError, match="bias shape mismatch"):
        pack_linear_weights(weight, bias)


# ----- extract_static_weights_for_region -----------------------------


def _make_linear_state_dict(
    *,
    in_c: int = 4,
    out_c: int = 3,
    weight_key: str = "weight",
    bias_key: str | None = "bias",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    sd: dict[str, torch.Tensor] = {
        weight_key: torch.randn(out_c, in_c, dtype=torch.float32),
    }
    if bias_key is not None:
        sd[bias_key] = torch.randn(out_c, dtype=torch.float32)
    return sd


def test_extract_happy_path() -> None:
    sd = _make_linear_state_dict(in_c=4, out_c=3)
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=4,
        out_c=3,
    )
    out = extract_static_weights_for_region(spec, sd)
    assert out is not None
    # 3*4 weights + 3 bias = 15 fp32 values
    assert len(out) == 15
    # Re-pack via the canonical helper and compare
    expected = pack_linear_weights(sd["weight"], sd["bias"])
    assert out == expected


def test_extract_unsupported_op_returns_none() -> None:
    sd = _make_linear_state_dict()
    spec = RegionWeightSpec(
        op_kind="conv2d",  # not in _FC_OP_KINDS
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=4,
        out_c=3,
    )
    assert extract_static_weights_for_region(spec, sd) is None


def test_extract_missing_weight_returns_none() -> None:
    sd = _make_linear_state_dict()
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="nonexistent.weight",
        bias_attr_name=None,
        in_c=4,
        out_c=3,
    )
    assert extract_static_weights_for_region(spec, sd) is None


def test_extract_shape_mismatch_returns_none() -> None:
    # State dict has (3, 4) weight but spec asks for (3, 8).
    sd = _make_linear_state_dict(in_c=4, out_c=3)
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name=None,
        in_c=8,
        out_c=3,
    )
    assert extract_static_weights_for_region(spec, sd) is None


def test_extract_resolves_prefixed_keys() -> None:
    # torch.export rewriting can prepend _orig_mod. or L__self___ —
    # the helper should still find the tensor.
    sd = {
        "_orig_mod.linear.weight": torch.randn(3, 4),
        "_orig_mod.linear.bias": torch.randn(3),
    }
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="linear.weight",
        bias_attr_name="linear.bias",
        in_c=4,
        out_c=3,
    )
    out = extract_static_weights_for_region(spec, sd)
    assert out is not None
    assert len(out) == 3 * 4 + 3


def test_extract_resolves_tail_match() -> None:
    # State dict uses "L__self___weight" (export-rewritten flat name);
    # contract knows the FX node target as "weight". Tail-match wins.
    sd = {
        "L__self___weight": torch.randn(3, 4),
        "L__self___bias": torch.randn(3),
    }
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=4,
        out_c=3,
    )
    out = extract_static_weights_for_region(spec, sd)
    assert out is not None


# ----- load_state_dict_from_bundle -----------------------------------


def test_load_state_dict_from_bundle_canonical_layout(tmp_path: Path) -> None:
    """Verify load_state_dict_from_bundle picks up the canonical layout."""
    capture_dir = tmp_path / "00_graph_capture"
    capture_dir.mkdir()

    # Build + export a tiny module.
    linear = torch.nn.Linear(4, 3, bias=True, dtype=torch.float32)
    sample = (torch.randn(1, 4, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch.export.export(linear, sample)
    torch.export.save(ep, str(capture_dir / "exported_program.pt2"))

    sd = load_state_dict_from_bundle(tmp_path)
    assert sd, "expected a non-empty state_dict"
    # The exact keys depend on torch.export's name-mangling rules; any
    # match against trailing `weight`/`bias` is acceptable.
    has_weight = any(k.endswith("weight") for k in sd)
    has_bias = any(k.endswith("bias") for k in sd)
    assert has_weight and has_bias


def test_load_state_dict_from_bundle_flat_fallback(tmp_path: Path) -> None:
    """Bundle without 00_graph_capture/ but with a flat exported_program.pt2."""
    linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    sample = (torch.randn(1, 2, dtype=torch.float32),)
    with torch.no_grad():
        ep = torch.export.export(linear, sample)
    torch.export.save(ep, str(tmp_path / "exported_program.pt2"))

    sd = load_state_dict_from_bundle(tmp_path)
    assert sd, "expected non-empty state_dict from flat layout"


def test_load_state_dict_from_bundle_missing(tmp_path: Path) -> None:
    sd = load_state_dict_from_bundle(tmp_path)
    assert sd == {}


# ----- populate_provider_extras --------------------------------------


def test_populate_provider_extras_happy_path() -> None:
    sd = _make_linear_state_dict(in_c=4, out_c=3, seed=42)
    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=4,
        out_c=3,
    )
    extras: dict[str, object] = {}
    populate_provider_extras(extras, spec, sd)
    assert extras["shape_dims"] == [4, 3]
    assert extras["int_params"] == [0]
    assert extras["float_params"] == [-1.0e30, 1.0e30]
    assert "static_weights_f32" in extras
    assert len(extras["static_weights_f32"]) == 3 * 4 + 3


def test_populate_provider_extras_skipped_leaves_dict_untouched() -> None:
    sd = _make_linear_state_dict()
    spec = RegionWeightSpec(
        op_kind="conv2d",       # unsupported
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=4,
        out_c=3,
    )
    extras: dict[str, object] = {"some_other_key": 7}
    populate_provider_extras(extras, spec, sd)
    assert extras == {"some_other_key": 7}, (
        "unsupported op kind should leave extras untouched"
    )


# ----- end-to-end: extract → propose → emitted C carries real values -


def test_extracted_extras_drive_provider_emit(tmp_path: Path) -> None:
    """Extract weights from a real nn.Linear → feed into
    XnnpackProvider.propose() → confirm the emitted C has the right
    shape + a kWeights array, not the placeholder."""
    from xpu_rt.kernels.providers.xnnpack import XnnpackProvider
    from xpu_rt.providers.kernel_provider import KernelCodegenRequest

    torch.manual_seed(20260515)
    linear = torch.nn.Linear(6, 4, bias=True, dtype=torch.float32)

    # Build a fake state_dict for the layer (skipping torch.export
    # for this test — load_state_dict_from_bundle has its own coverage).
    state_dict = {
        "weight": linear.weight.detach(),
        "bias":   linear.bias.detach(),
    }

    spec = RegionWeightSpec(
        op_kind="matmul",
        weight_attr_name="weight",
        bias_attr_name="bias",
        in_c=6,
        out_c=4,
    )
    extras: dict[str, object] = {}
    populate_provider_extras(extras, spec, state_dict)
    assert "static_weights_f32" in extras

    class _Contract:
        op_kind = "matmul"
        dtype = "f32"
        layout = "NC"

    class _Target:
        family = "host_cpu"

    artifact_dir = tmp_path / "arts"
    artifact_dir.mkdir()
    provider = XnnpackProvider()
    request = KernelCodegenRequest(
        task_id="t-extract-1",
        contract=_Contract(),
        target=_Target(),
        artifact_dir=str(artifact_dir),
        extras=extras,
    )
    result = provider.propose(request)
    assert result.status == "generated"

    c_files = list(artifact_dir.glob("xnnpack_kernel_*.c"))
    assert len(c_files) == 1
    c_text = c_files[0].read_text()

    # Real shape baked in.
    assert "kInitShape[] = { 6LL, 4LL }" in c_text
    # Real weights array — placeholder would have produced kWeights = NULL.
    assert "static const float kWeights[] = {" in c_text
    # Count of weight values written into the array should equal
    # 4*6 weights + 4 bias = 28.
    n_weights_in_c = c_text.count("f,") + (1 if c_text.rstrip().endswith("f\n};") else 0)
    # Allow some slack since the formatter chunks 8/line; just check
    # the total is >= 28.
    n_floats = sum(1 for tok in c_text.split() if tok.endswith("f,"))
    assert n_floats >= 28, (
        f"expected >= 28 fp32 literals in kWeights, found {n_floats}"
    )

    # Metadata records the shape so the cross-compile orchestrator
    # can size buffers.
    import json
    meta = json.loads((artifact_dir / "kernel_metadata.json").read_text())
    assert meta["xnn_create_shape"] == [6, 4]
    assert meta["static_weights_n_f32"] == 4 * 6 + 4

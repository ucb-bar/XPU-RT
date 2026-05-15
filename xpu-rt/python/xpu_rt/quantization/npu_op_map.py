"""NPU operator classification and ISA mapping.

Classifies every PyTorch ATen operator relevant to the quantized smolVLA model
into an NPU execution category and maps it to the corresponding NPU ISA
mnemonic.  This ensures complete coverage: every op from pi0-quant's 22
tracked operations has an NPU execution unit assignment.

NPU execution categories:

    **MXU_FP8** -- Matrix multiply units (systolic/tree).  FP8 E4M3 inputs,
    BF16 accumulation.  Maps to ``vmatmul.mxu0`` / ``vmatmul.mxu1``.

    **VPU_BF16** -- Vector processing unit.  BF16 elementwise and
    transcendental ops.  Maps to ``vadd.bf16``, ``vmul.bf16``, ``vexp.bf16``,
    etc.

    **XLU_BF16** -- Tensor transform/reduction unit.  BF16 reductions and
    transposes.  Maps to ``vredsum.bf16``, ``vredmax.bf16``, ``vtrpose.xlu``.

    **PASSTHROUGH** -- No compute; metadata-only ops (reshape, view, permute).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NpuOpCategory(str, Enum):
    """NPU execution unit classification for an ATen operator."""

    MXU_FP8 = "mxu_fp8"
    """Matrix unit: FP8 inputs, BF16 accumulation (32x32 tile, 32-cycle latency)."""

    VPU_BF16 = "vpu_bf16"
    """Vector unit: BF16 elementwise / transcendental (2-8 cycle latency)."""

    XLU_BF16 = "xlu_bf16"
    """Reduction / transform unit: BF16 reductions and transposes (4-cycle latency)."""

    PASSTHROUGH = "passthrough"
    """No compute: reshape, view, permute, cat, etc."""


@dataclass(frozen=True)
class NpuQuantDecision:
    """Per-operator quantization decision for NPU deployment.

    Attributes:
        category: NPU execution unit.
        input_dtype: Expected input dtype (``"fp8_e4m3"`` or ``"bf16"``).
        compute_dtype: Internal compute dtype (``"bf16"`` for all NPU units).
        output_dtype: Output dtype (``"bf16"`` for accumulators, ``"fp8_e4m3"`` if packed).
        scale_format: Scale register format (``"e8m0"`` for po2, ``None`` for unscaled).
        isa_mnemonic: NPU ISA instruction mnemonic (if directly mappable).
    """

    category: NpuOpCategory
    input_dtype: str
    compute_dtype: str = "bf16"
    output_dtype: str = "bf16"
    scale_format: str | None = None
    isa_mnemonic: str | None = None


# ---------------------------------------------------------------------------
# Complete operator mapping table
# ---------------------------------------------------------------------------
# Every op from pi0-quant's operator inventory is covered, plus additional
# ops needed for model infrastructure (softmax, embedding, etc.).
# Matrix ops (4):
#   nn.Linear, Conv2d, SDPA score matmul, SDPA AV matmul
# Vector ops (18 from pi0-quant VectorQuantMode):
#   add, sub, mul, div, pow, reciprocal, sqrt, sin, cos, tanh,
#   log2, exp, exp2, amax, sum(default), sum(dim)
# Additional model ops:
#   softmax, embedding, layer_norm, reshape, view, etc.

_OP_TABLE: dict[str, NpuQuantDecision] = {
    # ======================================================================
    # Matrix operations -> MXU (FP8 inputs, BF16 accumulation)
    # ======================================================================
    "aten.linear.default": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    "aten.mm.default": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    "aten.addmm.default": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    "aten.bmm.default": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    "aten.conv2d": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    "aten.convolution.default": NpuQuantDecision(
        NpuOpCategory.MXU_FP8,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vmatmul.mxu0",
    ),
    # ======================================================================
    # Elementwise binary -> VPU (BF16)
    # ======================================================================
    "aten.add.Tensor": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vadd.bf16",
    ),
    "aten.sub.Tensor": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vsub.bf16",
    ),
    "aten.mul.Tensor": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vmul.bf16",
    ),
    "aten.mul.Scalar": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vmul.bf16",
    ),
    "aten.div.Tensor": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vrecip.bf16",
    ),
    "aten.pow.Tensor_Scalar": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vmul.bf16",
    ),
    # ======================================================================
    # Elementwise unary -> VPU (BF16)
    # ======================================================================
    "aten.reciprocal.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vrecip.bf16",
    ),
    "aten.sqrt.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vsqrt.bf16",
    ),
    "aten.sin.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vsin.bf16",
    ),
    "aten.cos.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vcos.bf16",
    ),
    "aten.tanh.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vtanh.bf16",
    ),
    "aten.log2.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vlog2.bf16",
    ),
    "aten.exp.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vexp.bf16",
    ),
    "aten.exp2.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vexp2.bf16",
    ),
    "aten.relu.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vrelu.bf16",
    ),
    "aten.gelu.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        None,
    ),
    "aten.silu.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        None,
    ),
    # ======================================================================
    # Softmax -> VPU (ALWAYS BF16, never FP8)
    # ======================================================================
    "aten._softmax.default": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        None,
    ),
    "aten.softmax.int": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        None,
    ),
    # ======================================================================
    # Reductions -> XLU (BF16)
    # ======================================================================
    "aten.amax.default": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vredmax.bf16",
    ),
    "aten.sum.default": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vredsum.bf16",
    ),
    "aten.sum.dim_IntList": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vredsum.bf16",
    ),
    "aten.mean.dim": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vredsum.bf16",
    ),
    "aten.var.correction": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        None,
    ),
    # ======================================================================
    # Transpose -> XLU
    # ======================================================================
    "aten.t.default": NpuQuantDecision(
        NpuOpCategory.XLU_BF16,
        "bf16",
        "bf16",
        "bf16",
        None,
        "vtrpose.xlu",
    ),
    # ======================================================================
    # Quantize / dequantize -> VPU (hardware pack/unpack)
    # ======================================================================
    "npu.pack_bf16_to_fp8": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "bf16",
        "bf16",
        "fp8_e4m3",
        "e8m0",
        "vpack.bf16.fp8",
    ),
    "npu.unpack_fp8_to_bf16": NpuQuantDecision(
        NpuOpCategory.VPU_BF16,
        "fp8_e4m3",
        "bf16",
        "bf16",
        "e8m0",
        "vunpack.fp8.bf16",
    ),
    # ======================================================================
    # Passthrough / metadata-only ops
    # ======================================================================
    "aten.view.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.reshape.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.permute.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.expand.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.slice.Tensor": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.cat.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.split.Tensor": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.unsqueeze.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.squeeze.dim": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.contiguous.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.clone.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.embedding.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.select.int": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten._to_copy.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.detach.default": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "aten.clamp.default": NpuQuantDecision(NpuOpCategory.VPU_BF16, "bf16", "bf16", "bf16", None, None),
    "aten.abs.default": NpuQuantDecision(NpuOpCategory.VPU_BF16, "bf16", "bf16", "bf16", None, None),
    "aten.neg.default": NpuQuantDecision(NpuOpCategory.VPU_BF16, "bf16", "bf16", "bf16", None, None),
    # Passthrough ops from dynamo built-in normalization
    "passthrough.getitem": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.setitem": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.arange": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.empty_like": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.full": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.zeros": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.ones": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.where": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.masked_fill": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.index_select": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.gather": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
    "passthrough.scatter": NpuQuantDecision(NpuOpCategory.PASSTHROUGH, "any", "any", "any"),
}


# ---------------------------------------------------------------------------
# The 22 pi0-quant ops that must all be classified
# ---------------------------------------------------------------------------

PI0_QUANT_OPS: set[str] = {
    # Matrix ops (4)
    "aten.linear.default",
    "aten.mm.default",
    "aten.addmm.default",
    "aten.convolution.default",
    # Vector binary (4)
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    # Vector power (2)
    "aten.pow.Tensor_Scalar",
    "aten.reciprocal.default",
    # Vector unary transcendental (7)
    "aten.sqrt.default",
    "aten.sin.default",
    "aten.cos.default",
    "aten.tanh.default",
    "aten.log2.default",
    "aten.exp.default",
    "aten.exp2.default",
    # Reductions (3)
    "aten.amax.default",
    "aten.sum.default",
    "aten.sum.dim_IntList",
    # Softmax (always BF16)
    "aten._softmax.default",
    # Attention (composite, not in table directly — handled by attention module)
    "aten.bmm.default",
}


def classify_op(op_target: str) -> NpuOpCategory:
    """Classify a PyTorch ATen operator into an NPU execution category.

    Args:
        op_target: ATen operator target string (e.g., ``"aten.mm.default"``).

    Returns:
        The NPU execution category for this operator.

    Raises:
        KeyError: If the operator is not in the mapping table.
    """
    decision = _OP_TABLE.get(op_target)
    if decision is None:
        raise KeyError(f"Unmapped operator: {op_target!r}. Add it to npu_op_map._OP_TABLE.")
    return decision.category


def get_quant_decision(op_target: str) -> NpuQuantDecision:
    """Get the full quantization decision for an ATen operator.

    Args:
        op_target: ATen operator target string.

    Returns:
        ``NpuQuantDecision`` with dtype info and ISA mnemonic.

    Raises:
        KeyError: If the operator is not in the mapping table.
    """
    decision = _OP_TABLE.get(op_target)
    if decision is None:
        raise KeyError(f"Unmapped operator: {op_target!r}. Add it to npu_op_map._OP_TABLE.")
    return decision


def npu_isa_mnemonic(op_target: str) -> str | None:
    """Get the NPU ISA instruction mnemonic for an ATen operator.

    Args:
        op_target: ATen operator target string.

    Returns:
        ISA mnemonic string (e.g., ``"vmatmul.mxu0"``), or ``None`` if the op
        maps to a composite sequence or has no direct ISA equivalent.
    """
    decision = _OP_TABLE.get(op_target)
    return decision.isa_mnemonic if decision is not None else None


def validate_pi0_quant_coverage() -> list[str]:
    """Verify that all 22 pi0-quant operators have NPU classifications.

    Returns:
        List of uncovered operator names (empty if fully covered).
    """
    return [op for op in PI0_QUANT_OPS if op not in _OP_TABLE]


def all_op_targets() -> list[str]:
    """Return all mapped operator target strings."""
    return list(_OP_TABLE.keys())


__all__ = [
    "NpuOpCategory",
    "NpuQuantDecision",
    "PI0_QUANT_OPS",
    "all_op_targets",
    "classify_op",
    "get_quant_decision",
    "npu_isa_mnemonic",
    "validate_pi0_quant_coverage",
]

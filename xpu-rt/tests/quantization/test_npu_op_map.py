"""Tests for NPU operator classification and ISA mapping."""

from __future__ import annotations

import pytest
from compgen.quantization.npu_op_map import (
    PI0_QUANT_OPS,
    NpuOpCategory,
    classify_op,
    get_quant_decision,
    npu_isa_mnemonic,
    validate_pi0_quant_coverage,
)


class TestClassifyOp:
    def test_linear_maps_to_mxu(self) -> None:
        assert classify_op("aten.linear.default") == NpuOpCategory.MXU_FP8

    def test_mm_maps_to_mxu(self) -> None:
        assert classify_op("aten.mm.default") == NpuOpCategory.MXU_FP8

    def test_addmm_maps_to_mxu(self) -> None:
        assert classify_op("aten.addmm.default") == NpuOpCategory.MXU_FP8

    def test_bmm_maps_to_mxu(self) -> None:
        assert classify_op("aten.bmm.default") == NpuOpCategory.MXU_FP8

    def test_conv_maps_to_mxu(self) -> None:
        assert classify_op("aten.convolution.default") == NpuOpCategory.MXU_FP8

    def test_add_maps_to_vpu(self) -> None:
        assert classify_op("aten.add.Tensor") == NpuOpCategory.VPU_BF16

    def test_sub_maps_to_vpu(self) -> None:
        assert classify_op("aten.sub.Tensor") == NpuOpCategory.VPU_BF16

    def test_mul_maps_to_vpu(self) -> None:
        assert classify_op("aten.mul.Tensor") == NpuOpCategory.VPU_BF16

    def test_div_maps_to_vpu(self) -> None:
        assert classify_op("aten.div.Tensor") == NpuOpCategory.VPU_BF16

    def test_exp_maps_to_vpu(self) -> None:
        assert classify_op("aten.exp.default") == NpuOpCategory.VPU_BF16

    def test_sqrt_maps_to_vpu(self) -> None:
        assert classify_op("aten.sqrt.default") == NpuOpCategory.VPU_BF16

    def test_sin_maps_to_vpu(self) -> None:
        assert classify_op("aten.sin.default") == NpuOpCategory.VPU_BF16

    def test_cos_maps_to_vpu(self) -> None:
        assert classify_op("aten.cos.default") == NpuOpCategory.VPU_BF16

    def test_tanh_maps_to_vpu(self) -> None:
        assert classify_op("aten.tanh.default") == NpuOpCategory.VPU_BF16

    def test_log2_maps_to_vpu(self) -> None:
        assert classify_op("aten.log2.default") == NpuOpCategory.VPU_BF16

    def test_exp2_maps_to_vpu(self) -> None:
        assert classify_op("aten.exp2.default") == NpuOpCategory.VPU_BF16

    def test_reciprocal_maps_to_vpu(self) -> None:
        assert classify_op("aten.reciprocal.default") == NpuOpCategory.VPU_BF16

    def test_pow_maps_to_vpu(self) -> None:
        assert classify_op("aten.pow.Tensor_Scalar") == NpuOpCategory.VPU_BF16

    def test_softmax_maps_to_vpu_bf16(self) -> None:
        cat = classify_op("aten._softmax.default")
        assert cat == NpuOpCategory.VPU_BF16

    def test_softmax_is_bf16_only(self) -> None:
        decision = get_quant_decision("aten._softmax.default")
        assert decision.input_dtype == "bf16"
        assert decision.output_dtype == "bf16"
        assert decision.scale_format is None  # No FP8 scaling

    def test_amax_maps_to_xlu(self) -> None:
        assert classify_op("aten.amax.default") == NpuOpCategory.XLU_BF16

    def test_sum_maps_to_xlu(self) -> None:
        assert classify_op("aten.sum.default") == NpuOpCategory.XLU_BF16
        assert classify_op("aten.sum.dim_IntList") == NpuOpCategory.XLU_BF16

    def test_reshape_is_passthrough(self) -> None:
        assert classify_op("aten.reshape.default") == NpuOpCategory.PASSTHROUGH

    def test_view_is_passthrough(self) -> None:
        assert classify_op("aten.view.default") == NpuOpCategory.PASSTHROUGH

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(KeyError, match="Unmapped operator"):
            classify_op("aten.totally_fake_op.default")


class TestNpuIsaMnemonic:
    def test_matmul_mnemonic(self) -> None:
        assert npu_isa_mnemonic("aten.mm.default") == "vmatmul.mxu0"

    def test_add_mnemonic(self) -> None:
        assert npu_isa_mnemonic("aten.add.Tensor") == "vadd.bf16"

    def test_exp_mnemonic(self) -> None:
        assert npu_isa_mnemonic("aten.exp.default") == "vexp.bf16"

    def test_pack_mnemonic(self) -> None:
        assert npu_isa_mnemonic("npu.pack_bf16_to_fp8") == "vpack.bf16.fp8"

    def test_unpack_mnemonic(self) -> None:
        assert npu_isa_mnemonic("npu.unpack_fp8_to_bf16") == "vunpack.fp8.bf16"

    def test_softmax_no_direct_mnemonic(self) -> None:
        # Softmax is a composite op, no single ISA instruction
        assert npu_isa_mnemonic("aten._softmax.default") is None

    def test_passthrough_no_mnemonic(self) -> None:
        assert npu_isa_mnemonic("aten.view.default") is None


class TestQuantDecision:
    def test_matmul_fp8_inputs_bf16_accum(self) -> None:
        decision = get_quant_decision("aten.mm.default")
        assert decision.input_dtype == "fp8_e4m3"
        assert decision.compute_dtype == "bf16"
        assert decision.output_dtype == "bf16"
        assert decision.scale_format == "e8m0"

    def test_vpu_op_bf16_throughout(self) -> None:
        decision = get_quant_decision("aten.add.Tensor")
        assert decision.input_dtype == "bf16"
        assert decision.compute_dtype == "bf16"
        assert decision.output_dtype == "bf16"
        assert decision.scale_format is None


class TestPi0QuantCoverage:
    def test_all_22_ops_covered(self) -> None:
        """Every op from pi0-quant's operator inventory must be classified."""
        uncovered = validate_pi0_quant_coverage()
        assert uncovered == [], f"Uncovered pi0-quant ops: {uncovered}"

    def test_pi0_quant_ops_count(self) -> None:
        assert len(PI0_QUANT_OPS) == 22

    def test_matmul_ops_all_mxu(self) -> None:
        matmul_ops = [
            "aten.linear.default",
            "aten.mm.default",
            "aten.addmm.default",
            "aten.convolution.default",
            "aten.bmm.default",
        ]
        for op in matmul_ops:
            assert classify_op(op) == NpuOpCategory.MXU_FP8, f"{op} not MXU_FP8"

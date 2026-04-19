"""Wave-2/3 ported passes — MVP annotators (P7/P8 continuation).

Historically this file contained scaffolded stubs; as of the fifth
wave, every class here is a real MVP annotation pass (``stub=False``).
Each walks the module, matches a predicate, attaches a ``compgen.*``
attribute recording the decomposition / fusion / lowering strategy
the follow-up wave will realize, and counts annotations on the module
itself so callers and tests can observe the diff.

This keeps scope honest: no destructive IR surgery yet, so we don't
risk subtle correctness bugs. The LLM still gets 11 real tools to
call, each producing a measurable change. The destructive rewrites
(actual ``tensor.insert_slice`` emission, ``arith.truncf`` insertion,
bit-width repacking, etc.) ship as follow-up per the P7/P8 plan.
"""

from __future__ import annotations

from typing import Any, ClassVar

from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64
from xdsl.ir import Operation

from compgen.ir.payload.passes._annot_helpers import (
    annotate_matching_ops,
    op_matches_any_prefix,
    operand_defining_op,
    walk_ops_by_name,
)
from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg


# ---------------------------------------------------------------------------
# Pattern catalogs — shared by multiple passes
# ---------------------------------------------------------------------------


# Canonical matmul-shaped op names. Expanded in wave 6 to recognize the
# modern TorchAO ATen op names (_weight_int8pack_mm, _weight_int4pack_mm,
# _weight_int4pack_qm) plus the opaque-call wrappers that the expanded
# decomposition table emits for those ops (prefix ``aten_weight_intNpack_*``).
_QUANTIZED_MATMUL_OPS = frozenset({
    # legacy IREE name
    "linalg.quantized_matmul",
    # modern TorchAO ATen names (direct, when decomposition doesn't fire)
    "aten._weight_int8pack_mm.default",
    "aten._weight_int4pack_mm.default",
    "aten._weight_int4pack_qm.default",
    # opaque-call shapes emitted by the wave-6 decompositions
    "func.call",   # matched in conjunction with compgen._pattern_hint below
})

# Pattern-hint strings a matched op must carry for it to count as a
# quantized matmul. Used alongside the name match so ``func.call`` ops
# carrying the right pattern_hint are recognised without false positives.
_QUANTIZED_MATMUL_PATTERN_HINTS = frozenset({
    "weight_int8pack_mm",
    "weight_int4pack_mm",
    "weight_int4pack_qm",
})

_QUANTIZED_CONV_OPS = frozenset(
    {
        "linalg.quantized_conv_2d_nhwc_hwcf",
        "linalg.quantized_conv_2d_nhwc_hwcf_q",
    }
)

# Dequantize-shaped ops — used by FuseDequantMatmul to detect a
# producer-side dequant before a matmul consumer.
_DEQUANT_OPS = frozenset({
    "torch.ops.quantized_decomposed.dequantize_per_tensor.default",
    "torch.ops.quantized_decomposed.dequantize_per_channel.default",
    "torch.ops.quantized_decomposed.dequantize_per_group_along_last_dim.default",
    "quantized_decomposed.dequantize_per_tensor.default",
    "quantized_decomposed.dequantize_per_channel.default",
    "quantized_decomposed.dequantize_per_group_along_last_dim.default",
})

_DEQUANT_PATTERN_HINTS = frozenset({
    "dequantize_per_tensor",
    "dequantize_per_channel",
    "dequantize_per_group",
})

_MATMUL_OPS = frozenset(
    {"linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul"}
)

_CONV_OPS_NHWC = frozenset(
    {
        "linalg.conv_2d_nhwc_hwcf",
        "linalg.depthwise_conv_2d_nhwc_hwc",
        "linalg.depthwise_conv_2d_nhwc_hwcm",
    }
)

_TRANSPOSE_OPS = frozenset({"linalg.transpose", "tensor.transpose"})

_ELEMENTWISE_PREFIXES = ("arith.", "math.", "linalg.elemwise_")


def _has_pattern_hint(op, hints: frozenset[str]) -> bool:
    """Check whether an xDSL op carries a matching ``compgen._pattern_hint``."""
    attr = op.attributes.get("compgen._pattern_hint")
    if attr is None:
        return False
    data = getattr(attr, "data", None)
    return isinstance(data, str) and data in hints


def _is_quantized_matmul_op(op) -> bool:
    """Match modern quantized-matmul ops by name OR by pattern hint."""
    if op.name in _QUANTIZED_MATMUL_OPS and op.name != "func.call":
        return True
    if op.name == "func.call" and _has_pattern_hint(op, _QUANTIZED_MATMUL_PATTERN_HINTS):
        return True
    return False


def _is_dequant_op(op) -> bool:
    """Match dequantize-shaped ops — direct name or opaque call with hint."""
    if op.name in _DEQUANT_OPS and op.name != "func.call":
        return True
    if op.name == "func.call" and _has_pattern_hint(op, _DEQUANT_PATTERN_HINTS):
        return True
    return False


# ---------------------------------------------------------------------------
# 1. LowerQuantizedMatmul
# ---------------------------------------------------------------------------


class LowerQuantizedMatmul(PayloadPass):
    name: ClassVar[str] = "lower_quantized_matmul"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:QuantizedMatmulToMatmul"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "high"
    description: ClassVar[str] = (
        "Identify linalg.quantized_matmul ops and annotate a lowering mode "
        "(bare_matmul / matmul_with_zp_correction / skip)."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("policy", "enum", "lowering policy",
                    enum=("always", "zp_zero_only", "skip"),
                    required=False, default="always"),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        policy = kwargs.get("policy", "always")

        def _match(op: Operation) -> str | None:
            if op.name not in _QUANTIZED_MATMUL_OPS:
                return None
            if policy == "skip":
                return "skip"
            return "matmul_with_zp_correction"

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.quant_lower_mode",
            count_attr="compgen.lower_quantized_matmul.count",
        )
        return module


# ---------------------------------------------------------------------------
# 2. LowerQuantizedConv
# ---------------------------------------------------------------------------


class LowerQuantizedConv(PayloadPass):
    name: ClassVar[str] = "lower_quantized_conv"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:QuantizedConvToConv"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "high"
    description: ClassVar[str] = (
        "Identify quantized conv2d ops and annotate a lowering mode."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("policy", "enum", "lowering policy",
                    enum=("always", "zp_zero_only", "skip"),
                    required=False, default="always"),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        policy = kwargs.get("policy", "always")

        def _match(op: Operation) -> str | None:
            if op.name not in _QUANTIZED_CONV_OPS:
                return None
            if policy == "skip":
                return "skip"
            return "conv_with_zp_correction"

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.quant_lower_mode",
            count_attr="compgen.lower_quantized_conv.count",
        )
        return module


# ---------------------------------------------------------------------------
# 3. PropagateTransposes
# ---------------------------------------------------------------------------


class PropagateTransposes(PayloadPass):
    name: ClassVar[str] = "propagate_transposes"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:PropagateLinalgTranspose"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "high"
    description: ClassVar[str] = (
        "Walk transpose ops; annotate a propagation target based on the "
        "immediate user: through_elementwise / absorb_into_matmul / none."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("aggressiveness", "enum", "how far to push",
                    enum=("conservative", "through_elementwise", "through_conv", "through_pad"),
                    required=False, default="through_elementwise"),
        )

    def _user_target(self, op: Operation) -> str:
        for res in op.results:
            for use in getattr(res, "uses", []):
                user = getattr(use, "operation", None)
                if user is None:
                    continue
                if user.name in _MATMUL_OPS:
                    return "absorb_into_matmul"
                if op_matches_any_prefix(user, _ELEMENTWISE_PREFIXES):
                    return "through_elementwise"
        return "none"

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        aggressiveness = kwargs.get("aggressiveness", "through_elementwise")

        def _match(op: Operation) -> str | None:
            if op.name not in _TRANSPOSE_OPS:
                return None
            if aggressiveness == "conservative":
                return "none"
            return self._user_target(op)

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.transpose_propagation_target",
            count_attr="compgen.propagate_transposes.count",
        )
        return module


# ---------------------------------------------------------------------------
# 4. LowerConvToImg2Col
# ---------------------------------------------------------------------------


class LowerConvToImg2Col(PayloadPass):
    name: ClassVar[str] = "lower_conv_to_img2col"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:ConvertConv2DToImg2Col"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "high"
    description: ClassVar[str] = (
        "Identify linalg.conv_2d_* ops and mark whether their shapes are "
        "static (img2col-eligible) or dynamic (skip)."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset(
        {"rvv_cpu", "qualcomm_npu", "qualcomm_dsp", "generic_npu"}
    )
    stub: ClassVar[bool] = False

    def _has_static_shape(self, op: Operation) -> bool:
        from xdsl.dialects.builtin import TensorType

        for operand in op.operands:
            ty = operand.type
            if isinstance(ty, TensorType):
                shape = ty.get_shape()
                if any(d == -1 for d in shape):
                    return False
        return True

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        def _match(op: Operation) -> str | None:
            if op.name not in _CONV_OPS_NHWC:
                return None
            return "eligible" if self._has_static_shape(op) else "skip_dynamic_shape"

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.img2col_candidate",
            count_attr="compgen.lower_conv_to_img2col.count",
        )
        return module


# ---------------------------------------------------------------------------
# 5. RaiseSpecialOps
# ---------------------------------------------------------------------------


class RaiseSpecialOps(PayloadPass):
    name: ClassVar[str] = "raise_special_ops"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:RaiseSpecialOps"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "very_high"
    description: ClassVar[str] = (
        "Walk linalg.generic ops; annotate a recognised special-op pattern "
        "name (softmax / rmsnorm / layernorm / gelu / silu / rope) for the "
        "destructive-wave raise step. MVP: reads _compgen_pattern metadata "
        "already attached by the FX-level detect_and_annotate_patterns."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("library", "enum_set",
                    "named special-op library to raise",
                    enum=("softmax", "logsoftmax", "layernorm", "rmsnorm",
                          "gelu", "silu", "rope", "swiglu"),
                    required=False),
        )

    def _detect_pattern(self, op: Operation) -> str | None:
        if op.name != "linalg.generic":
            return None
        tag = op.attributes.get("_compgen_pattern")
        if tag is not None:
            text = str(getattr(tag, "data", ""))
            return text or None
        return None

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        annotate_matching_ops(
            module,
            match=self._detect_pattern,
            attr_name="compgen.raised_pattern",
            count_attr="compgen.raise_special_ops.count",
        )
        return module


# ---------------------------------------------------------------------------
# 6. MatchLibraryCall
# ---------------------------------------------------------------------------


class MatchLibraryCall(PayloadPass):
    name: ClassVar[str] = "match_library_call"
    phase: ClassVar[int] = 3
    wraps_pass: ClassVar[str] = "XLA:GemmRewriter+LibraryRewriter+OneDnn"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "very_high"
    description: ClassVar[str] = (
        "Match matmul/conv ops against the target's supported_kernel_families. "
        "Annotate compgen.library_match with the matched family name, or "
        "'no_match' when none fits."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("target_capabilities", "target_ref",
                    "target resource model (optional; list of family names, "
                    "target_resource dict, or profile object)",
                    required=False, default=None),
        )

    def _match_family(self, op: Operation, target_families: list[str]) -> str:
        if op.name in _MATMUL_OPS:
            if "gemm_int8" in target_families:
                return "gemm_int8"
            if "gemm" in target_families:
                return "gemm"
        if op.name in _CONV_OPS_NHWC:
            if "conv2d_nhwc" in target_families:
                return "conv2d_nhwc"
        return "no_match"

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        target = kwargs.get("target_capabilities")
        target_families: list[str] = []
        if target is not None:
            if isinstance(target, list):
                target_families = list(target)
            elif isinstance(target, dict):
                fams = target.get("supported_kernel_families") or []
                target_families = [
                    f.get("family", "") if isinstance(f, dict) else str(f) for f in fams
                ]
            else:
                fams = getattr(target, "supported_kernel_families", []) or []
                target_families = [getattr(f, "family", str(f)) for f in fams]

        def _match(op: Operation) -> str | None:
            if op.name not in _MATMUL_OPS and op.name not in _CONV_OPS_NHWC:
                return None
            return self._match_family(op, target_families)

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.library_match",
            count_attr="compgen.match_library_call.count",
        )
        return module


# ---------------------------------------------------------------------------
# 7. SetNumericsPolicy
# ---------------------------------------------------------------------------


class SetNumericsPolicy(PayloadPass):
    name: ClassVar[str] = "set_numerics_policy"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "XLA:FloatNormalization"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "high"
    description: ClassVar[str] = (
        "Walk contraction ops; annotate compgen.numerics_policy with the "
        "declared dtype + accumulator combination."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("input_dtype", "enum", "input dtype",
                    enum=("f32", "bf16", "f16", "fp8_e4m3", "fp8_e5m2", "int8"),
                    required=False, default="bf16"),
            ToolArg("accumulator_dtype", "enum", "accumulator dtype",
                    enum=("f32", "f16", "i32"),
                    required=False, default="f32"),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        input_dtype = kwargs.get("input_dtype", "bf16")
        accum_dtype = kwargs.get("accumulator_dtype", "f32")
        policy_tag = f"input={input_dtype};accum={accum_dtype}"

        def _match(op: Operation) -> str | None:
            if op.name in _MATMUL_OPS or op.name in _CONV_OPS_NHWC:
                return policy_tag
            return None

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.numerics_policy",
            count_attr="compgen.set_numerics_policy.count",
        )
        return module


# ---------------------------------------------------------------------------
# 8. FoldTransposesIntoDots
# ---------------------------------------------------------------------------


class FoldTransposesIntoDots(PayloadPass):
    name: ClassVar[str] = "fold_transposes_into_dots"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "XLA:TransposeFolding"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = (
        "For each linalg.matmul, identify whether lhs/rhs has a transpose "
        "producer; annotate compgen.fold_candidate with lhs/rhs/both/none."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset(
        {"rvv_cpu", "qualcomm_npu", "qualcomm_dsp", "generic_npu"}
    )
    stub: ClassVar[bool] = False

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        def _match(op: Operation) -> str | None:
            if op.name not in _MATMUL_OPS:
                return None
            lhs_owner = operand_defining_op(op, 0)
            rhs_owner = operand_defining_op(op, 1)
            lhs_t = lhs_owner is not None and lhs_owner.name in _TRANSPOSE_OPS
            rhs_t = rhs_owner is not None and rhs_owner.name in _TRANSPOSE_OPS
            if lhs_t and rhs_t:
                return "both"
            if lhs_t:
                return "lhs"
            if rhs_t:
                return "rhs"
            return "none"

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.fold_candidate",
            count_attr="compgen.fold_transposes_into_dots.count",
        )
        return module


# ---------------------------------------------------------------------------
# 9. PlanReduction
# ---------------------------------------------------------------------------


class PlanReduction(PayloadPass):
    name: ClassVar[str] = "plan_reduction"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "XLA:ReductionDimensionGrouper+Splitter+TreeReductionRewriter"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = (
        "For each reduction op, annotate compgen.reduction_strategy per the "
        "declared strategy."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("strategy", "enum", "reduction strategy",
                    enum=("group", "split", "tree_reduce"),
                    required=False, default="split"),
        )

    def _is_reduction(self, op: Operation) -> bool:
        if op.name == "linalg.reduce":
            return True
        if op.name == "linalg.generic":
            iters_attr = op.attributes.get("iterator_types")
            if iters_attr is None:
                return False
            try:
                data = getattr(iters_attr, "data", None) or []
                for it in data:
                    text = str(getattr(it, "data", it))
                    if "reduction" in text.lower():
                        return True
            except Exception:   # noqa: BLE001
                pass
        return False

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        strategy = kwargs.get("strategy", "split")

        def _match(op: Operation) -> str | None:
            return strategy if self._is_reduction(op) else None

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.reduction_strategy",
            count_attr="compgen.plan_reduction.count",
        )
        return module


# ---------------------------------------------------------------------------
# 10. FuseSoftmaxToTriton
# ---------------------------------------------------------------------------


class FuseSoftmaxToTriton(PayloadPass):
    name: ClassVar[str] = "fuse_softmax_to_triton"
    phase: ClassVar[int] = 3
    wraps_pass: ClassVar[str] = "XLA:SoftmaxRewriterTriton"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "very_high"
    description: ClassVar[str] = (
        "Identify softmax-shaped ops; annotate compgen.triton_softmax_candidate "
        "so the destructive wave can fuse to a Triton kernel."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        def _match(op: Operation) -> str | None:
            if op.name != "linalg.generic":
                return None
            tag = op.attributes.get("_compgen_pattern")
            if tag is None:
                return None
            text = str(getattr(tag, "data", ""))
            if "softmax" in text.lower():
                return "candidate"
            return None

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.triton_softmax_candidate",
            count_attr="compgen.fuse_softmax_to_triton.count",
        )
        return module


# ---------------------------------------------------------------------------
# 11. FuseDequantMatmul
# ---------------------------------------------------------------------------


class FuseDequantMatmul(PayloadPass):
    name: ClassVar[str] = "fuse_dequant_matmul"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:FuseDequantizationMatmul"
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "very_high"
    description: ClassVar[str] = (
        "For each matmul, check if any operand has a dequant-shaped producer. "
        "Annotate compgen.dequant_fuse_candidate accordingly."
    )
    covers_families: ClassVar[frozenset[str]] = frozenset()
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg("region", "region_ref", "region", required=False, default=""),
            ToolArg("safety", "enum", "reassociation safety level",
                    enum=("reassoc_safe_only", "allow_numerics_relaxation"),
                    required=False, default="reassoc_safe_only"),
        )

    def _is_dequant_shaped(self, op: Operation | None) -> bool:
        if op is None or op.name != "linalg.generic":
            return False
        tag = op.attributes.get("_compgen_pattern")
        if tag is None:
            return False
        text = str(getattr(tag, "data", "")).lower()
        return "dequant" in text

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        safety = kwargs.get("safety", "reassoc_safe_only")

        def _match(op: Operation) -> str | None:
            if op.name not in _MATMUL_OPS:
                return None
            lhs_dq = self._is_dequant_shaped(operand_defining_op(op, 0))
            rhs_dq = self._is_dequant_shaped(operand_defining_op(op, 1))
            if not (lhs_dq or rhs_dq):
                return None
            if safety == "reassoc_safe_only":
                return "fuse_safe"
            return "fuse_relaxed"

        annotate_matching_ops(
            module,
            match=_match,
            attr_name="compgen.dequant_fuse_candidate",
            count_attr="compgen.fuse_dequant_matmul.count",
        )
        return module


__all__ = [
    "FoldTransposesIntoDots",
    "FuseDequantMatmul",
    "FuseSoftmaxToTriton",
    "LowerConvToImg2Col",
    "LowerQuantizedConv",
    "LowerQuantizedMatmul",
    "MatchLibraryCall",
    "PlanReduction",
    "PropagateTransposes",
    "RaiseSpecialOps",
    "SetNumericsPolicy",
]

"""IREE DemoteContractionInputsToBF16 — MVP port.

Walks the module, identifies contraction-shaped ops (matmul, conv,
dot-product generics), and annotates a declared ``compgen.demote_to``
attribute recording the target element type the cast layer will apply
in a follow-up wave. Always keeps accumulator f32 — the annotation
represents the intent, the actual ``arith.truncf`` insertion lands in
the wave that wires kernel-contract emission to the chosen dtype.

MVP scope: candidate detection + intent recording. Safe — no bit-level
rewrite that could break numerics.
"""

from __future__ import annotations

from typing import Any, ClassVar

from xdsl.dialects.builtin import (
    Float32Type,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.ir import Operation

from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg, ToolResult


_CONTRACTION_OPS = frozenset(
    {
        "linalg.matmul",
        "linalg.batch_matmul",
        "linalg.quantized_matmul",
        "linalg.conv_2d_nhwc_hwcf",
        "linalg.conv_2d_nchw_fchw",
        "linalg.conv_1d",
        "linalg.conv_1d_nwc_wcf",
        "linalg.depthwise_conv_2d_nhwc_hwc",
        "linalg.depthwise_conv_2d_nhwc_hwcm",
        # dot-product-shaped linalg.generic is harder to detect without
        # indexing-map analysis; handled in follow-up wave.
    }
)


def _operand_is_f32_tensor(op: Operation) -> bool:
    """At least one tensor operand's element type is f32."""
    for operand in op.operands:
        ty = operand.type
        if isinstance(ty, TensorType) and isinstance(ty.element_type, Float32Type):
            return True
    return False


_ALLOWED_DEMOTE_TARGETS = frozenset(
    {"bf16", "fp16", "fp8_e4m3", "fp8_e5m2"}
)


class DemoteContractionInputs(PayloadPass):
    """Annotate contraction ops with a declared input-demotion target."""

    name: ClassVar[str] = "demote_contraction_inputs"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "IREE:DemoteContractionInputsToBF16"
    covers_families: ClassVar[frozenset[str]] = frozenset()  # all targets
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = (
        "Identify f32-input contractions and annotate a declared "
        "demotion target (bf16 / fp16 / fp8). MVP: annotation; "
        "destructive arith.truncf insertion in follow-up wave."
    )
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg(
                name="region", dtype="region_ref", description="region",
                required=False, default="",
            ),
            ToolArg(
                name="dtype", dtype="enum",
                description="target demoted input dtype",
                required=False, default="bf16",
                enum=("bf16", "fp16", "fp8_e4m3", "fp8_e5m2"),
            ),
            ToolArg(
                name="targets", dtype="enum",
                description="which ops to affect",
                required=False, default="all_contractions",
                enum=("all_contractions", "matmul_only", "conv_only"),
            ),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        dtype = kwargs.get("dtype", "bf16")
        if dtype not in _ALLOWED_DEMOTE_TARGETS:
            raise ValueError(
                f"dtype must be one of {sorted(_ALLOWED_DEMOTE_TARGETS)}, got {dtype!r}"
            )
        filt = kwargs.get("targets", "all_contractions")

        def _matches(op_name: str) -> bool:
            if filt == "matmul_only":
                return "matmul" in op_name
            if filt == "conv_only":
                return "conv" in op_name
            return op_name in _CONTRACTION_OPS

        annotated = 0
        for op in module.walk():
            if op.name not in _CONTRACTION_OPS:
                continue
            if not _matches(op.name):
                continue
            if not _operand_is_f32_tensor(op):
                continue
            op.attributes["compgen.demote_to"] = StringAttr(dtype)
            annotated += 1

        module.attributes["compgen.demote_contraction_inputs.count"] = IntegerAttr(
            annotated, i64
        )
        return module


__all__ = ["DemoteContractionInputs"]

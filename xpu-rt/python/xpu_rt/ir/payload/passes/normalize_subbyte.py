"""XLA SubByteNormalization — MVP port.

Walks the module, finds tensor operands whose element type is a
sub-byte integer (i4 / i2 / u4 / u2 / i1), and attaches a canonical
``compgen.subbyte_packing`` attribute declaring the packing strategy
(``bit_pack`` / ``byte_pack`` / ``target_native``). Counts annotated
tensors into a module-level attribute.

MVP scope: annotation + counting. Full bit-width-rewrite happens in a
follow-up wave once targets declare preferred packings via the v2
target YAML (``target_resource.v2.supported_dtypes``).
"""

from __future__ import annotations

from typing import Any, ClassVar

from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    TensorType,
    i64,
)
from xdsl.ir import Operation

from compgen.ir.payload.passes.base import PayloadPass
from compgen.llm.registry import AutocompCostImpact, ToolArg

_SUBBYTE_BITS = frozenset({1, 2, 4})


def _element_bitwidth(tensor_type: TensorType) -> int | None:
    et = tensor_type.element_type
    if isinstance(et, IntegerType):
        return int(getattr(et, "width").data) if hasattr(et, "width") else None
    return None


def _op_has_subbyte_tensor(op: Operation) -> bool:
    for operand in op.operands:
        ty = operand.type
        if isinstance(ty, TensorType):
            bw = _element_bitwidth(ty)
            if bw is not None and bw in _SUBBYTE_BITS:
                return True
    for result in op.results:
        ty = result.type
        if isinstance(ty, TensorType):
            bw = _element_bitwidth(ty)
            if bw is not None and bw in _SUBBYTE_BITS:
                return True
    return False


class NormalizeSubByte(PayloadPass):
    """Identify ops touching sub-byte tensors; annotate packing strategy."""

    name: ClassVar[str] = "normalize_subbyte"
    phase: ClassVar[int] = 2
    wraps_pass: ClassVar[str] = "XLA:SubByteNormalization"
    covers_families: ClassVar[frozenset[str]] = frozenset()  # applies everywhere
    autocomp_cost_impact: ClassVar[AutocompCostImpact] = "medium"
    description: ClassVar[str] = (
        "Identify ops with i1/i2/i4/u2/u4 tensors and annotate a "
        "packing strategy. MVP: annotation + count; destructive bit "
        "rewrite deferred to follow-up wave."
    )
    stub: ClassVar[bool] = False

    def tool_args(self) -> tuple[ToolArg, ...]:
        return (
            ToolArg(
                name="region",
                dtype="region_ref",
                description="region",
                required=False,
                default="",
            ),
            ToolArg(
                name="packing",
                dtype="enum",
                description="target packing strategy",
                required=False,
                default="target_native",
                enum=("bit_pack", "byte_pack", "target_native"),
            ),
        )

    def run(self, module: ModuleOp, **kwargs: Any) -> ModuleOp:
        packing = kwargs.get("packing", "target_native")
        annotated = 0
        for op in module.walk():
            if not _op_has_subbyte_tensor(op):
                continue
            op.attributes["compgen.subbyte_packing"] = StringAttr(packing)
            annotated += 1

        module.attributes["compgen.normalize_subbyte.count"] = IntegerAttr(annotated, i64)
        return module


__all__ = ["NormalizeSubByte"]

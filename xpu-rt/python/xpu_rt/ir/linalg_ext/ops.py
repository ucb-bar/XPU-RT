"""Named high-level structured ops for the ``compgen.linalg_ext`` dialect.

Each op is ``Pure`` and carries explicit typed operands so the dialect
is usable as both a destination for ``raise_special_ops`` and a source
for library dispatch + kernel generation.

Semantics match the corresponding PyTorch op, which is also what MLIR
upstream's linalg_ext + torch-mlir families express:

- :class:`SoftmaxOp` -- ``torch.nn.functional.softmax(x, dim)``
- :class:`LayerNormOp` -- ``torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)``
- :class:`RMSNormOp` -- root-mean-square norm as in LLaMA / RMSNorm papers
- :class:`RoPEOp` -- rotary position embedding (Su et al. 2021)
- :class:`SwiGLUOp` -- SwiGLU activation: ``silu(gate) * up`` (as in LLaMA MLP)
- :class:`GeluOp` -- Gaussian Error Linear Unit, with exact / tanh approximation
- :class:`SiluOp` -- Sigmoid-weighted Linear Unit (aka Swish)
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import (
    FloatAttr,
    Float64Type,
    IntegerAttr,
    IntegerType,
    StringAttr,
)
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


def _ia(v: int) -> IntegerAttr:
    return IntegerAttr(v, IntegerType(64))


def _fa(v: float) -> FloatAttr:
    return FloatAttr(float(v), Float64Type())


# --- Softmax ------------------------------------------------------------------


@irdl_op_definition
class SoftmaxOp(IRDLOperation):
    """``softmax(input, dim)`` along a single axis.

    Semantics: ``out[..., i, ...] = exp(x - max(x, axis=dim)) /
    sum(exp(x - max(x, axis=dim)), axis=dim)``.
    """

    name = "compgen.linalg_ext.softmax"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    dim = prop_def(IntegerAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        input: SSAValue,
        dim: int | IntegerAttr,
        result_type: Attribute,
    ) -> None:
        if isinstance(dim, int):
            dim = _ia(dim)
        super().__init__(
            operands=[input],
            result_types=[result_type],
            properties={"dim": dim},
        )

    def verify_(self) -> None:
        if self.dim.value.data < 0:
            raise VerifyException(
                f"{self.name}: dim must be non-negative (resolve negative "
                f"axes at import time), got {self.dim.value.data}"
            )


# --- LayerNorm ----------------------------------------------------------------


@irdl_op_definition
class LayerNormOp(IRDLOperation):
    """``layer_norm(input, normalized_shape, weight, bias, eps)``.

    Operands:
        input: the input tensor.
        weight: affine scale tensor (or absent for ``elementwise_affine=False``).
        bias: affine shift tensor (optional).

    Properties:
        eps: numerical stabilizer.
        axis: the first dimension being normalized (the last rank -
            ``len(normalized_shape) + axis`` dims are normalized).
    """

    name = "compgen.linalg_ext.layer_norm"

    input = operand_def(Attribute)
    weight = opt_operand_def(Attribute)
    bias = opt_operand_def(Attribute)
    result = result_def(Attribute)

    eps = prop_def(FloatAttr)
    axis = opt_prop_def(IntegerAttr)

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    traits = traits_def(Pure())

    def __init__(
        self,
        input: SSAValue,
        result_type: Attribute,
        *,
        weight: SSAValue | None = None,
        bias: SSAValue | None = None,
        eps: float | FloatAttr = 1e-5,
        axis: int | IntegerAttr | None = None,
    ) -> None:
        if isinstance(eps, float):
            eps = _fa(eps)
        properties: dict[str, Attribute] = {"eps": eps}
        if axis is not None:
            if isinstance(axis, int):
                axis = _ia(axis)
            properties["axis"] = axis
        operands = [
            [input],
            [weight] if weight is not None else [],
            [bias] if bias is not None else [],
        ]
        super().__init__(
            operands=operands,
            result_types=[result_type],
            properties=properties,
        )

    def verify_(self) -> None:
        if self.eps.value.data <= 0:
            raise VerifyException(
                f"{self.name}: eps must be strictly positive, "
                f"got {self.eps.value.data}"
            )


# --- RMSNorm ------------------------------------------------------------------


@irdl_op_definition
class RMSNormOp(IRDLOperation):
    """``rms_norm(input, weight, eps)``.

    ``out = x * weight / sqrt(mean(x**2, axis=-1, keepdim=True) + eps)``.
    """

    name = "compgen.linalg_ext.rms_norm"

    input = operand_def(Attribute)
    weight = opt_operand_def(Attribute)
    result = result_def(Attribute)

    eps = prop_def(FloatAttr)

    traits = traits_def(Pure())

    def __init__(
        self,
        input: SSAValue,
        result_type: Attribute,
        *,
        weight: SSAValue | None = None,
        eps: float | FloatAttr = 1e-6,
    ) -> None:
        if isinstance(eps, float):
            eps = _fa(eps)
        operands = [[input], [weight] if weight is not None else []]
        super().__init__(
            operands=operands,
            result_types=[result_type],
            properties={"eps": eps},
        )

    def verify_(self) -> None:
        if self.eps.value.data <= 0:
            raise VerifyException(
                f"{self.name}: eps must be strictly positive, "
                f"got {self.eps.value.data}"
            )


# --- RoPE ---------------------------------------------------------------------


@irdl_op_definition
class RoPEOp(IRDLOperation):
    """``rope(q, k, cos, sin)`` -- rotary position embedding applied to q + k.

    Produces a pair (q_rot, k_rot) with the same shape as q, k.
    """

    name = "compgen.linalg_ext.rope"

    q = operand_def(Attribute)
    k = operand_def(Attribute)
    cos = operand_def(Attribute)
    sin = operand_def(Attribute)
    q_out = result_def(Attribute)
    k_out = result_def(Attribute)

    # Which dim carries the rotary feature (usually the last dim).
    feature_dim = opt_prop_def(IntegerAttr)
    # Optional variant tag: "neox", "gpt_j", "half_half" etc.
    variant = opt_prop_def(StringAttr)

    traits = traits_def(Pure())

    _VALID_VARIANTS: ClassVar[frozenset[str]] = frozenset({
        "neox", "gpt_j", "half_half", "llama",
    })

    def __init__(
        self,
        q: SSAValue,
        k: SSAValue,
        cos: SSAValue,
        sin: SSAValue,
        q_result: Attribute,
        k_result: Attribute,
        *,
        feature_dim: int | IntegerAttr | None = None,
        variant: str | StringAttr | None = None,
    ) -> None:
        properties: dict[str, Attribute] = {}
        if feature_dim is not None:
            if isinstance(feature_dim, int):
                feature_dim = _ia(feature_dim)
            properties["feature_dim"] = feature_dim
        if variant is not None:
            if isinstance(variant, str):
                variant = StringAttr(variant)
            properties["variant"] = variant
        super().__init__(
            operands=[q, k, cos, sin],
            result_types=[q_result, k_result],
            properties=properties,
        )

    def verify_(self) -> None:
        if self.variant is not None and self.variant.data not in self._VALID_VARIANTS:
            raise VerifyException(
                f"{self.name}: variant must be one of "
                f"{sorted(self._VALID_VARIANTS)}, got {self.variant.data!r}"
            )


# --- SwiGLU -------------------------------------------------------------------


@irdl_op_definition
class SwiGLUOp(IRDLOperation):
    """``swiglu(gate, up) = silu(gate) * up`` (LLaMA-style MLP gate)."""

    name = "compgen.linalg_ext.swiglu"

    gate = operand_def(Attribute)
    up = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(Pure())

    def __init__(
        self,
        gate: SSAValue,
        up: SSAValue,
        result_type: Attribute,
    ) -> None:
        super().__init__(
            operands=[gate, up],
            result_types=[result_type],
        )


# --- GELU ---------------------------------------------------------------------


@irdl_op_definition
class GeluOp(IRDLOperation):
    """``gelu(input, approximate)`` -- Gaussian Error Linear Unit.

    Approximation ∈ {"none", "tanh"}: matches PyTorch's
    ``torch.nn.functional.gelu(approximate=...)``.
    """

    name = "compgen.linalg_ext.gelu"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    approximate = opt_prop_def(StringAttr)

    traits = traits_def(Pure())

    _VALID: ClassVar[frozenset[str]] = frozenset({"none", "tanh"})

    def __init__(
        self,
        input: SSAValue,
        result_type: Attribute,
        *,
        approximate: str | StringAttr = "none",
    ) -> None:
        if isinstance(approximate, str):
            approximate = StringAttr(approximate)
        super().__init__(
            operands=[input],
            result_types=[result_type],
            properties={"approximate": approximate},
        )

    def verify_(self) -> None:
        if self.approximate is not None and self.approximate.data not in self._VALID:
            raise VerifyException(
                f"{self.name}: approximate must be one of "
                f"{sorted(self._VALID)}, got {self.approximate.data!r}"
            )


# --- SiLU ---------------------------------------------------------------------


@irdl_op_definition
class SiluOp(IRDLOperation):
    """``silu(x) = x * sigmoid(x)`` (aka Swish-1)."""

    name = "compgen.linalg_ext.silu"

    input = operand_def(Attribute)
    result = result_def(Attribute)

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue, result_type: Attribute) -> None:
        super().__init__(operands=[input], result_types=[result_type])


LINALG_EXT_OPS: list[type[IRDLOperation]] = [
    SoftmaxOp,
    LayerNormOp,
    RMSNormOp,
    RoPEOp,
    SwiGLUOp,
    GeluOp,
    SiluOp,
]


__all__ = [
    "GeluOp",
    "LINALG_EXT_OPS",
    "LayerNormOp",
    "RMSNormOp",
    "RoPEOp",
    "SiluOp",
    "SoftmaxOp",
    "SwiGLUOp",
]

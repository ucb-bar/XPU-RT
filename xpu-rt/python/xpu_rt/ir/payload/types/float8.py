"""FP8 element types for Payload IR.

xDSL's builtin dialect does not ship Float8 types. MLIR upstream does
(``Float8E4M3FNType``, ``Float8E5M2Type``) and TorchAO / PyTorch both
produce tensors typed as ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``.
Until xDSL gains these natively, CompGen ships its own implementations
that mirror MLIR's semantics so that a future MLIR round-trip works and
so that Phase-2 numerics passes (``set_numerics_policy``,
``demote_contraction_inputs``) can see FP8 as a real type rather than
a silent ``f16`` demotion.

Two types, matching MLIR exactly:

- ``Float8E4M3FNType`` -- 1 sign, 4 exponent, 3 mantissa bits. Bias 7.
  Finite range ±448. No infinity; NaN is encoded only as ``S.1111.111``
  (the "FN" = "Finite, NaN-only"). This is the H100 / TorchAO
  preferred activation + weight format.
- ``Float8E5M2Type`` -- 1 sign, 5 exponent, 2 mantissa bits. Bias 15.
  IEEE-754-like: has Inf and NaN. Matches the H100 gradient format.

Both types are tensor-legal: ``TensorType(Float8E4M3FNType(), [16, 16])``
parses and verifies without special-case wiring. They also participate
in the ``FixedBitwidthType`` hierarchy so buffer-layout passes can
compute element sizes correctly (1 byte per element).
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import FixedBitwidthType
from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition


class _Float8Base(ParametrizedAttribute, FixedBitwidthType):
    """Shared implementation for FP8 types.

    Each subclass is a parameterless type attribute. Format metadata
    lives on class-level UPPERCASE constants (xDSL's IRDL machinery
    requires that convention for const ClassVars) and is exposed via
    lowercase instance properties for readability.
    """

    EXPONENT_BITS: ClassVar[int] = 0
    MANTISSA_BITS: ClassVar[int] = 0
    HAS_INFINITY: ClassVar[bool] = False
    HAS_NAN: ClassVar[bool] = False
    MAX_FINITE: ClassVar[float] = 0.0

    @property
    def bitwidth(self) -> int:
        return 8

    @property
    def compile_time_size(self) -> int:
        return 1

    @property
    def exponent_bits(self) -> int:
        return self.EXPONENT_BITS

    @property
    def mantissa_bits(self) -> int:
        return self.MANTISSA_BITS

    @property
    def has_infinity(self) -> bool:
        return self.HAS_INFINITY

    @property
    def has_nan(self) -> bool:
        return self.HAS_NAN

    @property
    def max_finite(self) -> float:
        return self.MAX_FINITE

    @property
    def exponent_bias(self) -> int:
        """IEEE-style exponent bias: ``2^(exponent_bits - 1) - 1``."""
        return (1 << (self.EXPONENT_BITS - 1)) - 1


@irdl_attr_definition
class Float8E4M3FNType(_Float8Base):
    """FP8 with 4 exponent + 3 mantissa bits, finite-only, NaN-only.

    Matches ``mlir::Float8E4M3FNType``. The "FN" suffix marks the
    format as *Finite, NaN-only*: there is no infinity encoding, and
    NaN occupies the single bit pattern ``S.1111.111``. This gives
    ±448 as the finite range, which is larger than an IEEE-compliant
    E4M3 would allow.

    This is the preferred activation + weight format on H100 for
    TorchAO's ``float8_weight_only`` / ``float8_dynamic_activation``
    paths.
    """

    name = "compgen.float8_e4m3fn"

    EXPONENT_BITS: ClassVar[int] = 4
    MANTISSA_BITS: ClassVar[int] = 3
    HAS_INFINITY: ClassVar[bool] = False
    HAS_NAN: ClassVar[bool] = True
    MAX_FINITE: ClassVar[float] = 448.0


@irdl_attr_definition
class Float8E5M2Type(_Float8Base):
    """FP8 with 5 exponent + 2 mantissa bits, IEEE-like.

    Matches ``mlir::Float8E5M2Type``. Has full IEEE-style Inf + NaN
    encodings. The trade-off vs E4M3FN: E5M2 has wider dynamic range
    (±57344) but only 2 mantissa bits of precision, so it's the
    gradient/activation format when dynamic range matters more than
    precision.
    """

    name = "compgen.float8_e5m2"

    EXPONENT_BITS: ClassVar[int] = 5
    MANTISSA_BITS: ClassVar[int] = 2
    HAS_INFINITY: ClassVar[bool] = True
    HAS_NAN: ClassVar[bool] = True
    MAX_FINITE: ClassVar[float] = 57344.0


__all__ = [
    "Float8E4M3FNType",
    "Float8E5M2Type",
]

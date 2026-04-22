"""Regression tests for dynamic-shape + bf16 hazards in decompositions.

Both failures originally surfaced while importing real SmolVLA:

1. ``u6 should be of base attribute builtin.int`` — ``torch.export``
   emits ``SymInt`` dims for image-tile counts; xDSL's ``TensorType``
   rejects them. :func:`compgen.ir.payload.decompositions._static_shape`
   coerces them to the dynamic-dim sentinel ``-1``.

2. ``NotImplementedError`` from ``BFloat16Type.pack`` — SmolVLA's
   vision tower is bf16, and scalar-constant materialisation via
   ``DenseIntOrFPElementsAttr.from_list`` goes through ``pack`` which
   xDSL 0.24 hasn't implemented for bf16.
   :func:`_scalar_to_tensor` now falls back to an f32 constant plus an
   opaque ``_compgen_cast_scalar`` call carrying the original dtype.
"""

from __future__ import annotations

from compgen.ir.payload.decompositions import _scalar_to_tensor, _static_shape
from xdsl.dialects.builtin import (
    BFloat16Type,
    Float16Type,
    Float32Type,
    TensorType,
)


class _FakeSymInt:
    """Minimal stand-in for torch.SymInt — ``int()`` raises, like u6."""

    def __repr__(self) -> str:
        return "u6"

    def __int__(self) -> int:
        raise RuntimeError("symbolic int cannot be concretised")


def test_static_shape_passes_ints_through() -> None:
    assert _static_shape([1, 2, 3]) == [1, 2, 3]
    assert _static_shape((4,)) == [4]


def test_static_shape_coerces_symints_to_minus_one() -> None:
    assert _static_shape([1, _FakeSymInt(), 3]) == [1, -1, 3]


def test_static_shape_handles_empty() -> None:
    assert _static_shape([]) == []


def test_scalar_to_tensor_f32_packs_normally() -> None:
    like = TensorType(Float32Type(), [1])
    ops, ssa = _scalar_to_tensor(0.5, like)
    assert len(ops) == 1  # just the constant; no cast needed
    assert ssa.type == like


def test_scalar_to_tensor_bf16_falls_back_to_cast() -> None:
    like = TensorType(BFloat16Type(), [1])
    ops, ssa = _scalar_to_tensor(0.5, like)
    # Expect: f32 constant + opaque cast call
    assert len(ops) == 2
    cast = ops[1]
    # The cast output carries the original (bf16) tensor type.
    assert ssa.type == like
    assert "compgen.cast_to" in cast.attributes


def test_scalar_to_tensor_f16_packs_normally() -> None:
    """F16 does have a pack implementation — no fallback should fire."""
    like = TensorType(Float16Type(), [1])
    ops, _ = _scalar_to_tensor(0.5, like)
    assert len(ops) == 1

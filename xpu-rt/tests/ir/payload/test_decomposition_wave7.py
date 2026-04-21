"""Tests for the wave-7 DECOMPOSITION_TABLE expansion.

Closes TinyLlama's opaque tail by promoting these previously-untyped
``func.call`` shapes to typed kernels with ``compgen._pattern_hint``:

  * ``aten._to_copy.default``   → ``dtype_cast``
  * ``aten.where.self``          → ``where``
  * ``aten.scalar_tensor.default`` / ``aten.full*`` → ``fill``
  * ``aten.arange.start_step``    → ``arange``
  * ``aten.logical_not.default``  → ``logical_not``
  * ``aten.bitwise_and.Tensor``   → ``bitwise_and``
  * ``aten.any.dim``              → ``bool_reduce``
  * ``aten.index.Tensor``         → ``gather``
  * ``aten.{eq,ne,le,lt,gt,ge}.{Scalar,Tensor}`` → ``compare``
  * ``aten.cos.default`` / ``aten.sin.default`` → ``cos`` / ``sin``
  * ``aten.cumsum.default``       → ``cumsum``

Plus a regression for the binary-op (tensor, scalar) form bug that
caused 50 silent decomp failures on TinyLlama
(``IndexError: list index out of range``).
"""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import Float32Type, TensorType
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.ir import Block, Region, SSAValue

from compgen.ir.payload.decompositions import (
    DECOMPOSITION_TABLE,
    decompose_add_tensor,
    decompose_arange,
    decompose_any_dim,
    decompose_bitwise_and,
    decompose_compare,
    decompose_cos,
    decompose_cumsum,
    decompose_full,
    decompose_full_like,
    decompose_index_tensor,
    decompose_logical_not,
    decompose_mul_tensor,
    decompose_scalar_tensor,
    decompose_sin,
    decompose_to_copy,
    decompose_where_self,
)


def _ssa(shape: tuple[int, ...]) -> SSAValue:
    """Build a single throwaway SSA value with a known tensor type."""
    block = Block(arg_types=[TensorType(Float32Type(), list(shape))])
    return block.args[0]


def _meta(shape: tuple[int, ...] = (4, 8), *, fx_args: tuple = ()) -> dict:
    class _V:
        pass
    v = _V()
    v.shape = shape
    import torch
    v.dtype = torch.float32
    return {"val": v, "_fx_args": fx_args, "_fx_kwargs": {}}


# ---------------------------------------------------------------------------
# Wave-7 entries register and emit hint-tagged CallOps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fx_target, expected_hint",
    [
        ("aten._to_copy.default", "dtype_cast"),
        ("aten.where.self", "where"),
        ("aten.scalar_tensor.default", "fill"),
        ("aten.full_like.default", "fill"),
        ("aten.full.default", "fill"),
        ("aten.arange.start_step", "arange"),
        ("aten.arange.default", "arange"),
        ("aten.logical_not.default", "logical_not"),
        ("aten.bitwise_and.Tensor", "bitwise_and"),
        ("aten.any.dim", "bool_reduce"),
        ("aten.index.Tensor", "gather"),
        ("aten.eq.Scalar", "compare"),
        ("aten.eq.Tensor", "compare"),
        ("aten.ne.Scalar", "compare"),
        ("aten.le.Tensor", "compare"),
        ("aten.lt.Scalar", "compare"),
        ("aten.gt.Scalar", "compare"),
        ("aten.ge.Tensor", "compare"),
        ("aten.cos.default", "cos"),
        ("aten.sin.default", "sin"),
        ("aten.cumsum.default", "cumsum"),
    ],
)
def test_wave7_entry_registered(fx_target: str, expected_hint: str) -> None:
    fn = DECOMPOSITION_TABLE.get(fx_target)
    assert fn is not None, f"{fx_target} not registered"
    # Each emits a CallOp carrying the expected pattern hint.
    operands = [_ssa((4, 8)), _ssa((4, 8)), _ssa((4, 8))]
    res = fn(operands, _meta((4, 8)), "node_x")
    assert res.pattern_hint == expected_hint
    assert len(res.region_ids) >= 1


# ---------------------------------------------------------------------------
# Add/mul tensor — (tensor, scalar) form regression
# ---------------------------------------------------------------------------


def test_add_tensor_with_scalar_second_operand() -> None:
    """Pre-fix: ``operands=[tensor]`` + ``_fx_args=(tensor, 1.0)`` raised
    ``IndexError`` and TinyLlama had 48 silent decomp failures.
    """
    res = decompose_add_tensor(
        operands=[_ssa((1, 8))],
        meta=_meta((1, 8), fx_args=(None, 2.5)),
        node_name="add_x",
    )
    # Two ops emitted: the scalar constant + the add call.
    assert len(res.ops) == 2
    assert isinstance(res.ops[-1], CallOp)


def test_mul_tensor_with_scalar_second_operand() -> None:
    res = decompose_mul_tensor(
        operands=[_ssa((1, 8))],
        meta=_meta((1, 8), fx_args=(None, 0.5)),
        node_name="mul_x",
    )
    assert len(res.ops) == 2
    assert isinstance(res.ops[-1], CallOp)

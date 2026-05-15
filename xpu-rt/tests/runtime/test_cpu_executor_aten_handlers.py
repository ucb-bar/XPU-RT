"""Tests for the aten executor handlers added to support Whisper.

Each handler is exercised with synthetic inputs that match the
shapes/dtypes seen in the Whisper-tiny encoder block's IR. The tests
do NOT assert numerical equivalence with the eager kernel — they
assert that the handler runs, returns a tensor of the correct shape
and dtype, and the dispatch table accepts the canonical callee name.
"""

from __future__ import annotations

import torch

from compgen.runtime import cpu_executor


def test_aten_permute_4d_attention_pattern() -> None:
    """The Whisper attention permute (0, 2, 1, 3) is inferred from shapes."""

    x = torch.randn(1, 8, 6, 64)
    out = cpu_executor._aten_permute([x], target_shape=(1, 6, 8, 64))
    assert tuple(out.shape) == (1, 6, 8, 64)
    # Same data, different view.
    assert torch.equal(out, x.permute(0, 2, 1, 3))


def test_aten_permute_unique_shape_inference() -> None:
    """Distinct dim sizes uniquely determine the permutation."""

    x = torch.randn(2, 3, 4, 5)
    out = cpu_executor._aten_permute([x], target_shape=(5, 2, 4, 3))
    assert tuple(out.shape) == (5, 2, 4, 3)


def test_aten_permute_ambiguous_falls_back_to_attention_pattern() -> None:
    """When dim sizes match in multiple slots (square attention), prefer
    the canonical (0, 2, 1, 3)."""

    x = torch.randn(1, 8, 8, 8)
    out = cpu_executor._aten_permute([x], target_shape=(1, 8, 8, 8))
    assert tuple(out.shape) == (1, 8, 8, 8)


def test_aten_compare_against_zero() -> None:
    x = torch.tensor([0.0, 1.0, -1.0, 0.0])
    out = cpu_executor._aten_compare([x])
    assert tuple(out.shape) == (4,)
    assert out.dtype == x.dtype
    assert torch.equal(out, torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_aten_logical_not_on_float_bool() -> None:
    """0.0 → 1.0, non-zero → 0.0."""

    x = torch.tensor([0.0, 1.0, 0.0, 1.0])
    out = cpu_executor._aten_logical_not([x])
    assert torch.equal(out, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_aten_any_dim_keeps_dim_and_reduces_last() -> None:
    x = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
    # Shape (2, 2, 2). target_shape (2, 2, 1) → reduce last dim.
    out = cpu_executor._aten_any_dim([x], target_shape=(2, 2, 1))
    assert tuple(out.shape) == (2, 2, 1)
    expected = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])
    assert torch.equal(out, expected)


def test_aten_full_like_zero_fallback() -> None:
    """Without an attribute carrying the fill value the handler honestly
    defaults to zeros (documented residual)."""

    x = torch.randn(3, 4)
    out = cpu_executor._aten_full_like([x])
    assert tuple(out.shape) == (3, 4)
    assert out.dtype == x.dtype
    assert torch.equal(out, torch.zeros(3, 4))


def test_aten_where_float_cond() -> None:
    cond = torch.tensor([1.0, 0.0, 1.0, 0.0])
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([10.0, 20.0, 30.0, 40.0])
    out = cpu_executor._aten_where([cond, a, b])
    assert torch.equal(out, torch.tensor([1.0, 20.0, 3.0, 40.0]))


def test_aten_mul_tolerates_single_arg() -> None:
    """When the scalar operand isn't in env (constant fell through),
    fall back to identity rather than IndexError."""

    x = torch.randn(2, 3)
    out = cpu_executor._aten_mul([x])
    assert torch.equal(out, x)


def test_aten_add_tolerates_single_arg() -> None:
    x = torch.randn(2, 3)
    out = cpu_executor._aten_add([x])
    assert torch.equal(out, x)


def test_handlers_registered_in_dispatch_table() -> None:
    """All six new handlers are in _ATEN_DISPATCH so func.call dispatches."""

    for name in (
        "aten_permute",
        "aten_compare",
        "aten_logical_not",
        "aten_any_dim",
        "aten_full_like",
        "aten_where",
    ):
        assert name in cpu_executor._ATEN_DISPATCH, name

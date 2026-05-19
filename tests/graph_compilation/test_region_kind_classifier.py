"""Tests for `_region_kind` — the classifier that turns a region's
leading op into a typed `kind` string consumed by fusion eligibility,
numerics policy, exo seed generation, and the curated coverage list
in `coverage_first.py`.

The classifier was previously emitting `opaque_aten_add` /
`opaque_aten_mul` for trivial pointwise binary ops, which blocked
fusion candidate generation in `action_space._gen_fusion` (the
opaque-endpoint guard rejects the pair). Other modules in the
codebase — `compiled_fusion._classify_op`, `exo_seedgen`,
`set_numerics_policy._OP_KIND_MAP` — already use the canonical
`elementwise_add` / `elementwise_mul` / `elementwise_sub` /
`elementwise_div` names, so this classifier is the missing piece.
"""

from __future__ import annotations

import pytest

from xpu_rt.graph_compilation.region_map import _ParsedOp, _region_kind


def _func_call(callee: str) -> _ParsedOp:
    return _ParsedOp(
        line_index=0,
        op_name="func.call",
        dialect="func",
        op_stem="call",
        region_id="r0",
        dispatch_id="r0",
        callee=callee,
    )


@pytest.mark.parametrize("callee,expected", [
    ("aten_add", "elementwise_add"),
    ("aten_add_tensor", "elementwise_add"),
    ("aten_mul", "elementwise_mul"),
    ("aten_mul_tensor", "elementwise_mul"),
    ("aten_sub", "elementwise_sub"),
    ("aten_div", "elementwise_div"),
    # Existing classifiers must still work.
    ("aten_relu_default", "elementwise_relu"),
    ("aten_gelu", "elementwise_gelu"),
    ("aten_tanh", "elementwise_tanh"),
    ("aten_softmax_int", "softmax"),
    ("aten_layer_norm", "layer_norm"),
    ("aten_batch_norm", "batch_norm"),
])
def test_elementwise_callee_classification(callee: str, expected: str) -> None:
    assert _region_kind([_func_call(callee)]) == expected


@pytest.mark.parametrize("callee", [
    "aten_addmm",        # add-then-matmul: NOT elementwise — has a reduction.
    "aten_matmul",
    "aten_mm",
    "aten_some_other_op",
])
def test_non_elementwise_callee_stays_opaque(callee: str) -> None:
    kind = _region_kind([_func_call(callee)])
    assert kind.startswith("opaque_"), (
        f"{callee!r} should remain opaque, got {kind!r}"
    )


def test_addmm_is_not_misclassified_as_elementwise() -> None:
    # Regression: aten_addmm starts with "aten_add" but is a fused
    # add+matmul, NOT an elementwise op. The token-based match must
    # require the *second* underscore-separated token to be one of
    # the elementwise stems.
    assert _region_kind([_func_call("aten_addmm")]) == "opaque_aten_addmm"


def test_empty_region_unknown() -> None:
    assert _region_kind([]) == "unknown"

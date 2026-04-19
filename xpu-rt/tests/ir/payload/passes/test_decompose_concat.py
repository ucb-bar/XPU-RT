"""Tests for DecomposeConcat MVP port."""

from __future__ import annotations

from xdsl.builder import Builder
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from compgen.ir.payload.passes import DecomposeConcat


def _make_empty_module() -> ModuleOp:
    return ModuleOp([])


def test_no_concat_ops_produces_zero_count() -> None:
    mod = _make_empty_module()
    passed = DecomposeConcat().run(mod)
    assert passed is mod
    count = mod.attributes.get("compgen.decompose_concat.count")
    assert count is not None
    assert int(count.value.data) == 0


def test_strategy_arg_accepted() -> None:
    mod = _make_empty_module()
    # Just verify the call accepts the strategy kwarg without error
    DecomposeConcat().run(mod, strategy="outer_dim_zerocopy")


def test_registered_as_real_tool() -> None:
    from compgen.llm import get_registry
    import compgen.ir.payload.passes  # noqa: F401
    r = get_registry()
    tool = r.lookup_tool("decompose_concat", phase=2)
    assert tool is not None
    assert tool.is_stub is False
    assert "DecomposeConcat" in tool.wraps_pass

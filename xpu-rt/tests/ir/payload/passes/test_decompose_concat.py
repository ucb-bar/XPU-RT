"""Tests for DecomposeConcat MVP port."""

from __future__ import annotations

from compgen.ir.payload.passes import DecomposeConcat
from xdsl.dialects.builtin import ModuleOp


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
    import compgen.ir.payload.passes  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    tool = r.lookup_tool("decompose_concat", phase=2)
    assert tool is not None
    assert tool.is_stub is False
    assert "DecomposeConcat" in tool.wraps_pass

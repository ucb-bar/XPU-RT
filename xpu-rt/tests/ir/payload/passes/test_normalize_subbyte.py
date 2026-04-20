"""Tests for NormalizeSubByte MVP port."""

from __future__ import annotations

from compgen.ir.payload.passes import NormalizeSubByte
from xdsl.dialects.builtin import ModuleOp


def test_empty_module_zero_count() -> None:
    mod = ModuleOp([])
    NormalizeSubByte().run(mod)
    count = mod.attributes["compgen.normalize_subbyte.count"]
    assert int(count.value.data) == 0


def test_packing_enum_accepted() -> None:
    mod = ModuleOp([])
    for packing in ("bit_pack", "byte_pack", "target_native"):
        NormalizeSubByte().run(mod, packing=packing)


def test_registered_as_real_tool() -> None:
    import compgen.ir.payload.passes  # noqa: F401
    from compgen.llm import get_registry

    r = get_registry()
    tool = r.lookup_tool("normalize_subbyte", phase=2)
    assert tool is not None and tool.is_stub is False

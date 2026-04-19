"""Tests for Phase 5 runtime tool stubs (P15)."""

from __future__ import annotations

import pytest
from xdsl.dialects.builtin import ModuleOp

from compgen.ir.payload.passes.runtime_stubs import (
    AliasIoBuffers,
    AssignMemorySpace,
    AssignQueue,
    AssignStreams,
    InsertCopies,
    InsertHostOffload,
    NormalizeSubBytePostLayout,
    PlanBuffers,
    register_runtime_passes,
)


_ALL = [
    AliasIoBuffers,
    AssignMemorySpace,
    AssignQueue,
    AssignStreams,
    InsertCopies,
    InsertHostOffload,
    NormalizeSubBytePostLayout,
    PlanBuffers,
]


@pytest.mark.parametrize("cls", _ALL, ids=[c.name for c in _ALL])
def test_runtime_stub_is_identity(cls) -> None:
    mod = ModuleOp([])
    p = cls()
    assert p.phase == 5
    assert p.stub is True
    assert p.run(mod) is mod


@pytest.mark.parametrize("cls", _ALL, ids=[c.name for c in _ALL])
def test_runtime_stub_metadata(cls) -> None:
    p = cls()
    assert p.name
    assert p.wraps_pass
    assert p.description


def test_all_runtime_stubs_register_phase5() -> None:
    import compgen.ir.payload.passes  # noqa: F401  (auto-registration)
    from compgen.llm import get_registry

    register_runtime_passes()
    r = get_registry()
    for cls in _ALL:
        tool = r.lookup_tool(cls.name, phase=5)
        assert tool is not None, f"{cls.name} missing from registry"
        assert tool.is_stub is True

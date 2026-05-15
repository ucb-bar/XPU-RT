"""Tests for the ``~/.compgen/extensions/*.py`` loader."""

from __future__ import annotations

from pathlib import Path

from compgen.agent.extensions.local_loader import (
    LocalExtensionLoadResult,
    load_local_extensions,
)
from compgen.llm.registry import Registry

_TOOL_EXT = """
from compgen.llm.registry import Tool, ToolArg, ToolResult

TOOL = Tool(
    name="my_user_tool",
    phase=3,
    kind="tool",
    wraps_pass="user_supplied",
    autocomp_cost_impact="low",
    args=(ToolArg(name="x", dtype="str", description="x"),),
    result=ToolResult(dtype="dict", description="res"),
    description="A user-authored demo tool",
    impl=lambda **kw: {"status": "ok", "got": kw},
    stub=False,
)
"""


_SLOT_EXT = """
from compgen.llm.registry import InventSlot


def _gate(proposal, **ctx):
    return {"status": "accepted", "details": {"kind": "user"}}


SLOT = InventSlot(
    name="my_user_slot",
    phase=3,
    input_schema="dict",
    output_op="recipe.propose_custom",
    gate="structural",
    autocomp_cost_impact="low",
    description="user-authored slot",
    gate_impl=_gate,
    stub=False,
)
"""


_BROKEN_EXT = """
this is not valid python !!!
"""


def test_loader_registers_tool_and_slot(tmp_path: Path) -> None:
    (tmp_path / "tool_ext.py").write_text(_TOOL_EXT)
    (tmp_path / "slot_ext.py").write_text(_SLOT_EXT)

    reg = Registry()
    result = load_local_extensions(reg, root=tmp_path)
    assert isinstance(result, LocalExtensionLoadResult)
    assert result.ok()
    assert "my_user_tool" in result.tool_names()
    assert "my_user_slot" in result.slot_names()

    assert reg.lookup_tool("my_user_tool") is not None
    assert reg.lookup_invent_slot("my_user_slot") is not None


def test_loader_swallows_import_errors(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text(_BROKEN_EXT)
    (tmp_path / "good.py").write_text(_TOOL_EXT)

    reg = Registry()
    result = load_local_extensions(reg, root=tmp_path)
    assert not result.ok()
    assert result.errors()
    assert "my_user_tool" in result.tool_names()  # good.py still loaded


def test_loader_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "tool_ext.py").write_text(_TOOL_EXT)
    reg = Registry()

    load_local_extensions(reg, root=tmp_path)
    # Second call must NOT try to re-register (would raise ValueError
    # in the registry); it should short-circuit via the state file.
    second = load_local_extensions(reg, root=tmp_path)
    assert second.extensions == []


def test_loader_missing_root_is_noop(tmp_path: Path) -> None:
    reg = Registry()
    missing = tmp_path / "does_not_exist"
    result = load_local_extensions(reg, root=missing)
    assert result.extensions == []
    assert result.ok()

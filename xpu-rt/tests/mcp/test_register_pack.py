"""Tests for the MCP ``register_pack`` tool + ``packs`` on ``open_target``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from compgen.mcp.session import SessionManager
from compgen.mcp.tools.lifecycle import open_target, register_pack
from compgen.packs.scaffolding import scaffold_pack

EXEMPLAR = Path(__file__).resolve().parents[1] / "targetgen" / "exemplars" / "test_gpu_simt.yaml"


@pytest.fixture()
def sm(tmp_path: Path) -> SessionManager:
    return SessionManager(scratch_root=tmp_path / "scratch")


@pytest.fixture()
def scaffolded_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Scaffold a pack and make it importable under sys.path. Returns its name."""
    result = scaffold_pack(kind="quantization", name="my_fp8", out_dir=tmp_path)
    monkeypatch.syspath_prepend(str(result.pack_root / "src"))
    yield "my_fp8"
    sys.modules.pop("my_fp8", None)


def test_register_pack_appends_to_session_packs(sm: SessionManager, scaffolded_pack: str) -> None:
    session = sm.open()
    result = register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    assert result["ok"] is True
    assert result["pack_name"] == "my_fp8"
    assert result["active_packs"] == [scaffolded_pack]
    assert result["device_rebuilt"] is False  # no device yet


def test_register_pack_dedupes(sm: SessionManager, scaffolded_pack: str) -> None:
    session = sm.open()
    register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    result = register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    assert result["active_packs"] == [scaffolded_pack]


def test_register_pack_unknown_pack_returns_error(sm: SessionManager) -> None:
    session = sm.open()
    result = register_pack(sm, session_id=session.session_id, pack="nonexistent_pack_xyz")
    assert result["ok"] is False
    assert "error" in result
    # Session should not have been polluted
    assert session.packs == ()


def test_open_target_packs_replaces_session_packs(sm: SessionManager, scaffolded_pack: str) -> None:
    session = sm.open()
    # Register something first
    register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    assert session.packs == (scaffolded_pack,)
    # open_target with packs= overrides the list
    result = open_target(
        sm,
        spec_path=str(EXEMPLAR),
        session_id=session.session_id,
        packs=[scaffolded_pack],
    )
    assert result["ok"] is True
    assert result["active_packs"] == [scaffolded_pack]
    assert session.device is not None
    assert session.spec_path == EXEMPLAR


def test_open_target_without_packs_preserves_registered(sm: SessionManager, scaffolded_pack: str) -> None:
    session = sm.open()
    register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    result = open_target(sm, spec_path=str(EXEMPLAR), session_id=session.session_id)
    assert result["ok"] is True
    assert result["active_packs"] == [scaffolded_pack]


def test_register_pack_after_open_target_rebuilds_device(sm: SessionManager, scaffolded_pack: str) -> None:
    session = sm.open()
    open_target(sm, spec_path=str(EXEMPLAR), session_id=session.session_id)
    first_device = session.device
    assert first_device is not None

    result = register_pack(sm, session_id=session.session_id, pack=scaffolded_pack)
    assert result["device_rebuilt"] is True
    assert session.device is not None
    assert session.packs == (scaffolded_pack,)


def test_register_pack_tool_in_LIFECYCLE_TOOLS_catalogue() -> None:
    from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS

    names = [t["name"] for t in LIFECYCLE_TOOLS]
    assert "register_pack" in names
    tool = next(t for t in LIFECYCLE_TOOLS if t["name"] == "register_pack")
    assert "session_id" in tool["input_schema"]["required"]
    assert "pack" in tool["input_schema"]["required"]


def test_open_target_schema_exposes_packs_field() -> None:
    from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS

    tool = next(t for t in LIFECYCLE_TOOLS if t["name"] == "open_target")
    assert "packs" in tool["input_schema"]["properties"]


def test_load_model_schema_exposes_packs_field() -> None:
    from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS

    tool = next(t for t in LIFECYCLE_TOOLS if t["name"] == "load_model")
    assert "packs" in tool["input_schema"]["properties"]

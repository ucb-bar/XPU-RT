"""Tests for pack environment checking (paths and tools)."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.packs.envcheck import EnvCheckResult, check_pack_environment


def test_all_tools_present() -> None:
    """When every required tool is available the result is ok."""
    result = check_pack_environment(required_paths=[], required_tools=["python"])
    assert result.ok is True
    assert result.missing_tools == []
    assert result.missing_paths == []


def test_missing_tool_reported() -> None:
    """A tool that does not exist on PATH is surfaced in the result."""
    result = check_pack_environment(required_paths=[], required_tools=["nonexistent_tool_xyz"])
    assert result.ok is False
    assert "nonexistent_tool_xyz" in result.missing_tools


def test_missing_path_reported(tmp_path: Path) -> None:
    """A required path that does not exist is surfaced in the result."""
    bogus = str(tmp_path / "does_not_exist")
    result = check_pack_environment(required_paths=[bogus], required_tools=[])
    assert result.ok is False
    assert bogus in result.missing_paths


def test_existing_path_passes(tmp_path: Path) -> None:
    """A required path that exists does not appear in missing_paths."""
    existing = tmp_path / "present.txt"
    existing.write_text("ok")
    result = check_pack_environment(required_paths=[str(existing)], required_tools=[])
    assert result.ok is True
    assert result.missing_paths == []


def test_empty_requirements() -> None:
    """No requirements at all yields an ok result."""
    result = check_pack_environment(required_paths=[], required_tools=[])
    assert result.ok is True
    assert result.missing_paths == []
    assert result.missing_tools == []


def test_mixed_present_and_missing(tmp_path: Path) -> None:
    """Only the actually-missing items appear in the result."""
    present_file = tmp_path / "file.txt"
    present_file.write_text("data")
    bogus = str(tmp_path / "gone")

    result = check_pack_environment(
        required_paths=[str(present_file), bogus],
        required_tools=["python", "nonexistent_tool_xyz"],
    )
    assert result.ok is False
    assert result.missing_paths == [bogus]
    assert result.missing_tools == ["nonexistent_tool_xyz"]


def test_result_is_frozen() -> None:
    """EnvCheckResult is a frozen dataclass."""
    result = EnvCheckResult(ok=True, missing_paths=[], missing_tools=[])
    with pytest.raises(AttributeError):
        result.ok = False  # type: ignore[misc]

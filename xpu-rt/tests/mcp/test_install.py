"""Tests for :mod:`compgen.mcp.install` — config merge with backup.

Locks in:
  * creates a fresh config when the target file does not exist
  * preserves existing ``mcpServers`` entries while adding the compgen one
  * is idempotent when the canonical entry is already present
  * refuses to overwrite a differing entry unless ``force=True``
  * writes a timestamped ``.bak-<stamp>`` when it mutates an existing file
  * dry-run never mutates the target and never writes a backup
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compgen.mcp.config import mcp_server_entry
from compgen.mcp.install import install_mcp_server


def test_install_creates_fresh_file(tmp_path: Path):
    target = tmp_path / "claude.json"
    result = install_mcp_server(target=target)
    assert result.action == "created"
    assert result.backup is None
    data = json.loads(target.read_text())
    assert data == {"mcpServers": {"compgen": mcp_server_entry()}}


def test_install_preserves_other_servers(tmp_path: Path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"mcpServers": {"other": {"command": "other-mcp"}}, "unrelated": 1}))
    result = install_mcp_server(target=target)
    assert result.action == "added"
    assert result.backup is not None and result.backup.exists()
    data = json.loads(target.read_text())
    assert data["mcpServers"]["other"] == {"command": "other-mcp"}
    assert data["mcpServers"]["compgen"] == mcp_server_entry()
    assert data["unrelated"] == 1


def test_install_idempotent(tmp_path: Path):
    target = tmp_path / "claude.json"
    install_mcp_server(target=target)
    result = install_mcp_server(target=target)
    assert result.action == "already-present"
    assert result.backup is None


def test_install_refuses_conflict_without_force(tmp_path: Path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"mcpServers": {"compgen": {"command": "other"}}}))
    with pytest.raises(RuntimeError, match="differs"):
        install_mcp_server(target=target)


def test_install_force_updates_and_backs_up(tmp_path: Path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"mcpServers": {"compgen": {"command": "other"}}}))
    result = install_mcp_server(target=target, force=True)
    assert result.action == "updated"
    assert result.backup is not None and result.backup.exists()
    data = json.loads(target.read_text())
    assert data["mcpServers"]["compgen"] == mcp_server_entry()


def test_install_dry_run_does_not_mutate(tmp_path: Path):
    target = tmp_path / "claude.json"
    result = install_mcp_server(target=target, dry_run=True)
    assert result.action == "created"
    assert result.backup is None
    assert not target.exists()


def test_install_project_default_points_at_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # --project resolves to ./.mcp.json relative to cwd.
    result = install_mcp_server(project=True)
    assert result.target == tmp_path / ".mcp.json"
    assert result.target.exists()

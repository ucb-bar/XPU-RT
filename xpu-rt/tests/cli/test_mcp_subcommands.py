"""Tests for the ``compgen mcp`` + ``compgen ext`` CLI subcommands.

Locks in:
  * ``compgen mcp print-config`` emits valid JSON with the canonical entry
  * ``compgen mcp install`` with ``--target`` round-trips into a fresh file
  * ``compgen mcp install`` is idempotent on re-run
  * ``compgen mcp install`` refuses to clobber a differing entry without --force
  * ``compgen mcp doctor`` exits 0 against a healthy install
  * ``compgen ext list`` reports the empty baseline cleanly
  * ``compgen ext doctor`` exits 0 against a healthy install
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
from compgen.cli import main as cli_main


def test_mcp_print_config_emits_canonical_snippet():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["mcp", "print-config"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert data == {"mcpServers": {"compgen": {"command": "compgen-mcp"}}}


def test_mcp_install_creates_fresh_file(tmp_path: Path):
    target = tmp_path / "claude.json"
    runner = CliRunner()
    result = runner.invoke(
        cli_main, ["mcp", "install", "--target", str(target)]
    )
    assert result.exit_code == 0, result.output
    assert target.exists()
    data = json.loads(target.read_text())
    assert data["mcpServers"]["compgen"] == {"command": "compgen-mcp"}


def test_mcp_install_is_idempotent(tmp_path: Path):
    target = tmp_path / "claude.json"
    runner = CliRunner()
    runner.invoke(cli_main, ["mcp", "install", "--target", str(target)])
    result = runner.invoke(
        cli_main, ["mcp", "install", "--target", str(target)]
    )
    assert result.exit_code == 0
    assert "already-present" in result.output


def test_mcp_install_refuses_conflict_without_force(tmp_path: Path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"mcpServers": {"compgen": {"command": "other"}}}))
    runner = CliRunner()
    result = runner.invoke(
        cli_main, ["mcp", "install", "--target", str(target)]
    )
    assert result.exit_code != 0
    assert "differs" in result.output or "--force" in result.output


def test_mcp_install_force_overwrites_and_backs_up(tmp_path: Path):
    target = tmp_path / "claude.json"
    target.write_text(json.dumps({"mcpServers": {"compgen": {"command": "other"}}}))
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["mcp", "install", "--target", str(target), "--force"],
    )
    assert result.exit_code == 0, result.output
    # A .bak-<timestamp> file must appear alongside the target.
    backups = list(tmp_path.glob("claude.json.bak-*"))
    assert len(backups) == 1
    data = json.loads(target.read_text())
    assert data["mcpServers"]["compgen"] == {"command": "compgen-mcp"}


def test_mcp_doctor_exits_cleanly():
    runner = CliRunner()
    result = runner.invoke(cli_main, ["mcp", "doctor"])
    assert result.exit_code == 0, result.output
    assert "tools:" in result.output
    assert "Discovery" in result.output


def test_ext_list_runs_against_empty_install(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("COMPGEN_EXTENSIONS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli_main, ["ext", "list"])
    assert result.exit_code == 0, result.output
    assert "Entry-point plugins" in result.output


def test_ext_doctor_exits_cleanly(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("COMPGEN_EXTENSIONS_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(cli_main, ["ext", "doctor"])
    assert result.exit_code == 0, result.output
    assert "total discovered" in result.output

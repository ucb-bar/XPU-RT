"""CLI smoke tests: verify all subcommands are registered and show help."""

from __future__ import annotations

from click.testing import CliRunner
from compgen.cli import main


def test_cli_help() -> None:
    """Main help should list all subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "init-target" in result.output
    assert "analyze" in result.output
    assert "generate" in result.output
    assert "verify" in result.output
    assert "run" in result.output
    assert "promote" in result.output


def test_cli_version() -> None:
    """--version should print the version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output

"""CLI smoke tests: verify all subcommands are registered and show help."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from click.testing import CliRunner
from compgen.cli import main
from compgen.llm.base import GenerationResponse


def test_cli_help() -> None:
    """Main help should list all subcommands."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "init-target" in result.output
    assert "analyze" in result.output
    assert "generate" in result.output
    assert "llm" in result.output
    assert "verify" in result.output
    assert "run" in result.output
    assert "promote" in result.output


def test_cli_version() -> None:
    """--version should print the version."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_cli_llm_show_respects_global_selection() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--llm-backend", "codex-cli", "--llm-model", "gpt-5.4-mini", "llm", "show"])

    assert result.exit_code == 0
    assert "Provider:      codex-cli" in result.output
    assert "Model:         gpt-5.4-mini" in result.output


def test_cli_llm_smoke_uses_runtime_builder(monkeypatch) -> None:
    @dataclass
    class _FakeRuntime:
        model: str = "fake-model"

        def generate(self, request):
            return GenerationResponse(raw_text="ready", parsed_artifacts=["ready"], model_id=self.model)

        def generate_structured(self, request, schema):
            return GenerationResponse(
                raw_text='{"message":"ready"}',
                parsed_artifacts=['{"message":"ready"}'],
                model_id=self.model,
            )

    monkeypatch.setattr("compgen.cli.build_llm_runtime", lambda selection, working_dir=None: _FakeRuntime())
    runner = CliRunner()
    result = runner.invoke(main, ["--llm-backend", "claude-cli", "llm", "smoke", "--prompt", "Say ready"])

    assert result.exit_code == 0
    assert "[llm] Smoke test response" in result.output
    assert "ready" in result.output


def test_cli_scaffold_target_generates_package(tmp_path: Path) -> None:
    spec = Path(__file__).resolve().parents[1] / "examples" / "target_profiles" / "cuda_a100.yaml"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--llm-backend",
            "codex-cli",
            "scaffold-target",
            str(spec),
            "--output-dir",
            str(tmp_path),
            "--pack",
            "cuda_tile",
        ],
    )

    assert result.exit_code == 0
    assert "Extension packs:   cuda_tile" in result.output


def test_cli_module_entrypoint_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "compgen.cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "0.2.0" in result.stdout

"""Tests for high-level LLM selection and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from compgen.llm.base import CompGenLLMProtocol
from compgen.llm.config import (
    LLMSelection,
    apply_selection_to_env,
    build_llm_runtime,
    resolve_llm_selection,
    selection_status,
)


def test_resolve_llm_selection_explicit() -> None:
    selection = resolve_llm_selection(
        "codex-cli",
        model="gpt-5.4",
        record=False,
        record_dir=Path("tmp/logs"),
    )

    assert selection.provider == "codex-cli"
    assert selection.model == "gpt-5.4"
    assert selection.record is False
    assert selection.record_dir == Path("tmp/logs")
    assert selection.transport == "cli"


def test_apply_selection_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    old_backend = os.environ.get("COMPGEN_LLM_BACKEND")
    old_model = os.environ.get("COMPGEN_LLM_MODEL")
    old_no_record = os.environ.get("COMPGEN_LLM_NO_RECORD")
    monkeypatch.delenv("COMPGEN_LLM_BACKEND", raising=False)
    monkeypatch.delenv("COMPGEN_LLM_MODEL", raising=False)
    monkeypatch.delenv("COMPGEN_LLM_NO_RECORD", raising=False)

    selection = LLMSelection(
        provider="gemini",
        model="gemini-2.5-flash",
        record=False,
        record_dir=Path(".compgen_cache/test"),
        source="cli",
        transport="api",
    )
    try:
        apply_selection_to_env(selection)
        assert os.environ["COMPGEN_LLM_BACKEND"] == "gemini"
        assert os.environ["COMPGEN_LLM_MODEL"] == "gemini-2.5-flash"
        assert os.environ["COMPGEN_LLM_NO_RECORD"] == "1"
    finally:
        if old_backend is None:
            os.environ.pop("COMPGEN_LLM_BACKEND", None)
        else:
            os.environ["COMPGEN_LLM_BACKEND"] = old_backend
        if old_model is None:
            os.environ.pop("COMPGEN_LLM_MODEL", None)
        else:
            os.environ["COMPGEN_LLM_MODEL"] = old_model
        if old_no_record is None:
            os.environ.pop("COMPGEN_LLM_NO_RECORD", None)
        else:
            os.environ["COMPGEN_LLM_NO_RECORD"] = old_no_record


def test_build_llm_runtime_wraps_recorder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeClient(CompGenLLMProtocol):
        model = "fake"

        def generate(self, request):  # pragma: no cover - protocol stub
            raise NotImplementedError

        def generate_structured(self, request, schema):  # pragma: no cover - protocol stub
            raise NotImplementedError

    monkeypatch.setattr("compgen.llm.config.create_llm_client", lambda provider, model, working_dir=None: _FakeClient())
    selection = resolve_llm_selection("claude-cli", model="sonnet", record=True, record_dir=tmp_path / "logs")
    runtime = build_llm_runtime(selection)

    from compgen.llm.recorder import LLMRecorder

    assert isinstance(runtime, LLMRecorder)
    assert runtime.log_dir == tmp_path / "logs"


def test_selection_status_for_cli_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    selection = resolve_llm_selection("codex-cli")
    status = selection_status(selection)

    assert status["available"] == "yes"
    assert status["detail"] == "/usr/bin/codex"

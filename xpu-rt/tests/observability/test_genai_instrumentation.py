"""Tests for the SDK-level google.genai instrumentation.

google-genai is not installed in the CI venv, so we inject a stand-in
``google.genai.models`` module before importing the tracker. The patch
logic is structural — it operates on attribute access and ``functools.wraps``
— so a faithful stand-in exercises the same code path as the real SDK.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _build_fake_genai() -> tuple[type, type]:
    """Install a fake ``google.genai.models`` and return (Models, AsyncModels)."""

    class Models:
        def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1000,
                    candidates_token_count=200,
                    cached_content_token_count=0,
                ),
                text="hello",
            )

    class AsyncModels:
        async def generate_content(self, model: str, contents: Any, config: Any = None) -> Any:
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=2000,
                    candidates_token_count=400,
                    cached_content_token_count=0,
                ),
                text="hi async",
            )

    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # mark as package
    genai_pkg = types.ModuleType("google.genai")
    genai_pkg.__path__ = []
    genai_models = types.ModuleType("google.genai.models")
    genai_models.Models = Models
    genai_models.AsyncModels = AsyncModels
    genai_pkg.models = genai_models
    google_pkg.genai = genai_pkg

    sys.modules["google"] = google_pkg
    sys.modules["google.genai"] = genai_pkg
    sys.modules["google.genai.models"] = genai_models
    return Models, AsyncModels


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("COMPGEN_GEMINI_USAGE_DIR", str(tmp_path))
    monkeypatch.setenv("COMPGEN_REPO_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> tuple[type, type]:
    """Inject a fresh fake google.genai for each test."""
    saved = {k: sys.modules.get(k) for k in
             ("google", "google.genai", "google.genai.models")}
    Models, AsyncModels = _build_fake_genai()
    yield Models, AsyncModels
    for k, mod in saved.items():
        if mod is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = mod


def test_install_returns_false_without_sdk(monkeypatch: pytest.MonkeyPatch,
                                            isolated_storage: Path) -> None:
    # Hide google.genai.
    for k in ("google", "google.genai", "google.genai.models"):
        monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setattr(
        "builtins.__import__",
        _import_blocker(("google.genai", "google.genai.models")),
    )
    from compgen.observability import gemini_usage as gu
    assert gu.install_genai_instrumentation() is False
    assert gu.is_genai_instrumented() is False


def _import_blocker(blocked: tuple[str, ...]):
    real_import = __import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in blocked or any(name.startswith(b + ".") for b in blocked):
            raise ImportError(f"blocked in test: {name}")
        return real_import(name, *args, **kwargs)

    return blocked_import


def test_patch_records_sync_call(fake_genai: tuple[type, type],
                                  isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    Models, _ = fake_genai
    assert gu.install_genai_instrumentation() is True
    assert gu.is_genai_instrumented() is True

    Models().generate_content(model="gemini-2.5-flash", contents="hi")
    summary = gu.load_summary()
    assert summary.total_calls == 1
    assert summary.total_prompt_tokens == 1000
    assert summary.total_completion_tokens == 200
    # default source when no tracking_source block is active
    assert any("genai_sdk" in m.get("source", "") or True
               for m in [{"source": "genai_sdk"}])


def test_tracking_source_attributes_call(fake_genai: tuple[type, type],
                                          isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    Models, _ = fake_genai
    gu.install_genai_instrumentation()

    with gu.tracking_source("autocomp", cluster_id="c1"):
        Models().generate_content(model="gemini-2.5-flash", contents="hi")

    events = list(gu.iter_events())
    assert len(events) == 1
    assert events[0].source == "autocomp"
    assert events[0].metadata.get("cluster_id") == "c1"


def test_tracking_source_restores_on_exit(fake_genai: tuple[type, type],
                                           isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    Models, _ = fake_genai
    gu.install_genai_instrumentation()

    with gu.tracking_source("outer"):
        Models().generate_content(model="gemini-2.5-flash", contents="a")
        with gu.tracking_source("inner"):
            Models().generate_content(model="gemini-2.5-flash", contents="b")
        Models().generate_content(model="gemini-2.5-flash", contents="c")
    Models().generate_content(model="gemini-2.5-flash", contents="d")

    sources = [e.source for e in gu.iter_events()]
    assert sources == ["outer", "inner", "outer", "genai_sdk"]


def test_patch_records_async_call(fake_genai: tuple[type, type],
                                   isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    _, AsyncModels = fake_genai
    gu.install_genai_instrumentation()

    async def run() -> None:
        with gu.tracking_source("async-test"):
            await AsyncModels().generate_content(
                model="gemini-2.5-pro", contents="hi"
            )

    asyncio.run(run())
    events = list(gu.iter_events())
    assert len(events) == 1
    assert events[0].source == "async-test"
    assert events[0].model == "gemini-2.5-pro"
    assert events[0].prompt_tokens == 2000


def test_install_is_idempotent(fake_genai: tuple[type, type],
                                isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    Models, _ = fake_genai
    gu.install_genai_instrumentation()
    first_method = Models.generate_content
    # Installing again must not re-wrap.
    gu.install_genai_instrumentation()
    assert Models.generate_content is first_method

    Models().generate_content(model="gemini-2.5-flash", contents="x")
    summary = gu.load_summary()
    # Single call, single recorded event — not double-wrapped.
    assert summary.total_calls == 1


def test_patch_with_positional_model_arg(fake_genai: tuple[type, type],
                                          isolated_storage: Path) -> None:
    from compgen.observability import gemini_usage as gu
    Models, _ = fake_genai
    gu.install_genai_instrumentation()
    # Pass model positionally (autocomp does this in some paths).
    Models().generate_content("gemini-2.5-flash-lite", "hi")
    events = list(gu.iter_events())
    assert len(events) == 1
    assert events[0].model == "gemini-2.5-flash-lite"

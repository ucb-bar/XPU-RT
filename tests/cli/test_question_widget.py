"""Tests for the Claude-Code-style question widget."""

from __future__ import annotations

import pytest

from xpu_rt.ui.question import QuestionAnswer, QuestionOption, ask_user_question


def test_noninteractive_fallback_picks_default(monkeypatch):
    monkeypatch.setenv("XPURT_NONINTERACTIVE", "1")
    answer = ask_user_question(
        "Pick one:",
        [
            QuestionOption(label="A", value="a"),
            QuestionOption(label="B", value="b", hint="recommended"),
        ],
        default_value="b",
    )
    assert isinstance(answer, QuestionAnswer)
    assert answer.first == "b"
    assert not answer.skipped


def test_noninteractive_fallback_first_when_no_default(monkeypatch):
    monkeypatch.setenv("XPURT_NONINTERACTIVE", "1")
    answer = ask_user_question(
        "Pick one:",
        [QuestionOption(label="A", value="a"), QuestionOption(label="B", value="b")],
    )
    assert answer.first == "a"


def test_freeform_fallback_reads_stdin(monkeypatch, capsys):
    monkeypatch.setenv("XPURT_NONINTERACTIVE", "1")
    inputs = iter(["10.44.120.201\n"])
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(inputs).strip())
    answer = ask_user_question(
        "Board host:",
        (),
        kind="freeform",
        allow_skip=True,
    )
    assert answer.freeform_text == "10.44.120.201"


def test_noninteractive_empty_options_skips(monkeypatch):
    monkeypatch.setenv("XPURT_NONINTERACTIVE", "1")
    answer = ask_user_question("title", ())
    assert answer.skipped

"""Tests for :mod:`compgen.agent.self_extension.sandbox`."""

from __future__ import annotations

import pytest

from compgen.agent.self_extension.sandbox import (
    DEFAULT_IMPORT_ALLOWLIST,
    SandboxResult,
    SandboxViolation,
    sandbox_invoke,
)


def test_sandbox_runs_trivial_function_and_returns_value() -> None:
    source = """
def run(x, y):
    return {"sum": x + y}
"""
    result = sandbox_invoke(source, "run", kwargs={"x": 3, "y": 4})
    assert result.ok
    assert result.value == {"sum": 7}
    assert result.error is None
    assert result.violations == []


def test_sandbox_reports_forbidden_import() -> None:
    source = """
import socket

def run():
    return socket.gethostname()
"""
    result = sandbox_invoke(source, "run")
    assert not result.ok
    assert result.first_violation() is not None
    assert result.first_violation().kind == "forbidden_import"
    assert "socket" in result.first_violation().detail


def test_sandbox_allows_math_and_compgen() -> None:
    source = """
import math
from compgen.llm.registry import Tool

def run():
    return {"pi": math.pi, "has_Tool": Tool.__name__}
"""
    result = sandbox_invoke(source, "run")
    assert result.ok
    assert result.value["has_Tool"] == "Tool"


def test_sandbox_surfaces_syntax_error() -> None:
    source = "def run(:\n    return 1\n"
    result = sandbox_invoke(source, "run")
    assert not result.ok
    assert any("SyntaxError" in v.detail for v in result.violations)


def test_sandbox_missing_entry_point_reported() -> None:
    source = "def something_else():\n    return 1\n"
    result = sandbox_invoke(source, "run")
    assert not result.ok
    assert "run" in (result.error or "")


def test_sandbox_catches_runtime_exception() -> None:
    source = """
def run():
    raise ValueError('boom')
"""
    result = sandbox_invoke(source, "run")
    assert not result.ok
    assert "ValueError" in (result.error or "")


def test_sandbox_timeout_best_effort() -> None:
    """A long-running authored tool must not hang the test.

    Sandbox timeout uses SIGALRM — only reliable on the main thread
    of a POSIX process. If the timeout can't be installed (e.g. on
    non-main thread) the test falls through without waiting.
    """
    source = """
def run():
    while True:
        pass
"""
    # 0.2 s budget keeps the test snappy.
    result = sandbox_invoke(source, "run", timeout_s=0.2)
    assert not result.ok
    assert any(v.kind in {"timeout", "exec_error"} for v in result.violations)


def test_sandbox_forbidden_builtins_not_exposed_by_name() -> None:
    source = """
def run():
    # eval is filtered out of __builtins__ by name
    try:
        return {"eval": eval('1+1')}
    except NameError as exc:
        return {"error": str(exc)}
"""
    result = sandbox_invoke(source, "run")
    assert result.ok
    assert "error" in result.value


def test_sandbox_explicit_empty_allowlist_blocks_all_imports() -> None:
    source = """
import math

def run():
    return math.pi
"""
    result = sandbox_invoke(source, "run", allow_imports=frozenset())
    assert not result.ok
    v = result.first_violation()
    assert v is not None and v.kind == "forbidden_import"

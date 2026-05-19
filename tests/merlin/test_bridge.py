"""Tests for MerlinBridge resolution: import → subprocess → unavailable."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from xpu_rt.targets.backends.merlin.bridge import (
    MerlinBridge,
    MerlinCallResult,
    MerlinUnavailableError,
)


def _write_fake_merlin(tmp_path: Path, *, tool: str, body: str) -> Path:
    """Stand up a fake $MERLIN_ROOT layout with one tools/<tool>.py."""
    tools = tmp_path / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "__init__.py").write_text("")
    (tools / f"{tool}.py").write_text(textwrap.dedent(body))
    return tmp_path


def test_bridge_python_api_path(tmp_path: Path) -> None:
    """Python-API path: setup_parser + main(args) → MerlinCallResult."""
    root = _write_fake_merlin(
        tmp_path,
        tool="echo_tool",
        body="""
        import argparse

        def setup_parser(p: argparse.ArgumentParser) -> None:
            p.add_argument("--message", required=True)

        def main(args: argparse.Namespace) -> int:
            print(f"ECHO: {args.message}")
            return 0
        """,
    )
    bridge = MerlinBridge(merlin_root=root)
    result = bridge.call("echo_tool", cli_args=["--message", "hi"])
    assert isinstance(result, MerlinCallResult)
    assert result.via == "python-api"
    assert result.returncode == 0
    assert "ECHO: hi" in result.stdout
    # Clean up sys.path so subsequent tests are independent.
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.modules.pop("tools", None)
    sys.modules.pop("tools.echo_tool", None)


def test_bridge_python_api_nonzero_exit(tmp_path: Path) -> None:
    """Non-zero return propagates; stderr is captured."""
    root = _write_fake_merlin(
        tmp_path,
        tool="fail_tool",
        body="""
        import argparse, sys

        def setup_parser(p: argparse.ArgumentParser) -> None:
            pass

        def main(args: argparse.Namespace) -> int:
            print("oh no", file=sys.stderr)
            return 7
        """,
    )
    bridge = MerlinBridge(merlin_root=root)
    result = bridge.call("fail_tool")
    assert result.returncode == 7
    assert "oh no" in result.stderr
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.modules.pop("tools", None)
    sys.modules.pop("tools.fail_tool", None)


def test_bridge_subprocess_fallback(tmp_path: Path) -> None:
    """prefer_python_api=False forces the subprocess path."""
    root = _write_fake_merlin(
        tmp_path,
        tool="sub_tool",
        body="""
        import argparse

        def setup_parser(p: argparse.ArgumentParser) -> None:
            p.add_argument("--n", type=int, default=1)

        def main(args: argparse.Namespace) -> int:
            print("SUB:", args.n * 2)
            return 0

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            setup_parser(parser)
            raise SystemExit(main(parser.parse_args()))
        """,
    )
    bridge = MerlinBridge(merlin_root=root, prefer_python_api=False)
    result = bridge.call("sub_tool", cli_args=["--n", "21"])
    assert result.via == "subprocess"
    assert result.returncode == 0
    assert "SUB: 42" in result.stdout


def test_bridge_unavailable_when_root_missing(tmp_path: Path) -> None:
    """No merlin_root on disk + no python-api import → typed error."""
    bridge = MerlinBridge(
        merlin_root=tmp_path / "does-not-exist",
        prefer_python_api=False,
    )
    with pytest.raises(MerlinUnavailableError) as exc_info:
        bridge.call("anything")
    assert exc_info.value.tool == "anything"
    assert exc_info.value.merlin_root == tmp_path / "does-not-exist"


def test_bridge_available_probe(tmp_path: Path) -> None:
    assert MerlinBridge(merlin_root=tmp_path).available() is False
    (tmp_path / "tools").mkdir()
    assert MerlinBridge(merlin_root=tmp_path).available() is True

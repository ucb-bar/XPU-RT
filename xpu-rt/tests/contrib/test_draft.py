"""Tests for :mod:`compgen.contrib.draft`.

We use a real git worktree-like layout under tmp_path so the
branch / commit steps are exercised without touching the actual
repo. ``draft_pr(commit=False, create_branch=False)`` covers the
file-copy + pytest-synthesis path on its own for cheaper assertions.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from compgen.contrib import draft_pr, list_extensions, status
from compgen.contrib.draft import _synthesize_test

_TOOL_EXT = """
from compgen.llm.registry import Tool, ToolArg, ToolResult

TOOL = Tool(
    name="my_user_tool",
    phase=3,
    kind="tool",
    wraps_pass="user_supplied",
    autocomp_cost_impact="low",
    args=(ToolArg(name="x", dtype="str", description="x"),),
    result=ToolResult(dtype="dict", description="res"),
    description="A user-authored demo tool",
    impl=lambda **kw: {"status": "ok", "got": kw},
    stub=False,
)
"""


def _bootstrap_ext_root(tmp_path: Path, invocation_counts: int = 5) -> Path:
    """Populate a fake ~/.compgen/extensions with one tool + a state file."""
    root = tmp_path / "ext"
    root.mkdir()
    (root / "my_user_tool.py").write_text(_TOOL_EXT)
    state_path = root / "_state.json"
    # NB: no ``loaded_modules`` — every test's scratch Registry needs to
    # actually load the tool so list_extensions sees it.
    state_path.write_text(
        json.dumps(
            {
                "accepted_invocations": {
                    "my_user_tool": [{"session_id": f"s_{i}", "step_index": i} for i in range(invocation_counts)],
                },
            },
            indent=2,
        )
    )
    return root


def _bootstrap_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "python" / "compgen").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0'\n")
    # Initialise a fresh git repo so the draft commit workflow can run.
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@x", "-c", "user.name=t", "commit", "--allow-empty", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


# ---------------------------------------------------------------------------
# Catalogue helpers
# ---------------------------------------------------------------------------


def test_list_extensions_returns_counts(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path, invocation_counts=4)
    exts = list_extensions(root=root)
    names = {e.name for e in exts}
    assert "my_user_tool" in names
    row = next(e for e in exts if e.name == "my_user_tool")
    assert row.accepted_invocations == 4
    assert row.eligible  # 4 >= MIN_ACCEPTED_INVOCATIONS=3


def test_list_extensions_marks_ineligible_when_below_threshold(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path, invocation_counts=1)
    [row] = [e for e in list_extensions(root=root) if e.name == "my_user_tool"]
    assert not row.eligible
    assert ">=" in row.eligibility_reason


def test_status_emits_summary(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path, invocation_counts=5)
    info = status(root=root)
    assert info["total"] >= 1
    names = [e["name"] for e in info["extensions"]]
    assert "my_user_tool" in names


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def test_draft_pr_without_tests_or_commit_copies_files(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path, invocation_counts=3)
    repo = _bootstrap_repo(tmp_path)

    result = draft_pr(
        "my_user_tool",
        source_root=root,
        repo_root=repo,
        run_tests=False,
        commit=False,
        create_branch=False,
    )
    assert not result.errors, result.errors
    assert result.upstream_module is not None
    assert result.upstream_module.exists()
    assert result.upstream_test is not None
    assert result.upstream_test.exists()

    # The upstream test imports the upstream module.
    body = result.upstream_test.read_text()
    assert "compgen.agent.invent_slots.contrib.my_user_tool" in body

    # Copied verbatim — the original `name="my_user_tool"` survives.
    assert "my_user_tool" in result.upstream_module.read_text()

    # gh command is always surfaced but never invoked.
    assert "gh pr create" in result.gh_command


def test_draft_pr_unknown_slot_reports_error(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path)
    repo = _bootstrap_repo(tmp_path)
    result = draft_pr(
        "no_such_slot_anywhere",
        source_root=root,
        repo_root=repo,
        run_tests=False,
        commit=False,
        create_branch=False,
    )
    assert result.errors
    assert any("no extension file found" in e for e in result.errors)


def test_draft_pr_with_branch_and_commit(tmp_path: Path) -> None:
    root = _bootstrap_ext_root(tmp_path, invocation_counts=3)
    repo = _bootstrap_repo(tmp_path)
    subprocess.run(
        ["git", "-c", "user.email=t@x", "-c", "user.name=t", "config", "user.email", "t@x"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@x", "-c", "user.name=t", "config", "user.name", "tester"],
        cwd=repo,
        check=True,
    )

    result = draft_pr(
        "my_user_tool",
        source_root=root,
        repo_root=repo,
        run_tests=False,
        commit=True,
        create_branch=True,
    )
    assert result.branch == "contrib/my-user-tool"
    assert result.committed
    # Branch should exist.
    branches = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "contrib/my-user-tool" in branches


def test_synthesize_test_embeds_slot_name_and_accepted_invocations() -> None:
    body = _synthesize_test("widget_fusion", [{"k": "v", "i": 1}])
    assert "widget_fusion" in body
    assert "compgen.agent.invent_slots.contrib.widget_fusion" in body
    assert "Accepted-invocation log" in body

"""Draft an upstream contribution from a local user extension.

The core operation is :func:`draft_pr`. Everything else is read-only
introspection that the CLI and tests use to surface "what local
extensions do I have" and "which of them are eligible to land?".

Upstream layout assumed here::

    python/compgen/agent/invent_slots/contrib/<slot>.py
    tests/agent/invent_slots/contrib/test_<slot>.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.agent.extensions.local_loader import (
    DEFAULT_ROOT as EXT_ROOT,
    STATE_FILENAME,
    _load_state,
    _state_path,
    load_local_extensions,
)
from compgen.llm.registry import Registry

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContribExtension:
    """One local extension and its upstream-readiness summary."""

    name: str                            # slot or tool name
    kind: str                            # "tool" | "slot"
    source_path: Path
    accepted_invocations: int
    eligible: bool
    eligibility_reason: str = ""


@dataclass
class ContribDraftResult:
    """Outcome of one :func:`draft_pr` call."""

    slot_name: str
    branch: str
    upstream_module: Path | None = None
    upstream_test: Path | None = None
    pytest_passed: bool = False
    committed: bool = False
    gh_command: str = ""
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Catalogue helpers
# ---------------------------------------------------------------------------


def _ext_root(override: Path | None) -> Path:
    return Path(override).expanduser() if override is not None else EXT_ROOT


def _build_registry(root: Path) -> tuple[Registry, dict[str, Any]]:
    """Load every user extension into a scratch registry + return state."""
    reg = Registry()
    load_local_extensions(reg, root=root)
    state = _load_state(root)
    return reg, state


def list_extensions(root: Path | None = None) -> list[ContribExtension]:
    """Return one :class:`ContribExtension` per user-authored tool/slot.

    Scans ``root`` (defaults to ``~/.compgen/extensions``), builds a
    scratch registry, and joins it with the accepted-invocations log
    from ``_state.json``. An extension file that no longer loads (e.g.
    import errors) is skipped silently — list only surfaces healthy
    entries.
    """
    r = _ext_root(root)
    reg, state = _build_registry(r)
    accepted = state.get("accepted_invocations", {}) or {}

    out: list[ContribExtension] = []
    # We need a mapping name -> source path. Walk the local root once.
    py_files = {p.stem: p for p in r.glob("*.py") if p.is_file()}

    for tool in reg.list_tools():
        src = py_files.get(tool.name, Path(""))
        count = len(accepted.get(tool.name, []))
        eligible, reason = _eligibility(tool_name=tool.name, count=count)
        out.append(ContribExtension(
            name=tool.name, kind="tool",
            source_path=src, accepted_invocations=count,
            eligible=eligible, eligibility_reason=reason,
        ))
    for slot in reg.list_invent_slots():
        src = py_files.get(slot.name, Path(""))
        count = len(accepted.get(slot.name, []))
        eligible, reason = _eligibility(tool_name=slot.name, count=count)
        out.append(ContribExtension(
            name=slot.name, kind="slot",
            source_path=src, accepted_invocations=count,
            eligible=eligible, eligibility_reason=reason,
        ))
    out.sort(key=lambda e: e.name)
    return out


_MIN_ACCEPTED_INVOCATIONS = 3


def _eligibility(*, tool_name: str, count: int) -> tuple[bool, str]:
    if count < _MIN_ACCEPTED_INVOCATIONS:
        return False, (
            f"need >={_MIN_ACCEPTED_INVOCATIONS} accepted invocations "
            f"(got {count})"
        )
    return True, "ready"


def status(root: Path | None = None) -> dict[str, Any]:
    """Roll-up for ``compgen contrib status``."""
    exts = list_extensions(root=root)
    eligible = [e for e in exts if e.eligible]
    return {
        "root": str(_ext_root(root)),
        "total": len(exts),
        "eligible": len(eligible),
        "extensions": [
            {
                "name": e.name, "kind": e.kind,
                "accepted_invocations": e.accepted_invocations,
                "eligible": e.eligible,
                "reason": e.eligibility_reason,
                "source": str(e.source_path),
            }
            for e in exts
        ],
    }


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------


def _find_source_file(root: Path, name: str) -> Path | None:
    exact = root / f"{name}.py"
    if exact.exists():
        return exact
    for p in root.glob("*.py"):
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            # Heuristic: file defines a TOOL/SLOT with the given name.
            if f'name="{name}"' in text or f"name='{name}'" in text:
                return p
    return None


def _slug(name: str) -> str:
    return name.replace("_", "-").lower()


def _repo_root() -> Path:
    """Best-effort repo root — falls back to CWD."""
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (p / "pyproject.toml").exists():
            return p
    return cwd


def _synthesize_test(
    slot_name: str, invocations: list[dict[str, Any]],
) -> str:
    """Generate a smoke test that imports the contrib module."""
    header = f'''"""Auto-generated contrib regression test for ``{slot_name}``.

Generated by ``compgen.contrib.draft``. Locks the copied invent-slot
against accidental breakage; customise freely once the real upstream
semantics land.
"""

from __future__ import annotations


def test_{slot_name}_module_imports() -> None:
    import importlib
    mod = importlib.import_module(
        "compgen.agent.invent_slots.contrib.{slot_name}"
    )
    assert mod is not None


def test_{slot_name}_registration_roundtrips() -> None:
    from compgen.llm.registry import Registry
    from compgen.agent.extensions.local_loader import _import_file
    from pathlib import Path
    import compgen.agent.invent_slots.contrib.{slot_name} as mod
    src = Path(mod.__file__)
    reg = Registry()
    module = _import_file(src)
    # Collect declared tools / slots via the loader's helper.
    from compgen.agent.extensions.local_loader import _register_from_module
    tools, slots = _register_from_module(module, reg)
    assert tools or slots, "module declared neither TOOL nor SLOT"
'''
    if invocations:
        header += "\n\n# Accepted-invocation log (truncated): " + \
            json.dumps(invocations[:5], indent=2, default=str) + "\n"
    return header


def _git(
    args: list[str], *, cwd: Path,
) -> tuple[int, str, str]:
    """Run a git command and return (rc, stdout, stderr) — never raises."""
    try:
        p = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "git not installed"


def _pytest(test_file: Path, *, cwd: Path) -> tuple[bool, str]:
    try:
        p = subprocess.run(
            ["uv", "run", "pytest", str(test_file), "-q", "--no-header"],
            cwd=cwd, capture_output=True, text=True, check=False, timeout=120,
        )
    except Exception as exc:   # noqa: BLE001
        return False, f"pytest invocation failed: {exc}"
    return p.returncode == 0, p.stdout + p.stderr


def draft_pr(
    slot_name: str,
    *,
    source_root: Path | None = None,
    repo_root: Path | None = None,
    run_tests: bool = True,
    commit: bool = True,
    create_branch: bool = True,
) -> ContribDraftResult:
    """Draft an upstream contribution for ``slot_name``.

    Reads the local extension file matching ``slot_name``, copies it
    into ``repo_root/python/compgen/agent/invent_slots/contrib/<slot>.py``,
    synthesises a regression test, optionally runs pytest, and commits
    everything on a new ``contrib/<slug>`` branch.

    The function never pushes to a remote. It prints the ``gh pr create``
    command so the human reviewer can invoke it manually.
    """
    result = ContribDraftResult(slot_name=slot_name, branch=f"contrib/{_slug(slot_name)}")
    src_root = _ext_root(source_root)
    repo = Path(repo_root).expanduser() if repo_root is not None else _repo_root()

    source_file = _find_source_file(src_root, slot_name)
    if source_file is None:
        result.errors.append(
            f"no extension file found for {slot_name!r} under {src_root}"
        )
        return result

    # Target paths inside the repo.
    upstream_dir = repo / "python" / "compgen" / "agent" / "invent_slots" / "contrib"
    upstream_dir.mkdir(parents=True, exist_ok=True)
    pkg_init = upstream_dir / "__init__.py"
    if not pkg_init.exists():
        pkg_init.write_text(
            '"""User-contributed invent slots (auto-generated from local extensions)."""\n'
        )
    upstream_module = upstream_dir / f"{slot_name}.py"
    result.upstream_module = upstream_module

    test_dir = repo / "tests" / "agent" / "invent_slots" / "contrib"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_init = test_dir / "__init__.py"
    if not test_init.exists():
        test_init.write_text("")
    upstream_test = test_dir / f"test_{slot_name}.py"
    result.upstream_test = upstream_test

    # Copy + test synthesis.
    shutil.copyfile(source_file, upstream_module)
    accepted = (_load_state(src_root).get("accepted_invocations", {}) or {}).get(
        slot_name, []
    )
    upstream_test.write_text(_synthesize_test(slot_name, accepted))

    # Optional pytest.
    if run_tests:
        ok, output = _pytest(upstream_test, cwd=repo)
        result.pytest_passed = ok
        if not ok:
            result.errors.append(f"pytest failed:\n{output[-2000:]}")
    else:
        result.pytest_passed = True

    # Branch + commit.
    if create_branch:
        rc, out, err = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
        if rc != 0:
            result.errors.append(f"git rev-parse failed: {err}")
        else:
            current = out.strip()
            if current != result.branch:
                rc2, _, err2 = _git(
                    ["checkout", "-B", result.branch], cwd=repo,
                )
                if rc2 != 0:
                    result.errors.append(f"git checkout -B failed: {err2}")

    if commit and not result.errors:
        _git(["add", str(upstream_module), str(upstream_test)], cwd=repo)
        msg = (
            f"feat(contrib): graduate local extension {slot_name}\n\n"
            f"Auto-generated by compgen contrib draft.\n"
            f"Source: {source_file}\n"
            f"Accepted invocations recorded: {len(accepted)}"
        )
        rc, _, err = _git(["commit", "-m", msg], cwd=repo)
        if rc == 0:
            result.committed = True
        else:
            result.errors.append(f"git commit failed: {err}")

    # gh pr create reminder (always printed as a string, never executed).
    result.gh_command = (
        f"gh pr create --title 'feat(contrib): graduate {slot_name}' "
        f"--body 'Auto-drafted by compgen contrib draft; see commit for details.'"
    )

    return result


__all__ = [
    "ContribDraftResult",
    "ContribExtension",
    "draft_pr",
    "list_extensions",
    "status",
]

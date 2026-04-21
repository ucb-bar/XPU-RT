"""Install (merge) the CompGen MCP server into a client config file.

Targets Claude Code's ``~/.claude.json`` by default, and project-level
``.mcp.json`` when ``--project`` is used. The merge is conservative:

* The target file is read as JSON. Missing file -> treated as empty.
* A ``.bak-<timestamp>`` copy of the pre-existing file is written
  alongside before any mutation.
* The ``mcpServers.compgen`` key is set to
  :func:`compgen.mcp.config.mcp_server_entry`. If the key already exists
  and differs, the install raises unless ``force=True``.

The function returns a :class:`InstallResult` summarising what happened
so the CLI can echo it back to the user.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from compgen.mcp.config import SERVER_NAME, mcp_server_entry


@dataclass
class InstallResult:
    target: Path
    backup: Path | None
    action: str  # "created" | "added" | "already-present" | "updated"
    entry: dict


def default_claude_config() -> Path:
    """Path Claude Code reads for user-level MCP servers."""
    return Path.home() / ".claude.json"


def default_project_config(cwd: Path | None = None) -> Path:
    """Path Claude Code reads for project-level MCP servers."""
    return (cwd or Path.cwd()) / ".mcp.json"


def _read(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must be a JSON object at the top level")
    return data


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bak = path.with_suffix(path.suffix + f".bak-{stamp}")
    shutil.copy2(path, bak)
    return bak


def install_mcp_server(
    *,
    target: Path | None = None,
    project: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> InstallResult:
    """Merge the CompGen MCP server into a client config file.

    Args:
        target: Explicit destination. If ``None``, uses
            :func:`default_project_config` when ``project`` is True,
            else :func:`default_claude_config`.
        project: Write to the current working directory's ``.mcp.json``.
        force: Overwrite an existing ``mcpServers.compgen`` entry that
            differs from the canonical one.
        dry_run: Compute the action + backup path but do not write.

    Returns:
        An :class:`InstallResult` describing the outcome. When
        ``dry_run=True`` the result's ``backup`` is always ``None``.
    """
    dest = target or (default_project_config() if project else default_claude_config())
    existing = _read(dest)
    servers = existing.setdefault("mcpServers", {}) if not dry_run else {
        **existing.get("mcpServers", {})
    }
    entry = mcp_server_entry()

    if SERVER_NAME in servers:
        if servers[SERVER_NAME] == entry:
            return InstallResult(
                target=dest, backup=None, action="already-present", entry=entry
            )
        if not force:
            raise RuntimeError(
                f"{dest} already has an mcpServers.{SERVER_NAME} entry "
                f"({servers[SERVER_NAME]!r}) that differs from the canonical one. "
                "Pass --force to overwrite."
            )
        action = "updated"
    else:
        action = "created" if not dest.exists() else "added"

    if dry_run:
        return InstallResult(target=dest, backup=None, action=action, entry=entry)

    backup = _backup(dest)
    existing["mcpServers"][SERVER_NAME] = entry
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(existing, indent=2) + "\n")
    return InstallResult(target=dest, backup=backup, action=action, entry=entry)


__all__ = [
    "InstallResult",
    "default_claude_config",
    "default_project_config",
    "install_mcp_server",
]

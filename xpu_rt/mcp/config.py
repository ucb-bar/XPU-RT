"""Canonical MCP server config snippet for Claude Code + compatible clients.

Having one source of truth means ``xpu_rt mcp print-config``,
``xpu_rt mcp install``, and the module docstring in
:mod:`xpu_rt.mcp.server` cannot drift apart.
"""

from __future__ import annotations

import json
from typing import Any

SERVER_NAME = "xpu-rt"
SERVER_COMMAND = "xpu-rt-mcp"


def mcp_server_entry() -> dict[str, Any]:
    """Return the dict that goes under ``mcpServers.xpu_rt`` in a client config."""
    return {"command": SERVER_COMMAND}


def mcp_server_json(*, indent: int = 2) -> str:
    """Render the full ``{"mcpServers": {"xpu-rt": {...}}}`` snippet."""
    return json.dumps({"mcpServers": {SERVER_NAME: mcp_server_entry()}}, indent=indent)


__all__ = ["SERVER_COMMAND", "SERVER_NAME", "mcp_server_entry", "mcp_server_json"]

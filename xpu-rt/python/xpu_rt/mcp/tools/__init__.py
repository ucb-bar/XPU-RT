"""MCP tool handlers for CompGen.

Each submodule defines one or more pure-Python callables that accept
a ``SessionManager`` + keyword args and return a JSON-serialisable
dict. The callables are re-exported here as a flat namespace so
``server.py`` can iterate them when it wires the MCP SDK decorators.

Exported tool dicts take the shape::

    {
      "name": "open_target",
      "description": "...",
      "input_schema": {...},     # JSON schema for MCP tool discovery
      "handler": callable,        # def (sm: SessionManager, **kwargs) -> dict
      "phase": "lifecycle",       # lifecycle | inspect | transform | job
    }
"""

from __future__ import annotations

from compgen.mcp.tools.diagnose import DIAGNOSE_TOOLS
from compgen.mcp.tools.inspect import INSPECT_TOOLS
from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS
from compgen.mcp.tools.recipe_apply import APPLY_RECIPE_TOOLS
from compgen.mcp.tools.recovery import RECOVERY_TOOLS
from compgen.mcp.tools.transform import TRANSFORM_TOOLS

ALL_TOOLS: list[dict] = [
    *LIFECYCLE_TOOLS,
    *INSPECT_TOOLS,
    *DIAGNOSE_TOOLS,
    *TRANSFORM_TOOLS,
    *RECOVERY_TOOLS,
    *APPLY_RECIPE_TOOLS,
]

__all__ = [
    "ALL_TOOLS",
    "APPLY_RECIPE_TOOLS",
    "DIAGNOSE_TOOLS",
    "INSPECT_TOOLS",
    "LIFECYCLE_TOOLS",
    "RECOVERY_TOOLS",
    "TRANSFORM_TOOLS",
]

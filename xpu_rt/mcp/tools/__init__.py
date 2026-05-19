"""MCP tool handlers for XPU-RT.

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

from typing import Any

from xpu_rt.mcp.tools.agent_decision import AGENT_DECISION_TOOLS
from xpu_rt.mcp.tools.autotune import AUTOTUNE_TOOLS
from xpu_rt.mcp.tools.batch import BATCH_TOOLS
from xpu_rt.mcp.tools.bench import BENCH_TOOLS
from xpu_rt.mcp.tools.compile import COMPILE_TOOLS
from xpu_rt.mcp.tools.conformance import CONFORMANCE_TOOLS
from xpu_rt.mcp.tools.decisions import DECISION_TOOLS
from xpu_rt.mcp.tools.diagnose import DIAGNOSE_TOOLS
from xpu_rt.mcp.tools.dispatch import DISPATCH_TOOLS
from xpu_rt.mcp.tools.embedded import EMBEDDED_TOOLS
from xpu_rt.mcp.tools.explain import EXPLAIN_TOOLS
from xpu_rt.mcp.tools.graduate import GRADUATE_TOOLS
from xpu_rt.mcp.tools.graph_digest import GRAPH_DIGEST_TOOLS
from xpu_rt.mcp.tools.inspect import INSPECT_TOOLS
from xpu_rt.mcp.tools.kernel import KERNEL_TOOLS
from xpu_rt.mcp.tools.knowledge import KNOWLEDGE_TOOLS
from xpu_rt.mcp.tools.lifecycle import LIFECYCLE_TOOLS
from xpu_rt.mcp.tools.recipe_apply import APPLY_RECIPE_TOOLS
from xpu_rt.mcp.tools.recovery import RECOVERY_TOOLS
from xpu_rt.mcp.tools.refinement import REFINEMENT_TOOLS
from xpu_rt.mcp.tools.suggest import SUGGEST_TOOLS
from xpu_rt.mcp.tools.targets import TARGET_TOOLS
from xpu_rt.mcp.tools.transform import TRANSFORM_TOOLS
from xpu_rt.mcp.tools.vendor_dialect import VENDOR_DIALECT_TOOLS


def _optimize_tools() -> list[dict]:
    """Imported lazily to avoid an import cycle with xpu_rt.agent."""
    from xpu_rt.agent.mcp_optimizer import OPTIMIZE_TOOLS

    return OPTIMIZE_TOOLS


def _pack_mcp_tools() -> list[dict]:
    """Discover + load tools from the ``xpu_rt.mcp.tools`` entry-point group.

    Each entry resolves to a single tool dict or an iterable of tool
    dicts. Validation is performed inside the plugins registry — entries
    that fail are logged and skipped (never raise) so a single broken
    pack doesn't prevent XPU-RT from starting.

    Returns the flat list to append to ``ALL_TOOLS``.
    """
    try:
        from xpu_rt.plugins import GROUP_MCP_TOOLS, discover_all, registry
    except Exception:  # noqa: BLE001
        return []

    discover_all()
    out: list[dict] = []
    for plugin in registry().get(GROUP_MCP_TOOLS):
        obj = plugin.object
        items = obj if isinstance(obj, (list, tuple)) else [obj]
        for item in items:
            # Annotate provenance so `xpu_rt mcp tools` can render
            # `[pack: <dist>]` next to pack-owned entries.
            t = dict(item)
            t.setdefault("_pack", plugin.distribution or plugin.name)
            out.append(t)
    return out


def _bridge_tools() -> list[dict]:
    """ToolCard-bridged MCP tools.

    Loaded lazily so a malformed ToolCard YAML cannot prevent the rest
    of the MCP server from starting; the bridge's own logging plus the
    audit catch the malformed card on the next CI run.
    """

    try:
        from xpu_rt.mcp.tool_bridge import bridge_tools

        return bridge_tools()
    except Exception:  # noqa: BLE001
        return []


_IN_TREE_TOOLS: list[dict] = [
    *LIFECYCLE_TOOLS,
    *INSPECT_TOOLS,
    *DIAGNOSE_TOOLS,
    *TRANSFORM_TOOLS,
    *RECOVERY_TOOLS,
    *APPLY_RECIPE_TOOLS,
    *EXPLAIN_TOOLS,
    *GRADUATE_TOOLS,
    *BATCH_TOOLS,
    *SUGGEST_TOOLS,
    *VENDOR_DIALECT_TOOLS,
    *KERNEL_TOOLS,
    *DISPATCH_TOOLS,
    *BENCH_TOOLS,
    *KNOWLEDGE_TOOLS,
    *GRAPH_DIGEST_TOOLS,
    *DECISION_TOOLS,
    *AGENT_DECISION_TOOLS,
    *REFINEMENT_TOOLS,
    *AUTOTUNE_TOOLS,
    *EMBEDDED_TOOLS,
    *CONFORMANCE_TOOLS,
    *COMPILE_TOOLS,
    *TARGET_TOOLS,
    *_optimize_tools(),
    *_bridge_tools(),
]

# Cached merged list (in-tree + entry-point-discovered pack tools).
# ``None`` until the first read of ``ALL_TOOLS`` — see ``__getattr__``.
_ALL_TOOLS_CACHE: list[dict] | None = None


def get_all_tools() -> list[dict]:
    """Return the merged in-tree + pack-discovered tools list.

    Entry-point discovery is deferred until the first call so importing
    ``xpu_rt.mcp.tools.embedded`` (or any other submodule) doesn't
    trigger pack-side resolution at package-init time. That deferral
    avoids circular-import traps when a pack imports from
    ``xpu_rt.mcp.tools.embedded`` at its own module-load time.
    """
    global _ALL_TOOLS_CACHE
    if _ALL_TOOLS_CACHE is None:
        _ALL_TOOLS_CACHE = [*_IN_TREE_TOOLS, *_pack_mcp_tools()]
    return _ALL_TOOLS_CACHE


def __getattr__(name: str) -> Any:
    if name == "ALL_TOOLS":
        return get_all_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALL_TOOLS",
    "get_all_tools",
    "APPLY_RECIPE_TOOLS",
    "AUTOTUNE_TOOLS",
    "BATCH_TOOLS",
    "BENCH_TOOLS",
    "COMPILE_TOOLS",
    "CONFORMANCE_TOOLS",
    "DECISION_TOOLS",
    "DIAGNOSE_TOOLS",
    "DISPATCH_TOOLS",
    "EXPLAIN_TOOLS",
    "GRADUATE_TOOLS",
    "GRAPH_DIGEST_TOOLS",
    "INSPECT_TOOLS",
    "KERNEL_TOOLS",
    "KNOWLEDGE_TOOLS",
    "LIFECYCLE_TOOLS",
    "EMBEDDED_TOOLS",
    "RECOVERY_TOOLS",
    "REFINEMENT_TOOLS",
    "SUGGEST_TOOLS",
    "TRANSFORM_TOOLS",
    "VENDOR_DIALECT_TOOLS",
]

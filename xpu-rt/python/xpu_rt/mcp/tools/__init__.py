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

from typing import Any

from compgen.mcp.tools.autotune import AUTOTUNE_TOOLS
from compgen.mcp.tools.batch import BATCH_TOOLS
from compgen.mcp.tools.bench import BENCH_TOOLS
from compgen.mcp.tools.compile import COMPILE_TOOLS
from compgen.mcp.tools.conformance import CONFORMANCE_TOOLS
from compgen.mcp.tools.decisions import DECISION_TOOLS
from compgen.mcp.tools.diagnose import DIAGNOSE_TOOLS
from compgen.mcp.tools.dispatch import DISPATCH_TOOLS
from compgen.mcp.tools.embedded import EMBEDDED_TOOLS
from compgen.mcp.tools.explain import EXPLAIN_TOOLS
from compgen.mcp.tools.graduate import GRADUATE_TOOLS
from compgen.mcp.tools.graph_digest import GRAPH_DIGEST_TOOLS
from compgen.mcp.tools.inspect import INSPECT_TOOLS
from compgen.mcp.tools.kernel import KERNEL_TOOLS
from compgen.mcp.tools.knowledge import KNOWLEDGE_TOOLS
from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS
from compgen.mcp.tools.recipe_apply import APPLY_RECIPE_TOOLS
from compgen.mcp.tools.recovery import RECOVERY_TOOLS
from compgen.mcp.tools.refinement import REFINEMENT_TOOLS
from compgen.mcp.tools.suggest import SUGGEST_TOOLS
from compgen.mcp.tools.targets import TARGET_TOOLS
from compgen.mcp.tools.transform import TRANSFORM_TOOLS
from compgen.mcp.tools.vendor_dialect import VENDOR_DIALECT_TOOLS


def _optimize_tools() -> list[dict]:
    """Imported lazily to avoid an import cycle with compgen.agent."""
    from compgen.agent.mcp_optimizer import OPTIMIZE_TOOLS

    return OPTIMIZE_TOOLS


def _pack_mcp_tools() -> list[dict]:
    """Discover + load tools from the ``compgen.mcp.tools`` entry-point group.

    Each entry resolves to a single tool dict or an iterable of tool
    dicts. Validation is performed inside the plugins registry — entries
    that fail are logged and skipped (never raise) so a single broken
    pack doesn't prevent CompGen from starting.

    Returns the flat list to append to ``ALL_TOOLS``.
    """
    try:
        from compgen.plugins import GROUP_MCP_TOOLS, discover_all, registry
    except Exception:  # noqa: BLE001
        return []

    discover_all()
    out: list[dict] = []
    for plugin in registry().get(GROUP_MCP_TOOLS):
        obj = plugin.object
        items = obj if isinstance(obj, (list, tuple)) else [obj]
        for item in items:
            # Annotate provenance so `compgen mcp tools` can render
            # `[pack: <dist>]` next to pack-owned entries.
            t = dict(item)
            t.setdefault("_pack", plugin.distribution or plugin.name)
            out.append(t)
    return out


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
    *REFINEMENT_TOOLS,
    *AUTOTUNE_TOOLS,
    *EMBEDDED_TOOLS,
    *CONFORMANCE_TOOLS,
    *COMPILE_TOOLS,
    *TARGET_TOOLS,
    *_optimize_tools(),
]

# Cached merged list (in-tree + entry-point-discovered pack tools).
# ``None`` until the first read of ``ALL_TOOLS`` — see ``__getattr__``.
_ALL_TOOLS_CACHE: list[dict] | None = None


def get_all_tools() -> list[dict]:
    """Return the merged in-tree + pack-discovered tools list.

    Entry-point discovery is deferred until the first call so importing
    ``compgen.mcp.tools.embedded`` (or any other submodule) doesn't
    trigger pack-side resolution at package-init time. That deferral
    avoids circular-import traps when a pack imports from
    ``compgen.mcp.tools.embedded`` at its own module-load time.
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

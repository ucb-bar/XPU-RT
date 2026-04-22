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

from compgen.mcp.tools.autotune import AUTOTUNE_TOOLS
from compgen.mcp.tools.batch import BATCH_TOOLS
from compgen.mcp.tools.bench import BENCH_TOOLS
from compgen.mcp.tools.decisions import DECISION_TOOLS
from compgen.mcp.tools.diagnose import DIAGNOSE_TOOLS
from compgen.mcp.tools.dispatch import DISPATCH_TOOLS
from compgen.mcp.tools.explain import EXPLAIN_TOOLS
from compgen.mcp.tools.graduate import GRADUATE_TOOLS
from compgen.mcp.tools.graph_digest import GRAPH_DIGEST_TOOLS
from compgen.mcp.tools.inspect import INSPECT_TOOLS
from compgen.mcp.tools.kernel import KERNEL_TOOLS
from compgen.mcp.tools.knowledge import KNOWLEDGE_TOOLS
from compgen.mcp.tools.lifecycle import LIFECYCLE_TOOLS
from compgen.mcp.tools.embedded import EMBEDDED_TOOLS
from compgen.mcp.tools.recipe_apply import APPLY_RECIPE_TOOLS
from compgen.mcp.tools.recovery import RECOVERY_TOOLS
from compgen.mcp.tools.refinement import REFINEMENT_TOOLS
from compgen.mcp.tools.suggest import SUGGEST_TOOLS
from compgen.mcp.tools.transform import TRANSFORM_TOOLS
from compgen.mcp.tools.vendor_dialect import VENDOR_DIALECT_TOOLS


def _optimize_tools() -> list[dict]:
    """Imported lazily to avoid an import cycle with compgen.agent."""
    from compgen.agent.mcp_optimizer import OPTIMIZE_TOOLS
    return OPTIMIZE_TOOLS


ALL_TOOLS: list[dict] = [
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
    *_optimize_tools(),
]

__all__ = [
    "ALL_TOOLS",
    "APPLY_RECIPE_TOOLS",
    "AUTOTUNE_TOOLS",
    "BATCH_TOOLS",
    "BENCH_TOOLS",
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

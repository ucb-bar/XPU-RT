"""MCP (Model Context Protocol) server surface for CompGen.

Exposes CompGen's LLM-driven compilation backbone as a stdio MCP
server named ``compgen-mcp`` that Claude Code (or any MCP client)
can plug into. The server is a thin adaptor over
:mod:`compgen.agent.llm_driver` — no logic lives here that doesn't
already exist in the Python surface.

Package layout:

- ``session`` — per-client session manager + :class:`LLMDrivenCompiler`
  lifetimes.
- ``async_jobs`` — job-id pattern for long-running Triton JIT calls
  that would otherwise stall the stdio pipe.
- ``tools.lifecycle`` — open_target, load_model, compile, bundle_export.
- ``tools.inspect`` — view_recipe, diff_recipe, list_phase_tools,
  get_dossier, session_summary.
- ``tools.transform`` — invoke_tool, propose_invent_slot,
  verify_proposal, step_proposal.
- ``server`` — MCP stdio entry (``compgen-mcp`` script).

Tool handlers in ``tools/`` are pure functions so the test suite can
drive them directly without spawning a subprocess. The MCP SDK
dependency is optional: if ``mcp`` is not installed, the Python
surface still works; only the ``compgen-mcp`` CLI requires it.
"""

from __future__ import annotations

from compgen.mcp.session import McpSession, SessionManager

__all__ = ["McpSession", "SessionManager"]

"""MCP stdio server entry point for CompGen.

Adapts :data:`compgen.mcp.tools.ALL_TOOLS` into MCP SDK tool handlers.
The server is only instantiated when ``compgen-mcp`` is invoked from
the shell; importing :mod:`compgen.mcp` does not require the MCP SDK.

Usage::

    pip install compgen
    compgen-mcp                 # or: compgen mcp serve

Claude Code configuration (``.mcp.json``)::

    {
      "mcpServers": {
        "compgen": {
          "command": "compgen-mcp"
        }
      }
    }

The canonical config snippet lives in :mod:`compgen.mcp.config`;
``compgen mcp print-config`` and ``compgen mcp install`` both read it
from there so docstring + CLI can't drift.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import structlog

from compgen.mcp.session import SessionManager
from compgen.mcp.tools import ALL_TOOLS
from compgen.mcp.transcript import McpTranscriptRecorder


def _route_logs_to_stderr() -> None:
    """Force every logger to stderr so nothing pollutes the stdio JSON-RPC stream.

    The MCP transport expects newline-delimited JSON on stdout; a single
    log line on stdout breaks every downstream parser. Called at server
    startup before any tool dispatch.
    """

    # Route the stdlib root logger to stderr.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stderr_handler)

    # Pin structlog to the same stderr stream.
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


log = structlog.get_logger()


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    sm: SessionManager,
    tool_by_name: dict[str, dict[str, Any]],
    recorder: McpTranscriptRecorder | None = None,
) -> dict[str, Any]:
    """Dispatch one MCP tool call, record it, and return the result dict.

    Extracted from the async call-tool handler so the same dispatch path
    can be driven by tests (and non-SDK callers) without the MCP SDK.
    """

    tool = tool_by_name.get(name)
    # H2 — Phase-scoped discovery: when the strict gating env var is
    # set AND the session has a ``current_phase``, refuse any tool not
    # in the per-phase allowlist. Default off so legacy tests / callers
    # see no behavior change.
    import os as _os

    if (
        tool is not None
        and _os.environ.get("COMPGEN_STRICT_PHASE_GATING") == "1"
    ):
        from compgen.mcp.phase_taxonomy import is_tool_allowed_in_phase

        session_id_for_phase = arguments.get("session_id") or ""
        try:
            session = sm.get(session_id_for_phase) if session_id_for_phase else None
        except KeyError:
            session = None
        current_phase = getattr(session, "current_phase", None) if session else None
        if current_phase and not is_tool_allowed_in_phase(name, current_phase):
            blocked = {
                "ok": False,
                "status": "blocked",
                "blocked_reason": "phase_violation",
                "tool": name,
                "current_phase": current_phase,
            }
            if recorder is not None:
                recorder.record(
                    tool=name,
                    args=arguments,
                    result=blocked,
                    session_id=session_id_for_phase or "unknown",
                    duration_ms=0.0,
                    error="phase_violation",
                )
            return blocked

    # H3 — Capability gating: when the strict env flag is set AND a
    # session is provided, refuse high-risk tools that don't carry
    # their required tokens / role.
    if (
        tool is not None
        and _os.environ.get("COMPGEN_STRICT_CAPABILITIES") == "1"
    ):
        from compgen.mcp.capabilities import missing_capabilities

        session_id_for_caps = arguments.get("session_id") or ""
        try:
            session = sm.get(session_id_for_caps) if session_id_for_caps else None
        except KeyError:
            session = None
        if session is not None:
            missing, role_mismatch = missing_capabilities(
                tool_name=name,
                session_caps=getattr(session, "capabilities", frozenset()),
                caller_role=getattr(session, "caller_role", "agent"),
            )
            if missing or role_mismatch is not None:
                blocked = {
                    "ok": False,
                    "status": "blocked",
                    "blocked_reason": "capability_missing",
                    "tool": name,
                    "missing_tokens": sorted(missing),
                    "role_mismatch": role_mismatch,
                }
                if recorder is not None:
                    recorder.record(
                        tool=name,
                        args=arguments,
                        result=blocked,
                        session_id=session_id_for_caps or "unknown",
                        duration_ms=0.0,
                        error="capability_missing",
                    )
                return blocked

    if tool is None:
        result: dict[str, Any] = {
            "ok": False,
            "error": f"Unknown tool: {name!r}",
            "available": sorted(tool_by_name.keys()),
        }
        if recorder is not None:
            recorder.record(
                tool=name,
                args=arguments,
                result=result,
                session_id=arguments.get("session_id") or "unknown",
                duration_ms=0.0,
                error=result["error"],
            )
        return result

    # H1 — Snapshot session state BEFORE the call so we can diff after.
    from compgen.mcp.tool_delta import (
        Cost,
        SideEffects,
        ToolDelta,
        _snapshot_decision_keys,
        _snapshot_modules,
        build_state_changes,
        canonical_args_hash,
        now_timestamp,
    )

    pre_recipe_hash, pre_payload_hash = _snapshot_modules(sm)
    pre_decisions = _snapshot_decision_keys(sm)
    envelope_timestamp = now_timestamp()
    args_hash = canonical_args_hash(arguments)

    started = time.perf_counter()
    error: str | None = None
    try:
        result = tool["handler"](sm, **arguments)
    except Exception as exc:  # noqa: BLE001
        log.exception("mcp.tool.failed", tool=name)
        error = f"{type(exc).__name__}: {exc}"
        result = {"ok": False, "error": error, "tool": name}
    duration_ms = (time.perf_counter() - started) * 1000.0

    session_id = (
        arguments.get("session_id") or (result.get("session_id") if isinstance(result, dict) else None) or "unknown"
    )

    # H1 — Build the typed ``ToolDelta`` envelope. Additive: the raw
    # ``result`` dict still flows back unchanged to the caller; the
    # envelope flows to the recorder so the trace bus has a structured
    # record. Honest residual: ``state_changes`` only diffs the
    # cheap-to-hash IR + decision-registry keys today; full op-level
    # diffs are a follow-up.
    state_changes = build_state_changes(
        sm=sm,
        pre_recipe=pre_recipe_hash,
        pre_payload=pre_payload_hash,
        pre_decisions=pre_decisions,
    )
    envelope = ToolDelta(
        tool=name,
        args_hash=args_hash,
        timestamp=envelope_timestamp,
        state_changes=state_changes,
        side_effects=SideEffects(),
        return_value=result if isinstance(result, dict) else None,
        cost=Cost(wall_ms=duration_ms),
        status="error" if error else "ok",
        blocked_reason=None,
    )
    log.debug(
        "mcp.tool.delta",
        tool=name,
        args_hash=envelope.args_hash,
        recipe_changed=state_changes.recipe_hash_before != state_changes.recipe_hash_after,
        payload_changed=state_changes.payload_hash_before != state_changes.payload_hash_after,
        new_decisions=len(state_changes.decisions),
        status=envelope.status,
    )

    if recorder is not None:
        recorder.record(
            tool=name,
            args=arguments,
            result=result,
            session_id=session_id,
            duration_ms=duration_ms,
            error=error,
        )
    return result


def _require_mcp() -> Any:
    try:
        import mcp  # type: ignore[import-not-found]
        import mcp.server  # type: ignore[import-not-found]
        import mcp.server.stdio  # type: ignore[import-not-found]
        import mcp.types  # type: ignore[import-not-found]

        return mcp
    except ImportError as exc:
        sys.stderr.write(
            "compgen-mcp requires the 'mcp' package.\n"
            "Reinstall with: pip install --upgrade compgen\n"
            f"Import error: {exc}\n"
        )
        sys.exit(2)


def _serialise(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def _run_async_server() -> None:
    """Run the MCP stdio server event loop."""
    import asyncio

    _route_logs_to_stderr()
    mcp_mod = _require_mcp()
    from mcp.server import Server  # type: ignore[import-not-found]
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found]
    from mcp.types import TextContent, Tool  # type: ignore[import-not-found]

    # Surface every discoverable extension (entry-point plugins, vendor
    # dialects, user-space ~/.compgen/extensions/*.py) to the registries
    # before tools are enumerated, so an installed kernel-provider or
    # dropped-in extension is visible on the first list_tools call.
    try:
        from compgen.plugins import discover_everything

        discovery = discover_everything()
        log.info(
            "mcp.discovery.complete",
            total=discovery.total(),
            user_space_root=discovery.user_space_root,
            vendor_dialects=len(discovery.vendor_dialects),
            user_space_tools=len(discovery.user_space_tools),
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("mcp.discovery.failed", error=str(exc))

    sm = SessionManager()
    # Wrap the transcript recorder so every MCP tool invocation also
    # lands on the active trace bus (no-op when no bus is installed).
    from compgen.trace import TracingMcpTranscriptRecorder

    recorder = TracingMcpTranscriptRecorder.wrap(McpTranscriptRecorder.from_env())
    server: Any = Server("compgen")

    tool_by_name = {t["name"]: t for t in ALL_TOOLS}

    @server.list_tools()
    async def _list_tools() -> list[Any]:  # type: ignore[misc]
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"],
            )
            for t in ALL_TOOLS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:  # type: ignore[misc]
        result = dispatch_tool(
            name,
            arguments,
            sm=sm,
            tool_by_name=tool_by_name,
            recorder=recorder,
        )
        return [TextContent(type="text", text=_serialise(result))]

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())


def main() -> None:
    """Entry-point for the ``compgen-mcp`` script.

    Honours ``--version`` and ``--help`` ahead of starting the
    blocking server. Without arguments the script behaves as
    before — a long-running stdio MCP transport that an editor or
    Claude Code spawns and connects to.
    """
    import sys

    args = sys.argv[1:]
    if args and args[0] in ("-V", "--version"):
        from compgen import __version__

        print(f"compgen-mcp {__version__}")
        sys.exit(0)
    if args and args[0] in ("-h", "--help"):
        from compgen import __version__

        print(
            f"compgen-mcp {__version__} — Compgen Model Context Protocol server\n"
            "\n"
            "Usage:\n"
            "  compgen-mcp                 # blocking stdio server (typical use)\n"
            "  compgen-mcp --version       # print version + exit\n"
            "  compgen-mcp --help          # this message\n"
            "\n"
            "Wire into Claude Code by adding the following to your\n"
            "~/.config/claude-code/mcp.json:\n"
            "\n"
            '  {"mcpServers": {"compgen": {"command": "compgen-mcp"}}}\n'
        )
        sys.exit(0)
    _run_async_server()


if __name__ == "__main__":
    main()


__all__ = ["dispatch_tool", "main"]

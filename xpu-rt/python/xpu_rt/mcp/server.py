"""MCP stdio server entry point for CompGen.

Adapts :data:`compgen.mcp.tools.ALL_TOOLS` into MCP SDK tool handlers.
The server is only instantiated when ``compgen-mcp`` is invoked from
the shell; importing :mod:`compgen.mcp` does not require the MCP SDK.

Usage::

    pip install compgen[mcp]
    compgen-mcp

Claude Code configuration (``.mcp.json``)::

    {
      "mcpServers": {
        "compgen": {
          "command": "compgen-mcp"
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import structlog

from compgen.mcp.session import SessionManager
from compgen.mcp.tools import ALL_TOOLS
from compgen.mcp.transcript import McpTranscriptRecorder

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
        arguments.get("session_id")
        or (result.get("session_id") if isinstance(result, dict) else None)
        or "unknown"
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
        import mcp   # type: ignore[import-not-found]
        import mcp.server   # type: ignore[import-not-found]
        import mcp.server.stdio   # type: ignore[import-not-found]
        import mcp.types   # type: ignore[import-not-found]

        return mcp
    except ImportError as exc:
        sys.stderr.write(
            "compgen-mcp requires the optional 'mcp' package.\n"
            "Install with: pip install compgen[mcp]\n"
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

    mcp_mod = _require_mcp()
    from mcp.server import Server   # type: ignore[import-not-found]
    from mcp.server.stdio import stdio_server   # type: ignore[import-not-found]
    from mcp.types import TextContent, Tool   # type: ignore[import-not-found]

    sm = SessionManager()
    recorder = McpTranscriptRecorder.from_env()
    server: Any = Server("compgen")

    tool_by_name = {t["name"]: t for t in ALL_TOOLS}

    @server.list_tools()
    async def _list_tools() -> list[Any]:   # type: ignore[misc]
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["input_schema"],
            )
            for t in ALL_TOOLS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:   # type: ignore[misc]
        result = dispatch_tool(
            name, arguments, sm=sm, tool_by_name=tool_by_name, recorder=recorder,
        )
        return [TextContent(type="text", text=_serialise(result))]

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())


def main() -> None:
    """Entry-point for the ``compgen-mcp`` script."""
    _run_async_server()


if __name__ == "__main__":
    main()


__all__ = ["dispatch_tool", "main"]

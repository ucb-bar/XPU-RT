"""Boot compgen-mcp over stdio and drive the JSON-RPC handshake.

Catches regressions the in-process :func:`compgen.mcp.server.dispatch_tool`
tests can't see: SDK load failures, initialize-hang bugs, tool-list
drift. Run via the ``mcp`` workflow; also handy locally as
``uv run python scripts/ci/mcp_stdio_smoke.py``.

The MCP Python SDK uses newline-delimited JSON on stdio (one JSON-RPC
message per line), not LSP-style Content-Length framing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXEMPLAR_TARGET = REPO_ROOT / "tests" / "targetgen" / "exemplars" / "test_gpu_simt.yaml"

REQUIRED_TOOLS = {
    "open_target",
    "load_model",
    "compile",
    "bundle_export",
    "register_pack",
}


def _send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("MCP server closed stdout before response")
    return json.loads(line.decode("utf-8"))


def main() -> int:
    if not EXEMPLAR_TARGET.exists():
        print(f"exemplar not found: {EXEMPLAR_TARGET}", file=sys.stderr)
        return 2

    proc = subprocess.Popen(
        ["compgen-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )
    try:
        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-stdio-smoke", "version": "0"},
            },
        })
        init = _recv(proc)
        assert "result" in init, f"initialize failed: {init}"
        print(f"initialize ok: server={init['result'].get('serverInfo')}")

        # SDK requires a notification before the server accepts request traffic.
        _send(proc, {
            "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
        })

        _send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        tools_resp = _recv(proc)
        tools = [t["name"] for t in tools_resp["result"]["tools"]]
        print(f"tools/list ok: {len(tools)} tools")
        missing = REQUIRED_TOOLS - set(tools)
        if missing:
            print(f"missing required tools: {sorted(missing)}", file=sys.stderr)
            return 3

        _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "open_target",
                "arguments": {"spec_path": str(EXEMPLAR_TARGET)},
            },
        })
        call_resp = _recv(proc)
        content = call_resp["result"]["content"]
        assert content, f"empty content: {call_resp}"
        payload = json.loads(content[0]["text"])
        assert payload.get("ok") is True, f"open_target failed: {payload}"
        print(f"tools/call open_target ok: target_id={payload.get('target_id')}")

        return 0
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())

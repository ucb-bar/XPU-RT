"""Real MCP end-to-end: spawn ``compgen-mcp`` over stdio + drive it as a client.

This is the proper end-to-end test: it uses the ``mcp`` SDK's
:class:`StdioServerParameters` + :class:`ClientSession` to spawn
``compgen-mcp`` as a subprocess, negotiates the protocol (``initialize``
+ ``tools/list``), and dispatches every CompGen MCP tool through real
JSON-RPC over stdio. Nothing short-circuits the protocol layer.

After the compile finishes we read the trace JSONL + IR dumps that the
server wrote (both live under the session's scratch dir) to confirm the
observability surface behaves end-to-end.

Run::

    uv run python scripts/mcp_llama_stdio_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "tests/targetgen/exemplars/test_gpu_simt.yaml"
MODEL_PATH = REPO / "examples/llama_block.py"


def _load_json(block: object) -> dict:
    """MCP ``CallToolResult.content`` is a list of ``TextContent`` — parse the JSON."""
    if hasattr(block, "structuredContent") and block.structuredContent:
        return dict(block.structuredContent)
    for item in getattr(block, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {}


async def _call(session: ClientSession, name: str, args: dict) -> dict:
    print(f"\n==> mcp.tools/call {name}({_compact(args)})", flush=True)
    reply = await session.call_tool(name, args)
    body = _load_json(reply)
    print(f"    ok={body.get('ok')!r}", flush=True)
    return body


def _compact(d: dict) -> str:
    parts = []
    for k, v in d.items():
        s = str(v)
        if len(s) > 44:
            s = s[:41] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _require_ok(body: dict, label: str) -> None:
    if not body.get("ok", False):
        print(f"[fail] {label}: {body}")
        sys.exit(1)


def _verify_trace(output_dir: Path) -> None:
    trace = output_dir / "trace" / "trace.jsonl"
    assert trace.exists(), f"missing trace file: {trace}"
    events = [json.loads(line) for line in trace.read_text().splitlines() if line]
    kinds = Counter(e["kind"] for e in events)
    print(f"\n[trace] {trace} ({len(events)} events)")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:20s} {count}")
    parented = [e for e in events if e["parent_event_id"]]
    ids = {e["event_id"] for e in events}
    orphans = [e for e in parented if e["parent_event_id"] not in ids]
    assert not orphans, f"orphan events: {orphans[:3]}"
    print(f"[trace] parented events: {len(parented)} (no orphans)")


def _verify_ir_dumps(output_dir: Path) -> None:
    dumps = sorted((output_dir / "ir_dumps").glob("*.mlir"))
    index_path = output_dir / "ir_dumps" / "index.json"
    assert dumps, f"no IR dumps under {output_dir / 'ir_dumps'}"
    assert index_path.exists(), "missing index.json"
    index = json.loads(index_path.read_text())
    print(f"\n[ir_dumps] {len(dumps)} files; index has {index['count']} entries")
    for entry in index["entries"][:6]:
        print(f"  {entry['index']:04d} {entry['name']:40s} {entry['phase']:7s} {entry['ir_hash']}")
    if len(index["entries"]) > 6:
        print(f"  ... ({len(index['entries']) - 6} more)")
    final = output_dir / "ir_dumps" / "final.mlir"
    assert final.exists(), "missing final.mlir"
    print(f"[ir_dumps] final.mlir size={final.stat().st_size} bytes")


async def run() -> None:
    assert SPEC_PATH.exists(), f"missing spec: {SPEC_PATH}"
    assert MODEL_PATH.exists(), f"missing model: {MODEL_PATH}"

    env = dict(os.environ)
    env["COMPGEN_DUMP_IR"] = "1"  # ensure IR dumps on inside the subprocess
    env.setdefault("COMPGEN_SESSION_DIR", str(REPO / "sessions" / "mcp_stdio_llama"))
    Path(env["COMPGEN_SESSION_DIR"]).mkdir(parents=True, exist_ok=True)

    # Spawn ``compgen-mcp`` as a real subprocess over stdio.
    params = StdioServerParameters(command="compgen-mcp", args=[], env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            # Protocol initialize.
            init = await session.initialize()
            print(f"[mcp.init] server={init.serverInfo.name} protocol={init.protocolVersion}")

            # Enumerate tools and sanity-check our new ones are visible.
            listing = await session.list_tools()
            names = {t.name for t in listing.tools}
            print(f"[mcp.tools/list] {len(names)} tools available")
            for needle in ("open_target", "load_model", "analyze_graph", "focus_chunk", "bundle_export"):
                marker = "✓" if needle in names else "✗"
                print(f"  [{marker}] {needle}")
                assert needle in names, f"missing tool: {needle}"

            # open_target
            body = await _call(session, "open_target", {"spec_path": str(SPEC_PATH)})
            _require_ok(body, "open_target")
            session_id = body["session_id"]
            print(f"    session_id={session_id} target={body['target_id']}")

            # load_model — triggers compile_model inside the server.
            body = await _call(
                session,
                "load_model",
                {
                    "session_id": session_id,
                    "model_path": str(MODEL_PATH),
                    "llm": "mock",
                    "budget": 2,
                },
            )
            _require_ok(body, "load_model")
            compile_out = Path(body["output_dir"])
            print(
                f"    num_ops={body.get('num_ops')} stages_run={body.get('stages_run')} "
                f"output_dir={compile_out}"
            )

            # analyze_graph — our new digest tool over MCP.
            body = await _call(session, "analyze_graph", {"session_id": session_id, "full": True})
            _require_ok(body, "analyze_graph")
            print("[digest] prompt summary:")
            for line in (body.get("summary") or "").splitlines():
                print(f"  {line}")
            digest = body.get("digest", {})
            assert digest.get("dtype_spectrum"), "digest dtype_spectrum empty"
            assert digest.get("memory_footprint_bytes", 0) > 0
            print(
                f"[digest] patterns={len(digest.get('pattern_histogram', {}))} "
                f"regions={len(digest.get('region_index', []))} "
                f"bottlenecks={digest.get('bottleneck_ops')}"
            )

            # focus_chunk — our new chunk-view tool over MCP.
            region = (digest.get("region_index") or [""])[0]
            body = await _call(
                session,
                "focus_chunk",
                {"session_id": session_id, "selector": {"region_id": region}},
            )
            _require_ok(body, "focus_chunk")
            chunk = body["chunk"]
            print(
                f"[chunk] region={chunk['region_id']} pattern={chunk['pattern_type']} "
                f"ops={len(chunk['ops'])}"
            )
            print(f"[chunk] knobs.granularity={chunk['decision_knobs']['granularity_candidates']}")
            print(f"[chunk] knobs.memory_tier={chunk['decision_knobs']['memory_tier_candidates']}")
            print(f"[chunk] knobs.tile_candidates={len(chunk['decision_knobs']['tile_candidates'])}")
            print(f"[chunk] dof.archetypes={chunk['dof_description']['archetypes']}")
            print(f"[chunk] dof.axes={chunk['dof_description']['axes']}")
            assert chunk["decision_knobs"]["granularity_candidates"]
            assert chunk["dof_description"]["archetypes"]

            # bundle_export — write the final bundle.
            bundle_out = Path(env["COMPGEN_SESSION_DIR"]) / session_id / "bundle"
            body = await _call(
                session,
                "bundle_export",
                {"session_id": session_id, "output_dir": str(bundle_out)},
            )
            _require_ok(body, "bundle_export")
            print(f"    bundle written to {bundle_out}")

    # Server subprocess has exited; verify the artifacts it wrote.
    _verify_trace(compile_out)
    _verify_ir_dumps(compile_out)
    print("\n[done] real MCP stdio end-to-end compile of llama_block succeeded.")


if __name__ == "__main__":
    asyncio.run(run())

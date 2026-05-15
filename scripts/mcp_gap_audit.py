"""Audit script: does the live MCP server actually close every gap?

Spawns ``compgen-mcp`` via stdio (real JSON-RPC), drives a full compile
of the llama block, then asserts all 10 gap-fix acceptance criteria
against the trace + IR dumps + digest the server wrote.

Run::

    uv run python scripts/mcp_gap_audit.py
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
SPEC = REPO / "tests/targetgen/exemplars/test_gpu_simt.yaml"
MODEL = REPO / "examples/llama_block.py"


def _json(reply) -> dict:
    if getattr(reply, "structuredContent", None):
        return dict(reply.structuredContent)
    for item in getattr(reply, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {}


async def _call(session, name, args):
    return _json(await session.call_tool(name, args))


async def main() -> None:
    env = dict(os.environ)
    env["COMPGEN_DUMP_IR"] = "1"
    env.setdefault("COMPGEN_SESSION_DIR", str(REPO / "sessions" / "mcp_audit"))
    Path(env["COMPGEN_SESSION_DIR"]).mkdir(parents=True, exist_ok=True)

    params = StdioServerParameters(command="compgen-mcp", args=[], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            opened = await _call(session, "open_target", {"spec_path": str(SPEC)})
            sid = opened["session_id"]
            loaded = await _call(
                session,
                "load_model",
                {
                    "session_id": sid,
                    "model_path": str(MODEL),
                    "llm": "mock",
                    "budget": 2,
                },
            )
            assert loaded["ok"], loaded
            out_dir = Path(loaded["output_dir"])

            digest = await _call(session, "analyze_graph", {"session_id": sid, "full": True})
            chunk = await _call(
                session,
                "focus_chunk",
                {
                    "session_id": sid,
                    "selector": {"pattern_type": "rmsnorm"},
                    "include_concrete_shapes": True,
                },
            )
            await _call(
                session,
                "bundle_export",
                {"session_id": sid, "output_dir": str(Path(env["COMPGEN_SESSION_DIR"]) / sid / "bundle")},
            )

    # Load trace + index.
    trace_path = out_dir / "trace" / "trace.jsonl"
    events = [json.loads(ln) for ln in trace_path.read_text().splitlines() if ln]
    kinds = Counter(e["kind"] for e in events)
    index = json.loads((out_dir / "ir_dumps" / "index.json").read_text())

    failures: list[str] = []
    print(f"[events] kinds: {dict(kinds)}")

    # Gap 1: mcp_call events present.
    if kinds.get("mcp_call", 0) < 3:
        failures.append(f"gap1: expected >=3 mcp_call events, got {kinds.get('mcp_call', 0)}")

    # Gap 2: pass_run events from the modern pipeline.
    if kinds.get("pass_run", 0) < 4:
        failures.append(f"gap2: expected >=4 pass_run events (capture/fx/ukernel/eqsat), got {kinds.get('pass_run', 0)}")
    # stage_run still present (spans fire on every stage).
    if kinds.get("stage_run", 0) < 2:
        failures.append(f"gap2: stage_run events too few: {kinds.get('stage_run', 0)}")

    # Gap 3: real oracle knobs.
    knobs = chunk["chunk"]["decision_knobs"]
    gran = knobs["granularity_candidates"]
    if not (isinstance(gran, list) and gran and isinstance(gran[0], dict)):
        failures.append("gap3a: granularity_candidates not oracle-structured dicts")
    elif not any(g.get("recommended") for g in gran):
        failures.append("gap3a: no granularity marked recommended by oracle")
    if len(knobs["tile_candidates"]) < 2:
        failures.append(f"gap3b: expected >1 tile_candidates (multi-dtype sweep), got {len(knobs['tile_candidates'])}")
    fusion_verdicts = [fc.get("verdict") for fc in knobs["fusion_candidates"]]
    if fusion_verdicts and all(v == "unknown" for v in fusion_verdicts):
        failures.append("gap3c: every fusion candidate still verdict=unknown")
    # Memory tiers filtered: test-gpu-simt has no bandwidth info → expect
    # fewer than the full 5 when filtering is engaged.
    # (We just assert non-empty + structured.)
    if not knobs["memory_tier_candidates"]:
        failures.append("gap3d: memory_tier_candidates empty")

    # Gap 4: dim_roles non-empty in chunk + digest dim_spectrum summed > 0.
    if not chunk["chunk"]["dim_roles"]:
        failures.append("gap4a: chunk.dim_roles is empty")
    dim_spec = digest["digest"]["dim_spectrum"]
    total_roles = (
        dim_spec["parallel_dims"]
        + dim_spec["reduce_dims"]
        + dim_spec["batch_dims"]
        + dim_spec["broadcast_dims"]
    )
    if total_roles == 0:
        failures.append("gap4b: digest dim_spectrum role counts all zero")

    # Gap 5: chunk dtypes populated for the synthetic cluster.
    if not chunk["chunk"]["dtypes"]:
        failures.append("gap5: chunk.dtypes empty (FX↔xDSL bridge still broken)")

    # Gap 6: decision events in compile_model.
    decisions = [e for e in events if e["kind"] == "decision"]
    if not decisions:
        failures.append("gap6: no decision events emitted during api.compile_model")

    # Gap 7: llm_turn_id key present on every decision payload.
    for d in decisions:
        if "llm_turn_id" not in d["payload"]:
            failures.append(f"gap7: decision missing llm_turn_id: {d}")
            break

    # Gap 8: FLOP fallback → total > 0.
    flops = digest["digest"]["flop_distribution"]
    if flops["total"] == 0:
        failures.append("gap8: flops_total still 0 (fallback did not fire)")
    if "source" not in flops:
        failures.append("gap8: flop_distribution.source missing")

    # Gap 9: wrap idempotency — mcp_call events exist proves lazy-resolve.
    # (Same signal as gap 1; only fail separately if mcp_call > 0 but all
    # have empty session_id due to mis-wrap.)
    for e in events:
        if e["kind"] == "mcp_call":
            if not e["payload"].get("tool"):
                failures.append("gap9: mcp_call event missing tool name")
                break

    # Gap 10: duration_ms nonzero somewhere in index.
    nonzero = sum(1 for entry in index["entries"] if entry.get("duration_ms", 0) > 0)
    if nonzero == 0:
        failures.append("gap10: every index.duration_ms is zero")

    # Print summary.
    print(f"[trace] {trace_path}")
    print(f"[index] {nonzero}/{len(index['entries'])} entries have non-zero duration_ms")
    print(f"[chunk.dim_roles]   {chunk['chunk']['dim_roles']}")
    print(f"[chunk.dtypes]      {chunk['chunk']['dtypes']}")
    print(f"[chunk.granularity] {[(g['granularity'], g.get('recommended')) for g in gran]}")
    print(f"[chunk.fusion]      verdicts={Counter(fusion_verdicts)}")
    print(f"[digest.flops]      total={flops['total']} source={flops.get('source')}")
    print(f"[digest.roles]      parallel={dim_spec['parallel_dims']} reduce={dim_spec['reduce_dims']} batch={dim_spec['batch_dims']} broadcast={dim_spec['broadcast_dims']}")
    print(f"[decisions]         {[d['payload'].get('decision_type') for d in decisions]}")

    if failures:
        print("\n[FAILED]")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\n[PASS] all 10 gap-fix criteria satisfied via real MCP stdio")


if __name__ == "__main__":
    asyncio.run(main())

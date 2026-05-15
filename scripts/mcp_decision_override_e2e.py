"""End-to-end: agent overrides an oracle decision via real MCP stdio.

Flow:

1. Spawn ``compgen-mcp`` via stdio.
2. ``open_target`` — loads the SIMT-GPU target.
3. Dry-run a compile to learn which decision sites will appear. We do
   this by calling ``load_model`` once; on the server side, every
   stage plugin enqueues its sites BEFORE writing IR, and we capture
   the first site id from the resulting ``list_decisions`` response.
4. Inspect the IR the oracle-default compile produced.
5. Start a fresh session, ``apply_decision`` an override for one
   matmul encoding BEFORE compile, then ``load_model``, then verify
   the IR got the override value instead of the oracle value.

Any deviation and we fail loudly. This proves the agent's write path
is real.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
    reply = await session.call_tool(name, args)
    return _json(reply)


def _env_for_subprocess() -> dict:
    env = dict(os.environ)
    env["COMPGEN_DUMP_IR"] = "1"
    env.setdefault("COMPGEN_SESSION_DIR", str(REPO / "sessions" / "mcp_override_audit"))
    Path(env["COMPGEN_SESSION_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


async def _run_baseline() -> tuple[Path, list[dict]]:
    """Compile with no overrides. Return compile output dir + site list."""
    env = _env_for_subprocess()
    params = StdioServerParameters(command="compgen-mcp", args=[], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            opened = _json(await session.call_tool("open_target", {"spec_path": str(SPEC)}))
            sid = opened["session_id"]
            loaded = await _call(session, "load_model", {
                "session_id": sid,
                "model_path": str(MODEL),
                "llm": "mock",
                "budget": 2,
            })
            out = Path(loaded["output_dir"])
            sites = await _call(session, "list_decisions", {
                "session_id": sid,
                "kind": "encoding",
            })
            return out, sites.get("sites", [])


async def _run_with_override(site_id: str, chosen_id: str, rationale: str) -> Path:
    env = _env_for_subprocess()
    params = StdioServerParameters(command="compgen-mcp", args=[], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            opened = _json(await session.call_tool("open_target", {"spec_path": str(SPEC)}))
            sid = opened["session_id"]
            # Pre-commit the agent's pick BEFORE load_model triggers the
            # pipeline that enqueues the site.
            applied = await _call(session, "apply_decision", {
                "session_id": sid,
                "site_id": site_id,
                "chosen_id": chosen_id,
                "rationale": rationale,
            })
            assert applied["ok"], f"apply_decision failed: {applied}"
            loaded = await _call(session, "load_model", {
                "session_id": sid,
                "model_path": str(MODEL),
                "llm": "mock",
                "budget": 2,
            })
            return Path(loaded["output_dir"])


def _encoding_of_site_in_final_mlir(out_dir: Path, region_id: str) -> str:
    """Scrape the encoding attribute applied to the op whose region_id matches."""
    final = out_dir / "ir_dumps" / "final.mlir"
    text = final.read_text()
    # Find the line where ``compgen.region_id = "rmsnorm_0"`` appears and
    # extract its ``compgen.encoding`` attribute value.
    import re

    pattern = re.compile(
        rf'compgen\.region_id\s*=\s*"{re.escape(region_id)}".*?compgen\.encoding\s*=\s*"([^"]+)"'
    )
    m = pattern.search(text)
    if not m:
        # The attribute order can flip; try the reverse.
        rev = re.compile(
            rf'compgen\.encoding\s*=\s*"([^"]+)".*?compgen\.region_id\s*=\s*"{re.escape(region_id)}"'
        )
        m = rev.search(text)
    if not m:
        raise RuntimeError(
            f"couldn't find encoding for region_id={region_id!r} in {final}"
        )
    return m.group(1)


async def main() -> None:
    print("=== step 1: baseline compile (oracle picks) ===")
    baseline_out, sites = await _run_baseline()
    print(f"baseline compile_out={baseline_out}")
    print(f"baseline encoding sites={len(sites)}")
    assert sites, "no encoding decision sites enqueued"

    # Pick a matmul site and note its oracle pick.
    matmul_site = None
    for s in sites:
        if s["context"].get("is_matmul"):
            matmul_site = s
            break
    if matmul_site is None:
        matmul_site = sites[0]
    site_id = matmul_site["site_id"]
    region_id = site_id.split(":", 1)[1]
    oracle_choice = matmul_site["oracle_recommended_id"]
    print(f"chosen site   : {site_id}")
    print(f"oracle picked : {oracle_choice}")
    baseline_encoding = _encoding_of_site_in_final_mlir(baseline_out, region_id)
    print(f"baseline IR encoding for {region_id!r}: {baseline_encoding}")
    assert baseline_encoding == oracle_choice, (
        f"baseline IR should match oracle pick; got {baseline_encoding!r}"
    )

    # Pick the OTHER candidate as the override.
    others = [c for c in matmul_site["candidates"] if c["id"] != oracle_choice]
    assert others, "site has only one candidate; nothing to override with"
    override_id = others[0]["id"]
    print(f"\n=== step 2: override with {override_id!r} and recompile ===")

    override_out = await _run_with_override(
        site_id,
        override_id,
        rationale="agent prefers this to test the override path",
    )
    override_encoding = _encoding_of_site_in_final_mlir(override_out, region_id)
    print(f"override IR encoding for {region_id!r}: {override_encoding}")

    if override_encoding != override_id:
        print(
            f"\n[FAIL] expected encoding to become {override_id!r}, "
            f"saw {override_encoding!r}"
        )
        sys.exit(1)

    # Also check the trace carries a decision event with source="agent".
    trace_path = override_out / "trace" / "trace.jsonl"
    agent_decisions = []
    for line in trace_path.read_text().splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event["kind"] == "decision" and event["payload"].get("source") == "agent":
            agent_decisions.append(event)
    if not agent_decisions:
        print("[FAIL] no decision event with source='agent' in trace")
        sys.exit(1)

    # Every agent decision should reference our site.
    matched = [
        d for d in agent_decisions if d["payload"].get("site_id") == site_id
    ]
    if not matched:
        print(f"[FAIL] no agent decision references site_id={site_id!r}")
        print(f"agent decisions: {[d['payload'].get('site_id') for d in agent_decisions]}")
        sys.exit(1)

    # The baseline compile should have NO agent decisions on that site,
    # only fallback_oracle.
    baseline_trace = [
        json.loads(ln)
        for ln in (baseline_out / "trace" / "trace.jsonl").read_text().splitlines()
        if ln
    ]
    baseline_sources = [
        e["payload"].get("source")
        for e in baseline_trace
        if e["kind"] == "decision" and e["payload"].get("site_id") == site_id
    ]
    if not baseline_sources or not all(s == "fallback_oracle" for s in baseline_sources):
        print(f"[FAIL] baseline decision sources for this site: {baseline_sources}")
        sys.exit(1)

    print(
        f"\n[PASS] agent override round-tripped\n"
        f"       baseline encoding = {baseline_encoding!r} (source=fallback_oracle)\n"
        f"       override encoding = {override_encoding!r} (source=agent)\n"
        f"       trace event_id    = {matched[0]['event_id']}"
    )


if __name__ == "__main__":
    asyncio.run(main())

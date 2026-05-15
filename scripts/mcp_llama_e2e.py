"""End-to-end MCP-driven compile of the llama block.

This script drives compilation the way an MCP client would: every
compiler action goes through :func:`compgen.mcp.server.dispatch_tool`,
which is the same code path the stdio MCP server uses. Nothing is done
via direct Python imports of compile internals.

After each stage we verify the observability features:

1. Trace  → ``<output_dir>/trace/trace.jsonl`` exists and contains
   ``llm_prompt`` / ``llm_response`` / ``mcp_call`` / ``pass_run``
   / ``stage_run`` / ``analysis_run`` / ``decision`` / ``ir_dump`` events
   linked by ``parent_event_id``.
2. IR dumps → ``<output_dir>/ir_dumps/NNN_<pass>_<before|after>.mlir``
   plus ``final.mlir`` plus ``index.json``.
3. Graph digest → non-empty ``analyze_graph`` response + ``focus_chunk``
   returns both oracle-enumerated knobs and an open-ended DoF view.

Run::

    COMPGEN_DUMP_IR=1 uv run python scripts/mcp_llama_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

# Make sure the compile path writes IR dumps even when the MCP lifecycle
# tool does not propagate ``dump_ir`` explicitly.
os.environ.setdefault("COMPGEN_DUMP_IR", "1")

# Redirect the session dir to a deterministic path under the repo so the
# trace mirror is easy to inspect.
REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "sessions" / "mcp_e2e_llama"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["COMPGEN_SESSION_DIR"] = str(OUT_ROOT)

from compgen.mcp.server import dispatch_tool  # noqa: E402
from compgen.mcp.session import SessionManager  # noqa: E402
from compgen.mcp.tools import ALL_TOOLS  # noqa: E402
from compgen.mcp.transcript import McpTranscriptRecorder  # noqa: E402
from compgen.trace import TracingMcpTranscriptRecorder  # noqa: E402

SPEC_PATH = REPO / "tests/targetgen/exemplars/test_gpu_simt.yaml"
MODEL_PATH = REPO / "examples/llama_block.py"

TOOL_BY_NAME = {t["name"]: t for t in ALL_TOOLS}


def _call(
    sm: SessionManager,
    recorder: McpTranscriptRecorder,
    name: str,
    args: dict,
) -> dict:
    print(f"\n==> MCP {name}({_compact(args)})", flush=True)
    out = dispatch_tool(
        name,
        args,
        sm=sm,
        tool_by_name=TOOL_BY_NAME,
        recorder=recorder,
    )
    print(f"    ok={out.get('ok')!r}", flush=True)
    return out


def _compact(d: dict) -> str:
    parts = []
    for k, v in d.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def _require_ok(result: dict, label: str) -> None:
    if not result.get("ok", False):
        print(f"[fail] {label}: {result}")
        sys.exit(1)


def _verify_trace(compile_out_dir: Path) -> None:
    trace = compile_out_dir / "trace" / "trace.jsonl"
    assert trace.exists(), f"missing trace file: {trace}"
    events = [json.loads(line) for line in trace.read_text().splitlines() if line]
    kinds = Counter(e["kind"] for e in events)
    print(f"\n[trace] {trace} ({len(events)} events)")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind:20s} {count}")
    # Every paired span has start+end; confirm pass_run has at least one pair
    start_ends = Counter(
        (e["kind"], e["phase"]) for e in events if e["phase"] in {"start", "end"}
    )
    print(f"[trace] start_ends: {dict(start_ends)}")
    # Parent-linkage smoke check: at least one child event references an
    # existing parent.
    ids = {e["event_id"] for e in events}
    children = [e for e in events if e["parent_event_id"]]
    orphans = [e for e in children if e["parent_event_id"] not in ids]
    assert not orphans, f"orphan events: {orphans[:3]}"
    print(f"[trace] parented events: {len(children)} (no orphans)")


def _verify_ir_dumps(compile_out_dir: Path) -> None:
    dumps = sorted((compile_out_dir / "ir_dumps").glob("*.mlir"))
    index_path = compile_out_dir / "ir_dumps" / "index.json"
    assert dumps, f"no IR dumps under {compile_out_dir / 'ir_dumps'}"
    assert index_path.exists(), f"missing index.json"
    index = json.loads(index_path.read_text())
    print(f"\n[ir_dumps] {len(dumps)} files; index has {index['count']} entries")
    for entry in index["entries"][:6]:
        print(f"  {entry['index']:04d} {entry['name']:40s} {entry['phase']:7s} {entry['ir_hash']}")
    if len(index["entries"]) > 6:
        print(f"  ... ({len(index['entries']) - 6} more)")
    final = compile_out_dir / "ir_dumps" / "final.mlir"
    assert final.exists(), "missing final.mlir"
    print(f"[ir_dumps] final.mlir size={final.stat().st_size} bytes")


def _print_digest_summary(digest_result: dict) -> None:
    print("\n[digest] prompt summary:")
    for line in (digest_result.get("summary") or "").splitlines():
        print(f"  {line}")


def main() -> None:
    print(f"repo           : {REPO}")
    print(f"spec           : {SPEC_PATH}")
    print(f"model          : {MODEL_PATH}")
    print(f"session root   : {OUT_ROOT}")
    assert SPEC_PATH.exists(), f"missing spec: {SPEC_PATH}"
    assert MODEL_PATH.exists(), f"missing model: {MODEL_PATH}"

    sm = SessionManager()
    raw_recorder = McpTranscriptRecorder.from_env()
    recorder = TracingMcpTranscriptRecorder.wrap(raw_recorder)

    # 1) open_target → attaches a CompGenDevice to a session.
    result = _call(
        sm,
        recorder,
        "open_target",
        {"spec_path": str(SPEC_PATH)},
    )
    _require_ok(result, "open_target")
    session_id = result["session_id"]
    print(f"    session_id={session_id}  target={result['target_id']}")

    # 2) load_model → runs compile_model under the hood (installs bus,
    #    wires IR dump writer, runs pipeline, emits final.mlir).
    result = _call(
        sm,
        recorder,
        "load_model",
        {
            "session_id": session_id,
            "model_path": str(MODEL_PATH),
            "llm": "mock",
            "budget": 2,
        },
    )
    _require_ok(result, "load_model")
    print(f"    num_ops={result.get('num_ops')}  stages_run={result.get('stages_run')}")

    compile_out_dir = Path(result["output_dir"]).resolve()
    print(f"    compile output : {compile_out_dir}")

    # 3) analyze_graph → our new MCP tool: shape-free digest.
    result = _call(
        sm,
        recorder,
        "analyze_graph",
        {"session_id": session_id, "full": True},
    )
    _require_ok(result, "analyze_graph")
    _print_digest_summary(result)
    digest = result.get("digest", {})
    assert digest.get("dtype_spectrum"), "digest dtype_spectrum empty"
    assert digest.get("memory_footprint_bytes", 0) > 0, "digest memory footprint zero"
    print(
        f"[digest] patterns={len(digest.get('pattern_histogram', {}))}, "
        f"regions={len(digest.get('region_index', []))}, "
        f"flops={digest.get('flop_distribution', {}).get('total')}"
    )

    # 4) focus_chunk → our new MCP tool: chunk view with knobs+DoF.
    first_region = (digest.get("region_index") or [""])[0]
    result = _call(
        sm,
        recorder,
        "focus_chunk",
        {"session_id": session_id, "selector": {"region_id": first_region}},
    )
    _require_ok(result, "focus_chunk")
    chunk = result["chunk"]
    print(
        f"[chunk] region={chunk['region_id']} pattern={chunk['pattern_type']} "
        f"ops={len(chunk['ops'])}"
    )
    print(
        f"[chunk] knobs.granularity={chunk['decision_knobs']['granularity_candidates']}"
    )
    print(
        f"[chunk] knobs.memory_tier={chunk['decision_knobs']['memory_tier_candidates']}"
    )
    print(
        f"[chunk] knobs.tile_candidates={len(chunk['decision_knobs']['tile_candidates'])}"
    )
    print(
        f"[chunk] dof.archetypes={chunk['dof_description']['archetypes']}"
    )
    print(
        f"[chunk] dof.axes={chunk['dof_description']['axes']}"
    )
    assert chunk["decision_knobs"]["granularity_candidates"], "no knob granularity candidates"
    assert chunk["dof_description"]["archetypes"], "no DoF archetypes"

    # 5) bundle_export → writes bundle (payload.mlir etc)
    result = _call(
        sm,
        recorder,
        "bundle_export",
        {"session_id": session_id, "output_dir": str(OUT_ROOT / session_id / "bundle")},
    )
    _require_ok(result, "bundle_export")

    # 6) Verify trace + IR dumps that compile_model wrote.
    _verify_trace(compile_out_dir)
    _verify_ir_dumps(compile_out_dir)

    print("\n[done] MCP-only end-to-end compile of llama_block succeeded.")


if __name__ == "__main__":
    main()

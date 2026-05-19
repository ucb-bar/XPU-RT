"""One-shot Spike+gemmini eval for a Claude-Code-authored kernel.

Used during the Path-A pilot where I (Claude Code in-loop) produce a
kernel inline in chat, write it to disk, then invoke this script to
run the real :class:`CRiscvEvaluator` against the chosen contract.

Reads:
  --manifest  the SmolVLA subset manifest.json
  --region    region_sig_hash (or substring of region_id) to look up
              the contract
  --kernel    path to the kernel .c source

Writes a JSONL row to ``--out`` capturing: contract summary, eval
report (correct / cycles / compile_log / runtime_log), per-step
Claude-session token delta (before + after snapshot), kernel size
proxy (chars / lines).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from xpu_rt.benchmarks.smolvla_subset import load_contracts
from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import CRiscvEvaluator
from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse
from xpu_rt.observability import claude_session


def _find_contract(manifest: Path, region: str):
    contracts = load_contracts(manifest)
    region_l = region.lower()
    # Match by region_sig_hash (saved as <hash>.json) OR by substring of region_id.
    for c in contracts:
        if region_l in c.region_id.lower():
            return c
    # Try matching by hash via the on-disk manifest entries.
    body = json.loads(manifest.read_text())
    for r in body.get("contracts", []):
        if r["region_sig_hash"].startswith(region_l):
            for c in contracts:
                if (
                    tuple(tuple(s) for s in r["input_shapes"]) == c.input_shapes
                    and tuple(tuple(s) for s in r["output_shapes"]) == c.output_shapes
                ):
                    return c
    raise SystemExit(f"no contract matched region={region!r} in {manifest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--region", required=True, help="contract region_id substring or sig_hash prefix")
    parser.add_argument("--kernel", type=Path, required=True, help="path to candidate C kernel")
    parser.add_argument("--out", type=Path, required=True, help="JSONL row sink (appends)")
    parser.add_argument("--action", default="claude-r1", help="strategy label recorded in the row")
    parser.add_argument("--target", default="gemmini_mx")
    parser.add_argument("--rounds-so-far", type=int, default=1)
    parser.add_argument(
        "--baseline-snapshot",
        type=Path,
        default=None,
        help="path to a prior session snapshot JSON to compute token delta",
    )
    parser.add_argument("--timeout-s", type=int, default=120)
    args = parser.parse_args(argv)

    contract = _find_contract(args.manifest, args.region)
    contract_payload = {
        "region_id": contract.region_id,
        "op_family": contract.op_family,
        "input_shapes": [list(s) for s in contract.input_shapes],
        "output_shapes": [list(s) for s in contract.output_shapes],
        "dtypes": list(contract.dtypes),
        "target_name": contract.target_name,
    }

    kernel_code = args.kernel.read_text()
    chars = len(kernel_code)
    lines = kernel_code.count("\n") + 1

    # Run the evaluator.
    evaluator = CRiscvEvaluator(
        contract=contract,
        timeout_s=args.timeout_s,
        keep_workdir=False,
    )
    t0 = time.perf_counter()
    report = evaluator.evaluate(
        ProposeResponse(
            kernel_code=kernel_code,
            language="c",
            action=args.action,
        )
    )
    wall_s = time.perf_counter() - t0

    # Snapshot the session AFTER this eval call.
    after = claude_session.read_snapshot()

    # Compute delta vs baseline if provided.
    delta_payload: dict | None = None
    if args.baseline_snapshot and args.baseline_snapshot.is_file():
        before_body = json.loads(args.baseline_snapshot.read_text())
        # Reconstruct a SessionSnapshot from JSON.
        from xpu_rt.observability.claude_session import SessionSnapshot, TokenCounts

        before = SessionSnapshot(session_path=Path(before_body.get("session_path", "")))
        before.rows_seen = before_body.get("rows_seen", 0)
        before.rows_with_usage = before_body.get("rows_with_usage", 0)
        for m, t in before_body.get("by_model", {}).items():
            before.by_model[m] = TokenCounts(
                input_tokens=int(t.get("input_tokens", 0)),
                output_tokens=int(t.get("output_tokens", 0)),
                cache_read_tokens=int(t.get("cache_read_tokens", 0)),
                cache_creation_5m_tokens=int(t.get("cache_creation_5m_tokens", 0)),
                cache_creation_1h_tokens=int(t.get("cache_creation_1h_tokens", 0)),
            )
        delta_payload = claude_session.delta(before, after)

    row = {
        "backend": "claude_code_xpu_rt",
        "contract": contract_payload,
        "action": args.action,
        "rounds_so_far": args.rounds_so_far,
        "kernel_chars": chars,
        "kernel_lines": lines,
        "correct": report.correct,
        "cycles": report.cycles,
        "score": report.score,
        "compile_log": (report.compile_log or "")[-1500:],
        "runtime_log": (report.runtime_log or "")[-1500:],
        "diff_summary": report.diff_summary,
        "meta": report.metadata,
        "wall_s": wall_s,
        "session_delta": delta_payload,
        "session_total_after": after.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    # Also stdout a compact summary so I can see it in chat.
    summary = {
        "region": contract_payload["region_id"].rsplit(".", 2)[-2] + "." + contract_payload["region_id"].rsplit(".", 1)[-1],
        "shape": f"{contract_payload['input_shapes'][0]} x {contract_payload['input_shapes'][1]}",
        "correct": report.correct,
        "cycles": report.cycles,
        "wall_s": round(wall_s, 1),
        "kernel_lines": lines,
        "delta_input": delta_payload["total_delta"]["input_tokens"] if delta_payload else None,
        "delta_output": delta_payload["total_delta"]["output_tokens"] if delta_payload else None,
        "delta_cost_usd": delta_payload["estimated_cost_delta_usd"] if delta_payload else None,
    }
    print(json.dumps(summary, indent=2))
    if not report.correct:
        # Surface the actual error head so I can fix in round 2.
        log_tail = (report.compile_log or report.runtime_log or "")[-800:]
        if log_tail:
            print("=== log tail (head 800) ===")
            print(log_tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

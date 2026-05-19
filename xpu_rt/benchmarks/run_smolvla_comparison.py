"""Phase A / B driver: run KB-vanilla bridge + XPU-RT/KB v2 on each
SmolVLA contract and emit JSONL rows for the aggregator.

Mirrors plan 2 § A.3 and § A.4. Both backends use Gemini-2.5-flash so
the model is held constant; the only varying dimension is the **prompting
+ memory + evaluator** architecture.

Usage::

    uv run python -m xpu_rt.benchmarks.run_smolvla_comparison \\
        --manifest /tmp/xpu_rt_smolvla_subset/manifest.json \\
        --out      results/comparison/smolvla_subset \\
        --target   gemmini_mx \\
        --rounds   4 \\
        --limit    5     # take the top-occurrence 5 contracts only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

from xpu_rt.benchmarks.smolvla_subset import load_contracts
from xpu_rt.kernels.kernelblaster_gemmini_bridge import KernelBlasterGemminiBridge
from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    KernelBlasterV2,
    KernelGeneratorLLM,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import CRiscvEvaluator
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory import target_knowledge as tk
from xpu_rt.observability import gemini_usage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def _spend_snapshot() -> dict[str, float]:
    s = gemini_usage.load_summary()
    return {
        "cumulative_usd": s.total_cost_usd,
        "tokens_in": s.total_prompt_tokens,
        "tokens_out": s.total_completion_tokens,
        "calls": s.total_calls,
    }


def _run_kb_vanilla(
    *,
    contract: KernelContract,
    target_card: tk.TargetKnowledgeCard,
    rounds: int,
    model: str,
) -> dict:
    pre = _spend_snapshot()
    t0 = time.perf_counter()
    bridge = KernelBlasterGemminiBridge(
        target_card=target_card,
        max_rounds=rounds,
        model=model,
    )
    result = bridge.run(contract)
    wall = time.perf_counter() - t0
    post = _spend_snapshot()
    return {
        "backend": "kb_vanilla",
        "contract": _summarise_contract(contract),
        "rounds": result.rounds,
        "compile": result.compile,
        "intrinsic_use_rate": result.intrinsic_use_rate,
        "intrinsic_matched": result.intrinsic_matched,
        "intrinsic_total": result.intrinsic_total,
        "shape_consistency": result.shape_consistency,
        "shape_missing": result.shape_missing,
        "final_strategy": result.final_strategy,
        "tokens_in": post["tokens_in"] - pre["tokens_in"],
        "tokens_out": post["tokens_out"] - pre["tokens_out"],
        "cost_usd": post["cumulative_usd"] - pre["cumulative_usd"],
        "wall_s": wall,
        "attempts": result.attempts,
    }


def _run_xpu_rt(
    *,
    contract: KernelContract,
    target_card: tk.TargetKnowledgeCard,
    rounds: int,
    model: str,
    eval_timeout_s: int,
) -> dict:
    pre = _spend_snapshot()
    t0 = time.perf_counter()
    evaluator = CRiscvEvaluator(
        contract=contract,
        timeout_s=eval_timeout_s,
    )
    loop = KernelBlasterV2(
        card=target_card,
        generator=KernelGeneratorLLM(model=model),
        evaluator=evaluator,
        config=AgentLoopConfig(max_iterations=rounds, accept_threshold=1.0),
    )
    result = loop.run(contract)
    wall = time.perf_counter() - t0
    post = _spend_snapshot()
    pr = result.to_provider_result()
    return {
        "backend": "xpu_rt_kb_v2",
        "contract": _summarise_contract(contract),
        "rounds": pr.iterations_used,
        "found": pr.found,
        "correct": pr.correct,
        "speedup": pr.speedup,
        "cycles": pr.latency_us if pr.latency_us > 0 else None,
        "plan": pr.plan,
        "state_hash": pr.metadata.get("state_hash", ""),
        "aborted": pr.metadata.get("aborted", False),
        "abort_reason": pr.metadata.get("abort_reason", ""),
        "tokens_in": post["tokens_in"] - pre["tokens_in"],
        "tokens_out": post["tokens_out"] - pre["tokens_out"],
        "cost_usd": post["cumulative_usd"] - pre["cumulative_usd"],
        "wall_s": wall,
        "attempts": [
            {
                "attempt": c.attempt,
                "action": c.proposal.action,
                "correct": c.report.correct,
                "score": c.report.score,
                "cycles": c.report.cycles,
                "diff_summary": c.report.diff_summary,
            }
            for c in result.history
        ],
    }


def _summarise_contract(c: KernelContract) -> dict:
    return {
        "region_id": c.region_id,
        "op_family": c.op_family,
        "input_shapes": [list(s) for s in c.input_shapes],
        "output_shapes": [list(s) for s in c.output_shapes],
        "dtypes": list(c.dtypes),
        "target_name": c.target_name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase A driver: KB-vanilla vs XPU-RT on SmolVLA subset.")
    parser.add_argument("--manifest", type=Path, required=True, help="subset manifest.json path")
    parser.add_argument("--out", type=Path, required=True, help="results dir")
    parser.add_argument("--target", default="gemmini_mx")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N contracts (0 = all).",
    )
    parser.add_argument(
        "--backend",
        choices=("both", "kb_vanilla", "xpu_rt"),
        default="both",
    )
    parser.add_argument(
        "--eval-timeout-s",
        type=int,
        default=90,
        help="Spike-eval timeout per candidate (XPU-RT side only).",
    )
    parser.add_argument(
        "--abort-on-spend-usd",
        type=float,
        default=5.0,
        help="Abort cleanly if the *incremental* spend during this run exceeds this many USD.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    args.out.mkdir(parents=True, exist_ok=True)

    contracts = load_contracts(args.manifest)
    if args.limit > 0:
        contracts = contracts[: args.limit]
    logger.info("loaded %d contracts from %s", len(contracts), args.manifest)

    card = tk.load(args.target)

    kb_path = args.out / "kb_vanilla.jsonl"
    xr_path = args.out / "xpu_rt.jsonl"
    pre = _spend_snapshot()
    print(json.dumps({"event": "run_start", "pre_spend": pre}, indent=2))

    for i, contract in enumerate(contracts):
        # Stamp the chosen target into the contract (it carries
        # target_name="gemmini_mx" from the selector, but allow override).
        contract = KernelContract(
            region_id=contract.region_id,
            op_family=contract.op_family,
            input_shapes=contract.input_shapes,
            output_shapes=contract.output_shapes,
            dtypes=contract.dtypes,
            layout=contract.layout,
            target_name=args.target,
            hardware_key=contract.hardware_key,
            objective=contract.objective,
            constraints=contract.constraints,
            provider_hints=contract.provider_hints,
        )

        post = _spend_snapshot()
        incremental = post["cumulative_usd"] - pre["cumulative_usd"]
        if incremental > args.abort_on_spend_usd:
            print(
                json.dumps(
                    {
                        "event": "spend_cap_hit",
                        "incremental_usd": incremental,
                        "cap": args.abort_on_spend_usd,
                        "processed": i,
                    },
                    indent=2,
                )
            )
            break

        print(json.dumps({"event": "contract_start", "i": i, "region_id": contract.region_id}))

        if args.backend in ("both", "kb_vanilla"):
            row = _run_kb_vanilla(
                contract=contract,
                target_card=card,
                rounds=args.rounds,
                model=args.model,
            )
            with kb_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"  kb_vanilla: compile={row['compile']} "
                f"intr={row['intrinsic_use_rate']:.2f} "
                f"shape={row['shape_consistency']} "
                f"rounds={row['rounds']} "
                f"cost=${row['cost_usd']:.4f} wall={row['wall_s']:.1f}s"
            )

        if args.backend in ("both", "xpu_rt"):
            row = _run_xpu_rt(
                contract=contract,
                target_card=card,
                rounds=args.rounds,
                model=args.model,
                eval_timeout_s=args.eval_timeout_s,
            )
            with xr_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"  xpu_rt:     correct={row['correct']} "
                f"cycles={row.get('cycles')} "
                f"speedup={row['speedup']:.2f} "
                f"rounds={row['rounds']} "
                f"cost=${row['cost_usd']:.4f} wall={row['wall_s']:.1f}s"
            )

    post = _spend_snapshot()
    summary = {
        "event": "run_end",
        "pre_spend": pre,
        "post_spend": post,
        "incremental_usd": post["cumulative_usd"] - pre["cumulative_usd"],
        "kb_vanilla_jsonl": str(kb_path) if kb_path.exists() else None,
        "xpu_rt_jsonl": str(xr_path) if xr_path.exists() else None,
    }
    (args.out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

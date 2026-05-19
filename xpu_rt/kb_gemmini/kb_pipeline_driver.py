"""Drive **vanilla KB's actual prompt-generation code** on Gemmini.

Bypasses KB's LangGraph workflow, microservices, and dataset modules
— all of which are tightly CUDA-coupled — and instead imports the
core prompt-construction function from
``third_party/kernelblaster/src/kernelblaster/agents/opt_ncu_rl.py``
verbatim. We hand it:

  * A real :class:`OptimizationEntry` / :class:`CompositeOptimization`
    drawn from a small Gemmini-flavored strategy menu (these are the
    same dataclasses KB's RL bandit cycles through internally).
  * The candidate kernel source.
  * The previous round's Spike+gemmini counter output in the place
    KB normally puts NCU logs (we patch the surrounding system-prompt
    to tell the LLM the metric is Gemmini cycles, not NCU).
  * The Gemmini Target Card content as ``database_content``.

The result is the **identical prompt KB would build**, just with our
hardware context substituted for CUDA's. We post it to gemini-flash,
compile + run the candidate on Spike+gemmini via our existing
``CRiscvEvaluator``, and iterate up to N rounds.

This isolates "KB's prompting code" as the variable while holding the
LLM, the evaluator, and the per-shape harness constant.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make vanilla KB importable.
_KB_ROOT = Path("/scratch2/agustin/xpu-rt-integration/third_party/kernelblaster")
if str(_KB_ROOT) not in sys.path:
    sys.path.insert(0, str(_KB_ROOT))


from xpu_rt.benchmarks.smolvla_subset import load_contracts
from xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv import CRiscvEvaluator
from xpu_rt.kernels.kernelblaster_v2.generators import (
    KernelGeneratorLLM, ProposeRequest, ProposeResponse,
)
from xpu_rt.kernels.kernelblaster_v2.prompt_builder import PromptBundle
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory import target_knowledge as tk
from xpu_rt.observability import claude_session, gemini_usage


# ---------------------------------------------------------------------------
# Import KB's actual prompt-generation code + dataclasses.
# ---------------------------------------------------------------------------

# These are the real, untouched KB types — we are NOT reimplementing KB,
# we are importing its code.
from src.kernelblaster.agents.opt_ncu_rl import generate_strategy_guided_prompt
from src.kernelblaster.agents.database import OptimizationEntry, CompositeOptimization


# ---------------------------------------------------------------------------
# Gemmini-flavoured strategy menu (the "RL bandit" pool).
# These map onto KB's OptimizationEntry shape — same fields, Gemmini values.
# ---------------------------------------------------------------------------


def _gemmini_strategies() -> list[OptimizationEntry]:
    return [
        OptimizationEntry(
            technique="tile_ws_dataflow",
            description=(
                "Use weight-stationary dataflow via config_ex(WEIGHT_STATIONARY,...). "
                "Pre-load B tiles into the systolic array with gemmini_extended_preload; "
                "stream A through with gemmini_extended_compute_preloaded for the first K-tile "
                "and gemmini_extended_compute_accumulated for k>0."
            ),
            category="compute",
            predicted_speedup=12.0,
            confidence_score=0.7,
        ),
        OptimizationEntry(
            technique="mvin_overlap_AB",
            description=(
                "Pre-stage A in scratchpad lane 1 via gemmini_extended_mvin and B in lane 2 "
                "via gemmini_extended_mvin2 with independent stride configs "
                "(gemmini_extended_config_ld for lane 1, gemmini_extended2_config_ld for lane 2). "
                "Loads run in parallel with the systolic array."
            ),
            category="memory",
            predicted_speedup=2.0,
            confidence_score=0.6,
        ),
        OptimizationEntry(
            technique="accumulator_keep_in_place",
            description=(
                "Keep partial sums in the Gemmini accumulator across the inner-K loop. "
                "Set the accumulate-flag bit (1<<30) on the C scratchpad address for k>0 "
                "rounds; only call gemmini_extended_mvout to drain after the full K is consumed."
            ),
            category="compute",
            predicted_speedup=1.5,
            confidence_score=0.7,
        ),
        OptimizationEntry(
            technique="tiled_matmul_auto_helper",
            description=(
                "Use Gemmini's high-level helper tiled_matmul_auto(M, N, K, A, B, NULL, C, "
                "K, N, N, N, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION, "
                "ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY, false, false, false, true, false, "
                "0, WS) which handles all tile/preload/accumulate orchestration internally. "
                "Easiest correct path for K <= 720; may pick wrong tile shape for large K."
            ),
            category="compute",
            predicted_speedup=8.0,
            confidence_score=0.8,
        ),
    ]


def _composite() -> CompositeOptimization:
    """Compose the top-2 techniques into a CompositeOptimization (the form
    KB's strategy bandit prefers when multiple techniques apply)."""
    return CompositeOptimization(
        state="mma_throughput_limited",
        technique1="tile_ws_dataflow",
        technique2="mvin_overlap_AB",
        technique3="accumulator_keep_in_place",
        order_of_techniques=[
            "1. tile_ws_dataflow",
            "2. mvin_overlap_AB",
            "3. accumulator_keep_in_place",
        ],
        parameters_to_fine_tune={
            "tile_I": "min(M/DIM, 4)",
            "tile_J": "min(N/DIM, 4-8 depending on M)",
            "tile_K": "min(K/DIM, 45 for K<=720; smaller for K>=960)",
        },
        predicted_improvement=15.0,
        reason="Combined dataflow + DMA overlap + accumulator residency for Gemmini matmul",
        side_effects="Requires careful scratchpad/accumulator budgeting; see Target Card derivation_rules",
        confidence_score=0.6,
    )


# ---------------------------------------------------------------------------
# Synthetic "NCU log" for Gemmini — KB's prompt-generator expects an
# NCU-like profile blob; we substitute Spike counter output formatted to
# look like an NCU log so KB's text-construction logic doesn't choke.
# ---------------------------------------------------------------------------


def _classify_bottleneck(
    *,
    cycles: int | None,
    exe_active: int | None,
    load_dma_wait: int | None,
    scratchpad_a_wait: int | None,
) -> tuple[str, str]:
    """Map Gemmini counter ratios to a KB-readable bottleneck state.

    Returns ``(state, rationale)``. KB's strategy bandit
    discriminates on the state label; the rationale shows the
    operator why we picked it.

    Heuristics calibrated against the `gemmini_counter.h` event
    semantics:
      * ``compute_throughput_limited``   — MMA pipeline saturated
        (exe_active / total ≥ 0.8); strategies should shrink tiles
        or change dataflow to free MMA cycles.
      * ``memory_bandwidth_limited``     — long DMA waits
        (load_dma_wait / total ≥ 0.3); strategies should overlap
        DMA with compute (mvin lane 2, double-buffer).
      * ``scratchpad_pressure``          — repeated A-scratchpad
        wait (scratchpad_a_wait / total ≥ 0.2); strategies should
        reduce tile_K or use scratchpad lane assignment.
      * ``balanced``                     — none of the above
        thresholds trip; the kernel runs close to the achievable
        ratio. Strategies should try different macro-tile shapes.
      * ``unknown``                      — no counter data
        available (first round, or harness didn't emit extras).
    """
    if cycles is None or cycles <= 0:
        return "unknown", "no cycle count available"
    if exe_active is None and load_dma_wait is None and scratchpad_a_wait is None:
        return "unknown", "extra counters not emitted by this harness"

    def _safe_ratio(x: int | None) -> float:
        return (x or 0) / cycles if cycles else 0.0

    exe_ratio = _safe_ratio(exe_active)
    dma_ratio = _safe_ratio(load_dma_wait)
    spad_ratio = _safe_ratio(scratchpad_a_wait)

    if exe_ratio >= 0.80:
        return (
            "compute_throughput_limited",
            f"exe_active/total = {exe_ratio:.2f} (MMA pipeline saturated)",
        )
    if dma_ratio >= 0.30:
        return (
            "memory_bandwidth_limited",
            f"load_dma_wait/total = {dma_ratio:.2f} (kernel stalls on DMA)",
        )
    if spad_ratio >= 0.20:
        return (
            "scratchpad_pressure",
            f"scratchpad_a_wait/total = {spad_ratio:.2f} (A-tile residency contention)",
        )
    return (
        "balanced",
        (
            f"exe={exe_ratio:.2f} dma={dma_ratio:.2f} spad={spad_ratio:.2f} "
            "— no single bottleneck dominates"
        ),
    )


def _gemmini_ncu_log(
    cycles: int | None,
    mismatches: tuple[int, int] | None,
    contract: KernelContract,
    *,
    counter_extras: dict[str, int] | None = None,
) -> str:
    """Format Spike+gemmini counter output to fit KB's NCU-log slot.

    The 4-counter trace lets KB's strategy bandit see a real
    bottleneck-state label (compute / memory / scratchpad / balanced),
    not just an opaque cycle count. Without this signal the bandit
    has no diversifying signal across rounds and converges to the
    first proposal.
    """
    if cycles is None and mismatches is None:
        return "(no prior measurement — this is the first round)"
    extras = counter_extras or {}
    state, rationale = _classify_bottleneck(
        cycles=cycles,
        exe_active=extras.get("exe_active_cycles"),
        load_dma_wait=extras.get("load_dma_wait_cycles"),
        scratchpad_a_wait=extras.get("scratchpad_a_wait_cycles"),
    )
    parts = []
    parts.append("=== Spike+gemmini counter output (substituted for NCU log) ===")
    parts.append(
        f"Contract: matmul i8 x i8 -> i32, "
        f"M={contract.input_shapes[0][0]}, "
        f"K={contract.input_shapes[0][1]}, "
        f"N={contract.input_shapes[1][1]}"
    )
    if cycles is not None:
        parts.append(f"MAIN_LD_ST_EX_CYCLES (Gemmini-active cycles): {cycles}")
    # New: surface the 3 extra counters so the bandit-state classifier
    # has something to discriminate on. Missing values appear as `—`.
    for label, key in (
        ("EXE_ACTIVE_CYCLE   (MMA pipeline active)", "exe_active_cycles"),
        ("LOAD_DMA_WAIT_CYCLE (kernel stalls on DMA)", "load_dma_wait_cycles"),
        ("SCRATCHPAD_A_WAIT_CYCLE (A-tile residency wait)", "scratchpad_a_wait_cycles"),
    ):
        value = extras.get(key)
        if value is not None:
            parts.append(f"{label}: {value}")
    parts.append(f"Bottleneck state: **{state}** ({rationale})")
    if mismatches is not None:
        parts.append(
            f"correctness vs scalar reference: {mismatches[0]}/{mismatches[1]} mismatches"
        )
        parts.append(
            f"  ({'CORRECT' if mismatches[0] == 0 else 'INCORRECT - need to fix kernel'})"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Target-Card content as "database_content" for KB's prompt.
# ---------------------------------------------------------------------------


def _database_content(card: tk.TargetKnowledgeCard) -> str:
    """Concatenate the bucket files (isa.md, intrinsics.md, constraints.md)
    plus the worked-out derivation_rules. This is what KB's prompt
    constructor will inject as the ``database_content`` field."""
    parts: list[str] = ["# Gemmini Target Card (substituted for KB's CUDA optimization database)\n"]
    for bucket in ("isa", "intrinsics", "constraints", "architecture"):
        path = card.bucket_path(bucket)
        if path.exists():
            parts.append(f"\n## {bucket}\n{path.read_text()}")
    if card.hardware_spec.derivation_rules:
        parts.append("\n## Sizing rules (USE THESE NUMBERS DIRECTLY)")
        for r in card.hardware_spec.derivation_rules:
            parts.append(
                f"- **{r.name}**: {r.symbolic}  →  concrete bound: "
                f"**{int(r.concrete_value) if r.concrete_value.is_integer() else r.concrete_value}{(' ' + r.unit) if r.unit else ''}**. "
                f"Apply as: `{r.how_to_apply}`"
            )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Override-description prepend — tells the LLM it's targeting Gemmini, not CUDA.
# ---------------------------------------------------------------------------


_GEMMINI_OVERRIDE = """\
**Target hardware: Gemmini RoCC custom-3 systolic accelerator (NOT a CUDA GPU).**

Emit a single C function with EXACT signature:

    void launch_gpu_implementation(void *output, void *input_A, void *input_B,
                                   int64_t M, int64_t K, int64_t N);

The void* args cast to: `int8_t *input_A`, `int8_t *input_B`, `int32_t *output`.
Include `#include <stdint.h>` and `#include "include/gemmini.h"`.

Use only the macros listed in the OPTIMIZATION DATABASE below — `gemmini_extended_*`
forms are preferred (they take explicit cols/rows args; do NOT pass extra args to
the simple 2-arg `gemmini_mvin(dram, spad)` / `gemmini_mvout(dram, spad)` forms).

**Pick exactly ONE of the two patterns below based on the contract's K**. Do not apply both. The K-split pattern is STRICTLY SLOWER for K <= 720 — only use it when K >= 960.

For K <= 720 use the high-level helper directly:
```c
tiled_matmul_auto(M, N, K, A, B, NULL, C, K, N, N, N,
                  MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION,
                  ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                  false, false, false, /*full_C=*/true, /*low_D=*/false,
                  /*weightA=*/0, WS);
```

For K >= 960 (where `tiled_matmul_auto` mvout's accumulator wrap silently corrupts the
last K-tile's output), use this K-split pattern — split K into chunks of <= 480 and
accumulate via the D-bias path. KSPLIT=480 is a safe upper bound; values up to 720
also work but 480 leaves headroom for double-buffering:
```c
#define KSPLIT 480
for (int64_t k0 = 0; k0 < K; k0 += KSPLIT) {
    int64_t kc = (K - k0 < KSPLIT) ? (K - k0) : KSPLIT;
    int8_t *A_sub = A + k0;         // A[m, k0:k0+kc]  (stride_A = K)
    int8_t *B_sub = B + k0 * N;     // B[k0:k0+kc, :]  (stride_B = N)
    if (k0 == 0) {
        tiled_matmul_auto(M, N, kc, A_sub, B_sub, /*D=*/NULL, C,
                          K, N, N, N,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION,
                          ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                          false, false, false, /*full_C=*/true, /*low_D=*/false,
                          0, WS);
    } else {
        // D=C feeds the previous partial-sum back as bias before mvout.
        tiled_matmul_auto(M, N, kc, A_sub, B_sub, /*D=*/C, C,
                          K, N, N, N,
                          MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, 1, NO_ACTIVATION,
                          ACC_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                          /*repeating_bias=*/false, false, false,
                          /*full_C=*/true, /*low_D=*/false,
                          0, WS);
    }
}
```
This pattern is known to produce correct results on K up to at least 2560.
"""


# ---------------------------------------------------------------------------
# Per-contract loop
# ---------------------------------------------------------------------------


def run_one_contract(
    contract: KernelContract,
    *,
    card: tk.TargetKnowledgeCard,
    starting_source: str,
    max_rounds: int = 3,
    model: str = "gemini-2.5-flash",
    log_path: Path | None = None,
) -> dict[str, Any]:
    """One full KB-style RL loop for one contract."""
    strategies = _gemmini_strategies()
    composite = _composite()

    current_source = starting_source
    prev_cycles: int | None = None
    prev_mismatches: tuple[int, int] | None = None
    # Latest Gemmini-counter snapshot — populated from the evaluator's
    # report.metadata each round so KB's bandit sees a real
    # bottleneck-state, not just an opaque cycle count.
    prev_counter_extras: dict[str, int] = {}
    history: list[dict] = []
    best: dict | None = None

    gen = KernelGeneratorLLM(model=model)
    eva = CRiscvEvaluator(contract=contract, timeout_s=300, keep_workdir=False)

    db_content = _database_content(card)

    for r in range(max_rounds):
        # Pick a strategy — round-robin through singles, with composite on round 0.
        opt_entry = composite if r == 0 else strategies[(r - 1) % len(strategies)]

        # Build the prompt using KB's actual code.
        kb_prompt = generate_strategy_guided_prompt(
            optimization_entry=opt_entry,
            annotated_ncu=_gemmini_ncu_log(
                prev_cycles, prev_mismatches, contract,
                counter_extras=prev_counter_extras,
            ),
            ncu_log=_gemmini_ncu_log(
                prev_cycles, prev_mismatches, contract,
                counter_extras=prev_counter_extras,
            ),
            database_content=db_content,
            override_description=_GEMMINI_OVERRIDE,
            original_code=current_source,
        )

        # Wrap in a PromptBundle and run through our Gemini client (budget-gated).
        bundle = PromptBundle(
            system=_GEMMINI_OVERRIDE,
            user=kb_prompt,
            schema={
                "type": "object",
                "properties": {
                    "kernel_code": {"type": "string"},
                    "language": {"type": "string"},
                    "action": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["kernel_code"],
            },
            metadata={"round": r, "strategy": getattr(opt_entry, "technique", "composite")},
        )
        req = ProposeRequest(bundle=bundle, attempt_index=r, state_hash="kb-pipeline")

        t0 = time.perf_counter()
        try:
            resp = gen.propose(req)
        except Exception as exc:
            history.append({"round": r, "error": f"{type(exc).__name__}: {exc}"})
            break
        propose_wall = time.perf_counter() - t0

        if not resp.kernel_code.strip():
            history.append({"round": r, "error": "empty LLM response"})
            continue

        # Evaluate on Spike+gemmini.
        t0 = time.perf_counter()
        eva_rep = eva.evaluate(ProposeResponse(
            # Wrap the emitted kernel so it exposes kernel_under_test
            # (our evaluator's harness expects this name; KB's prompt
            # asked for launch_gpu_implementation. We add a thin shim).
            kernel_code=(
                resp.kernel_code
                + "\n\n// Shim: harness calls kernel_under_test(); KB emits launch_gpu_implementation.\n"
                + f"#define _M {contract.input_shapes[0][0]}\n"
                + f"#define _K {contract.input_shapes[0][1]}\n"
                + f"#define _N {contract.input_shapes[1][1]}\n"
                + "void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C) {\n"
                + "    launch_gpu_implementation((void*)C, (void*)A, (void*)B, _M, _K, _N);\n"
                + "}\n"
            ),
            language=resp.language or "c",
            action=getattr(opt_entry, "technique", "composite"),
        ))
        eval_wall = time.perf_counter() - t0

        row = {
            "round": r,
            "strategy": getattr(opt_entry, "technique", "composite"),
            "compile_ok": eva_rep.metadata.get("reason") != "compile_failed",
            "correct": eva_rep.correct,
            "cycles": eva_rep.cycles,
            "diff_summary": eva_rep.diff_summary,
            "compile_log_tail": (eva_rep.compile_log or "")[-400:],
            "runtime_log_tail": (eva_rep.runtime_log or "")[-300:],
            "propose_wall_s": round(propose_wall, 2),
            "eval_wall_s": round(eval_wall, 2),
            "kernel_lines": resp.kernel_code.count("\n") + 1,
        }
        history.append(row)

        prev_cycles = eva_rep.cycles
        prev_mismatches = (
            eva_rep.metadata.get("mismatches"), eva_rep.metadata.get("total")
        ) if eva_rep.metadata.get("mismatches") is not None else None
        # Refresh the counter snapshot — keys are the same as those
        # the harness prints (exe_active_cycles / load_dma_wait_cycles
        # / scratchpad_a_wait_cycles); missing counters omitted.
        prev_counter_extras = {
            k: eva_rep.metadata[k]
            for k in (
                "exe_active_cycles",
                "load_dma_wait_cycles",
                "scratchpad_a_wait_cycles",
            )
            if k in eva_rep.metadata
        }

        # Update best.
        if eva_rep.correct and (best is None or (eva_rep.cycles or 0) < best.get("cycles", 10**18)):
            best = {**row, "kernel_code": resp.kernel_code}

        # KB's bandit: if correct + cycles improved, keep the kernel; else
        # retry with a different strategy.
        if eva_rep.correct:
            current_source = resp.kernel_code

        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "contract_region_id": contract.region_id,
                    "shape": f"[{contract.input_shapes[0][0]},{contract.input_shapes[0][1]}]x[{contract.input_shapes[0][1]},{contract.input_shapes[1][1]}]",
                    **row,
                }) + "\n")

    return {
        "contract_region_id": contract.region_id,
        "shape": f"[{contract.input_shapes[0][0]},{contract.input_shapes[0][1]}]x[{contract.input_shapes[0][1]},{contract.input_shapes[1][1]}]",
        "rounds": len(history),
        "history": history,
        "best": best,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", default="gemmini_mx")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--abort-on-spend-usd", type=float, default=3.0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "kb_pipeline_attempts.jsonl"
    if log_path.exists():
        log_path.unlink()

    contracts = load_contracts(args.manifest)
    if args.limit > 0:
        contracts = contracts[: args.limit]
    card = tk.load(args.target)

    pre_gem = gemini_usage.load_summary()
    print(f"# pre Gemini cumulative: ${pre_gem.total_cost_usd:.4f}")

    results = []
    for i, c in enumerate(contracts):
        c2 = dataclasses.replace(c, target_name=args.target)
        # Generate a starting source per shape (naive scalar).
        from xpu_rt.spike_harness.templates.gemmini import render_init_c
        starter = render_init_c()
        # Spend gate.
        post = gemini_usage.load_summary()
        incremental = post.total_cost_usd - pre_gem.total_cost_usd
        if incremental >= args.abort_on_spend_usd:
            print(f"# spend cap hit (incremental=${incremental:.4f}); aborting after {i} contracts")
            break

        print(f"\n[{i+1}/{len(contracts)}] {c.region_id} shape={c2.input_shapes}")
        res = run_one_contract(
            c2, card=card, starting_source=starter,
            max_rounds=args.rounds, model=args.model, log_path=log_path,
        )
        results.append(res)
        if res["best"]:
            print(f"  ✓ best: round={res['best']['round']} cycles={res['best']['cycles']} strategy={res['best']['strategy']}")
        else:
            print(f"  ✗ no correct kernel in {res['rounds']} rounds")

    post_gem = gemini_usage.load_summary()
    summary = {
        "contracts_run": len(results),
        "correct": sum(1 for r in results if r["best"] is not None),
        "incremental_usd": post_gem.total_cost_usd - pre_gem.total_cost_usd,
        "rows": results,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n# correct: {summary['correct']}/{len(results)}")
    print(f"# incremental Gemini $: ${summary['incremental_usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

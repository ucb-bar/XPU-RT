"""Experiment 5: translation validation of the memory planner.

Runs :func:`translation_validate_memory_plan` against real planner
outputs on several workloads, with and without Z3, and emits a
calibration table comparing Z3 to plain-Python check time. Includes
a negative-control row that intentionally corrupts a clean solution
to demonstrate the TV catches the kind of bug it claims to catch.

Outputs:

* ``build/experiments/exp5_memory_tv/results.jsonl`` — one row per
  (workload, planner, use_z3) plus the negative control.
* ``build/experiments/exp5_memory_tv/summary.md`` — table + honest
  framing of where Z3 actually buys anything.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from xpu_rt.solve.memory_plan_tv import (
    TVResult,
    translation_validate_memory_plan,
)
from xpu_rt.solve.memory_planner import (
    BufferAllocation,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
    plan_memory,
)

try:
    from xpu_rt.solve.memory_planner_greedy import plan_memory_greedy

    HAVE_GREEDY = True
except Exception:  # noqa: BLE001
    HAVE_GREEDY = False


OUT_DIR = Path("build/experiments/exp5_memory_tv")


def simple_chain_buffers() -> MemoryPlanInput:
    """4 buffers with disjoint lifetimes on a single tier."""

    buffers = (
        BufferSpec("c0", 128, 0, 2, ("sram",), alignment=16),
        BufferSpec("c1", 128, 2, 4, ("sram",), alignment=16),
        BufferSpec("c2", 128, 4, 6, ("sram",), alignment=16),
        BufferSpec("c3", 128, 6, 8, ("sram",), alignment=16),
    )
    return MemoryPlanInput(
        buffers=buffers,
        tier_capacities=(TierCapacity("sram", capacity_bytes=4096),),
    )


def overlapping_kv_cache() -> MemoryPlanInput:
    """12 buffers across 2 tiers with attention-like overlap."""

    buffers: list[BufferSpec] = []
    # 4 K-cache, 4 V-cache buffers (long-lived, in SRAM).
    for i in range(4):
        buffers.append(BufferSpec(f"k{i}", 2048, 0, 16, ("sram", "dram"), alignment=64))
        buffers.append(BufferSpec(f"v{i}", 2048, 0, 16, ("sram", "dram"), alignment=64))
    # 4 attention scratch buffers (short, overlap with each other).
    for i in range(4):
        buffers.append(
            BufferSpec(f"attn_scratch{i}", 1024, i * 2, i * 2 + 4, ("sram",), alignment=64)
        )
    return MemoryPlanInput(
        buffers=tuple(buffers),
        tier_capacities=(
            TierCapacity("sram", capacity_bytes=64 * 1024, weight=1.0),
            TierCapacity("dram", capacity_bytes=1024 * 1024, weight=4.0),
        ),
    )


def random_stress(n: int = 100, seed: int = 0) -> MemoryPlanInput:
    """``n`` buffers with random sizes / lifetimes on a single tier."""

    rng = random.Random(seed)
    buffers: list[BufferSpec] = []
    horizon = 64
    for i in range(n):
        size = rng.choice([64, 128, 256, 512, 1024])
        start = rng.randint(0, horizon - 4)
        end = start + rng.randint(1, 12)
        buffers.append(BufferSpec(f"r{i}", size, start, end, ("sram",), alignment=64))
    return MemoryPlanInput(
        buffers=tuple(buffers),
        tier_capacities=(TierCapacity("sram", capacity_bytes=1024 * 1024),),
    )


def pathological_aliasing(n: int = 20) -> MemoryPlanInput:
    """Interleaved lifetimes maximizing disjointness obligations."""

    buffers: list[BufferSpec] = []
    for i in range(n):
        # Two groups with heavily overlapping live windows.
        start = i % 4
        end = start + (n // 2 if i % 2 == 0 else n)
        buffers.append(
            BufferSpec(f"p{i}", 256 + (i * 16), start, end, ("sram",), alignment=32)
        )
    return MemoryPlanInput(
        buffers=tuple(buffers),
        tier_capacities=(TierCapacity("sram", capacity_bytes=2 * 1024 * 1024),),
    )


WORKLOADS_QUICK: list[tuple[str, Callable[[], MemoryPlanInput]]] = [
    ("simple_chain_buffers", simple_chain_buffers),
    ("overlapping_kv_cache", overlapping_kv_cache),
    ("pathological_aliasing_small", lambda: pathological_aliasing(n=10)),
]

WORKLOADS_FULL: list[tuple[str, Callable[[], MemoryPlanInput]]] = [
    ("simple_chain_buffers", simple_chain_buffers),
    ("overlapping_kv_cache", overlapping_kv_cache),
    ("random_stress_100", lambda: random_stress(n=100, seed=0)),
    ("pathological_aliasing_20", lambda: pathological_aliasing(n=20)),
]


def _run_planner(
    planner_name: str,
    problem: MemoryPlanInput,
) -> tuple[MemoryPlanSolved | None, str]:
    """Returns (plan, status_word). plan is None when the planner fails."""

    if planner_name == "milp":
        response, plan = plan_memory(problem)
        return plan, response.status.value
    if planner_name == "greedy":
        if not HAVE_GREEDY:
            return None, "unavailable"
        # plan_memory_greedy returns the plan directly (no SolverResponse envelope).
        plan = plan_memory_greedy(problem)
        return plan, plan.status
    raise ValueError(f"unknown planner: {planner_name}")


def _row(
    *,
    workload: str,
    planner: str,
    use_z3: bool,
    tv: TVResult,
    n_buffers: int,
    note: str = "",
) -> dict:
    z3_ratio: float | None
    if tv.python_time_ms > 0 and tv.z3_time_ms > 0:
        z3_ratio = tv.z3_time_ms / tv.python_time_ms
    else:
        z3_ratio = None
    return {
        "workload": workload,
        "planner": planner,
        "use_z3": use_z3,
        "n_buffers": n_buffers,
        "n_pairs_checked": tv.n_pairs_checked,
        "proved": tv.proved,
        "n_violations": len(tv.violations),
        "violations": [asdict(v) for v in tv.violations[:5]],
        "z3_time_ms": tv.z3_time_ms,
        "python_time_ms": tv.python_time_ms,
        "z3_overhead_ratio": z3_ratio,
        "note": note,
    }


def _negative_control(problem: MemoryPlanInput, solution: MemoryPlanSolved) -> dict:
    """Force an overlap between the first two same-tier overlapping buffers."""

    # Find a pair that shares a tier and has overlapping lifetimes.
    spec_by_id = {b.buffer_id: b for b in problem.buffers}
    by_tier: dict[str, list[BufferAllocation]] = {}
    for a in solution.buffers:
        by_tier.setdefault(a.tier, []).append(a)

    corrupt_buffers = list(solution.buffers)
    target_idx: tuple[int, int] | None = None
    for tier_id, allocs in by_tier.items():
        ids = [a.buffer_id for a in allocs]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_spec = spec_by_id[ids[i]]
                b_spec = spec_by_id[ids[j]]
                # Half-open overlap.
                if (
                    a_spec.lifetime_start < b_spec.lifetime_end
                    and b_spec.lifetime_start < a_spec.lifetime_end
                ):
                    # Find indices in the full buffers tuple.
                    idx_a = next(
                        k for k, x in enumerate(corrupt_buffers) if x.buffer_id == ids[i]
                    )
                    idx_b = next(
                        k for k, x in enumerate(corrupt_buffers) if x.buffer_id == ids[j]
                    )
                    target_idx = (idx_a, idx_b)
                    break
            if target_idx:
                break
        if target_idx:
            break

    if target_idx is None:
        return {
            "workload": "negative_control",
            "planner": "n/a",
            "note": "no overlapping pair to corrupt",
            "proved": None,
        }

    idx_a, idx_b = target_idx
    # Force identical offset → guaranteed byte-range overlap.
    corrupt_buffers[idx_b] = replace(
        corrupt_buffers[idx_b],
        offset_bytes=corrupt_buffers[idx_a].offset_bytes,
        aliases_with=None,
    )
    corrupt = replace(solution, buffers=tuple(corrupt_buffers))
    tv = translation_validate_memory_plan(problem, corrupt, use_z3=True)
    row = _row(
        workload="negative_control_corrupted",
        planner="injected",
        use_z3=True,
        tv=tv,
        n_buffers=len(problem.buffers),
        note=f"forced overlap on {corrupt_buffers[idx_a].buffer_id}/{corrupt_buffers[idx_b].buffer_id}",
    )
    row["caught_buffer_overlap"] = any(v["kind"] == "buffer_overlap" for v in row["violations"])
    return row


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# Exp 5: Memory-Planner Translation Validation\n")
    lines.append(
        "Validates that every concrete `MemoryPlanSolved` honors its `MemoryPlanInput`:\n"
        "(1) overlapping-lifetime buffers in the same tier have disjoint byte ranges,\n"
        "(2) per-tier peak does not exceed declared capacity,\n"
        "(3) offsets respect alignment, (4) `fixed_assignments` are honored.\n"
    )
    lines.append("## Per-workload TV verdicts\n")
    lines.append("| workload | planner | use_z3 | n_buffers | n_pairs | proved | n_viol | z3_ms | py_ms | z3/py |")
    lines.append("|---|---|---|---:|---:|:---:|---:|---:|---:|---:|")
    for r in rows:
        if r.get("workload", "").startswith("negative_control"):
            continue
        ratio = r.get("z3_overhead_ratio")
        ratio_s = f"{ratio:.1f}x" if isinstance(ratio, (int, float)) else "-"
        lines.append(
            f"| {r['workload']} | {r['planner']} | {r['use_z3']} | "
            f"{r['n_buffers']} | {r['n_pairs_checked']} | "
            f"{'Y' if r['proved'] else 'N'} | {r['n_violations']} | "
            f"{r['z3_time_ms']:.2f} | {r['python_time_ms']:.3f} | {ratio_s} |"
        )

    lines.append("\n## Calibration: Z3 vs plain-Python check time\n")
    z3_rows = [r for r in rows if r.get("use_z3") and r.get("python_time_ms", 0) > 0]
    if z3_rows:
        ratios = [
            r["z3_overhead_ratio"]
            for r in z3_rows
            if isinstance(r.get("z3_overhead_ratio"), (int, float))
        ]
        if ratios:
            lines.append(
                f"Z3/Python overhead ratio: min={min(ratios):.1f}x, "
                f"median={sorted(ratios)[len(ratios) // 2]:.1f}x, max={max(ratios):.1f}x "
                f"(n={len(ratios)}).\n"
            )

    lines.append("## Negative control (intentionally corrupted solution)\n")
    neg = [r for r in rows if r.get("workload", "").startswith("negative_control")]
    if neg:
        for r in neg:
            if r.get("proved") is None:
                lines.append(f"- skipped: {r.get('note')}\n")
                continue
            caught = r.get("caught_buffer_overlap")
            lines.append(
                f"- corrupted `{r.get('note')}`: proved={r.get('proved')}, "
                f"n_violations={r.get('n_violations')}, "
                f"caught_buffer_overlap={caught}.\n"
            )

    lines.append("## Honest framing\n")
    lines.append(
        "For concrete solver output with fixed integer offsets / sizes, every TV check is a\n"
        "trivial integer comparison. A plain Python `assert` is one to two orders of magnitude\n"
        "cheaper than building and discharging Z3 contexts per pair (see ratio above). The value\n"
        "of routing through Z3 here is:\n\n"
        "1. **Uniform counterexample shape.** Every solver-output bug surfaces as the same\n"
        "   structured `{kind, detail, z3_model}` payload that the rest of the obligation\n"
        "   machinery already consumes.\n"
        "2. **Forward-compat with parametric buffers.** When dynamic shapes show up the same\n"
        "   query becomes non-trivial — Python `assert` no longer suffices, Z3 does.\n"
        "3. **Drop-in obligation for `solve_request`.** Today the memory planner exits without\n"
        "   raising a `TRANSLATION_VALIDATION` obligation; this experiment shows what such a\n"
        "   pass would look like end-to-end and how much it costs.\n\n"
        "Whether to wire this into the planner's production exit path is a separate decision —\n"
        "this script only validates that the check itself is correct (positive cases) and\n"
        "catches what it claims to catch (negative control).\n"
    )

    (out_dir / "summary.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run only the small workloads")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    workloads = WORKLOADS_QUICK if args.quick else WORKLOADS_FULL

    rows: list[dict] = []
    last_clean: tuple[MemoryPlanInput, MemoryPlanSolved] | None = None
    planners = ["milp"] + (["greedy"] if HAVE_GREEDY else [])

    for wl_name, factory in workloads:
        problem = factory()
        for planner_name in planners:
            plan_t0 = time.perf_counter()
            plan, status = _run_planner(planner_name, problem)
            plan_time_ms = (time.perf_counter() - plan_t0) * 1000.0
            if plan is None:
                rows.append(
                    {
                        "workload": wl_name,
                        "planner": planner_name,
                        "planner_status": status,
                        "planner_time_ms": plan_time_ms,
                        "note": "planner did not return a solution",
                    }
                )
                continue
            for use_z3 in (True, False):
                tv = translation_validate_memory_plan(problem, plan, use_z3=use_z3)
                row = _row(
                    workload=wl_name,
                    planner=planner_name,
                    use_z3=use_z3,
                    tv=tv,
                    n_buffers=len(problem.buffers),
                )
                row["planner_status"] = status
                row["planner_time_ms"] = plan_time_ms
                rows.append(row)
            last_clean = (problem, plan)

    if last_clean is not None:
        problem, plan = last_clean
        rows.append(_negative_control(problem, plan))

    with (OUT_DIR / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    _write_summary(rows, OUT_DIR)
    print(f"wrote {OUT_DIR}/results.jsonl and summary.md ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

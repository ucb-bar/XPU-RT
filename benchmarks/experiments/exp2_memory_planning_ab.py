"""Memory planning A/B: greedy first-fit vs MILP.

Runs a synthetic + realistic workload set through the greedy planner
(:mod:`xpu_rt.solve.memory_planner_greedy`) and the MILP planner
(:mod:`xpu_rt.solve.memory_planner`) and writes a side-by-side report
to ``build/experiments/exp2/``.

Usage:
    uv run python scripts/experiments/exp2_memory_planning_ab.py --quick
    uv run python scripts/experiments/exp2_memory_planning_ab.py
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from xpu_rt.solve.memory_planner import (
    AliasCandidate,
    BufferSpec,
    MemoryPlanInput,
    MemoryPlanSolved,
    TierCapacity,
    plan_memory,
)
from xpu_rt.solve.memory_planner_greedy import plan_memory_greedy
from xpu_rt.solve.solver_types import SolverStatus


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "build" / "experiments" / "exp2"


@dataclass(frozen=True)
class WorkloadResult:
    workload: str
    planner: str
    status: str
    feasible: bool
    objective_value: float | None
    tier_peak_usage: dict[str, int]
    total_peak: int
    aliases_count: int
    fragmentation: int
    solve_time_ms: float


# ---------------------------------------------------------------- workloads


_ALIGN = 64


def simple_mlp_buffers() -> MemoryPlanInput:
    KB = 1024
    MB = 1024 * KB
    return MemoryPlanInput(
        buffers=(
            BufferSpec("W0", 64 * 128 * 4, 0, 6, ("scratch", "host"), alignment=_ALIGN, spill_cost=4.0),
            BufferSpec("W1", 128 * 32 * 4, 0, 6, ("scratch", "host"), alignment=_ALIGN, spill_cost=4.0),
            BufferSpec("x", 64 * 4, 0, 1, ("scratch", "host"), alignment=_ALIGN, spill_cost=2.0),
            BufferSpec("h0", 128 * 4, 1, 3, ("scratch", "host"), alignment=_ALIGN, spill_cost=3.0),
            BufferSpec("h1", 128 * 4, 2, 4, ("scratch", "host"), alignment=_ALIGN, spill_cost=3.0),
            BufferSpec("y", 32 * 4, 4, 6, ("scratch", "host"), alignment=_ALIGN, spill_cost=2.0),
        ),
        tier_capacities=(
            TierCapacity("scratch", capacity_bytes=64 * KB, weight=1.0),
            TierCapacity("host", capacity_bytes=64 * MB, weight=10.0),
        ),
        alias_candidates=(AliasCandidate("h0", "h1"),),
        objective_lambda=1e-6,
    )


def transformer_block_buffers(layers: int = 12) -> MemoryPlanInput:
    # Sizes are reduced by 1024 from a real GPT-style block so the
    # MILP's canonical post-pass (byte-granular first-fit walk) finishes
    # in a reasonable time. Relative ordering and lifetime overlap are
    # preserved, which is what the A/B comparison actually measures.
    KB = 1024
    MB = 1024 * KB
    d = 128
    seq = 64
    head_bytes = 4
    bufs: list[BufferSpec] = []
    aliases: list[AliasCandidate] = []

    bufs.append(
        BufferSpec(
            "kv_cache",
            size_bytes=2 * layers * seq * d * head_bytes,
            lifetime_start=0,
            lifetime_end=layers * 10 + 5,
            allowed_tiers=("scratch", "hbm"),
            alignment=_ALIGN,
            spill_cost=5.0,
        )
    )
    for L in range(layers):
        t0 = L * 10
        bufs.append(
            BufferSpec(
                f"W_qkv_{L}",
                3 * d * d * head_bytes,
                0,
                layers * 10 + 5,
                ("scratch", "hbm"),
                alignment=_ALIGN,
                spill_cost=2.0,
            )
        )
        bufs.append(
            BufferSpec(
                f"W_ffn_{L}",
                4 * d * d * head_bytes,
                0,
                layers * 10 + 5,
                ("scratch", "hbm"),
                alignment=_ALIGN,
                spill_cost=2.0,
            )
        )
        q = BufferSpec(f"Q_{L}", seq * d * head_bytes, t0, t0 + 3, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0)
        k = BufferSpec(f"K_{L}", seq * d * head_bytes, t0, t0 + 3, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0)
        v = BufferSpec(f"V_{L}", seq * d * head_bytes, t0, t0 + 3, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0)
        attn = BufferSpec(
            f"attn_{L}", seq * seq * head_bytes, t0 + 1, t0 + 4, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=10.0
        )
        ctx = BufferSpec(f"ctx_{L}", seq * d * head_bytes, t0 + 3, t0 + 5, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0)
        ffn_mid = BufferSpec(
            f"ffn_mid_{L}", seq * 4 * d * head_bytes, t0 + 5, t0 + 8, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0
        )
        out = BufferSpec(f"out_{L}", seq * d * head_bytes, t0 + 8, t0 + 11, ("scratch", "hbm"), alignment=_ALIGN, spill_cost=8.0)
        bufs.extend([q, k, v, attn, ctx, ffn_mid, out])
        aliases.append(AliasCandidate(q.buffer_id, ctx.buffer_id))
        aliases.append(AliasCandidate(ffn_mid.buffer_id, out.buffer_id))

    return MemoryPlanInput(
        buffers=tuple(bufs),
        tier_capacities=(
            TierCapacity("scratch", capacity_bytes=4 * MB, weight=1.0),
            TierCapacity("hbm", capacity_bytes=128 * MB, weight=4.0),
        ),
        alias_candidates=tuple(aliases),
        objective_lambda=1e-9,
        time_budget_ms=10_000,
    )


def random_overlap_stress(
    n_buffers: int = 200,
    overlap_density: float = 0.6,
    seed: int = 0,
) -> MemoryPlanInput:
    KB = 1024
    rng = random.Random(seed)
    horizon = max(10, int(n_buffers / max(overlap_density, 0.05)))
    bufs: list[BufferSpec] = []
    for i in range(n_buffers):
        size = rng.choice([1 * KB, 2 * KB, 4 * KB, 8 * KB, 16 * KB])
        start = rng.randint(0, horizon - 1)
        lifetime_extent = max(1, int(horizon * overlap_density))
        end = min(horizon, start + rng.randint(1, lifetime_extent))
        bufs.append(
            BufferSpec(
                buffer_id=f"b{i:03d}",
                size_bytes=size,
                lifetime_start=start,
                lifetime_end=end,
                allowed_tiers=("small", "medium", "large"),
                alignment=_ALIGN,
                spill_cost=1.0 + rng.random() * 4.0,
            )
        )
    return MemoryPlanInput(
        buffers=tuple(bufs),
        tier_capacities=(
            TierCapacity("small", capacity_bytes=256 * KB, weight=1.0),
            TierCapacity("medium", capacity_bytes=2 * KB * KB, weight=2.5),
            TierCapacity("large", capacity_bytes=64 * KB * KB, weight=8.0),
        ),
        objective_lambda=1e-9,
        time_budget_ms=10_000,
    )


def pathological_fragmentation_case() -> MemoryPlanInput:
    KB = 1024
    bufs: list[BufferSpec] = []
    for i in range(5):
        bufs.append(
            BufferSpec(
                f"big_{i}",
                size_bytes=64 * KB,
                lifetime_start=i * 10,
                lifetime_end=i * 10 + 3,
                allowed_tiers=("only",),
                alignment=_ALIGN,
                spill_cost=10.0,
            )
        )
    for j in range(50):
        small_size = (1 + (j % 4)) * 64
        start = j  # interleaves with bigs
        end = start + 100
        bufs.append(
            BufferSpec(
                f"small_{j}",
                size_bytes=small_size,
                lifetime_start=start,
                lifetime_end=end,
                allowed_tiers=("only",),
                alignment=_ALIGN,
                spill_cost=1.0,
            )
        )
    return MemoryPlanInput(
        buffers=tuple(bufs),
        tier_capacities=(TierCapacity("only", capacity_bytes=2 * KB * KB, weight=1.0),),
        objective_lambda=1e-9,
        time_budget_ms=10_000,
    )


WORKLOADS_FAST: list[tuple[str, Callable[[], MemoryPlanInput]]] = [
    ("simple_mlp", simple_mlp_buffers),
    ("transformer_block_x12", lambda: transformer_block_buffers(layers=12)),
    ("random_overlap_dense_seed0", lambda: random_overlap_stress(80, 0.6, 0)),
]
WORKLOADS_FULL: list[tuple[str, Callable[[], MemoryPlanInput]]] = [
    ("simple_mlp", simple_mlp_buffers),
    ("transformer_block_x12", lambda: transformer_block_buffers(layers=12)),
    ("random_overlap_n200_d0.6", lambda: random_overlap_stress(200, 0.6, 0)),
    ("random_overlap_n200_d0.3", lambda: random_overlap_stress(200, 0.3, 1)),
    ("pathological_fragmentation", pathological_fragmentation_case),
]


# ---------------------------------------------------------------- runners


def _max_concurrent_live_bytes(problem: MemoryPlanInput, tier: str, allocations: dict[str, str]) -> int:
    events: list[tuple[int, int, int]] = []
    for b in problem.buffers:
        if allocations.get(b.buffer_id) != tier:
            continue
        events.append((b.lifetime_start, 0, b.size_bytes))
        events.append((b.lifetime_end + 1, 1, -b.size_bytes))
    events.sort()
    live = 0
    peak = 0
    for _, _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def _summarize(
    name: str,
    planner: str,
    problem: MemoryPlanInput,
    plan: MemoryPlanSolved | None,
    elapsed_ms: float,
    status: str,
) -> WorkloadResult:
    if plan is None or plan.status == "infeasible":
        return WorkloadResult(
            workload=name,
            planner=planner,
            status=status,
            feasible=False,
            objective_value=None,
            tier_peak_usage={},
            total_peak=0,
            aliases_count=0,
            fragmentation=0,
            solve_time_ms=elapsed_ms,
        )
    allocations = {a.buffer_id: a.tier for a in plan.buffers}
    fragmentation = 0
    for tier_id, peak in plan.tier_peak_usage.items():
        live = _max_concurrent_live_bytes(problem, tier_id, allocations)
        fragmentation += max(0, peak - live)
    return WorkloadResult(
        workload=name,
        planner=planner,
        status=status,
        feasible=True,
        objective_value=plan.objective_value,
        tier_peak_usage=dict(plan.tier_peak_usage),
        total_peak=sum(plan.tier_peak_usage.values()),
        aliases_count=sum(1 for a in plan.buffers if a.aliases_with),
        fragmentation=fragmentation,
        solve_time_ms=elapsed_ms,
    )


def run_greedy(name: str, problem: MemoryPlanInput) -> WorkloadResult:
    t0 = time.perf_counter()
    plan = plan_memory_greedy(problem)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return _summarize(name, "greedy_first_fit", problem, plan, elapsed, plan.status)


def run_milp(name: str, problem: MemoryPlanInput) -> WorkloadResult:
    t0 = time.perf_counter()
    response, plan = plan_memory(problem, problem_id=f"exp2_{name}")
    elapsed = (time.perf_counter() - t0) * 1000.0
    status = response.status.value if isinstance(response.status, SolverStatus) else str(response.status)
    return _summarize(name, "milp", problem, plan, elapsed, status)


# ---------------------------------------------------------------- output


def _fmt_bytes(n: int) -> str:
    for unit, factor in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= factor:
            return f"{n / factor:.2f} {unit}"
    return f"{n} B"


def write_summary(results: list[WorkloadResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_workload: dict[str, dict[str, WorkloadResult]] = {}
    for r in results:
        by_workload.setdefault(r.workload, {})[r.planner] = r

    lines: list[str] = []
    lines.append("# Experiment 2: greedy vs MILP memory planning\n")
    lines.append(
        "One row per (workload, planner). `total_peak` sums tier peaks; "
        "`reduction%` is (greedy - milp)/greedy of total_peak.\n"
    )
    lines.append("| workload | planner | status | total_peak | aliases | fragmentation | solve_ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for wl in by_workload:
        for planner in ("greedy_first_fit", "milp"):
            r = by_workload[wl].get(planner)
            if r is None:
                continue
            lines.append(
                f"| {wl} | {planner} | {r.status} | {_fmt_bytes(r.total_peak)} | "
                f"{r.aliases_count} | {_fmt_bytes(r.fragmentation)} | {r.solve_time_ms:.1f} |"
            )

    lines.append("\n## Headline: peak-memory reduction (MILP vs greedy)\n")
    lines.append("| workload | greedy total_peak | milp total_peak | reduction% | solve_ms greedy | solve_ms milp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    big_wins: list[tuple[str, float]] = []
    for wl, slots in by_workload.items():
        g = slots.get("greedy_first_fit")
        m = slots.get("milp")
        if g is None or m is None or not g.feasible or not m.feasible:
            lines.append(
                f"| {wl} | "
                f"{_fmt_bytes(g.total_peak) if g and g.feasible else 'n/a'} | "
                f"{_fmt_bytes(m.total_peak) if m and m.feasible else 'n/a'} | "
                f"n/a | {g.solve_time_ms if g else 0:.1f} | {m.solve_time_ms if m else 0:.1f} |"
            )
            continue
        if g.total_peak == 0:
            reduction = 0.0
        else:
            reduction = 100.0 * (g.total_peak - m.total_peak) / g.total_peak
        if reduction > 10.0:
            big_wins.append((wl, reduction))
        lines.append(
            f"| {wl} | {_fmt_bytes(g.total_peak)} | {_fmt_bytes(m.total_peak)} | "
            f"{reduction:+.1f}% | {g.solve_time_ms:.1f} | {m.solve_time_ms:.1f} |"
        )

    if big_wins:
        lines.append("\n## Where MILP is worth it (>10% peak reduction)\n")
        for wl, red in sorted(big_wins, key=lambda kv: -kv[1]):
            lines.append(f"- **{wl}**: {red:.1f}% peak reduction")
    else:
        lines.append("\n## Where MILP is worth it\n\nNone of the run workloads showed > 10% peak reduction.\n")

    lines.append("\n## Caveats\n")
    lines.append("- QNN island demo workload skipped: `qrb5165_costs.json` -> `BufferSpec` translation is non-trivial; out of scope for this experiment.")
    lines.append("- Aliasing disabled in the greedy baseline by design (it is the MILP's advantage).")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def write_jsonl(results: list[WorkloadResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "workload": r.workload,
                        "planner": r.planner,
                        "status": r.status,
                        "feasible": r.feasible,
                        "objective_value": r.objective_value,
                        "tier_peak_usage": r.tier_peak_usage,
                        "total_peak_bytes": r.total_peak,
                        "aliases_count": r.aliases_count,
                        "fragmentation_bytes": r.fragmentation,
                        "solve_time_ms": r.solve_time_ms,
                    }
                )
                + "\n"
            )


def maybe_plot(results: list[WorkloadResult], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_workload: dict[str, dict[str, WorkloadResult]] = {}
    for r in results:
        by_workload.setdefault(r.workload, {})[r.planner] = r
    labels: list[str] = []
    reductions: list[float] = []
    for wl, slots in by_workload.items():
        g = slots.get("greedy_first_fit")
        m = slots.get("milp")
        if not (g and m and g.feasible and m.feasible and g.total_peak > 0):
            continue
        labels.append(wl)
        reductions.append(100.0 * (g.total_peak - m.total_peak) / g.total_peak)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(len(labels)))
    ax.bar(positions, reductions, color="#4c72b0")
    ax.set_ylabel("Peak memory reduction (%, MILP vs greedy)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "peak_reduction_vs_overlap.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip large stress workloads.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    workloads = WORKLOADS_FAST if args.quick else WORKLOADS_FULL
    results: list[WorkloadResult] = []
    for name, builder in workloads:
        print(f"[exp2] building workload: {name}")
        problem = builder()
        print(f"[exp2] running greedy: {name} ({len(problem.buffers)} buffers)")
        results.append(run_greedy(name, problem))
        print(f"[exp2] running milp:   {name}")
        results.append(run_milp(name, problem))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(results, args.output_dir)
    write_summary(results, args.output_dir)
    maybe_plot(results, args.output_dir)
    print(f"[exp2] wrote {args.output_dir}/summary.md and results.jsonl")


if __name__ == "__main__":
    main()

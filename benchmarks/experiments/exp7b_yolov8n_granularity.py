"""Experiment 7b: yolov8n granularity sweep on real QNN costs.

Chunks the 273-op ``yolov8n`` chain DAG into
``K in {1, 2, 4, 8, 16, 32, 273}`` topologically-consecutive groups
and runs the same three solvers (greedy, CP-SAT joint, MOSEK MILP) at
each K. Plots predicted makespan vs K and solver wall time vs K.

The op-level helpers (load, chain DAG, solver wrappers) are imported
from :mod:`exp7_real_perop_scheduling` so the two scripts share one
implementation.

Usage:
    uv run python scripts/experiments/exp7b_yolov8n_granularity.py [--quick]

``--quick`` runs ``K in {1, 4, 16, 32}`` with ``k_lookahead=1`` only.
The default sweeps ``K in {1, 2, 4, 8, 16, 32, 273}`` × ``{1, 4}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "xpu-rt" / "python"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments.exp7_real_perop_scheduling import (  # noqa: E402
    MOSEK_MAX_N,
    cpsat_joint,
    greedy_schedule,
    load_e2e_baselines,
    mosek_milp,
)
from xpu_rt.scheduler.qnn_real_workload import (  # noqa: E402
    BACKENDS,
    chunk_dag,
    load_cost_matrix,
    make_chain_dag,
)

COST_MATRIX_PATH = REPO_ROOT / "xpu-rt" / "data" / "profiled" / "qnn_cost_matrix.json"
OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp7b_granularity"

K_VALUES_FULL: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 273)
K_VALUES_QUICK: tuple[int, ...] = (1, 4, 16, 32)


@dataclass
class Exp7bResult:
    """One solver run for one (K, k_lookahead) cell."""

    K: int
    k_lookahead: int
    solver: str
    n_partitions: int
    makespan_us: float
    solver_time_ms: float
    feasible: bool
    status: str
    note: str = ""


def run_sweep(
    *,
    k_values: tuple[int, ...],
    lookaheads: tuple[int, ...],
    cpsat_timeout_ms: int,
    mosek_time_limit_s: float,
) -> list[Exp7bResult]:
    """Run the K × lookahead × solver sweep on yolov8n."""
    matrix = load_cost_matrix(COST_MATRIX_PATH)
    results: list[Exp7bResult] = []

    for kl in lookaheads:
        op_dag = make_chain_dag("yolov8n", matrix, k_lookahead=kl)
        for K in k_values:
            chunked = chunk_dag(op_dag, K)
            n = len(chunked.partition_ids)
            print(f"[exp7b] K={K:>3} k_lookahead={kl} n_partitions={n}")

            g = greedy_schedule(chunked)
            results.append(
                Exp7bResult(
                    K=K,
                    k_lookahead=kl,
                    solver="greedy",
                    n_partitions=n,
                    makespan_us=g.makespan_us,
                    solver_time_ms=g.solver_time_ms,
                    feasible=g.feasible,
                    status=g.status,
                    note=g.note,
                )
            )
            print(
                f"  greedy      makespan={g.makespan_us:>12.1f} µs "
                f"time={g.solver_time_ms:>7.2f} ms"
            )

            c = cpsat_joint(chunked, timeout_ms=cpsat_timeout_ms)
            results.append(
                Exp7bResult(
                    K=K,
                    k_lookahead=kl,
                    solver="cpsat_joint",
                    n_partitions=n,
                    makespan_us=c.makespan_us,
                    solver_time_ms=c.solver_time_ms,
                    feasible=c.feasible,
                    status=c.status,
                    note=c.note,
                )
            )
            print(
                f"  cpsat_joint makespan={c.makespan_us:>12.1f} µs "
                f"time={c.solver_time_ms:>7.2f} ms status={c.status}"
            )

            if n <= MOSEK_MAX_N:
                m = mosek_milp(chunked, time_limit_s=mosek_time_limit_s)
                results.append(
                    Exp7bResult(
                        K=K,
                        k_lookahead=kl,
                        solver="mosek_milp",
                        n_partitions=n,
                        makespan_us=m.makespan_us,
                        solver_time_ms=m.solver_time_ms,
                        feasible=m.feasible,
                        status=m.status,
                        note=m.note,
                    )
                )
                print(
                    f"  mosek_milp  makespan={m.makespan_us:>12.1f} µs "
                    f"time={m.solver_time_ms:>7.2f} ms status={m.status}"
                )
            else:
                results.append(
                    Exp7bResult(
                        K=K,
                        k_lookahead=kl,
                        solver="mosek_milp",
                        n_partitions=n,
                        makespan_us=float("inf"),
                        solver_time_ms=0.0,
                        feasible=False,
                        status="skipped",
                        note=f"n_partitions={n} > {MOSEK_MAX_N}",
                    )
                )
                print(f"  mosek_milp  skipped (n_partitions={n} > {MOSEK_MAX_N})")
    return results


def write_jsonl(results: list[Exp7bResult], path: Path) -> None:
    """Persist sweep results as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def _fmt_us(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:,.1f}"


def _fmt_ms(v: float) -> str:
    return f"{v:,.2f}"


def write_summary(
    results: list[Exp7bResult],
    path: Path,
    *,
    quick: bool,
    e2e: dict[str, dict[str, float]],
) -> None:
    """Render the one-page Markdown summary for A2."""
    by_lookahead: dict[int, dict[int, dict[str, Exp7bResult]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in results:
        by_lookahead[r.k_lookahead][r.K][r.solver] = r

    lines: list[str] = []
    lines.append("# Experiment 7b — yolov8n granularity sweep on real QNN costs\n\n")
    lines.append(f"Mode: {'quick' if quick else 'full'}\n\n")
    lines.append(
        "Solvers compared at each K: greedy (EFT), CP-SAT joint, MOSEK MILP. "
        "Backends (device-index order): "
        + ", ".join(BACKENDS)
        + ". Cross-backend transfer penalty: 100 µs.\n\n"
    )

    cpu_e2e = e2e.get("yolov8n", {}).get("CPU", float("inf"))
    lines.append(
        f"Best single-backend E2E baseline (yolov8n): **CPU = {cpu_e2e:,.0f} µs**.\n\n"
    )

    for kl in sorted(by_lookahead):
        lines.append(f"## k_lookahead = {kl}\n\n")
        lines.append(
            "| K | n_parts | greedy µs | greedy ms | CP-SAT µs | CP-SAT ms | CP-SAT status | MOSEK µs | MOSEK ms | MOSEK status |\n"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---|---:|---:|---|\n")
        for K in sorted(by_lookahead[kl]):
            bucket = by_lookahead[kl][K]
            g = bucket.get("greedy")
            c = bucket.get("cpsat_joint")
            m = bucket.get("mosek_milp")
            n_parts = (g or c or m).n_partitions
            lines.append(
                "| {K} | {n} | {gms} | {gt} | {cms} | {ct} | {cstat} | {mms} | {mt} | {mstat} |\n".format(
                    K=K,
                    n=n_parts,
                    gms=_fmt_us(g.makespan_us if (g and g.feasible) else float("inf")),
                    gt=_fmt_ms(g.solver_time_ms if g else 0.0),
                    cms=_fmt_us(c.makespan_us if (c and c.feasible) else float("inf")),
                    ct=_fmt_ms(c.solver_time_ms if c else 0.0),
                    cstat=(c.status if c else "n/a"),
                    mms=_fmt_us(m.makespan_us if (m and m.feasible) else float("inf")),
                    mt=_fmt_ms(m.solver_time_ms if (m and m.status != "skipped") else 0.0),
                    mstat=(m.status if m else "n/a") + (
                        f" ({m.note})" if (m and m.note) else ""
                    ),
                )
            )
        lines.append("\n")

    # Compute headlines per lookahead.
    lines.append("## Headline\n\n")
    for kl in sorted(by_lookahead):
        per_K = by_lookahead[kl]
        # Sweet-spot K (across all solvers) — minimum feasible makespan.
        best_K = None
        best_makespan = float("inf")
        best_solver = "n/a"
        for K in sorted(per_K):
            for solver, r in per_K[K].items():
                if r.feasible and math.isfinite(r.makespan_us) and r.makespan_us < best_makespan:
                    best_makespan = r.makespan_us
                    best_K = K
                    best_solver = solver
        # MOSEK cutoff: largest K with feasible MOSEK + smallest K skipped.
        mosek_largest_ok = None
        mosek_smallest_skipped = None
        for K in sorted(per_K):
            m = per_K[K].get("mosek_milp")
            if m is None:
                continue
            if m.status == "skipped":
                if mosek_smallest_skipped is None:
                    mosek_smallest_skipped = K
            elif m.feasible:
                mosek_largest_ok = K
        # CP-SAT at K=273.
        cpsat_273_ms = None
        if 273 in per_K and "cpsat_joint" in per_K[273]:
            cpsat_273_ms = per_K[273]["cpsat_joint"].solver_time_ms

        lines.append(f"### k_lookahead = {kl}\n\n")
        if best_K is not None:
            verdict = "beats" if best_makespan < cpu_e2e else "does NOT beat"
            lines.append(
                f"- **Sweet-spot K = {best_K}** with predicted makespan "
                f"**{best_makespan:,.0f} µs** ({best_solver}). This "
                f"{verdict} the CPU-E2E baseline ({cpu_e2e:,.0f} µs).\n"
            )
        if mosek_largest_ok is not None:
            lines.append(
                f"- **MOSEK runs up to K = {mosek_largest_ok}** "
                f"(n_partitions ≤ {MOSEK_MAX_N}). "
            )
            if mosek_smallest_skipped is not None:
                lines.append(
                    f"Skipped from **K = {mosek_smallest_skipped}** "
                    "onward (problem size exceeds the Exp 1 wall).\n"
                )
            else:
                lines.append("Ran at every K tested.\n")
        elif mosek_smallest_skipped is not None:
            lines.append(
                f"- **MOSEK skipped at every K in the sweep** "
                f"(smallest skipped K = {mosek_smallest_skipped}). "
                "All chosen K give n_partitions above the size wall.\n"
            )
        if cpsat_273_ms is not None:
            within_30 = cpsat_273_ms <= 30_000.0
            lines.append(
                f"- CP-SAT at K = 273 (full per-op): "
                f"solver wall = **{cpsat_273_ms / 1000.0:.2f} s** — "
                f"{'within' if within_30 else 'OVER'} the 30 s budget.\n"
            )

    # Transfer-cost dominance note: compare best per-op (K=273) to coarse K.
    lines.append(
        "\n## Notes\n\n"
        "- **Per-op transfer cost can dominate at high K.** With a flat "
        "100 µs cross-backend penalty and a chain DAG, every backend "
        "switch on the critical path pays 100 µs. At K=273 the worst-case "
        "transfer-cost contribution is K × 100 µs ≈ 27 ms, well above the "
        "best-backend op sum on DSP (≈ 60 ms). Schedules that switch "
        "backends pay this; the schedulers therefore prefer staying on a "
        "single backend unless the per-op gain exceeds the 100 µs.\n"
        "- The chain-DAG assumption under-estimates real parallelism; "
        "any 'sweet-spot K' here is a lower bound on the real benefit "
        "of multi-backend dispatch.\n"
    )

    path.write_text("".join(lines))


def maybe_plot(results: list[Exp7bResult], out_dir: Path) -> None:
    """Render makespan-vs-K and solver-time-vs-K line plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_lookahead: dict[int, dict[int, dict[str, Exp7bResult]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in results:
        by_lookahead[r.k_lookahead][r.K][r.solver] = r

    solvers = ("greedy", "cpsat_joint", "mosek_milp")

    # Makespan vs K (per lookahead, one curve per solver, log-x).
    fig, ax = plt.subplots(figsize=(7, 4))
    for kl in sorted(by_lookahead):
        Ks = sorted(by_lookahead[kl])
        for solver in solvers:
            xs: list[int] = []
            ys: list[float] = []
            for K in Ks:
                r = by_lookahead[kl][K].get(solver)
                if r is None or not r.feasible or not math.isfinite(r.makespan_us):
                    continue
                xs.append(K)
                ys.append(r.makespan_us)
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{solver} (k={kl})")
    ax.set_xscale("log")
    ax.set_xlabel("K (number of chunks)")
    ax.set_ylabel("predicted makespan (µs)")
    ax.set_title("yolov8n predicted makespan vs granularity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "makespan_vs_K.png", dpi=120)
    plt.close(fig)

    # Solver wall time vs K (per lookahead, one curve per solver, log-log).
    fig, ax = plt.subplots(figsize=(7, 4))
    for kl in sorted(by_lookahead):
        Ks = sorted(by_lookahead[kl])
        for solver in solvers:
            xs2: list[int] = []
            ys2: list[float] = []
            for K in Ks:
                r = by_lookahead[kl][K].get(solver)
                if r is None or r.status == "skipped":
                    continue
                xs2.append(K)
                ys2.append(max(r.solver_time_ms, 1e-3))  # avoid log(0)
            if xs2:
                ax.plot(xs2, ys2, marker="s", label=f"{solver} (k={kl})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (number of chunks)")
    ax.set_ylabel("solver wall time (ms)")
    ax.set_title("yolov8n solver time vs granularity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "solver_time_vs_K.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="K in {1,4,16,32}, k_lookahead=1 only.",
    )
    parser.add_argument(
        "--cpsat-timeout-ms",
        type=int,
        default=None,
        help="CP-SAT per-call timeout (default 10000 quick / 30000 full).",
    )
    parser.add_argument(
        "--mosek-time-limit-s",
        type=float,
        default=None,
        help="MOSEK per-call time limit in seconds (default 10 quick / 30 full).",
    )
    args = parser.parse_args()

    k_values = K_VALUES_QUICK if args.quick else K_VALUES_FULL
    lookaheads: tuple[int, ...] = (1,) if args.quick else (1, 4)
    cpsat_timeout_ms = args.cpsat_timeout_ms or (10000 if args.quick else 30000)
    mosek_time_limit_s = args.mosek_time_limit_s or (10.0 if args.quick else 30.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = run_sweep(
        k_values=k_values,
        lookaheads=lookaheads,
        cpsat_timeout_ms=cpsat_timeout_ms,
        mosek_time_limit_s=mosek_time_limit_s,
    )
    elapsed = time.perf_counter() - t0
    print(f"[exp7b] sweep done in {elapsed:.1f} s")
    e2e = load_e2e_baselines()
    write_jsonl(results, OUT_DIR / "results.jsonl")
    write_summary(results, OUT_DIR / "summary.md", quick=args.quick, e2e=e2e)
    maybe_plot(results, OUT_DIR)
    print(
        f"[exp7b] wrote {OUT_DIR / 'results.jsonl'}, {OUT_DIR / 'summary.md'}, "
        f"and plots."
    )


if __name__ == "__main__":
    main()

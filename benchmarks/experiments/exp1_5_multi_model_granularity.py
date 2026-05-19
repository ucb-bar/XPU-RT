"""Experiment 1.5 — multi-model x granularity scheduling sweep.

Sweeps M (number of concurrent models) x granularity x archetype and
runs three solvers per cell: greedy EFT, MOSEK MILP (via the scheduler
envelope), and CP-SAT joint. The goal is to surface the (M,
granularity, archetype) regime where CP-SAT decisively beats MOSEK on
either wall time (TLE) or makespan quality.

Outputs live under ``build/experiments/exp1_5/``:

* ``results.jsonl`` — one row per (cell, solver).
* ``summary.md`` — headline tables + decision-boundary paragraph.
* ``heatmap_solver_time.png`` — CP-SAT log10(wall time) per archetype.
* ``heatmap_makespan_gap.png`` — (MOSEK - CP-SAT) / MOSEK %.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiments._synthetic_dag import SyntheticDag, multi_model  # noqa: E402
from experiments.exp1_scheduling_ab import (  # noqa: E402
    cpsat_joint,
    greedy_schedule,
    mosek_milp,
)

OUT_DIR = REPO_ROOT / "build" / "experiments" / "exp1_5"

# Archetype -> (label, model_archetype, model_size).
ARCHETYPES: dict[str, tuple[str, int]] = {
    "chain_50": ("chain", 50),
    "transformer_L4": ("transformer", 4),
    "fan_out_20": ("fan_out", 20),
}

GRANULARITIES = ["per_op", "per_layer", "per_block", "per_model"]


@dataclass
class CellResult:
    archetype: str
    M: int
    granularity: str
    n_partitions: int
    num_devices: int
    solver: str
    makespan_us: float
    solver_time_ms: float
    feasible: bool
    status: str
    note: str = ""


def _make_dag(archetype_key: str, M: int, granularity: str, num_devices: int, seed: int) -> SyntheticDag:
    arch, size = ARCHETYPES[archetype_key]
    specs = [(arch, size) for _ in range(M)]
    return multi_model(specs, granularity=granularity, num_devices=num_devices, seed=seed)


def run_cell(
    archetype_key: str,
    M: int,
    granularity: str,
    *,
    num_devices: int,
    seed: int,
    cpsat_timeout_ms: int,
    mosek_time_limit_s: float,
    mosek_max_n: int,
) -> list[CellResult]:
    dag = _make_dag(archetype_key, M, granularity, num_devices=num_devices, seed=seed)
    n = len(dag.partition_ids)
    out: list[CellResult] = []

    g = greedy_schedule(dag)
    out.append(
        CellResult(
            archetype=archetype_key,
            M=M,
            granularity=granularity,
            n_partitions=n,
            num_devices=num_devices,
            solver="greedy",
            makespan_us=g.makespan_us,
            solver_time_ms=g.solver_time_ms,
            feasible=g.feasible,
            status=g.status,
        )
    )

    c = cpsat_joint(dag, timeout_ms=cpsat_timeout_ms)
    out.append(
        CellResult(
            archetype=archetype_key,
            M=M,
            granularity=granularity,
            n_partitions=n,
            num_devices=num_devices,
            solver="cpsat_joint",
            makespan_us=c.makespan_us,
            solver_time_ms=c.solver_time_ms,
            feasible=c.feasible,
            status=c.status,
        )
    )

    if n <= mosek_max_n:
        m = mosek_milp(dag, time_limit_s=mosek_time_limit_s)
        out.append(
            CellResult(
                archetype=archetype_key,
                M=M,
                granularity=granularity,
                n_partitions=n,
                num_devices=num_devices,
                solver="mosek_milp",
                makespan_us=m.makespan_us,
                solver_time_ms=m.solver_time_ms,
                feasible=m.feasible,
                status=m.status,
                note=m.note,
            )
        )
    else:
        out.append(
            CellResult(
                archetype=archetype_key,
                M=M,
                granularity=granularity,
                n_partitions=n,
                num_devices=num_devices,
                solver="mosek_milp",
                makespan_us=float("inf"),
                solver_time_ms=0.0,
                feasible=False,
                status="skipped_size",
                note=f"n_partitions={n}>{mosek_max_n}",
            )
        )
    return out


def run_sweep(quick: bool) -> list[CellResult]:
    cpsat_timeout_ms = 5000 if quick else 30000
    mosek_time_limit_s = 5.0 if quick else 30.0
    mosek_max_n = 60 if quick else 200

    Ms = [2, 4] if quick else [2, 4, 8]
    archetype_keys = ["transformer_L4", "fan_out_20"] if quick else list(ARCHETYPES.keys())

    cells: list[tuple[str, int, str]] = []
    for arch in archetype_keys:
        for M in Ms:
            for g in GRANULARITIES:
                cells.append((arch, M, g))

    results: list[CellResult] = []
    t_start = time.perf_counter()
    for i, (arch, M, g) in enumerate(cells, 1):
        cell_t0 = time.perf_counter()
        rs = run_cell(
            arch,
            M,
            g,
            num_devices=4,
            seed=42,
            cpsat_timeout_ms=cpsat_timeout_ms,
            mosek_time_limit_s=mosek_time_limit_s,
            mosek_max_n=mosek_max_n,
        )
        results.extend(rs)
        n = rs[0].n_partitions
        dt = time.perf_counter() - cell_t0
        cp = next(r for r in rs if r.solver == "cpsat_joint")
        mo = next(r for r in rs if r.solver == "mosek_milp")
        print(
            f"[exp1_5] {i}/{len(cells)} arch={arch} M={M} gran={g} n={n} "
            f"cp_ms={cp.solver_time_ms:.0f} ({cp.status}) "
            f"mo_ms={mo.solver_time_ms:.0f} ({mo.status}) wall={dt:.1f}s"
        )
    total = time.perf_counter() - t_start
    print(f"[exp1_5] sweep done in {total:.1f}s, {len(cells)} cells, {len(results)} rows")
    return results


def _by_cell(results: list[CellResult]) -> dict[tuple[str, int, str], dict[str, CellResult]]:
    out: dict[tuple[str, int, str], dict[str, CellResult]] = defaultdict(dict)
    for r in results:
        out[(r.archetype, r.M, r.granularity)][r.solver] = r
    return out


def write_jsonl(results: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")


def write_summary(results: list[CellResult], path: Path, quick: bool) -> None:
    cells = _by_cell(results)
    archetypes_seen = sorted({r.archetype for r in results})
    Ms_seen = sorted({r.M for r in results})

    lines: list[str] = []
    lines.append("# Experiment 1.5 — multi-model x granularity scheduling\n\n")
    lines.append(f"Mode: {'quick' if quick else 'full'}.\n\n")
    lines.append(
        "Each cell runs three solvers on a homogeneous M-copy multi-model DAG. "
        "MOSEK is skipped when `n_partitions` exceeds the policy threshold "
        "(quick=60, full=200).\n\n"
    )

    # Per-archetype MOSEK-feasibility table.
    lines.append("## MOSEK feasibility by archetype\n\n")
    for arch in archetypes_seen:
        lines.append(f"### {arch}\n\n")
        lines.append("| M \\ granularity | " + " | ".join(GRANULARITIES) + " |\n")
        lines.append("|---|" + "|".join("---" for _ in GRANULARITIES) + "|\n")
        for M in Ms_seen:
            row = [f"M={M}"]
            for g in GRANULARITIES:
                cell = cells.get((arch, M, g))
                if not cell:
                    row.append("-")
                    continue
                mo = cell.get("mosek_milp")
                if mo is None:
                    row.append("?")
                elif mo.status == "skipped_size":
                    row.append(f"skip(n={mo.n_partitions})")
                elif mo.feasible:
                    row.append(f"ok({mo.solver_time_ms:.0f}ms)")
                else:
                    row.append(f"TLE/fail({mo.status})")
            lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")

    # Makespan-gap table.
    lines.append("## Makespan gap: (MOSEK - CP-SAT) / MOSEK x 100\n\n")
    lines.append("Positive % = CP-SAT beats MOSEK. Only cells where both solvers were feasible.\n\n")
    for arch in archetypes_seen:
        lines.append(f"### {arch}\n\n")
        lines.append("| M \\ granularity | " + " | ".join(GRANULARITIES) + " |\n")
        lines.append("|---|" + "|".join("---" for _ in GRANULARITIES) + "|\n")
        for M in Ms_seen:
            row = [f"M={M}"]
            for g in GRANULARITIES:
                cell = cells.get((arch, M, g))
                if not cell:
                    row.append("-")
                    continue
                mo = cell.get("mosek_milp")
                cp = cell.get("cpsat_joint")
                if not (mo and cp and mo.feasible and cp.feasible and math.isfinite(mo.makespan_us) and mo.makespan_us > 0):
                    row.append("-")
                    continue
                gap = (mo.makespan_us - cp.makespan_us) / mo.makespan_us * 100.0
                row.append(f"{gap:+.2f}%")
            lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")

    # Decision boundary paragraph.
    lines.append("## Decision boundary\n\n")
    cpsat_status_by_size: list[tuple[int, float, str]] = []
    mosek_status_by_size: list[tuple[int, float, str, bool]] = []
    for _key, bucket in cells.items():
        cp = bucket.get("cpsat_joint")
        mo = bucket.get("mosek_milp")
        if cp:
            cpsat_status_by_size.append((cp.n_partitions, cp.solver_time_ms, cp.status))
        if mo:
            mosek_status_by_size.append((mo.n_partitions, mo.solver_time_ms, mo.status, mo.feasible))

    mosek_first_fail = None
    for n, ms, st, feas in sorted(mosek_status_by_size):
        if not feas:
            mosek_first_fail = (n, st)
            break
    mosek_skipped = sorted([(n, st) for n, _ms, st, feas in mosek_status_by_size if not feas and st == "skipped_size"])
    mosek_tle = sorted([(n, st) for n, _ms, st, feas in mosek_status_by_size if not feas and st != "skipped_size"])
    cpsat_max_time = max((ms for _n, ms, _st in cpsat_status_by_size), default=0.0)

    total_cells = len(cells)
    mosek_ok = sum(1 for b in cells.values() if (m := b.get("mosek_milp")) and m.feasible)
    cpsat_ok = sum(1 for b in cells.values() if (c := b.get("cpsat_joint")) and c.feasible)
    lines.append(
        f"- Total cells: **{total_cells}**. MOSEK feasible in **{mosek_ok}**, "
        f"CP-SAT feasible in **{cpsat_ok}**.\n"
    )
    if mosek_skipped:
        lines.append(
            f"- MOSEK skipped (size guard) in **{len(mosek_skipped)}** cells; "
            f"smallest such cell had n_partitions={mosek_skipped[0][0]}.\n"
        )
    if mosek_tle:
        lines.append(
            f"- MOSEK TLE / infeasible in **{len(mosek_tle)}** cells "
            f"(smallest at n_partitions={mosek_tle[0][0]}).\n"
        )
    if mosek_first_fail:
        lines.append(
            f"- First MOSEK failure (sorted by n_partitions) at n={mosek_first_fail[0]}, status=`{mosek_first_fail[1]}`.\n"
        )
    lines.append(f"- CP-SAT max wall time across the sweep: **{cpsat_max_time:.0f}ms**.\n")

    # Quantify makespan dominance among co-feasible cells.
    gaps: list[float] = []
    cpsat_wins = 0
    for bucket in cells.values():
        mo = bucket.get("mosek_milp")
        cp = bucket.get("cpsat_joint")
        if mo and cp and mo.feasible and cp.feasible and mo.makespan_us > 0 and math.isfinite(mo.makespan_us):
            gap = (mo.makespan_us - cp.makespan_us) / mo.makespan_us * 100.0
            gaps.append(gap)
            if gap > 2.0:
                cpsat_wins += 1
    if gaps:
        lines.append(
            f"- Co-feasible cells: **{len(gaps)}**. CP-SAT beats MOSEK by >2% in **{cpsat_wins}** "
            f"(mean gap {sum(gaps) / len(gaps):+.2f}%, max {max(gaps):+.2f}%, min {min(gaps):+.2f}%).\n"
        )

    path.write_text("".join(lines))


def maybe_plot(results: list[CellResult], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    cells = _by_cell(results)
    archetypes = sorted({r.archetype for r in results})
    Ms = sorted({r.M for r in results})
    grans = GRANULARITIES

    # Solver-time heatmap (one panel per archetype).
    fig, axes = plt.subplots(1, len(archetypes), figsize=(4 * len(archetypes), 3), squeeze=False)
    for ai, arch in enumerate(archetypes):
        ax = axes[0][ai]
        grid = np.full((len(Ms), len(grans)), np.nan)
        mosek_tle_mask = np.zeros((len(Ms), len(grans)), dtype=bool)
        for mi, M in enumerate(Ms):
            for gi, g in enumerate(grans):
                bucket = cells.get((arch, M, g))
                if not bucket:
                    continue
                cp = bucket.get("cpsat_joint")
                if cp and cp.solver_time_ms > 0:
                    grid[mi, gi] = math.log10(max(cp.solver_time_ms, 1e-3))
                mo = bucket.get("mosek_milp")
                if mo and not mo.feasible:
                    mosek_tle_mask[mi, gi] = True
        im = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(grans)))
        ax.set_xticklabels(grans, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(Ms)))
        ax.set_yticklabels([f"M={M}" for M in Ms])
        ax.set_title(f"{arch}: log10(CP-SAT ms)")
        for mi in range(len(Ms)):
            for gi in range(len(grans)):
                if mosek_tle_mask[mi, gi]:
                    ax.text(gi, mi, "X", ha="center", va="center", color="red", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("CP-SAT solve time (red X = MOSEK TLE/skipped)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_solver_time.png", dpi=120)
    plt.close(fig)

    # Makespan-gap heatmap.
    fig, axes = plt.subplots(1, len(archetypes), figsize=(4 * len(archetypes), 3), squeeze=False)
    for ai, arch in enumerate(archetypes):
        ax = axes[0][ai]
        grid = np.full((len(Ms), len(grans)), np.nan)
        for mi, M in enumerate(Ms):
            for gi, g in enumerate(grans):
                bucket = cells.get((arch, M, g))
                if not bucket:
                    continue
                mo = bucket.get("mosek_milp")
                cp = bucket.get("cpsat_joint")
                if mo and cp and mo.feasible and cp.feasible and mo.makespan_us > 0 and math.isfinite(mo.makespan_us):
                    grid[mi, gi] = (mo.makespan_us - cp.makespan_us) / mo.makespan_us * 100.0
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(grans)))
        ax.set_xticklabels(grans, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(Ms)))
        ax.set_yticklabels([f"M={M}" for M in Ms])
        ax.set_title(f"{arch}: (MOSEK-CPSAT)/MOSEK %")
        for mi in range(len(Ms)):
            for gi in range(len(grans)):
                v = grid[mi, gi]
                if not np.isnan(v):
                    ax.text(gi, mi, f"{v:+.1f}", ha="center", va="center", color="black", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Makespan gap; green = CP-SAT wins, NaN = MOSEK skipped/TLE")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_makespan_gap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Skip M=8 and chain_50 archetype.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep(quick=args.quick)
    write_jsonl(results, OUT_DIR / "results.jsonl")
    write_summary(results, OUT_DIR / "summary.md", quick=args.quick)
    maybe_plot(results, OUT_DIR)
    print(f"\n[exp1_5] wrote {OUT_DIR / 'results.jsonl'} and {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()

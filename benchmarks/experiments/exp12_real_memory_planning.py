"""Real-model memory planning A/B: greedy vs MILP, with aliases enabled.

Exp 2's headline ("MILP doesn't beat greedy on peak memory") ran on
synthetic transformer/MLP workloads with aliases disabled on both
sides. This experiment reruns the comparison on *real* QNN-converter
artifacts (dronet, yolov8n) and enables alias activation on the MILP
side so the planner can exploit the in-place opportunities the loader
extracts from the dataflow graph.

We run four cells per workload:

| planner | aliases? |
|---------|----------|
| greedy  | off      |  baseline
| greedy  | on       |  greedy with alias hints
| MILP    | off      |  MILP with NO alias candidates declared
| MILP    | on       |  MILP with alias candidates declared

A 60 s solver wall-clock cap is enforced; over-budget runs are
recorded with ``status="timeout"`` (the canonicalize-bug caveat from
Exp 2). The script does NOT modify the planners; it only feeds them
``MemoryPlanInput`` objects with and without ``alias_candidates``.

Usage:
    uv run python scripts/experiments/exp12_real_memory_planning.py --quick
    uv run python scripts/experiments/exp12_real_memory_planning.py
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from xpu_rt.scheduler.qnn_model_loader import ExtractionStats, extract_buffer_specs
from xpu_rt.solve.memory_planner import (
    MemoryPlanInput,
    MemoryPlanSolved,
    plan_memory,
)
from xpu_rt.solve.memory_planner_greedy import plan_memory_greedy
from xpu_rt.solve.solver_types import SolverStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "build" / "experiments" / "exp12_real_memory"

DRONET_PATH = Path("/tmp/qnn_build/dronet_net.json")
# Use the *quantized* yolov8n by default — smaller activations stay
# safely below the 100 MB canonicalize threshold.
YOLOV8N_PATH = Path("/tmp/yqnn/yolov8n_q_net.json")

SOLVER_TIMEOUT_S = 60.0


# ------------------------------------------------------------------ result row


@dataclass(frozen=True)
class Cell:
    workload: str
    planner: str
    aliases_declared: bool
    status: str
    feasible: bool
    objective_value: float | None
    tier_peak_usage: dict[str, int]
    total_peak: int
    aliases_activated: int
    num_buffers: int
    num_alias_candidates: int
    solve_time_ms: float
    timed_out: bool
    note: str = ""


# ------------------------------------------------------------------ runners


def _strip_aliases(problem: MemoryPlanInput) -> MemoryPlanInput:
    return replace(problem, alias_candidates=())


def _summarize(
    workload: str,
    planner: str,
    aliases_declared: bool,
    problem: MemoryPlanInput,
    plan: MemoryPlanSolved | None,
    status: str,
    elapsed_ms: float,
    timed_out: bool,
    note: str = "",
) -> Cell:
    if plan is None or not plan.buffers or plan.status == "infeasible":
        return Cell(
            workload=workload,
            planner=planner,
            aliases_declared=aliases_declared,
            status=status,
            feasible=False,
            objective_value=None,
            tier_peak_usage={},
            total_peak=0,
            aliases_activated=0,
            num_buffers=len(problem.buffers),
            num_alias_candidates=len(problem.alias_candidates),
            solve_time_ms=elapsed_ms,
            timed_out=timed_out,
            note=note,
        )
    return Cell(
        workload=workload,
        planner=planner,
        aliases_declared=aliases_declared,
        status=status,
        feasible=True,
        objective_value=plan.objective_value,
        tier_peak_usage=dict(plan.tier_peak_usage),
        total_peak=sum(plan.tier_peak_usage.values()),
        aliases_activated=sum(1 for a in plan.buffers if a.aliases_with),
        num_buffers=len(problem.buffers),
        num_alias_candidates=len(problem.alias_candidates),
        solve_time_ms=elapsed_ms,
        timed_out=timed_out,
        note=note,
    )


def _run_greedy(
    workload: str, problem: MemoryPlanInput, aliases_on: bool
) -> Cell:
    t0 = time.perf_counter()
    plan = plan_memory_greedy(problem, activate_aliases=aliases_on)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return _summarize(
        workload=workload,
        planner="greedy_first_fit",
        aliases_declared=aliases_on,
        problem=problem,
        plan=plan,
        status=plan.status,
        elapsed_ms=elapsed,
        timed_out=False,
    )


def _milp_worker(
    plan_input_pickled: bytes, problem_id: str, q: mp.Queue[Any]
) -> None:
    """Worker target — runs ``plan_memory`` in a separate process."""

    import pickle

    plan_input = pickle.loads(plan_input_pickled)
    t0 = time.perf_counter()
    try:
        response, plan = plan_memory(plan_input, problem_id=problem_id)
        elapsed = (time.perf_counter() - t0) * 1000.0
        status = (
            response.status.value
            if isinstance(response.status, SolverStatus)
            else str(response.status)
        )
        q.put(
            {
                "ok": True,
                "elapsed_ms": elapsed,
                "status": status,
                "plan_dict": plan.to_dict() if plan is not None else None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - t0) * 1000.0
        q.put({"ok": False, "elapsed_ms": elapsed, "error": repr(exc)})


def _run_milp_with_timeout(
    workload: str,
    problem: MemoryPlanInput,
    aliases_on: bool,
    timeout_s: float,
) -> Cell:
    """Run MILP in a child process and kill it after ``timeout_s``.

    This guards against the documented canonicalize-bug hang for
    large workloads.
    """

    import pickle

    payload = pickle.dumps(problem)
    ctx = mp.get_context("spawn")
    q: mp.Queue[Any] = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_milp_worker,
        args=(payload, f"exp12_{workload}_aliases={aliases_on}", q),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        return _summarize(
            workload=workload,
            planner="milp",
            aliases_declared=aliases_on,
            problem=problem,
            plan=None,
            status="timeout",
            elapsed_ms=timeout_s * 1000.0,
            timed_out=True,
            note=f"killed after {timeout_s:.0f}s wall-clock",
        )

    try:
        result = q.get_nowait()
    except Exception:  # noqa: BLE001
        return _summarize(
            workload=workload,
            planner="milp",
            aliases_declared=aliases_on,
            problem=problem,
            plan=None,
            status="error",
            elapsed_ms=0.0,
            timed_out=False,
            note="child exited without result",
        )

    if not result.get("ok"):
        return _summarize(
            workload=workload,
            planner="milp",
            aliases_declared=aliases_on,
            problem=problem,
            plan=None,
            status="error",
            elapsed_ms=result.get("elapsed_ms", 0.0),
            timed_out=False,
            note=str(result.get("error"))[:200],
        )

    plan_dict = result.get("plan_dict")
    plan: MemoryPlanSolved | None = None
    if plan_dict is not None:
        # Reconstruct enough of MemoryPlanSolved for summarization.
        from xpu_rt.solve.memory_planner import BufferAllocation

        plan = MemoryPlanSolved(
            schema_version=plan_dict["schema_version"],
            solver_backend=plan_dict["solver_backend"],
            status=plan_dict["status"],
            buffers=tuple(
                BufferAllocation(
                    buffer_id=row["buffer_id"],
                    tier=row["tier"],
                    offset_bytes=int(row["offset_bytes"]),
                    aliases_with=row.get("aliases_with"),
                )
                for row in plan_dict["buffers"]
            ),
            tier_peak_usage=dict(plan_dict["tier_peak_usage"]),
            objective_value=plan_dict.get("objective_value"),
            formulation_hash=plan_dict["formulation_hash"],
        )
    return _summarize(
        workload=workload,
        planner="milp",
        aliases_declared=aliases_on,
        problem=problem,
        plan=plan,
        status=str(result.get("status", "unknown")),
        elapsed_ms=float(result.get("elapsed_ms", 0.0)),
        timed_out=False,
    )


# ------------------------------------------------------------------ output


def _fmt_bytes(n: int) -> str:
    for unit, factor in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= factor:
            return f"{n / factor:.2f} {unit}"
    return f"{n} B"


def _write_jsonl(results: list[Cell], extractions: dict[str, ExtractionStats], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    with (out / "extractions.json").open("w") as f:
        json.dump({k: asdict(v) for k, v in extractions.items()}, f, indent=2)


def _write_summary(
    results: list[Cell], extractions: dict[str, ExtractionStats], out: Path
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Experiment 12: greedy vs MILP on real QNN models (aliases enabled)\n")
    lines.append(
        "Reruns the Exp 2 A/B but on real model buffer specs extracted from "
        "QNN converter `*_net.json` sidecars, with `activate_aliases=True` for the MILP cell.\n"
    )
    lines.append("## Extraction summary\n")
    lines.append("| workload | parser | ops | activations | statics | alias_cands | total_act | max_act |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for wl, s in extractions.items():
        lines.append(
            f"| {wl} | {s.parser} | {s.num_ops} | {s.num_activations} | "
            f"{s.num_static_params} | {s.alias_candidates_proposed} | "
            f"{_fmt_bytes(s.total_activation_bytes)} | "
            f"{_fmt_bytes(s.max_activation_bytes)} |"
        )

    lines.append("\n## Per-cell results\n")
    lines.append("| workload | planner | aliases | status | total_peak | activated | solve_ms | note |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    for r in results:
        lines.append(
            f"| {r.workload} | {r.planner} | "
            f"{'on' if r.aliases_declared else 'off'} | "
            f"{r.status} | "
            f"{_fmt_bytes(r.total_peak) if r.feasible else 'n/a'} | "
            f"{r.aliases_activated} | "
            f"{r.solve_time_ms:.1f} | {r.note} |"
        )

    # Headline: MILP-with-aliases vs greedy-baseline.
    lines.append("\n## Headline — MILP (aliases on) vs greedy (aliases off)\n")
    lines.append("| workload | greedy_off total_peak | milp_on total_peak | reduction% | milp_on solve_ms |")
    lines.append("|---|---:|---:|---:|---:|")
    by_key: dict[tuple[str, str, bool], Cell] = {
        (r.workload, r.planner, r.aliases_declared): r for r in results
    }
    headlines: list[tuple[str, float]] = []
    for wl in extractions:
        g = by_key.get((wl, "greedy_first_fit", False))
        m = by_key.get((wl, "milp", True))
        if g is None or m is None:
            continue
        if not (g.feasible and m.feasible) or g.total_peak == 0:
            reduction_str = "n/a"
            reduction = None
        else:
            reduction = 100.0 * (g.total_peak - m.total_peak) / g.total_peak
            reduction_str = f"{reduction:+.1f}%"
            headlines.append((wl, reduction))
        lines.append(
            f"| {wl} | "
            f"{_fmt_bytes(g.total_peak) if g.feasible else 'n/a'} | "
            f"{_fmt_bytes(m.total_peak) if m.feasible else 'n/a'} | "
            f"{reduction_str} | "
            f"{m.solve_time_ms:.1f} |"
        )

    # MILP-on vs MILP-off — isolates the alias contribution.
    lines.append("\n## Alias contribution — MILP-aliases-on vs MILP-aliases-off\n")
    lines.append("| workload | milp_off total_peak | milp_on total_peak | alias_reduction% | aliases_activated |")
    lines.append("|---|---:|---:|---:|---:|")
    for wl in extractions:
        mo = by_key.get((wl, "milp", False))
        mn = by_key.get((wl, "milp", True))
        if mo is None or mn is None:
            continue
        if not (mo.feasible and mn.feasible) or mo.total_peak == 0:
            reduction_str = "n/a"
        else:
            reduction = 100.0 * (mo.total_peak - mn.total_peak) / mo.total_peak
            reduction_str = f"{reduction:+.1f}%"
        lines.append(
            f"| {wl} | "
            f"{_fmt_bytes(mo.total_peak) if mo.feasible else 'n/a'} | "
            f"{_fmt_bytes(mn.total_peak) if mn.feasible else 'n/a'} | "
            f"{reduction_str} | "
            f"{mn.aliases_activated} |"
        )

    lines.append("\n## Verdict\n")
    if headlines:
        wins = [h for h in headlines if h[1] > 5.0]
        losses_or_ties = [h for h in headlines if h[1] <= 5.0]
        if wins:
            lines.append("Workloads where MILP-with-aliases beats greedy-baseline by >5%:")
            for wl, r in sorted(wins, key=lambda kv: -kv[1]):
                lines.append(f"  - **{wl}**: {r:+.1f}% peak reduction")
        if losses_or_ties:
            lines.append("\nWorkloads where MILP-with-aliases does NOT meaningfully beat greedy:")
            for wl, r in sorted(losses_or_ties, key=lambda kv: -kv[1]):
                lines.append(f"  - **{wl}**: {r:+.1f}%")
    else:
        lines.append("No feasible head-to-head cells (likely all timeouts).\n")

    lines.append("\n## Caveats\n")
    lines.append(
        "- Workloads pulled from QNN converter `*_net.json` (sidecar JSON). "
        "Static parameters excluded; only activations and graph i/o are planned.\n"
    )
    lines.append(
        "- Lifetimes encoded with half-step granularity so that an in-place "
        "elementwise op produces an output whose lifetime is strictly disjoint "
        "from its single-use input.\n"
    )
    lines.append(
        "- MILP guarded by a 60 s wall-clock cap; over-budget cells reported as "
        "`status=timeout`. This is the canonicalize-bug avoidance noted in Exp 2.\n"
    )
    lines.append(
        "- Greedy-with-aliases is shown for completeness; the headline compares "
        "the canonical greedy-aliases-off baseline against MILP-aliases-on.\n"
    )

    (out / "summary.md").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------ main


WorkloadBuilder = Callable[[], tuple[MemoryPlanInput, ExtractionStats]]


def _load_dronet() -> tuple[MemoryPlanInput, ExtractionStats]:
    return extract_buffer_specs(DRONET_PATH)


def _load_yolov8n() -> tuple[MemoryPlanInput, ExtractionStats]:
    return extract_buffer_specs(YOLOV8N_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="dronet only (skips yolov8n).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=SOLVER_TIMEOUT_S)
    args = parser.parse_args()

    workloads: list[tuple[str, WorkloadBuilder]] = [("dronet", _load_dronet)]
    if not args.quick and YOLOV8N_PATH.exists():
        workloads.append(("yolov8n_q", _load_yolov8n))

    extractions: dict[str, ExtractionStats] = {}
    results: list[Cell] = []

    for name, builder in workloads:
        print(f"[exp12] loading: {name}")
        problem, stats = builder()
        extractions[name] = stats
        no_alias_problem = _strip_aliases(problem)
        print(
            f"[exp12]   buffers={len(problem.buffers)} "
            f"alias_candidates={len(problem.alias_candidates)} "
            f"total_act={stats.total_activation_bytes:,}"
        )

        # Greedy, aliases off (baseline)
        print(f"[exp12] run: greedy aliases=off  {name}")
        results.append(_run_greedy(name, no_alias_problem, aliases_on=False))
        # Greedy, aliases on
        print(f"[exp12] run: greedy aliases=on   {name}")
        results.append(_run_greedy(name, problem, aliases_on=True))
        # MILP, aliases off
        print(f"[exp12] run: milp   aliases=off  {name}")
        results.append(
            _run_milp_with_timeout(name, no_alias_problem, aliases_on=False, timeout_s=args.timeout_s)
        )
        # MILP, aliases on
        print(f"[exp12] run: milp   aliases=on   {name}")
        results.append(
            _run_milp_with_timeout(name, problem, aliases_on=True, timeout_s=args.timeout_s)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(results, extractions, args.output_dir)
    _write_summary(results, extractions, args.output_dir)
    print(f"[exp12] wrote {args.output_dir}/summary.md and results.jsonl")


if __name__ == "__main__":
    main()

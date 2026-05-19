"""Exp 13: characterize the ``_canonicalize`` candidate+1 hang.

Background
----------
``xpu_rt.solve.memory_planner._canonicalize`` advances a candidate
offset by ``alignment`` bytes per iteration when the solver's
suggested offset conflicts with an already-placed buffer (lines
250-260 of ``memory_planner.py``). For multi-hundred-MB buffers the
scan is O(size_bytes / alignment) per buffer and dominates wall time.

This experiment:

1. Sweeps total-bytes per buffer through {1, 4, 16, 64, 128, 256,
   512, 1024} MiB with 4 overlapping buffers and times the full
   ``plan_memory`` MILP path under a 60 s wall-clock guard.
2. Compares to ``plan_memory_greedy`` on the same inputs.
3. Calls ``_canonicalize`` directly with synthetic offset hints so we
   isolate the post-pass cost from the solver core, confirming the
   linear scan signature.

Outputs
-------
``build/experiments/exp13_canonicalize/``:

* ``results.jsonl`` - one row per (planner, size, mode) measurement.
* ``timing_plot.png`` - wall time vs total bytes.
* ``summary.md`` - human-readable findings.

The script never modifies ``memory_planner.py``; it only observes.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from xpu_rt.solve.memory_planner import (
    BufferSpec,
    MemoryPlanInput,
    TierCapacity,
    _canonicalize,
    plan_memory,
)
from xpu_rt.solve.memory_planner_greedy import plan_memory_greedy

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "build" / "experiments" / "exp13_canonicalize"
ALIGN = 64
MB = 1024 * 1024
GB = 1024 * MB
TIMEOUT_S = 60.0


@dataclass(frozen=True)
class TimingRow:
    planner: str
    mode: str
    buffer_count: int
    per_buffer_bytes: int
    total_bytes: int
    elapsed_s: float
    timed_out: bool
    status: str
    note: str = ""


def _build_overlapping_input(per_buffer_bytes: int, buffer_count: int = 4) -> MemoryPlanInput:
    """All buffers share lifetime [0, 1] so MILP must give each its own slot.

    Tier capacity is sized to comfortably hold the full stack.
    """

    buffers = tuple(
        BufferSpec(
            buffer_id=f"b{i:02d}",
            size_bytes=per_buffer_bytes,
            lifetime_start=0,
            lifetime_end=1,
            allowed_tiers=("scratch",),
            alignment=ALIGN,
            spill_cost=1.0,
        )
        for i in range(buffer_count)
    )
    tier_cap = (buffer_count + 1) * per_buffer_bytes
    return MemoryPlanInput(
        buffers=buffers,
        tier_capacities=(TierCapacity("scratch", capacity_bytes=tier_cap, weight=1.0),),
        objective_lambda=1e-9,
        time_budget_ms=30_000,
    )


# ---------------------------- timeout harness ----------------------------


def _milp_worker(per_buffer_bytes: int, buffer_count: int, q: mp.Queue) -> None:
    try:
        plan_input = _build_overlapping_input(per_buffer_bytes, buffer_count)
        t0 = time.perf_counter()
        response, plan = plan_memory(plan_input)
        elapsed = time.perf_counter() - t0
        q.put({"ok": True, "elapsed_s": elapsed, "status": str(response.status)})
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "elapsed_s": -1.0, "status": "error", "error": repr(exc)})


def _greedy_worker(per_buffer_bytes: int, buffer_count: int, q: mp.Queue) -> None:
    try:
        plan_input = _build_overlapping_input(per_buffer_bytes, buffer_count)
        t0 = time.perf_counter()
        plan = plan_memory_greedy(plan_input)
        elapsed = time.perf_counter() - t0
        q.put({"ok": True, "elapsed_s": elapsed, "status": plan.status})
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "elapsed_s": -1.0, "status": "error", "error": repr(exc)})


def _canonicalize_worker(per_buffer_bytes: int, buffer_count: int, q: mp.Queue) -> None:
    """Direct ``_canonicalize`` micro-benchmark.

    Sets every buffer's solver-suggested offset to 0. Iteration
    order is sorted-by-buffer_id, so ``b00`` places at 0, then each
    subsequent buffer's candidate=0 hint conflicts with the prior
    placement and the candidate+1 loop must scan upward by
    ``alignment`` bytes per step through the full stack.
    """

    try:
        plan_input = _build_overlapping_input(per_buffer_bytes, buffer_count)
        # Force the pathology: every buffer has solver hint 0, so the
        # k-th buffer (after k-1 already placed) must scan through
        # k-1 buffers' worth of bytes via candidate+=alignment.
        offsets = {f"b{i:02d}": 0 for i in range(buffer_count)}
        tier_choice = {b.buffer_id: "scratch" for b in plan_input.buffers}
        t0 = time.perf_counter()
        _canonicalize(plan_input, tier_choice, offsets, alias_pairs=[])
        elapsed = time.perf_counter() - t0
        q.put({"ok": True, "elapsed_s": elapsed, "status": "ok"})
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "elapsed_s": -1.0, "status": "error", "error": repr(exc)})


def _run_with_timeout(
    target: Any, per_buffer_bytes: int, buffer_count: int, timeout_s: float = TIMEOUT_S
) -> tuple[bool, float, str, str]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(per_buffer_bytes, buffer_count, q))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return True, time.perf_counter() - t0, "timeout", ""
    try:
        result = q.get_nowait()
    except Exception:  # noqa: BLE001
        return False, time.perf_counter() - t0, "no_result", "queue empty"
    if not result.get("ok"):
        return False, result.get("elapsed_s", -1.0), result.get("status", "error"), result.get("error", "")
    return False, result["elapsed_s"], result["status"], ""


# ---------------------------- main sweep ----------------------------


def _sizes_mib(quick: bool) -> list[int]:
    if quick:
        return [1, 4, 16, 64]
    return [1, 4, 16, 64, 128, 256, 512, 1024]


def run_sweep(quick: bool = False, buffer_count: int = 4) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for size_mib in _sizes_mib(quick):
        per_buffer_bytes = size_mib * MB
        total_bytes = per_buffer_bytes * buffer_count
        print(f"\n=== size={size_mib} MiB/buffer  total={total_bytes/MB:.0f} MiB ===")

        for label, target in (
            ("canonicalize_direct", _canonicalize_worker),
            ("greedy", _greedy_worker),
            ("milp_full", _milp_worker),
        ):
            t_start = time.perf_counter()
            timed_out, elapsed, status, err = _run_with_timeout(
                target, per_buffer_bytes, buffer_count
            )
            wall = time.perf_counter() - t_start
            note = err if err else ""
            print(
                f"  {label:24s} elapsed={elapsed:7.3f}s wall={wall:7.3f}s "
                f"status={status} timed_out={timed_out}"
            )
            rows.append(
                TimingRow(
                    planner=label,
                    mode="overlapping_4buf",
                    buffer_count=buffer_count,
                    per_buffer_bytes=per_buffer_bytes,
                    total_bytes=total_bytes,
                    elapsed_s=elapsed if not timed_out else TIMEOUT_S,
                    timed_out=timed_out,
                    status=status,
                    note=note,
                )
            )
            # Don't keep escalating MILP sizes once it has already
            # timed out -- waste of wall time.
            if label == "milp_full" and timed_out and not quick:
                print("  ... MILP timed out; will continue but expect more timeouts")
    return rows


# ---------------------------- output ----------------------------


def _write_results(rows: list[TimingRow]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.jsonl"
    with results_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")
    print(f"\nwrote {results_path}")


def _plot(rows: list[TimingRow]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable: {exc}")
        return

    by_planner: dict[str, list[tuple[int, float, bool]]] = {}
    for row in rows:
        by_planner.setdefault(row.planner, []).append(
            (row.total_bytes, row.elapsed_s, row.timed_out)
        )

    fig, ax = plt.subplots(figsize=(9, 6))
    for planner, pts in sorted(by_planner.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] / MB for p in pts]
        ys = [p[1] for p in pts]
        markers = ["x" if p[2] else "o" for p in pts]
        ax.plot(xs, ys, "-", label=planner, alpha=0.8)
        for x, y, m in zip(xs, ys, markers):
            ax.plot([x], [y], m, color=ax.lines[-1].get_color())
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("total bytes (MiB, log2)")
    ax.set_ylabel("wall time (s, log10)")
    ax.set_title("exp13: _canonicalize hang characterization")
    ax.axhline(TIMEOUT_S, color="red", linestyle="--", alpha=0.4, label=f"timeout={TIMEOUT_S:.0f}s")
    ax.axhline(10.0, color="orange", linestyle=":", alpha=0.4, label="10 s threshold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plot_path = OUTPUT_DIR / "timing_plot.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"wrote {plot_path}")


def _summary(rows: list[TimingRow]) -> None:
    # Thresholds: smallest total_bytes where elapsed > 10s and > 60s
    def _threshold(planner: str, t_s: float) -> str:
        candidates = [
            r for r in rows if r.planner == planner and (r.timed_out or r.elapsed_s > t_s)
        ]
        if not candidates:
            return "not reached in sweep"
        first = min(candidates, key=lambda r: r.total_bytes)
        return f"{first.total_bytes / MB:.0f} MiB total ({first.per_buffer_bytes / MB:.0f} MiB/buf x {first.buffer_count})"

    lines: list[str] = []
    lines.append("# Exp 13 -- _canonicalize hang characterization\n")
    lines.append("## Setup\n")
    lines.append(
        f"- 4 buffers, shared lifetime [0,1], one tier (`scratch`), alignment={ALIGN}.\n"
        f"- Sweep per-buffer size in MiB: {_sizes_mib(False)}.\n"
        f"- Wall-clock guard: {TIMEOUT_S:.0f} s per call (separate process).\n\n"
    )
    lines.append("## Thresholds\n")
    for planner in ("canonicalize_direct", "milp_full", "greedy"):
        lines.append(f"- **{planner}**\n")
        lines.append(f"    - first exceeds 10 s at: {_threshold(planner, 10.0)}\n")
        lines.append(f"    - first exceeds 60 s (timeout) at: {_threshold(planner, 60.0)}\n")
    lines.append("\n## Per-size timings\n")
    lines.append("| size (MiB/buf) | total (MiB) | canonicalize_direct (s) | milp_full (s) | greedy (s) |\n")
    lines.append("|---:|---:|---:|---:|---:|\n")
    by_size: dict[int, dict[str, TimingRow]] = {}
    for row in rows:
        by_size.setdefault(row.per_buffer_bytes, {})[row.planner] = row
    for per_buf in sorted(by_size):
        bucket = by_size[per_buf]

        def _cell(name: str) -> str:
            r = bucket.get(name)
            if r is None:
                return "-"
            if r.timed_out:
                return f">{TIMEOUT_S:.0f} (TIMEOUT)"
            return f"{r.elapsed_s:.3f}"

        lines.append(
            f"| {per_buf / MB:.0f} | {per_buf * 4 / MB:.0f} | "
            f"{_cell('canonicalize_direct')} | {_cell('milp_full')} | {_cell('greedy')} |\n"
        )

    lines.append("\n## Notes\n")
    lines.append(
        "- ``canonicalize_direct`` calls ``_canonicalize`` with synthetic\n"
        "  reverse-order offset hints. This isolates the candidate+1 loop\n"
        "  from MILP solve cost.\n"
        "- ``milp_full`` is ``plan_memory`` end-to-end (MOSEK or HiGHS).\n"
        "- ``greedy`` is ``plan_memory_greedy``; first-fit-decreasing has\n"
        "  no candidate+1 sweep -- it uses an O(n) first-fit per buffer\n"
        "  over already-placed intervals.\n"
    )
    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("".join(lines))
    print(f"wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip 128MB-1GB points")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_sweep(quick=args.quick)
    _write_results(rows)
    _plot(rows)
    _summary(rows)


if __name__ == "__main__":
    main()

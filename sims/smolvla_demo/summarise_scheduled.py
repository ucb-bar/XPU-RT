#!/usr/bin/env python3
"""Compare the latency modes over all recorded episodes.

Reports success rate, sim time to finish, and the fraction of env steps spent
holding a stale action, plus the steady-state stall fraction the schedule
implies -- the measured number is over episodes that mostly end early, so the
two answer different questions and both are worth printing.

    python3 summarise_scheduled.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent


def steady_state_stall(latency_ms: float, covers_ms: float, replan: str) -> float | None:
    """Asymptotic fraction of time with nothing fresh to execute.

    on_empty  : a chunk is requested only once the previous runs out, so the
                cycle is covers + latency.
    pipelined : one inference is always in flight, so a chunk lands every
                latency ms regardless of how long the last one covered.
    """
    if latency_ms <= 0:
        return 0.0
    cycle = covers_ms + latency_ms if replan == "on_empty" else latency_ms
    return max(0.0, 1.0 - covers_ms / cycle)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "runs" / "scheduled")
    a = ap.parse_args()

    files = sorted(a.runs.glob("scheduled_rollout_*.json"))
    if not files:
        print(f"no runs in {a.runs}")
        return 1
    runs = [json.loads(f.read_text()) for f in files]
    runs.sort(key=lambda r: (r["latency_ms"], r.get("replan", "")))

    print(f"{'mode':<22}{'lat ms':>8}{'succ':>7}{'steps med':>11}"
          f"{'sim s med':>11}{'stall meas':>12}{'stall ss':>10}{'infers':>8}")
    print("-" * 89)
    for r in runs:
        eps = r["episodes"]
        name = r["latency_mode"]
        if r["latency_ms"] > 0:
            name = f"{name}/{r.get('replan','?')}"
        n_ok = sum(1 for e in eps if e["success"])
        steps = [e["steps"] for e in eps]
        stall = [e["stalled_steps"] / max(e["steps"], 1) for e in eps]
        ss = steady_state_stall(r["latency_ms"], r["chunk_covers_ms"],
                                r.get("replan", "on_empty"))
        print(f"{name:<22}{r['latency_ms']:>8.0f}{n_ok}/{len(eps):<5}"
              f"{st.median(steps):>11.0f}"
              f"{st.median(steps) / r['control_hz']:>11.2f}"
              f"{st.mean(stall) * 100:>11.1f}%"
              f"{ss * 100:>9.1f}%"
              f"{st.median([e['inferences'] for e in eps]):>8.0f}")

    base = next((r for r in runs if r["latency_ms"] == 0), None)
    if base:
        b = st.median([e["steps"] for e in base["episodes"]])
        print()
        print("  sim time to finish, relative to free inference:")
        for r in runs:
            if r["latency_ms"] == 0:
                continue
            m = st.median([e["steps"] for e in r["episodes"]])
            name = f"{r['latency_mode']}/{r.get('replan','?')}"
            print(f"    {name:<22}{m / b:>6.2f}x")
    print()
    print("  'stall meas' is over episodes that mostly end early; 'stall ss' is")
    print("  the asymptotic duty cycle the schedule implies. A policy whose chunk")
    print("  covers less wall time than it takes to produce cannot reach 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

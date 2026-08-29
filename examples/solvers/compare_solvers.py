#!/usr/bin/env python3
"""Several registered schedulers on one workload, side by side.

    .venv/bin/python examples/solvers/compare_solvers.py

WHAT THIS IS FOR. `xpu-rt/schedulers.py` is a flat registry -- MILP through
CVXPY (MOSEK / GUROBI / HIGHS / SCIP / CBC), CP-SAT, HEFT, PEFT, EDF,
critical-path, min-min, max-min, FIFO, round-robin, random-list, simulated
annealing, fastest-device. They share one signature:

    scheduler(workload, **kw) -> (t, alpha, fused_workload, fusion_map)

so any of them substitutes into the entry points with no other plumbing.

TWO AXES, AND THEY ARE NOT THE SAME AXIS. `--solver` picks the *strategy*
(`milp` = one global solve, `greedy` = list scheduling with periodic
refinement, `decomposed` = per-network MILP); `--scheduler` picks which entry
in the registry the `milp` strategy calls. Confusing them is easy:
`--scheduler heft --solver greedy` silently ignores the scheduler.

This drives `scripts/run_xpurt_schedule.py` rather than building a workload
itself. That is deliberate -- a second workload builder in an example is a
second thing that can disagree with the first.

TWO THINGS THIS IS CAREFUL ABOUT, both of which have produced wrong readings
in this project before.

**MOSEK is a bounded upper bound at this size, not the optimum.** The
monolithic MILP does not converge on a real multi-network workload;
`scripts/mosek_decompose_by_network.py` does, and its own header says the
result is a bound. "MOSEK 4.1 ms vs greedy 4.4 ms" invites the reading that
4.1 is optimal. It is not.

**Makespan is term 7 of 9.** Sorting solvers by makespan is the easiest way to
pick the wrong one. `xpu-rt/candidate_objective.py` ranks hard deadline misses
first and standalone kernel cycles LAST, and its worked examples are cases
where makespan gets it backwards. Use `scripts/compare_candidates.py` to
adjudicate; use this to see that the alternatives differ at all.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import REPO, head, note, step                      # noqa: E402

DEFAULT_SPEC = REPO / "data" / "toplevel" / "networks_k1_mb.json"

#: (label, extra argv). `greedy` needs no external solver, which is why it is
#: the one this example can always run.
#: (label, metrics-filename key, extra argv).
ARMS = [
    ("greedy",          "greedy",          ["--solver", "greedy"]),
    ("greedy_periodic", "greedy_periodic", ["--solver", "greedy_periodic"]),
    ("heft",            "heft",            ["--solver", "milp", "--scheduler", "heft"]),
    ("peft",            "peft",            ["--solver", "milp", "--scheduler", "peft"]),
    ("edf",             "edf",             ["--solver", "milp", "--scheduler", "edf"]),
    ("critical_path",   "critical_path",   ["--solver", "milp", "--scheduler", "critical_path"]),
    ("min_min",         "min_min",         ["--solver", "milp", "--scheduler", "min_min"]),
    ("fifo",            "fifo",            ["--solver", "milp", "--scheduler", "fifo"]),
]


def _metrics_for(spec: Path, arm_key: str) -> dict | None:
    """The metrics file THIS arm wrote.

    The filename carries the scheduler name, so match on it. Globbing and
    taking the newest looked right and was not: several specs share a stem
    prefix, so `networks_k1_mb` also matches `networks_k1_mb_B1`, and the
    table then reported one arm's numbers for every row -- which reads as
    "all the solvers agree" rather than as a bug.
    """
    p = REPO / "schedules" / f"scheduled_{spec.stem}_{arm_key}_profiled_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC,
                    help="workload spec. The default is small and UNCONTENDED, "
                         "which is why every arm returns the same schedule -- "
                         "see the note at the end. data/toplevel/ has "
                         "saturated ones (networks_k1_mb_3model_12hz.json), "
                         "which take minutes rather than seconds.")
    a = ap.parse_args(argv)
    global SPEC
    SPEC = a.spec

    head(f"Several registered schedulers on {SPEC.name}")

    if not SPEC.exists():
        print(f"SKIP: no spec at {SPEC}")
        return 0

    import schedulers                                            # noqa: E402
    step(1, f"registry: {len(schedulers.available_schedulers())} schedulers")
    note(", ".join(schedulers.available_schedulers()))

    step(2, f"solving {SPEC.relative_to(REPO)} several ways")
    rows = []
    for label, key, extra in ARMS:
        cmd = [sys.executable, str(REPO / "scripts" / "run_xpurt_schedule.py"),
               "--networks-json", str(SPEC)] + extra
        t0, wall0 = time.perf_counter(), time.time()
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        secs = time.perf_counter() - t0
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or [""]
            rows.append((label, None, None, secs, f"failed: {tail[0][:44]}"))
            continue
        m = _metrics_for(SPEC, key) or {}
        # `_ms` keys, not `_us`. Both exist and carry the SAME value: the
        # schedule's timebase is milliseconds and the `_us` suffix has been
        # wrong since metrics.py was written, so it is kept as an alias and
        # the honest name is the one to read.
        rows.append((label,
                     m.get("makespan_ms"),
                     m.get("op_deadline_miss_count"),
                     secs, "" if m else "no metrics file"))
        print(f"    {label:<20} ok  ({secs:.1f}s)")

    step(3, "results, deadline misses first")
    print()
    print(f"    {'arm':<20} {'misses':>8} {'makespan_ms':>13} {'wall (s)':>9}   note")
    print(f"    {'-'*20} {'-'*8} {'-'*13} {'-'*9}   {'-'*24}")
    for label, ms, miss, secs, why in sorted(
            rows, key=lambda r: (r[2] is None, r[2] or 0, r[1] or 0)):
        print(f"    {label:<20} {str(miss) if miss is not None else '-':>8} "
              f"{f'{ms:.4f}' if ms is not None else '-':>13} "
              f"{secs:>9.1f}   {why}")

    distinct = {(r[1], r[2]) for r in rows if r[1] is not None}
    print()
    if len(distinct) <= 1 and len(rows) > 1:
        note(f"""
EVERY ARM RETURNED THE SAME SCHEDULE, and on this workload that is the
expected answer rather than a broken example. {SPEC.name} has slack: with
enough machine time for every dispatch, a list scheduler, an
earliest-deadline one and a critical-path one all place the same work in the
same order, and there is nothing for a smarter policy to win.

Heuristics separate when the machine is CONTENDED. Re-run against a saturated
spec to see them diverge -- they take minutes rather than seconds:

    python examples/solvers/compare_solvers.py \\
        --spec data/toplevel/networks_k1_mb_3model_12hz.json

That an example's headline result is "no difference" is worth leaving in
place. A comparison that only ever runs on inputs where it looks impressive
teaches the wrong thing about when to reach for it.""")

    note("""
Ordered by DEADLINE MISSES, then makespan -- not by makespan alone. Makespan is
term 7 of the 9 in xpu-rt/candidate_objective.py; hard deadline misses are
term 1. A row at the top of a makespan-sorted table can be the worst choice
available for a workload whose networks have deadlines.

`op_deadline_miss` counts DISPATCHES, not instances. The per-instance number
comes from trace_metrics against a measured run, and the two legitimately
disagree -- see xpu-rt/trace_metrics.py, which exists because they once
disagreed silently.

To adjudicate two candidates properly:
    python scripts/compare_candidates.py --baseline-schedule A.json \\
        --candidate-schedule B.json --windows-from <spec>.json""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

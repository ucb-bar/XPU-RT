#!/usr/bin/env python3
"""Can the board's walker actually execute this schedule as written?

WHY A SEPARATE CHECK, WHEN THE SOLVERS ALREADY CONSTRAIN THIS
------------------------------------------------------------
The MILP/CP-SAT solvers enforce no-overlap as a CONSTRAINT, so a schedule that
comes straight out of one satisfies it by construction and this check is a
tautology. The schedules that do NOT come straight out of one are the reason
this exists:

  * `mosek_decompose_by_network.py` solves each network separately and STITCHES
    them sequentially under shared CPU_P/CPU_E capacity. The stitch is ordinary
    Python, outside any solver's constraint set.
  * `packing.combine_solved_windows` concatenates per-window solutions and then
    runs `overlap_fixer` push-back passes -- which is an admission that the
    combined result can overlap before that pass, and nothing re-checks it
    after.
  * any hand-edited, merged, or hot-swapped schedule.

WHAT THE WALKER DOES IF THIS FAILS, which is the point
------------------------------------------------------
`harness_xpurt` runs ONE worker per (core_kind, hart) and dispatches are
NON-PREEMPTIBLE. So a double-booked core does not fail loudly on the board: the
second dispatch simply waits, and the run comes out SLOWER than predicted. That
is indistinguishable, in a results table, from "the profile was optimistic" or
"contention" -- and it would be attributed to the hardware rather than to the
schedule. `ModelBlaster/pipeline/ingest_xpurt_schedule.py` catches unknown
machine labels, out-of-range core indices, dependency cycles and forward edges;
it does not catch temporal double-booking.

Everything here is host-side and costs no board time.

Exit 0 feasible, 1 infeasible, 2 could not be established.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))

from capabilities import (  # noqa: E402
    K1_CAPABILITIES, machine_type_prefix)
from schedule_trace import split_hardware_target  # noqa: E402

#: Physical harts on the K1: two 4-core clusters. A target indexing past this
#: is not a scheduling error, it is a schedule for a different machine.
K1_HARTS_PER_CLUSTER = 4


def intervals_by_machine(dispatches: Dict[str, dict]
                         ) -> Dict[str, List[Tuple[float, float, str]]]:
    """`{machine: [(start_ms, end_ms, key), ...]}`, start-sorted.

    A dispatch sharded across N machines occupies ALL of them for its whole
    duration -- that is what makes it one dispatch rather than N. So it is
    entered once per machine it holds.
    """
    out: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for key, d in dispatches.items():
        start = float(d.get("start_time", 0.0))
        end = start + float(d.get("duration", 0.0))
        cluster, cores = split_hardware_target(d.get("hardware_target", "CPU#0"))
        raw = str(d.get("hardware_target", ""))
        parts = [p for p in raw.split("+") if p] or [f"{cluster}#{cores[0]}"]
        for m in parts:
            out[m].append((start, end, key))
    for m in out:
        out[m].sort()
    return dict(out)


def find_overlaps(by_machine, tol_ms: float):
    """Every pair on one machine whose intervals intersect by more than `tol_ms`.

    A SWEEP, not an adjacent-pair scan. Comparing only neighbours in start
    order misses the case that matters most here: one long dispatch spanning
    several short ones. With A=[0,100], B=[10,20], C=[30,40], the neighbour
    scan finds A-B and then compares B against C -- which do not overlap -- and
    reports one conflict where there are two. Under-reporting the extent of a
    double-booking is exactly the wrong direction for a feasibility check,
    because the serialisation cost scales with how much was hidden.
    """
    bad = []
    for machine, ivs in sorted(by_machine.items()):
        active: List[Tuple[float, float, str]] = []
        for s1, e1, k1 in ivs:
            active = [iv for iv in active if iv[1] > s1 + tol_ms]
            for s0, e0, k0 in active:
                depth = min(e0, e1) - max(s0, s1)
                if depth > tol_ms:
                    bad.append({"machine": machine,
                                "overlap_ms": round(depth, 6),
                                "a": k0, "a_window": (s0, e0),
                                "b": k1, "b_window": (s1, e1)})
            active.append((s1, e1, k1))
    bad.sort(key=lambda r: -r["overlap_ms"])
    return bad


def find_dependency_violations(dispatches, tol_ms: float):
    """A dependency that has not finished when its dependent starts."""
    end = {k: float(d.get("start_time", 0.0)) + float(d.get("duration", 0.0))
           for k, d in dispatches.items()}
    bad = []
    for key, d in dispatches.items():
        start = float(d.get("start_time", 0.0))
        for dep in d.get("dependencies") or ():
            if dep not in end:
                bad.append({"key": key, "dep": dep, "why": "unknown dependency"})
                continue
            slip = end[dep] - start
            if slip > tol_ms:
                bad.append({"key": key, "dep": dep,
                            "why": f"dep ends {slip:.6f} ms after this starts"})
    return bad


def find_forward_edges(dispatches, tol_ms: float):
    """Dependencies that start STRICTLY later than the dispatch needing them.

    Related to the assertion `ingest_xpurt_schedule.py` makes at load time --
    every edge must point backward in its entry table or the walker deadlocks
    mid-run -- but deliberately weaker, and the difference is not pedantry.

    The obvious implementation sorts by `(start_time, key)` and flags any
    dependency that lands at a later index. It reports false conflicts on TIES,
    and ties are common and legitimate: a zero-duration op (`view`,
    `chunk2_c1`) finishes at the instant it starts, so its successor starts at
    exactly the same timestamp. Sorted by key, `..._dispatch_10` then precedes
    `..._dispatch_9` because "1" sorts before "9", and a perfectly good
    schedule is reported infeasible.

    Measured: `scheduled__iter_baseline_decomposed_profiled.json` is flagged by
    the key-ordered version at `yolov8_nano_dispatch_10 -> _9`, where both
    start at 14.66098 ms and dispatch 9 has duration 0.0. The real ingest
    topologically sorts, so it does not have this problem either.

    Only a strictly later start is unambiguously wrong under any tie-break, so
    that is what is flagged. A same-instant dependency that has NOT finished is
    already caught by `find_dependency_violations`.
    """
    start = {k: float(d.get("start_time", 0.0)) for k, d in dispatches.items()}
    bad = []
    for key, d in dispatches.items():
        for dep in d.get("dependencies") or ():
            if dep in start and start[dep] > start[key] + tol_ms:
                bad.append({"key": key, "dep": dep,
                            "delta_ms": round(start[dep] - start[key], 6)})
    return bad


def find_out_of_range_targets(dispatches, harts_per_cluster: int):
    bad = []
    for key, d in dispatches.items():
        raw = str(d.get("hardware_target", ""))
        for part in [p for p in raw.split("+") if p]:
            _, _, idx = part.partition("#")
            try:
                if idx and int(idx) >= harts_per_cluster:
                    bad.append({"key": key, "target": part,
                                "why": f"core #{idx} but the cluster has "
                                       f"{harts_per_cluster}"})
            except ValueError:
                bad.append({"key": key, "target": part,
                            "why": "unparseable core index"})
    return bad


def find_illegal_implementations(dispatches, capabilities=None):
    """Dispatches asking for an implementation their core cannot execute.

    WHY THIS IS NOT REDUNDANT WITH `capabilities.check_profile_hw_map`. That
    check runs when the workload is BUILT and sees one implementation per
    machine KIND -- it is the right guard for a Level-1 config that declares
    `profile_hw: {cpu_p: ime, cpu_e: rvv_x60}`. It cannot see a Level-2
    schedule, where the solver picks an implementation PER DISPATCH from
    `build_machine_combinations_with_impls` and writes it into the dispatch as
    `impl`. Nothing between that choice and the board re-checks it, and the
    two-block transformer schedule is the first artifact in this repo that
    exercises both clusters and both implementations at once.

    WHAT THE BOARD DOES IF THIS IS WRONG, which is why it is INFEASIBLE and
    not a warning. Unlike double-booking -- which merely runs slow -- an IME
    dispatch scheduled onto cluster 1 does not degrade. `smt.vmadot` is not
    implemented on harts 4-7: the process takes SIGILL and the run dies with
    no output at all, so the failure arrives as a missing results file rather
    than as a wrong number.

    A dispatch with no `impl` is not checked. Every schedule written before
    `postprocessing.output_scheduled_json` learned to record it is in that
    state, and inventing `rvv` for them would be legal everywhere and prove
    nothing.
    """
    caps = K1_CAPABILITIES if capabilities is None else capabilities
    bad = []
    for key, d in dispatches.items():
        impl = d.get("impl")
        if not impl:
            continue
        raw = str(d.get("hardware_target", ""))
        for part in [p for p in raw.split("+") if p]:
            kind = machine_type_prefix(part)
            allowed = caps.get(kind)
            if allowed is None or impl in allowed:
                continue
            bad.append({"key": key, "target": part, "impl": impl,
                        "why": f"{kind} executes {sorted(allowed)}, not "
                               f"{impl!r}"})
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--harts-per-cluster", type=int,
                    default=K1_HARTS_PER_CLUSTER)
    ap.add_argument("--tol-ms", type=float, default=1e-6,
                    help="overlap below this is float noise, not a conflict")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    schedule = json.load(open(a.schedule))
    dispatches = schedule.get("dispatches") or {}
    if not dispatches:
        print(f"{a.schedule}: no dispatches", file=sys.stderr)
        return 2

    by_machine = intervals_by_machine(dispatches)
    overlaps = find_overlaps(by_machine, a.tol_ms)
    dep_bad = find_dependency_violations(dispatches, a.tol_ms)
    fwd = find_forward_edges(dispatches, a.tol_ms)
    oor = find_out_of_range_targets(dispatches, a.harts_per_cluster)
    illegal = find_illegal_implementations(dispatches)

    makespan = max(float(d.get("start_time", 0.0)) + float(d.get("duration", 0.0))
                   for d in dispatches.values())
    busiest = max(((m, sum(e - s for s, e, _ in ivs))
                   for m, ivs in by_machine.items()),
                  key=lambda x: x[1], default=("-", 0.0))

    print(f"{len(dispatches)} dispatches over {len(by_machine)} machine(s), "
          f"makespan {makespan:.3f} ms")
    print(f"  busiest machine: {busiest[0]} at {busiest[1]:.3f} ms "
          f"({busiest[1] / makespan * 100:.1f}% of the makespan)")

    ok = True
    if overlaps:
        ok = False
        stolen = sum(r["overlap_ms"] for r in overlaps)
        print(f"\nINFEASIBLE: {len(overlaps)} double-booked interval(s) on a "
              f"single machine, {stolen:.3f} ms of overlap in total.")
        print("  The walker runs one worker per (core_kind, hart) and does not "
              "preempt, so it will SERIALISE these -- the run comes out slower "
              "than predicted, and it looks like contention.")
        for r in overlaps[:a.top]:
            print(f"    {r['machine']}: {r['a']} [{r['a_window'][0]:.3f}, "
                  f"{r['a_window'][1]:.3f}] vs {r['b']} "
                  f"[{r['b_window'][0]:.3f}, {r['b_window'][1]:.3f}] "
                  f"-- {r['overlap_ms']:.3f} ms")
        if len(overlaps) > a.top:
            print(f"    ... and {len(overlaps) - a.top} more")
    if dep_bad:
        ok = False
        print(f"\nINFEASIBLE: {len(dep_bad)} dependency violation(s).")
        for r in dep_bad[:a.top]:
            print(f"    {r['key']} <- {r['dep']}: {r['why']}")
    if fwd:
        ok = False
        print(f"\nINFEASIBLE: {len(fwd)} forward dependency edge(s); "
              f"ingest_xpurt_schedule refuses these and the walker would "
              f"deadlock mid-run.")
        for r in fwd[:a.top]:
            print(f"    {r['key']} depends on {r['dep']}, which starts "
                  f"{r['delta_ms']:.6f} ms later")
    if oor:
        ok = False
        print(f"\nINFEASIBLE: {len(oor)} target(s) outside this machine.")
        for r in oor[:a.top]:
            print(f"    {r['key']}: {r['target']} -- {r['why']}")
    if illegal:
        ok = False
        print(f"\nINFEASIBLE: {len(illegal)} dispatch(es) request an "
              f"implementation their core cannot execute. This one SIGILLs on "
              f"the board -- it does not run slowly, it produces no output.")
        for r in illegal[:a.top]:
            print(f"    {r['key']}: {r['impl']} on {r['target']} -- {r['why']}")
        if len(illegal) > a.top:
            print(f"    ... and {len(illegal) - a.top} more")

    if ok:
        print("\nFEASIBLE: no double-booking, no dependency violation, no "
              "forward edge, every target on the board, every implementation "
              "available where it was placed.")
    if a.json:
        json.dump({"schedule": a.schedule, "feasible": ok,
                   "n_dispatches": len(dispatches), "makespan_ms": makespan,
                   "overlaps": overlaps, "dependency_violations": dep_bad,
                   "forward_edges": fwd, "out_of_range_targets": oor,
                   "illegal_implementations": illegal},
                  open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

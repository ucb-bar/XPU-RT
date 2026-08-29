#!/usr/bin/env python3
"""Rank granularity candidates for a solved schedule, and emit a hint.

WHY THIS FILE EXISTS AGAIN. `ModelBlaster/scripts/decision_loop.py` shells out
to `XPU-RT/scripts/granularity_loop.py`, and both rewriters' docstrings and
half a dozen ModelBlaster notes reference it -- but the file was removed with
the IREE/FireSim path it was written for. `decision_loop --help` works and a
real run dies at its first step. This restores the driver against the CURRENT
tooling.

WHAT IT PRODUCES. `granularity_result.json`, the contract `decision_loop`
reads:

    {"candidates": [{"id", "type", "affected", "makespan_delta_us", ...}]}

`type` is `split_heavy_dispatch` or `fuse_linear_chain`, the two
`decision_loop.classify_realizability` accepts, and `affected` is a list of
`<network><instance>_dispatch_<id>` keys.

`makespan_delta_us` IS A PREDICTION AND IS LABELLED ONE. Every candidate
carries `predicted_basis` naming the arithmetic behind it. The number ranks
candidates; it does not decide them. The decision belongs to
`candidate_objective` after the rewrite has been built and measured -- see
`scripts/compare_candidates.py`. Ranking on a prediction and deciding on a
measurement are different jobs, and conflating them is how a rung gets accepted
because a heuristic liked it.

WHAT IT DELIBERATELY DOES NOT DO. It does not rewrite, build, profile or
accept. `decision_loop` drives those, and the acceptance term is not this
file's to choose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

import granularity_advisor as ga  # noqa: E402
import job_names  # noqa: E402

#: A dispatch at or below this is "tiny": fusing it saves more launch overhead
#: than it costs. The floor is measured -- the smallest dispatch on this board
#: costs 62 rdtime ticks at 24 MHz, i.e. 2.6 us (runbook section 3) -- and the
#: threshold is an order above it so a chain has to be genuinely overhead-bound.
TINY_MS = 0.026

#: Per-dispatch launch overhead used to PREDICT a fusion's saving. The measured
#: floor above is the best estimate available: it is what a dispatch costs when
#: it computes almost nothing.
LAUNCH_OVERHEAD_MS = 0.0026


def _disp_id(dispatch_key: str) -> int:
    """`'dronet0_dispatch_5'` -> 5, and -1 for a key without one."""
    if "_dispatch_" not in dispatch_key:
        return -1
    tail = dispatch_key.rsplit("_dispatch_", 1)[1]
    return int(tail) if tail.isdigit() else -1


def load_schedule(path):
    with open(path) as f:
        return json.load(f)


def known_networks(schedule, networks_json=None):
    if networks_json and os.path.exists(networks_json):
        spec = json.load(open(networks_json))
        nets = set(spec.get("networks") or ())
        if nets:
            return nets
    md = schedule.get("metadata") or {}
    return set(md.get("periodic_networks") or ()) or None


def split_candidates(records, periodic, known, max_per_type):
    """Dispatches longer than the tightest free slot they must fit inside.

    Same criterion as `compile_advice.blocking_advice`: a dispatch is a split
    candidate when it cannot fit the slot between periodic releases, because
    it then blocks non-preemptively for longer than the slot exists.
    """
    if not periodic:
        return []
    slot_ms = min(periodic.values())
    out = []
    for r in records:
        if r.duration <= slot_ms:
            continue
        out.append({
            "id": f"split:{r.dispatch_key}",
            "type": "split_heavy_dispatch",
            "network": r.base_id,
            "affected": [r.dispatch_key],
            # Splitting cannot remove work; what it removes is the part of the
            # dispatch that overruns the slot and therefore blocks.
            "makespan_delta_us": round(-(r.duration - slot_ms) * 1000.0, 3),
            "predicted_basis": (f"duration {r.duration:.3f} ms exceeds the "
                                f"tightest periodic free slot {slot_ms:.3f} ms; "
                                f"the overrun is what blocks"),
            "duration_ms": round(r.duration, 6),
            "free_slot_ms": round(slot_ms, 6),
        })
    out.sort(key=lambda c: c["makespan_delta_us"])
    return out[:max_per_type]


def fuse_candidates(records, known, max_per_type):
    """Maximal runs of tiny dispatches within one instance, in id order.

    A chain is a fusion candidate only if every member is tiny: fusing a tiny
    op into a large one saves one launch and makes a longer non-preemptible
    blocker, which is the trade the 36%-slower mlp_control rung already lost.
    """
    by_instance = defaultdict(list)
    for r in records:
        by_instance[r.instance_id].append(r)
    out = []
    for inst, rs in sorted(by_instance.items()):
        rs.sort(key=lambda r: _disp_id(r.dispatch_key))
        run = []
        def flush(run):
            if len(run) < 2:
                return
            ids = [_disp_id(x.dispatch_key) for x in run]
            out.append({
                "id": f"fuse:{inst}_{ids[0]}-{ids[-1]}",
                "type": "fuse_linear_chain",
                "network": run[0].base_id,
                "affected": [x.dispatch_key for x in run],
                "makespan_delta_us": round(
                    -LAUNCH_OVERHEAD_MS * (len(ids) - 1) * 1000.0, 3),
                "predicted_basis": (
                    f"{len(ids)} tiny dispatches (each <= {TINY_MS} ms) become "
                    f"one, saving {len(ids) - 1} launches at the measured "
                    f"{LAUNCH_OVERHEAD_MS * 1000:.1f} us floor"),
                "n_dispatches": len(ids),
            })
        for r in rs:
            if r.duration <= TINY_MS:
                run.append(r)
            else:
                flush(run); run = []
        flush(run)
    out.sort(key=lambda c: c["makespan_delta_us"])
    return out[:max_per_type]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks-json", default=None,
                    help="workload spec; supplies the real network names")
    ap.add_argument("--schedule", default=None,
                    help="a solved schedule to analyse. If absent, derived "
                         "from --networks-json + --baseline-solver.")
    ap.add_argument("--baseline-solver", default="greedy")
    ap.add_argument("--max-per-type", type=int, default=3)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emit-hint", default=None)
    a = ap.parse_args()

    sched_path = a.schedule
    if not sched_path and a.networks_json:
        stem = os.path.splitext(os.path.basename(a.networks_json))[0]
        guess = os.path.join(_REPO, "schedules",
                             f"scheduled_{stem}_{a.baseline_solver}_profiled.json")
        if os.path.exists(guess):
            sched_path = guess
    if not sched_path or not os.path.exists(sched_path):
        print(f"no schedule to analyse (looked for {sched_path!r}). Solve one "
              f"first with scripts/run_xpurt_schedule.py, or pass --schedule.",
              file=sys.stderr)
        return 2

    schedule = load_schedule(sched_path)
    known = known_networks(schedule, a.networks_json)
    records = ga.from_schedule_json(schedule, known)
    periodic, _ = ga.group_by_periodicity(records)

    cands = (split_candidates(records, periodic, known, a.max_per_type)
             + fuse_candidates(records, known, a.max_per_type))

    os.makedirs(a.out_dir, exist_ok=True)
    result = {
        "schedule": sched_path,
        "networks": sorted(known) if known else [],
        "periodic_periods_ms": {k: round(v, 4) for k, v in periodic.items()},
        "candidates": cands,
        "_note": ("makespan_delta_us is PREDICTED and ranks candidates only. "
                  "The verdict belongs to candidate_objective after the "
                  "rewrite is built and measured -- scripts/compare_candidates.py."),
    }
    out = os.path.join(a.out_dir, "granularity_result.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")
    for c in cands:
        print(f"  {c['type']:22s} {c['id']:38s} predicted {c['makespan_delta_us']:>10.1f} us")
    if not cands:
        print("  no candidate: no dispatch overruns its slot and no tiny chain "
              "is fusable. That is a result, not an empty run.")

    if a.emit_hint:
        splits = [c for c in cands if c["type"] == "split_heavy_dispatch"]
        by_net = defaultdict(list)
        for c in splits:
            did = int(c["affected"][0].rsplit("_dispatch_", 1)[1])
            by_net[c["network"]].append({"op": did, "n_splits": 2})
        hint = {
            "contract": "modelblaster.split_hints/v1",
            "reason": "granularity_loop: dispatches exceeding their periodic slot",
            "networks": [{"network": n, "split_ops": ops}
                         for n, ops in sorted(by_net.items())],
        }
        with open(a.emit_hint, "w") as f:
            json.dump(hint, f, indent=1)
        print(f"wrote {a.emit_hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

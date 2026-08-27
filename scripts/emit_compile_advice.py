#!/usr/bin/env python3
"""Emit compile_advice.json from measured K1 profiles and a schedule.

Deliberately reads *measurements*, not the solver's own predictions: the point
is to tell the compiler something it cannot already know, and the solver's
durations came from these same profiles anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "xpu-rt"))

from compile_advice import (  # noqa: E402
    blocking_advice, implementation_advice, load_profiles, overhead_advice,
    write_advice,
)


def is_linear_chain(graph_path: str) -> bool:
    """In/out degree <= 1 everywhere -- the precondition fusion.py requires."""
    if not os.path.exists(graph_path):
        return False
    g = json.load(open(graph_path))
    disp = g.get("dispatches", {})
    outdeg = {k: 0 for k in disp}
    for k, v in disp.items():
        deps = v.get("dependencies", []) or []
        if len(deps) > 1:
            return False
        for d in deps:
            if d in outdeg:
                outdeg[d] += 1
    return all(v <= 1 for v in outdeg.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default="gen")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", default="artifacts/k1_run/compile_advice.json")
    ap.add_argument("--models", default="mlp:mlp.q.int8,dronet:dronet.q.int8")
    ap.add_argument("--impls", default="RVV,scalar,IME")
    ap.add_argument("--baseline-impl", default="RVV")
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    periods = (sched.get("metadata") or {}).get("periodic_networks") or {}
    impls = [i for i in a.impls.split(",") if i]

    # The tightest periodic slot any long dispatch has to fit between releases.
    free_slot_ms = min(periods.values()) if periods else 0.0

    advice, notes = [], {}
    for spec in a.models.split(","):
        model, basename = spec.split(":")
        profs = load_profiles(a.gen_root, a.target, model, basename, impls)
        if not profs:
            print(f"WARN no profiles for {model}", file=sys.stderr)
            continue
        base = profs.get(a.baseline_impl, {})
        graph = os.path.join(a.gen_root, "vmfb", model, a.target,
                             a.baseline_impl, basename,
                             f"{basename}_dispatch_graph.json")
        chain = is_linear_chain(graph)
        notes[model] = {
            "implementations_profiled": sorted(profs),
            "n_dispatches": len(base),
            "total_median_ms": round(sum(r["median_ms"] for r in base.values()), 3),
            "linear_chain": chain,
        }
        advice += implementation_advice(model, profs, a.baseline_impl)
        advice += overhead_advice(model, base, chain)
        # Only a model that cannot fit its own period is blocking anything.
        total = sum(r["median_ms"] for r in base.values())
        period = periods.get(model)
        if period and total > period:
            advice += blocking_advice(model, base, free_slot_ms,
                                      misses=len(base))

    # Highest-priority first; the consumer is expected to apply a bounded number.
    advice.sort(key=lambda x: (x.priority, -x.evidence.service_time_us))
    write_advice(a.out, advice, schedule_id=os.path.basename(a.schedule),
                 notes=notes)

    actionable = [x for x in advice if x.recommendation != "unchanged"]
    print(f"wrote {a.out}")
    print(f"  {len(advice)} advice items, {len(actionable)} actionable")
    for x in actionable[:12]:
        print(f"  [p{x.priority}] {x.model}.{x.dispatch_id:<4} "
              f"{x.recommendation:<22} {x.rationale[:95]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

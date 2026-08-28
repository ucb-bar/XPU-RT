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
    blocking_advice, implementation_advice, load_profiles,
    load_profiles_by_cores, load_profiles_by_cores_csv, load_profiles_csv,
    overhead_advice, shard_advice, write_advice,
)
# The canonical granularity analysis. This file used to carry its own
# `is_linear_chain` over a dispatch-graph FILE and its own notion of a free
# slot; both already existed here, and the local copies were the worse of the
# two -- see the free-slot note below.
from granularity_advisor import (  # noqa: E402
    _free_slot_ms, _is_linear_chain, from_schedule_json, group_by_periodicity,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None,
                    help="measured trace CSV. Without it, "
                         "deadline_misses_attributed is 0 -- an unmeasured "
                         "miss count is not evidence.")
    ap.add_argument("--gen-root", default="gen")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", default="artifacts/k1_run/compile_advice.json")
    ap.add_argument("--models", default="mlp:mlp.q.int8,dronet:dronet.q.int8")
    ap.add_argument("--impls", default="RVV,scalar,IME")
    ap.add_argument("--baseline-impl", default="RVV")
    ap.add_argument("--profile-format", choices=("jsonl", "csv"),
                    default="jsonl",
                    help="which producer wrote the profiles. `jsonl` is "
                         "runtime/scripts/profile_k1.py (samples, "
                         "percentiles, cv). `csv` is ModelBlaster's "
                         "pipeline/profile_writer.py (IREE-shape results.csv, "
                         "one mean per dispatch, plus the `implementation` "
                         "column recording which kernel actually ran) -- the "
                         "only format the corrected rvv_x60 builds exist in.")
    a = ap.parse_args()
    if a.profile_format == "csv":
        get_profiles, get_profiles_by_cores = (load_profiles_csv,
                                               load_profiles_by_cores_csv)
    else:
        get_profiles, get_profiles_by_cores = (load_profiles,
                                              load_profiles_by_cores)

    sched = json.load(open(a.schedule))
    periods = (sched.get("metadata") or {}).get("periodic_networks") or {}
    impls = [i for i in a.impls.split(",") if i]

    # The tightest periodic FREE SLOT, not the tightest period.
    #
    # This previously read `min(periods.values())` while calling itself a slot.
    # A period is the whole interval between releases; the slot is what is left
    # of it after the model's own work -- period * (1 - utilization), computed
    # over a duration-weighted critical path. Using the raw period overstates
    # the room available by exactly the amount the model already occupies, so
    # every "this dispatch does not fit" judgement was measured against a
    # budget nobody has.
    #
    # granularity_advisor already computes the real thing; the local version was
    # a second, worse source of truth.
    # NOTE on group_by_periodicity's return: the second element is the
    # NON-periodic groups, not all of them. For a fully periodic workload it is
    # empty, so indexing it by a periodic base silently yields nothing and every
    # free slot comes out 0 -- which disables split and shard advice entirely
    # without saying so. Group the periodic records here instead.
    records = from_schedule_json(sched)
    periods_by_base, _non_periodic = group_by_periodicity(records)
    by_base: dict = {}
    for r in records:
        by_base.setdefault(r.base_id, []).append(r)
    slots = {base: _free_slot_ms(base, by_base[base], T)
             for base, T in periods_by_base.items() if base in by_base}

    def budget_for(model: str) -> float:
        """The budget a dispatch of `model` has to fit into, in ms.

        Two corrections over the previous `min(periods.values())`:

        * per model, not a global minimum. Taking the minimum across models
          meant one saturated model zeroed the budget for every other one, so a
          workload containing DroNet produced no advice about the MLP at all.
        * a ZERO free slot means the model already consumes its whole period --
          which is exactly when its long dispatches most need attention, not
          least. Gating on `slot > 0` silently excluded the saturated model, so
          the one case the advisor exists for produced nothing. When the slot
          is zero the period itself is the budget: an instance that cannot fit
          its own period is the finding.
        """
        slot = slots.get(model, 0.0)
        if slot > 0:
            return slot
        return periods_by_base.get(model, 0.0)

    def dispatch_budget(model: str, profile: dict) -> float:
        """Per-dispatch budget for a SATURATED model, in ms.

        For a model that fits its period, `budget_for` is the right test: a
        dispatch longer than the free slot is the thing blocking the slot.

        For a model that does not fit, that test finds nothing -- and finds
        nothing precisely when the model is in the worst trouble. DroNet needs
        113.7 ms against a 33.3 ms window, yet its largest single dispatch is
        22.9 ms, under the period. Comparing each dispatch to the whole period
        therefore reports "no dispatch is too long" about a model that misses
        every deadline.

        The honest per-dispatch target is its PROPORTIONAL SHARE of the window:
        if the instance is to fit at all, a dispatch responsible for x% of the
        work has x% of the window to do it in. For DroNet's heaviest
        convolution that is 33.3 * 22.87/113.7 = 6.7 ms -- which it exceeds on
        one core and meets on four, which is exactly the finding.
        """
        period = periods_by_base.get(model, 0.0)
        total = sum(float(r.get("median_ms") or 0.0) for r in profile.values())
        if period <= 0 or total <= period:
            return budget_for(model)
        return period / total  # scale factor; caller multiplies by cost

    free_slot_ms = min((v for v in slots.values() if v > 0), default=0.0)
    if slots:
        print("  free slots (ms): " + ", ".join(
            f"{b}={v:.3f}" + (" [saturated -> budget=period]" if v <= 0 else "")
            for b, v in sorted(slots.items())))

    measured_misses = {}
    if a.trace:
        import trace_metrics
        summary = trace_metrics.summarise_trace(
            trace_metrics.read_trace(a.trace), periods)
        measured_misses = {m: d["instance_deadline_misses"]
                           for m, d in (summary.get("per_model") or {}).items()}
        print("  measured misses: " + ", ".join(
            f"{m}={n}" for m, n in sorted(measured_misses.items())))

    advice, notes = [], {}
    for spec in a.models.split(","):
        model, basename = spec.split(":")
        profs = get_profiles(a.gen_root, a.target, model, basename, impls)
        if not profs:
            print(f"WARN no profiles for {model}", file=sys.stderr)
            continue
        base = profs.get(a.baseline_impl, {})
        # From the schedule's own records rather than a separate dispatch-graph
        # file, so the chain test and the cost data cannot disagree about which
        # graph they describe.
        chain = _is_linear_chain(by_base.get(model, []))
        notes[model] = {
            "implementations_profiled": sorted(profs),
            "n_dispatches": len(base),
            "total_median_ms": round(sum(r["median_ms"] for r in base.values()), 3),
            "linear_chain": chain,
            # What statistic the numbers above ARE. `results.csv` carries one
            # mean per dispatch and the harness that writes it takes a single
            # sample, so an advice document derived from it must not be read as
            # if it rested on a median over warm repetitions.
            "stat_basis": sorted({r.get("stat_basis", "median_of_reps")
                                  for r in base.values()}),
            # Which kernel actually ran. Curated kernels are looked up by exact
            # op name, so fused ops used to fall back to the scalar reference
            # inside builds labelled `rvv_x60`; a profile that cannot say which
            # kernel it timed cannot support advice about that kernel.
            "baseline_implementations": sorted(
                {r.get("implementation", "") for r in base.values()}) or None,
            "profile_csv": sorted({r["source_csv"] for r in base.values()
                                   if r.get("source_csv")}) or None,
            "core_counts_profiled": sorted(
                get_profiles_by_cores(a.gen_root, a.target, model, basename,
                                      a.baseline_impl)),
        }
        advice += implementation_advice(model, profs, a.baseline_impl)
        advice += overhead_advice(model, base, chain)
        # Only a model that cannot fit its own period is blocking anything.
        total = sum(r["median_ms"] for r in base.values())
        period = periods.get(model)
        if period and total > period:
            # `misses` is deadline misses ATTRIBUTED to this model. It used to
            # be passed as len(base) -- the number of DISPATCHES -- so every
            # split recommendation carried a dispatch count in a field named
            # deadline_misses_attributed, identically, for every item. Anyone
            # reading that field downstream was reading a mislabelled constant.
            #
            # The real figure comes from a measured trace, so when none is
            # supplied the honest value is 0 rather than a number that happens
            # to be available.
            advice += blocking_advice(model, base, budget_for(model),
                                      misses=measured_misses.get(model, 0))
        # Sharding is judged on measured scaling with core count, which no
        # single-topo profile can show.
        by_cores = get_profiles_by_cores(a.gen_root, a.target, model,
                                         basename, a.baseline_impl)
        if len(by_cores) > 1:
            period = periods.get(model, 0.0)
            if period and total > period:
                # Saturated: each dispatch gets its proportional share of the
                # window. shard_advice compares cost to a single budget, so
                # pass the share of the LARGEST dispatch -- the one that has to
                # shrink first for the instance to fit at all.
                biggest = max((float(r.get("median_ms") or 0.0)
                               for r in base.values()), default=0.0)
                target = period * (biggest / total) if total else 0.0
            else:
                target = budget_for(model)
            advice += shard_advice(model, by_cores, target)

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

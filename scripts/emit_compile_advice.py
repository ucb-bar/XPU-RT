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
    overhead_advice, shard_advice, unfuse_advice, write_advice,
)
# The canonical granularity analysis. This file used to carry its own
# `is_linear_chain` over a dispatch-graph FILE and its own notion of a free
# slot; both already existed here, and the local copies were the worse of the
# two -- see the free-slot note below.
from granularity_advisor import (  # noqa: E402
    _free_slot_ms, _is_linear_chain, from_schedule_json, group_by_periodicity,
)


def _backend_family(impl: str) -> str:
    """`rvv_x60` -> `rvv`: the curated kernel directory is per FAMILY.

    `unfuse_advice` globs `<kernels_dir>/<backend>_<op>_*.c` and the files are
    named `rvv_conv2d_s8_...`, not `rvv_x60_conv2d_s8_...` -- the same
    family/variant relationship `generate_kernels` calls `curated_aliases`.
    Passing the variant tag here matches nothing and the verb silently refuses,
    which looks identical to "no unfuse was warranted".
    """
    return (impl or "").split("_")[0] or impl


def _load_irs(specs) -> dict:
    """`{network: {dispatch_id: op}}` from repeated `<network>:<graph.json>`."""
    out = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--ir needs <network>:<graph.json>, got {spec!r}")
        net, path = spec.split(":", 1)
        with open(path) as fh:
            graph = json.load(fh)
        out[net] = {o["dispatch_id"]: o for o in graph.get("ops", [])
                    if o.get("dispatch_id") is not None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=None,
                    help="measured trace CSV. Without it, "
                         "deadline_misses_attributed is 0 -- an unmeasured "
                         "miss count is not evidence.")
    # Defaults describe the LIVE path (ModelBlaster/rvv_x60), not the retired
    # IREE one. They used to be `gen` / `RVV,scalar,IME` / `RVV` / `jsonl`,
    # which resolve nothing here: ModelBlaster writes `rvv_x60` and `scalar`
    # under `gen_mb`. The old defaults did not error -- `scalar` alone
    # resolved, so the "no profiles" warning never fired, `profs.get("RVV")`
    # returned {}, and the run wrote an EMPTY advice file and exited 0.
    # Measured: 118 advice items with the explicit flags, 0 with the defaults,
    # and nothing said why. The retired tree stays reachable by passing them.
    ap.add_argument("--ir", action="append", default=[],
                    help="<network>:<graph.json>, repeatable. Required for "
                         "`unfuse` advice: it is the one verb whose trigger is "
                         "not in the profile alone, needing the op's `sub_ops` "
                         "to know what unfusing would restore.")
    ap.add_argument("--kernels-dir", default=None,
                    help="curated kernel library, e.g. ModelBlaster/kernels/rvv. "
                         "Without it `unfuse_advice` refuses rather than "
                         "guessing a constituent has a kernel to land on.")
    ap.add_argument("--gen-root", default="gen_mb")
    ap.add_argument("--target", default="spacemit_x60")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", default="artifacts/k1_run/compile_advice.json")
    ap.add_argument("--models", default="mlp:mlp.q.int8,dronet:dronet.q.int8")
    ap.add_argument("--impls", default="rvv_x60,scalar")
    ap.add_argument("--baseline-impl", default="rvv_x60")
    ap.add_argument("--profile-format", choices=("jsonl", "csv"),
                    default="csv",
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

        Returns the share of the LARGEST dispatch, as a single ms budget.
        `blocking_advice` and `shard_advice` both compare every dispatch against
        one scalar, so the threshold has to be the heaviest dispatch's share --
        the one that must shrink first for the instance to fit at all. A literal
        per-dispatch share would be `cost * period/total` for every dispatch,
        which every dispatch exceeds whenever the model is saturated, so it
        would flag the entire model and say nothing.

        This used to return `period / total` -- a dimensionless scale factor,
        annotated "caller multiplies by cost" -- and no caller ever did, because
        no caller was ever written: the function was dead, `blocking_advice` got
        `budget_for` (the whole period for a saturated model), and the shard
        path recomputed the largest dispatch's share inline. So the case
        described above went exactly as described: a model needing 113.7 ms
        against a 33.3 ms period produced ten `unchanged` items and nothing
        actionable, because no single dispatch exceeded the full period.
        """
        period = periods_by_base.get(model, 0.0) or periods.get(model, 0.0)
        costs = [float(r.get("median_ms") or 0.0) for r in profile.values()]
        total = sum(costs)
        if period <= 0 or total <= period or not costs:
            return budget_for(model)
        return period * (max(costs) / total)

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

    irs_by_network = _load_irs(a.ir)
    advice, notes = [], {}
    for spec in a.models.split(","):
        model, basename = spec.split(":")
        profs = get_profiles(a.gen_root, a.target, model, basename, impls)
        if not profs:
            print(f"WARN no profiles for {model}", file=sys.stderr)
            continue
        # An absent BASELINE is not "no advice", it is "I could not read the
        # thing every comparison is made against". Silently yielding an empty
        # advice document for this is how the old defaults hid themselves.
        if a.baseline_impl not in profs:
            raise SystemExit(
                f"--baseline-impl {a.baseline_impl!r} has no profile for "
                f"{model}. Resolved: {sorted(profs)}. Requested: {impls}. "
                f"Looked under --gen-root {a.gen_root!r} with "
                f"--profile-format {a.profile_format}. Every comparison is "
                f"against the baseline, so continuing would write an empty "
                f"advice document and exit 0.")
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
        # `unfuse` had a producer, a bridge and a rewriter but NO CALLER here,
        # so the document could never contain it: the chain was complete
        # everywhere except at its own first step. Found on the board, where
        # the condition was reconstructed and the emitter still returned zero
        # unfuse items.
        ops_by_id = irs_by_network.get(model)
        if ops_by_id:
            advice += unfuse_advice(model, base, ops_by_id,
                                    kernels_dir=a.kernels_dir,
                                    backend=_backend_family(a.baseline_impl))
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
            advice += blocking_advice(model, base, dispatch_budget(model, base),
                                      misses=measured_misses.get(model, 0))
        # Sharding is judged on measured scaling with core count, which no
        # single-topo profile can show.
        by_cores = get_profiles_by_cores(a.gen_root, a.target, model,
                                         basename, a.baseline_impl)
        if len(by_cores) > 1:
            # The same budget `blocking_advice` was given, from the same
            # function. These two used to compute it separately -- the shard
            # path inline and correctly, the blocking path via `budget_for` and
            # wrongly -- so the two recommendations were made against different
            # thresholds for the same dispatch, and only one of them could be
            # right.
            advice += shard_advice(model, by_cores, dispatch_budget(model, base))

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

#!/usr/bin/env python3
"""Accept or reject a rewritten graph, on the terms the objective actually ranks.

THE GAP THIS CLOSES. `candidate_objective.accept()` is the project's stated
acceptance rule -- nine lexicographic terms, hard deadline misses first,
standalone kernel cycles LAST -- and until now nothing outside the test suite
called it. Every granularity rung was adjudicated by eye on a service-time
percentage, which is term 9 of 9, the one the module's own docstring says is
"never the deciding term". Its two worked examples are exactly the cases a
percentage gets backwards:

    a split making a kernel 5% slower in total cycles but letting DroNet meet
    30 Hz instead of missing 20% of deadlines is a WIN;
    a fusion making a model 10% faster in isolation but creating an 8 ms
    non-preemptible dispatch that breaks a 100 Hz MLP is a LOSS.

Neither can be seen without scheduling the rewritten graph, which is why a
rung that stops at "reprofiled on the board" has not been adjudicated at all.

WHAT IT SCORES. Two solved schedules. Both must come from the same workload
spec apart from the network under test, and from the same solver flags, or the
comparison is between different amounts of work rather than between two
graphs. `--windows-from` reads the spec so the deadline is the declared
`window_duration` rather than the period -- `D = windows_ms.get(m, T)` -- and
so the network names are known, which is what stops a name ending in a digit
from being split in the wrong place (see `job_names`).

TWO THINGS ARE CHECKED BEFORE ANY TERM IS COMPARED, because each of them
produces a verdict that looks perfectly well-formed and means nothing.

`pdb_hash` proves the two solves read DIFFERENT measured costs -- without it
the verdict is about scheduler noise, not about the rewrite.

Per-model INSTANCE COUNTS prove they scheduled the SAME AMOUNT OF WORK. This
one is not hypothetical: the 4 Hz baseline was re-solved without
`--max-periodic-iters 1`, the refinement loop grew mlp_control from 32
instances to 91, and the resulting file sat on disk under the same name as the
baseline three recorded verdicts had used. Nothing complained -- `pdb_hash`
still differed, every term still computed, and a figure rendered from it
reported the opposite verdict for the DroNet x2 rung. A split changes how many
dispatches an instance is made of; it must not change how many instances there
are, so unequal instance counts mean the flags differed, not the graph.

A tie is a REJECTION: `accept()` requires the candidate to be strictly better
on some term before any term it is worse on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
sys.path.insert(0, _HERE)

import candidate_objective as objective  # noqa: E402
from schedule_scoring import instances_per_model, score  # noqa: E402
import schedule_trace  # noqa: E402
import trace_metrics  # noqa: E402
import workload_spec  # noqa: E402


def _row(o):
    return (f"    misses={o.total_misses():<4} "
            f"worst_late={o.worst_lateness():<8.3f} "
            f"p99={o.worst_p99():<9.3f} "
            f"makespan={o.makespan_ms:<9.3f} "
            f"standalone={o.standalone_cycles}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-schedule", required=True)
    ap.add_argument("--candidate-schedule", required=True)
    ap.add_argument("--windows-from", default=None,
                    help="workload spec; supplies window_duration per network "
                         "AND the real network names")
    ap.add_argument("--critical-models", default="",
                    help="comma-separated; these carry the hard deadline term")
    ap.add_argument("--heavy-model", default=None)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--allow-instance-mismatch", action="store_true",
                    help="compare anyway when the two sides scheduled "
                         "different instance counts; the verdict is then "
                         "about the flags as much as the graph")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    windows, known, declared_periods = ({}, None, None)
    if a.windows_from:
        workload = json.load(open(a.windows_from))
        windows, known = workload_spec.windows_and_names(workload)
        declared_periods = workload_spec.periods_ms(workload)
    critical = tuple(m.strip() for m in a.critical_models.split(",") if m.strip())

    base_s = json.load(open(a.baseline_schedule))
    cand_s = json.load(open(a.candidate_schedule))

    # The check that the candidate was solved against different costs at all.
    # solve_hash is preferred where present: pdb_hash fingerprints only the profile CSVs,
    # so an additive solve and a --board-calibration re-solve of the same spec share it and
    # a pdb_hash-only check refuses the board-feedback comparison -- the one the headline
    # result rests on. solve_hash folds in the calibration table, solver and env, so those
    # two are distinguishable; identical solve_hash still means genuinely nothing differed.
    bm = (base_s.get("metadata") or {})
    cm = (cand_s.get("metadata") or {})
    bh, ch = bm.get("solve_hash"), cm.get("solve_hash")
    hash_kind = "solve_hash"
    if not (bh and ch):
        bh, ch = bm.get("pdb_hash"), cm.get("pdb_hash")
        hash_kind = "pdb_hash"
    if bh and ch and bh == ch:
        print(f"REFUSING: both schedules carry the same {hash_kind}, so they were "
              "solved against the SAME measured costs and switches. Whatever the "
              "verdict would be, it is not about the rewrite.", file=sys.stderr)
        if hash_kind == "pdb_hash":
            print("  (neither schedule carries solve_hash: re-solve with a build that "
                  "records solve provenance if these differ only by --board-calibration)",
                  file=sys.stderr)
        return 2

    # The check that they scheduled the same amount of work.
    bi = instances_per_model(base_s, known)
    ci = instances_per_model(cand_s, known)
    if bi != ci:
        msg = ("the two schedules hold different instance counts, so they are "
               "not two graphs -- they are two amounts of work:\n"
               f"    {a.baseline_label:>12}: {bi}\n"
               f"    {a.candidate_label:>12}: {ci}\n"
               "  solve BOTH sides with --max-periodic-iters 1; the refinement "
               "loop grows num_instances and the growth is not equal.")
        if not a.allow_instance_mismatch:
            print(f"REFUSING: {msg}", file=sys.stderr)
            return 2
        print(f"WARNING: {msg}", file=sys.stderr)

    _, base, _ = score(a.baseline_label, base_s, windows, critical,
                       a.heavy_model, known, declared_periods)
    _, cand, _ = score(a.candidate_label, cand_s, windows, critical,
                       a.heavy_model, known, declared_periods)

    ok, why = objective.accept(cand, base)
    order, _ = objective.compare(cand, base)

    print(f"{a.baseline_label:>12}  {_row(base)}")
    print(f"{a.candidate_label:>12}  {_row(cand)}")
    print()
    print(f"  VERDICT: {'ACCEPT' if ok else 'REJECT'}")
    print(f"  {why}")
    if not ok and order == 0:
        print("  (a tie is a rejection: the candidate must be strictly better "
              "on some term before any term it is worse on)")
    if a.json:
        json.dump({"baseline": a.baseline_schedule,
                   "candidate": a.candidate_schedule,
                   "accepted": bool(ok), "why": why,
                   "hash_kind": hash_kind,
                   "baseline_hash": bh, "candidate_hash": ch,
                   "baseline_pdb_hash": bm.get("pdb_hash"),
                   "candidate_pdb_hash": cm.get("pdb_hash"),
                   "baseline_solve_hash": bm.get("solve_hash"),
                   "candidate_solve_hash": cm.get("solve_hash"),
                   "baseline_instances": bi, "candidate_instances": ci,
                   "baseline_terms": objective.terms_dict(base),
                   "candidate_terms": objective.terms_dict(cand)},
                  open(a.json, "w"), indent=1, default=str)
        print(f"  wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Does an UPSTREAM control signal rescue adaptive candidate selection?

    python -m benchmarks.freshness_eval.upstream

Gate B found the hysteretic selector both pointless and unsafe, for two failures
that are independent and were tangled together:

  SATURATION  risk = observed_max_age / phi is measured DOWNSTREAM of the
              mitigation. Under every protective rung it is flat at 1.124 for
              B = 1, 2, 3 and 4 alike, so the signal cannot distinguish 65%
              offered load from 131%.
  LAG         the observation comes from the previous epoch, so the first
              high-contention epoch is always entered on a stale estimate. Here
              the lower rung fails by a 2.7x epoch overrun rather than gracefully,
              and one epoch of that is fatal.

This module changes the signal to one taken upstream of the mitigation: the
OFFERED soft work for the epoch about to be scheduled.

WHY THAT IS NOT AN ORACLE
-------------------------
It is the request count, not the outcome. A runtime performing admission control
necessarily knows how many jobs were submitted for the epoch it is about to
schedule -- that is what it admits or rejects. It does not know the resulting
validity, makespan, or input age; those still depend on the schedule it chooses.
`oracle_contention_aware` in adaptive.py is different: it reads the measured
output-validity of every candidate at that burst and picks the winner.

Because the signal is available AT the boundary rather than one epoch later, this
also removes the lag -- so the two failures are addressed together, and the
comparison below cannot attribute a change to one of them alone. That is stated
rather than hidden; separating them would need a workload where the low rung
degrades gracefully, which this one does not provide.

NO THRESHOLD FITTING
--------------------
The escalation threshold is SWEPT rather than chosen. Picking the value that
separates the measured outcomes would be fitting a one-parameter model to five
data points and reporting the fit as a result. The sweep shows which thresholds
are safe and what each retains, so a reader can see how much of the outcome is
the threshold's doing.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from benchmarks.freshness_eval.adaptive import (  # noqa: E402
    SELECTOR_RUNGS,
    TRAJECTORIES,
    CellTable,
    StrategyResult,
    _outcome,
    run_static,
)
from benchmarks.freshness_eval.headroom import load_rows  # noqa: E402

# Gemmini-side cost per instance, FireSim-measured medians divided by 1e6 under
# the assumed 1 GHz (see the plan's F1 table and the run manifests). Held here as
# named constants with provenance rather than inline numbers, and checked against
# the F1 utilisations by test_upstream.py.
MLP_MS = 0.546
DRONET_MS = 18.614
YOLO64_MS = 67.202
N_MLP = 30
N_DRONET = 6

BASE_GEMMINI_MS = N_DRONET * DRONET_MS + N_MLP * MLP_MS      # 128.064


def offered_utilisation(burst: int, epoch_ms: float = 300.0) -> float:
    """Offered gemmini load for a burst, as a fraction of the epoch.

    This is the physically meaningful form of the upstream signal: it is
    computable at the boundary from the request count plus the profile, and it is
    monotone in the burst, which is exactly what the downstream signal is not.
    """
    return (BASE_GEMMINI_MS + burst * YOLO64_MS) / epoch_ms


def run_upstream(table: CellTable, trajectory: Sequence[int], name: str,
                 epoch_ms: float, *, escalate_at: Sequence[int],
                 rungs: Sequence[str] = SELECTOR_RUNGS) -> StrategyResult:
    """Select on offered work, observed at the epoch boundary (no lag).

    `escalate_at` is ascending burst thresholds: rung i is used once the offered
    burst reaches escalate_at[i-1]. With rungs (C1, C2, C3) and escalate_at
    (3, 5): C1 below 3, C2 from 3, C3 from 5 (i.e. never, on a 0..4 grid).
    """
    if list(escalate_at) != sorted(escalate_at):
        raise ValueError(f"escalate_at must be ascending, got {escalate_at}")
    res = StrategyResult(strategy=f"upstream{tuple(escalate_at)}",
                         trajectory_name=name, epoch_ms=epoch_ms)
    prev: Optional[str] = None
    for e, b in enumerate(trajectory):
        level = sum(1 for t in escalate_at if b >= t)
        level = min(level, len(rungs) - 1)
        cid = rungs[level]
        o = _outcome(e, b, cid, table.get(cid, b))
        o.switched = prev is not None and cid != prev
        o.selector_reason = (f"offered B={b} (util {offered_utilisation(b, epoch_ms):.2f})"
                             f" -> level {level}")
        prev = cid
        res.epochs.append(o)
    return res


def candidate_thresholds(bursts: Sequence[int], n_rungs: int) -> List[Tuple[int, ...]]:
    """Every ascending threshold pair on the burst grid, plus the degenerate ones.

    Includes thresholds that never fire and thresholds that fire immediately, so
    the sweep contains the two ways a selector can collapse into a static policy.
    """
    lo, hi = min(bursts), max(bursts) + 2
    out: List[Tuple[int, ...]] = []
    for t1 in range(lo, hi):
        for t2 in range(t1, hi):
            out.append((t1, t2) if n_rungs >= 3 else (t1,))
    return sorted(set(out))


def evaluate(rows, *, phi: float, epoch_ms: float,
             trajectories: Optional[Dict[str, List[int]]] = None):
    table = CellTable(rows, phi)
    trajectories = trajectories or TRAJECTORIES
    bursts = table.bursts()
    thresholds = candidate_thresholds(bursts, len(SELECTOR_RUNGS))

    report: Dict[str, List[Dict[str, object]]] = {}
    for name, traj in (trajectories or {}).items():
        rowset: List[Dict[str, object]] = []
        # Reference: the best admissible STATIC rung, per validity target, is
        # what adaptation has to beat. Computed here from the same cells.
        statics = {}
        for cid in SELECTOR_RUNGS:
            r = run_static(table, cid, traj, name, epoch_ms)
            s = r.summary()
            statics[cid] = {
                "ok": all(e.makespan_ms <= epoch_ms for e in r.epochs),
                "valid": s["hard_output_valid_rate"],
                "soft": s["soft_completed"],
                "offered": s["soft_offered"],
            }
        for th in thresholds:
            r = run_upstream(table, traj, name, epoch_ms, escalate_at=th)
            s = r.summary()
            ok = all(e.makespan_ms <= epoch_ms for e in r.epochs)
            rowset.append({
                "thresholds": th,
                "admissible": ok,
                "valid": s["hard_output_valid_rate"],
                "soft": s["soft_completed"],
                "offered": s["soft_offered"],
                "switches": s["switch_count"],
                "rungs_used": sorted({e.candidate_id for e in r.epochs}),
            })
        report[name] = rowset
        report[f"{name}__statics"] = [dict(cid=k, **v) for k, v in statics.items()]
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="results/freshness_cand/*/aggregate.csv")
    ap.add_argument("--delta", type=float, default=20.0)
    ap.add_argument("--epoch-ms", type=float, default=300.0)
    args = ap.parse_args()

    pattern = (args.rows if os.path.isabs(args.rows)
               else os.path.join(_REPO, args.rows))
    rows = load_rows(pattern)
    phi = float(rows[0]["A0"]) + args.delta

    print(f"phi = A0 + {args.delta:g} = {phi:.3f} ms;  rungs {list(SELECTOR_RUNGS)}")
    print("\nupstream signal = offered soft work for the epoch about to be "
          "scheduled\n(known at the boundary because admission happens there; it "
          "is the request\ncount, NOT the outcome)\n")
    print("  offered gemmini utilisation by burst:")
    for b in range(5):
        print(f"    B={b}: {offered_utilisation(b, args.epoch_ms) * 100:5.1f}%")

    rep = evaluate(rows, phi=phi, epoch_ms=args.epoch_ms)
    for name in TRAJECTORIES:
        rowset = rep.get(name)
        if not rowset:
            continue
        print(f"\n=== trajectory {name}: {TRAJECTORIES[name]}")
        st = rep[f"{name}__statics"]
        print("  best admissible statics:")
        for s in st:
            print(f"    {s['cid']:<26} valid {s['valid']:.3f} "
                  f"soft {s['soft']}/{s['offered']} "
                  f"{'' if s['ok'] else ' INADMISSIBLE (epoch overrun)'}")
        print(f"  {'thresholds':>12} {'adm':>4} {'valid':>7} {'soft':>8} "
              f"{'sw':>3}  rungs used")
        seen = set()
        for r in rowset:
            key = (r["admissible"], round(r["valid"], 4), r["soft"],
                   tuple(r["rungs_used"]))
            if key in seen:
                continue          # collapse thresholds with identical behaviour
            seen.add(key)
            print(f"  {str(r['thresholds']):>12} {'yes' if r['admissible'] else 'NO':>4} "
                  f"{r['valid']:>7.3f} {r['soft']:>4}/{r['offered']:<3} "
                  f"{r['switches']:>3}  {','.join(x.replace('cand_', '') for x in r['rungs_used'])}")

        adm = [r for r in rowset if r["admissible"]]
        print(f"  -> {len(adm)}/{len(rowset)} threshold settings are admissible "
              f"(the downstream selector was admissible on "
              f"{'this' if name == 'ramp' else 'no'} trajectory)")
        if adm:
            best = max(adm, key=lambda r: r["soft"])
            rivals = [s for s in st if s["ok"] and s["valid"] >= best["valid"] - 1e-9]
            print(f"     best upstream: thresholds {best['thresholds']} valid "
                  f"{best['valid']:.3f} soft {best['soft']}/{best['offered']}")
            if rivals:
                br = max(rivals, key=lambda s: s["soft"])
                print(f"     best static at that validity: {br['cid']} soft "
                      f"{br['soft']}/{br['offered']}  -> upstream retains "
                      f"{best['soft'] - br['soft']:+d}")
            else:
                print("     no admissible static reaches that validity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

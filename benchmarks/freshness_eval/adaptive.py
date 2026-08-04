"""Phase 11: evaluate epoch-level adaptive candidate selection against statics.

A contention TRAJECTORY is a sequence of offered soft-burst levels, one per
epoch. Each strategy decides which precomputed candidate runs in each epoch, and
the epoch's outcome is looked up from the sweep's measured (candidate, B, phi)
cell. So the adaptive run is literally a stitching of precomputed schedules,
which is what "switch among precomputed schedules at epoch boundaries" means.

WHAT THIS COMPOSITION ASSUMES, AND WHY IT IS CHECKED
----------------------------------------------------
Composing per-epoch cells treats epochs as independent: no work is in flight
when an epoch ends. That is not free -- it is a property of this workload that
must hold, and `check_clean_boundaries` asserts it from the data
(makespan <= epoch_ms in every cell used). On this grid every schedule completes
at ~290.5 ms inside a 300 ms epoch, because the last control release is at
290 ms, so the assumption holds by measurement rather than by hope.

The consequence is stated rather than buried: because boundaries are clean here,
`transition_violations` is 0 for a real reason. It is NOT evidence that
switching is generally free. A workload whose makespan approached the epoch
would carry work across the boundary and this harness could not model it -- that
needs a simulator that runs through the switch, not a composition of cells.

FOUR STRATEGIES
---------------
  static_<cid>  one candidate for every epoch -- the "permanently conservative"
                comparison the research question is against
  adaptive      the hysteretic selector, reacting to the PREVIOUS epoch
  oracle_B      full knowledge of the contention level: per epoch, the candidate
                with the best measured output-validity AT THAT B. This is the
                oracle the specification asked for and is not deployable.
  (a post-hoc best-policy upper bound is deliberately NOT reused here; it
   answers a different question and conflating them overstates the headroom.)

The oracle_B upper bound matters for interpretation: adaptive can only ever
approach it, and the gap between adaptive and oracle_B is the price of reacting
one epoch late instead of knowing in advance.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))

from selector import (  # noqa: E402
    CandidateLevel,
    Selector,
    SelectorConfig,
)

# Named contention trajectories. Each is chosen to stress a different part of the
# selector, and the set is fixed in code so a trajectory cannot be quietly tuned
# until adaptive wins.
TRAJECTORIES: Dict[str, List[int]] = {
    # Slow ramp up and back down: the easy case, where reacting one epoch late
    # costs little because contention changes gradually.
    "ramp": [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
    # Step onto maximum contention and off again: the case that exposes
    # reaction lag, because the first high-contention epoch is always taken
    # with the previous epoch's (low) risk estimate.
    "step": [0, 0, 4, 4, 4, 4, 0, 0, 0, 0],
    # Alternating: designed to make a selector without hysteresis chatter, and
    # to make one with hysteresis look sluggish. Neither outcome is a bug; both
    # are the tradeoff being measured.
    "oscillate": [0, 4, 0, 4, 0, 4, 0, 4, 0, 4],
    # Sustained maximum: tests whether the candidate set has any answer at all.
    # If the selector saturates here, the finding is about the candidates.
    "sustained": [4] * 10,
}

# Metrics that are summed over epochs rather than averaged.
_SUM_KEYS = ("valid_count", "total_consumer_invocations", "soft_completed",
             "soft_offered")


@dataclass
class EpochOutcome:
    epoch: int
    offered_burst: int
    candidate_id: str
    output_valid_rate: float
    valid_count: int
    total_consumer_invocations: int
    freshness_success_rate: float
    deadline_success_rate: float
    max_input_age: Optional[float]
    soft_completed: int
    soft_offered: int
    makespan_ms: float
    switched: bool = False
    selector_reason: str = ""


@dataclass
class StrategyResult:
    strategy: str
    trajectory_name: str
    epoch_ms: float
    epochs: List[EpochOutcome] = field(default_factory=list)
    selector_log: List[Dict[str, object]] = field(default_factory=list)
    selector_overhead_us: float = 0.0

    def summary(self) -> Dict[str, object]:
        n = len(self.epochs)
        if not n:
            return {"strategy": self.strategy, "n_epochs": 0}
        tot_inv = sum(e.total_consumer_invocations for e in self.epochs)
        tot_valid = sum(e.valid_count for e in self.epochs)
        soft_done = sum(e.soft_completed for e in self.epochs)
        soft_off = sum(e.soft_offered for e in self.epochs)
        time_in: Dict[str, int] = {}
        for e in self.epochs:
            time_in[e.candidate_id] = time_in.get(e.candidate_id, 0) + 1
        switches = sum(1 for e in self.epochs if e.switched)
        return {
            "strategy": self.strategy,
            "trajectory": self.trajectory_name,
            "n_epochs": n,
            # Weighted by invocation count, not a mean of rates: epochs with
            # different consumer counts must not be given equal weight.
            "hard_output_valid_rate": tot_valid / tot_inv if tot_inv else None,
            "consumer_invocations": tot_inv,
            "soft_completed": soft_done,
            "soft_offered": soft_off,
            "soft_utility_fraction": soft_done / soft_off if soft_off else None,
            "switch_count": switches,
            "switches_per_epoch": switches / n,
            "epochs_in_candidate": time_in,
            "fraction_in_candidate": {k: v / n for k, v in time_in.items()},
            "worst_epoch_output_valid": min(e.output_valid_rate for e in self.epochs),
            "max_input_age": max(
                (e.max_input_age for e in self.epochs if e.max_input_age is not None),
                default=None),
            "selector_overhead_us_total": round(self.selector_overhead_us, 3),
            "selector_overhead_us_per_epoch": round(self.selector_overhead_us / n, 4),
            # 0 here is meaningful only because clean boundaries were CHECKED;
            # see check_clean_boundaries and the module docstring.
            "transition_violations": sum(
                1 for e in self.epochs if e.makespan_ms > self.epoch_ms),
        }




class CellTable:
    """Measured outcomes indexed by (candidate_id, offered_burst) at one phi."""

    def __init__(self, aggregate_rows: Sequence[Dict[str, str]], phi: float,
                 tol: float = 1e-6) -> None:
        self.phi = phi
        self._t: Dict[Tuple[str, int], Dict[str, str]] = {}
        for r in aggregate_rows:
            if abs(float(r["freshness_window"]) - phi) > tol:
                continue
            key = (r["policy"], int(float(r["contention_level"])))
            if key in self._t:
                # Multiple seeds: keep the first and let the caller average
                # upstream if needed. Silently overwriting would hide it.
                continue
            self._t[key] = r

    def has(self, candidate_id: str, burst: int) -> bool:
        return (candidate_id, burst) in self._t

    def get(self, candidate_id: str, burst: int) -> Dict[str, str]:
        try:
            return self._t[(candidate_id, burst)]
        except KeyError:
            raise KeyError(
                f"no measured cell for candidate {candidate_id!r} at burst "
                f"{burst} and phi={self.phi}. The trajectory visits a "
                f"(candidate, contention) pair that was never scheduled, so it "
                f"cannot be evaluated -- extend the sweep rather than "
                f"interpolating."
            ) from None

    def candidates(self) -> List[str]:
        return sorted({c for c, _ in self._t})

    def bursts(self) -> List[int]:
        return sorted({b for _, b in self._t})


def check_clean_boundaries(table: CellTable, candidates: Sequence[str],
                           trajectory: Sequence[int], epoch_ms: float
                           ) -> List[str]:
    """Report cells whose makespan exceeds the epoch.

    A cell that overruns leaves work in flight at the boundary, which this
    composition cannot represent. Returned as a list of problems rather than
    raised, so the caller can report an honest partial result instead of
    silently producing `transition_violations: 0`.
    """
    problems: List[str] = []
    for cid in candidates:
        for b in sorted(set(trajectory)):
            if not table.has(cid, b):
                continue
            mk = float(table.get(cid, b)["makespan_ms"])
            if mk > epoch_ms:
                problems.append(
                    f"{cid} at B={b}: makespan {mk:.1f} ms exceeds the "
                    f"{epoch_ms:.0f} ms epoch by {mk - epoch_ms:.1f} ms"
                )
    return problems


def _outcome(epoch: int, burst: int, cid: str, row: Dict[str, str]) -> EpochOutcome:
    def _f(k, default=0.0):
        v = row.get(k)
        return default if v in (None, "") else float(v)

    return EpochOutcome(
        epoch=epoch,
        offered_burst=burst,
        candidate_id=cid,
        output_valid_rate=_f("output_valid_rate"),
        valid_count=int(_f("valid_count")),
        total_consumer_invocations=int(_f("total_consumer_invocations")),
        freshness_success_rate=_f("freshness_success_rate"),
        deadline_success_rate=_f("deadline_success_rate"),
        max_input_age=(None if row.get("max_input_age") in (None, "")
                       else float(row["max_input_age"])),
        soft_completed=int(_f("soft_instances_completed")),
        soft_offered=burst,
        makespan_ms=_f("makespan_ms"),
    )


def run_static(table: CellTable, candidate_id: str, trajectory: Sequence[int],
               name: str, epoch_ms: float) -> StrategyResult:
    res = StrategyResult(strategy=f"static_{candidate_id}", trajectory_name=name,
                         epoch_ms=epoch_ms)
    for e, b in enumerate(trajectory):
        res.epochs.append(_outcome(e, b, candidate_id, table.get(candidate_id, b)))
    return res


def run_oracle_contention_aware(table: CellTable, candidates: Sequence[str],
                                trajectory: Sequence[int], name: str,
                                epoch_ms: float) -> StrategyResult:
    """Per epoch, the candidate with the best measured output-validity at that B.

    This is the specification's oracle -- selection with full knowledge of the
    contention level. Not deployable: it reads the current epoch's B before
    choosing. Tie-break is the LOWEST protection level, so the oracle does not
    get free credit for shedding soft work it did not need to shed.
    """
    res = StrategyResult(strategy="oracle_contention_aware", trajectory_name=name,
                         epoch_ms=epoch_ms)
    order = {c: i for i, c in enumerate(candidates)}
    prev = None
    for e, b in enumerate(trajectory):
        avail = [c for c in candidates if table.has(c, b)]
        if not avail:
            raise KeyError(f"no candidate has a measured cell at B={b}")
        best = min(
            avail,
            key=lambda c: (-float(table.get(c, b)["output_valid_rate"]), order[c]),
        )
        o = _outcome(e, b, best, table.get(best, b))
        o.switched = prev is not None and best != prev
        o.selector_reason = "oracle_full_knowledge"
        prev = best
        res.epochs.append(o)
    return res


def run_adaptive(table: CellTable, config: SelectorConfig,
                 trajectory: Sequence[int], phi: float, name: str,
                 epoch_ms: float, *, lag: int = 1) -> StrategyResult:
    """Closed loop: the candidate for epoch e is chosen from the age OBSERVED in
    epoch e-lag, under whichever candidate actually ran then.

    This is a genuine feedback loop, not a replay of a fixed observation
    sequence: the observation depends on the candidate the selector previously
    picked, so a bad early choice degrades its own next input.
    """
    res = StrategyResult(strategy="adaptive", trajectory_name=name,
                         epoch_ms=epoch_ms)
    sel = Selector(config)
    observed: List[Optional[float]] = []
    overhead = 0.0

    for e, b in enumerate(trajectory):
        src = e - lag
        obs = observed[src] if src >= 0 else None
        t0 = time.perf_counter()
        cid = sel.decide(e, obs, phi,
                         observation_from_epoch=(src if src >= 0 else None))
        overhead += (time.perf_counter() - t0) * 1e6
        row = table.get(cid, b)
        o = _outcome(e, b, cid, row)
        o.switched = sel.log[-1].switched
        o.selector_reason = sel.log[-1].reason
        res.epochs.append(o)
        observed.append(o.max_input_age)

    res.selector_log = sel.rows()
    res.selector_overhead_us = overhead
    return res


def default_selector_config(candidates: Sequence[str],
                            entry_risks: Sequence[float] = (0.0, 0.85, 1.10),
                            exit_risks: Sequence[float] = (-1.0, 0.70, 0.95),
                            *, min_residency: int = 1, cooldown: int = 1
                            ) -> SelectorConfig:
    """Thresholds expressed in RISK = observed_max_age / phi.

    0.85 and 1.10 are chosen relative to phi rather than fitted to the data:
    escalate once the observed age is within 15% of the window, and go to the
    strongest candidate once the window is actually being exceeded. They are
    parameters here so a sensitivity sweep can show whether the conclusion
    depends on them -- which it must not, or the result is a tuning artifact.
    """
    if len(candidates) > len(entry_risks):
        raise ValueError(
            f"{len(candidates)} candidates but only {len(entry_risks)} entry "
            f"thresholds"
        )
    return SelectorConfig(
        levels=tuple(
            CandidateLevel(cid, i, entry_risk=entry_risks[i],
                           exit_risk=exit_risks[i])
            for i, cid in enumerate(candidates)
        ),
        min_residency_epochs=min_residency,
        cooldown_epochs=cooldown,
    )


def evaluate_trajectories(
    aggregate_rows: Sequence[Dict[str, str]],
    candidates: Sequence[str],
    *,
    phi: float,
    epoch_ms: float,
    trajectories: Optional[Dict[str, List[int]]] = None,
    selector_config: Optional[SelectorConfig] = None,
    lag: int = 1,
) -> Tuple[List[StrategyResult], List[str]]:
    """Run every strategy on every trajectory. Returns (results, warnings)."""
    table = CellTable(aggregate_rows, phi)
    trajectories = trajectories or TRAJECTORIES
    cfg = selector_config or default_selector_config(candidates)

    warnings: List[str] = []
    results: List[StrategyResult] = []
    for name, traj in trajectories.items():
        problems = check_clean_boundaries(table, candidates, traj, epoch_ms)
        if problems:
            warnings.extend(
                f"trajectory {name}: {p} -- work is in flight at the epoch "
                f"boundary, which this composition cannot model, so "
                f"transition_violations for this trajectory is NOT meaningful"
                for p in problems
            )
        for cid in candidates:
            results.append(run_static(table, cid, traj, name, epoch_ms))
        results.append(run_adaptive(table, cfg, traj, phi, name, epoch_ms,
                                   lag=lag))
        results.append(
            run_oracle_contention_aware(table, candidates, traj, name,
                                        epoch_ms))
    return results, warnings


def write_artifacts(results: Sequence[StrategyResult], out_dir: str,
                    warnings: Sequence[str] = ()) -> None:
    os.makedirs(out_dir, exist_ok=True)

    per_epoch = [
        {"strategy": r.strategy, "trajectory": r.trajectory_name,
         **{k: v for k, v in vars(e).items()}}
        for r in results for e in r.epochs
    ]
    if per_epoch:
        cols = list(per_epoch[0])
        with open(os.path.join(out_dir, "adaptive_per_epoch.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(per_epoch)

    summaries = [r.summary() for r in results]
    if summaries:
        cols = sorted({k for s in summaries for k in s})
        lead = ["strategy", "trajectory", "hard_output_valid_rate",
                "soft_utility_fraction", "switch_count"]
        cols = [c for c in lead if c in cols] + [c for c in cols if c not in lead]
        with open(os.path.join(out_dir, "adaptive_summary.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for s in summaries:
                w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v)
                            for k, v in s.items()})

    # selector_log.csv -- a required artifact that Gate A's output dir lacked,
    # because no selector existed then.
    log_rows = [
        {"strategy": r.strategy, "trajectory": r.trajectory_name, **row}
        for r in results for row in r.selector_log
    ]
    if log_rows:
        with open(os.path.join(out_dir, "selector_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
            w.writeheader()
            w.writerows(log_rows)

    if warnings:
        with open(os.path.join(out_dir, "adaptive_warnings.txt"), "w") as f:
            f.write("\n".join(warnings) + "\n")


# --- MEASURED OUTCOME, and why adaptation failed here -----------------------
#
# Recorded next to the code that produced it, because the conclusion is negative
# and a negative conclusion is the easy one to lose.
#
# At phi = A0+20 over the four fixed trajectories, the deployable selector
# (1 epoch of observation lag) NEVER beat the best admissible static:
#
#   ramp       admissible; valid 0.907, soft 8/20 -- EXACTLY ties static admit1
#   step       INADMISSIBLE: ran C1 at B=4, makespan 805.9 ms in a 300 ms epoch
#   oscillate  INADMISSIBLE, same cause
#   sustained  INADMISSIBLE, same cause
#
# Two causes, and the second is the interesting one.
#
# 1. The protective mechanism is nearly free. Deferral costs no soft utility, so
#    a permanently conservative rung is barely worse than the best per-burst
#    choice. `headroom.py` bounds the whole opportunity at ONE soft instance out
#    of 10 offered, and at zero if bursts stay <= 3. There was almost nothing to
#    win before the selector was written.
#
# 2. THE OBSERVABLE SATURATES. risk = observed_max_age / phi, and under any
#    protective rung max_input_age is measured FLAT at 90.55 ms for B = 1, 2, 3
#    and 4 alike (risk 1.124), against 60.55 ms at B=0 (risk 0.752):
#
#              rung                     B=0    B=1    B=2    B=3    B=4
#              static_nominal          0.75   1.25   1.62  1.62*  6.21*
#              cand_c1_defer12         0.75   1.12   1.12   1.12  5.10*
#              cand_c2_defer12_admit2  0.75   1.12   1.12   1.12   1.12
#              cand_c2_defer12_admit1  0.75   1.12   1.12   1.12   1.12
#                                                   (* = schedule overruns)
#
#    So the signal cannot distinguish 65% offered load from 131%. The only values
#    that discriminate B=4 are the starred ones, and those are observable only
#    AFTER the overrunning schedule has already run. The mitigation masks the
#    disturbance it is mitigating: because protection successfully pins the input
#    age to one missed producer period, the age stops reporting how much
#    contention was offered.
#
#    The consequence is not a tuning problem. A threshold sweep on `step` shows
#    entry risks <= 0.752 DO keep it admissible -- but 0.752 is the B=0
#    observation, so such a selector escalates at zero contention, switches once,
#    never returns, and reproduces static admit1's numbers exactly (0.920,
#    4/16). It becomes safe by ceasing to adapt. Every threshold above 0.752
#    takes one full epoch of the 806 ms overrun. There is no setting that both
#    adapts and survives a step to B=4.
#
# What this does NOT show: that freshness-aware switching is useless in general.
# It shows that a selector observing only the protected quantity is blind on this
# workload. A signal measured UPSTREAM of the mitigation -- offered queue depth,
# admitted-vs-offered soft count, or the producer's own start-time slack -- does
# not saturate and is the obvious next thing to try. That is a design change, not
# a retune, and it is out of scope for this pass.
#
# --- driver -----------------------------------------------------------------
#
# The selector's option set deliberately EXCLUDES the nominal candidate. That is
# a measurement-driven choice, not a convenience: nominal is weakly dominated by
# C1 at every burst (equal at B=0, worse at every other) and its schedule
# overruns the epoch at B>=3, so handing a selector an option that is measured to
# be worse everywhere would test nothing. Nominal is still reported, as a static
# reference row, because the research question is stated against it.
SELECTOR_RUNGS = ("cand_c1_defer12", "cand_c2_defer12_admit2",
                  "cand_c2_defer12_admit1")
NOMINAL = "static_nominal"


def admissible_strategies(results: Sequence[StrategyResult]) -> Dict[str, bool]:
    """A strategy is inadmissible on a trajectory if any epoch it ran overruns.

    Composing an overrunning cell into an epoch sequence produces a rate, and
    that rate is not meaningful: work was still in flight when the next epoch
    began, which this harness cannot represent. Such a strategy must be excluded
    from "best safe static" rather than quietly competing with a number it has no
    right to.
    """
    return {f"{r.strategy}@{r.trajectory_name}":
            all(e.makespan_ms <= r.epoch_ms for e in r.epochs)
            for r in results}


def summarise(results: Sequence[StrategyResult]) -> str:
    ok = admissible_strategies(results)
    by_traj: Dict[str, List[StrategyResult]] = {}
    for r in results:
        by_traj.setdefault(r.trajectory_name, []).append(r)

    lines: List[str] = []
    for traj, rs in by_traj.items():
        lines.append(f"\ntrajectory {traj}: {TRAJECTORIES.get(traj)}")
        lines.append(f"  {'strategy':<34} {'valid':>7} {'soft':>9} {'switch':>6}  admissible")
        for r in sorted(rs, key=lambda r: r.strategy):
            s = r.summary()
            key = f"{r.strategy}@{r.trajectory_name}"
            lines.append(
                f"  {r.strategy:<34} "
                f"{s['hard_output_valid_rate']:>7.3f} "
                f"{s['soft_completed']:>4}/{s['soft_offered']:<4} "
                f"{s['switch_count']:>6}  {'yes' if ok[key] else 'NO (epoch overrun)'}"
            )
        # The comparison the research question asks for. It must be matched on
        # validity: comparing adaptive against the highest-utility static
        # regardless of validity would penalise adaptive for being more
        # conservative, which is not the question. The question is whether, AT
        # THE VALIDITY ADAPTIVE DELIVERED, some static could have delivered more
        # noncritical work.
        adapt_r = next((r for r in rs if r.strategy == "adaptive"), None)
        if adapt_r is None:
            continue
        adapt = adapt_r.summary()
        if not ok[f"adaptive@{traj}"]:
            lines.append(
                "  -> adaptive is INADMISSIBLE on this trajectory: it ran a "
                "candidate whose schedule overruns the epoch. Reacting to the "
                "previous epoch means the first high-contention epoch is always "
                "entered on a stale estimate, and here the lower rung's failure "
                "mode is a 2.7x epoch overrun rather than graceful degradation. "
                "No utility comparison is meaningful.")
            continue

        va = adapt["hard_output_valid_rate"]
        rivals = [r.summary() for r in rs
                  if r.strategy.startswith("static_")
                  and ok[f"{r.strategy}@{traj}"]
                  and r.summary()["hard_output_valid_rate"] >= va - 1e-9]
        if not rivals:
            lines.append(f"  -> no admissible static reaches adaptive's validity "
                         f"{va:.3f}; adaptive wins on validity alone")
            continue
        best = max(rivals, key=lambda s: s["soft_completed"])
        lines.append(
            f"  -> adaptive: valid {va:.3f}, soft "
            f"{adapt['soft_completed']}/{adapt['soft_offered']}, "
            f"{adapt['switch_count']} switches")
        lines.append(
            f"     best admissible static at validity >= {va:.3f}: "
            f"{best['strategy']} (valid {best['hard_output_valid_rate']:.3f}, "
            f"soft {best['soft_completed']}/{best['soft_offered']})")
        d = adapt["soft_completed"] - best["soft_completed"]
        lines.append(f"     adaptive retains {d:+d} soft instances vs that static"
                     + ("  <-- adaptation bought nothing" if d == 0 else ""))
    return "\n".join(lines)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Phase 11: adaptive candidate selection vs static strategies")
    ap.add_argument("--rows", default="results/freshness_cand/*/aggregate.csv",
                    help="glob of aggregate.csv files holding the measured cells")
    ap.add_argument("--output-dir", default="results/freshness_adaptive")
    ap.add_argument("--delta", type=float, default=20.0,
                    help="phi = A0 + delta, in ms")
    ap.add_argument("--epoch-ms", type=float, default=300.0)
    ap.add_argument("--lag", type=int, default=1,
                    help="epochs of observation lag; 1 is the deployable case")
    args = ap.parse_args()

    from benchmarks.freshness_eval.headroom import load_rows

    pattern = (args.rows if os.path.isabs(args.rows)
               else os.path.join(_REPO, args.rows))
    rows = load_rows(pattern)
    a0 = float(rows[0]["A0"])
    phi = a0 + args.delta

    candidates = [NOMINAL] + list(SELECTOR_RUNGS)
    results, warnings = evaluate_trajectories(
        rows, list(SELECTOR_RUNGS), phi=phi, epoch_ms=args.epoch_ms,
        lag=args.lag)
    # Nominal is not in the selector's option set, so evaluate_trajectories did
    # not produce its static rows. Add them: the research question is stated
    # against the unprotected schedule and it has to appear in the table.
    table = CellTable(rows, phi)
    for name, traj in TRAJECTORIES.items():
        if all(table.has(NOMINAL, b) for b in set(traj)):
            results.append(run_static(table, NOMINAL, traj, name, args.epoch_ms))

    out_dir = (args.output_dir if os.path.isabs(args.output_dir)
               else os.path.join(_REPO, args.output_dir))
    write_artifacts(results, out_dir, warnings)

    print(f"phi = A0 + {args.delta:g} = {phi:.3f} ms  (A0 = {a0:.3f})")
    print(f"selector rungs: {list(SELECTOR_RUNGS)}")
    print(f"nominal reported as a static reference only: {NOMINAL}")
    print(summarise(results))
    if warnings:
        print("\nWARNINGS (composition assumption violated):")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nwrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

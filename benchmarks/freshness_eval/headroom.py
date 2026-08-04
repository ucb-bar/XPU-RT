"""Phase 11 precondition: an UPPER BOUND on what candidate switching can buy.

This is computed BEFORE interpreting any selector result, and that ordering is
the point of the module.

`adaptive.py` runs the real hysteretic selector over contention trajectories and
compares it against static strategies. That answers "does the selector work".
This module answers the prior question: "is there anything for a selector to
win". The bound assumes a selector that knows the contention level exactly,
switches instantly and for free, and never oscillates -- so no real selector can
beat it.

Order matters because the two questions are easy to confuse. If the ceiling is
one soft instance, then a selector that beats static by one instance has not
demonstrated headroom -- it has saturated a ceiling that was already one instance
high, and the honest report is about the ceiling. Measuring the selector first
and reporting that it beat static would be true and misleading.

DEFINITIONS
-----------
admissible(rung, B)
    The rung's schedule FITS THE EPOCH at that burst. A rung that overruns is not
    a deployable choice there whatever its validity rate says, and its rate is
    measured over a longer trace with a different invocation count so it is not
    comparable either. This is what disqualifies C0 at B>=3 and C1 at B=4 on the
    canonical workload.

best safe static
    The rung with the most soft utility that is admissible AND meets the validity
    target at EVERY burst in the operating range. "Safe" has to mean safe across
    the whole range, because that is all a static choice can promise.

oracle-selector adaptive
    Per burst, the admissible rung with the most soft utility that still meets
    the target. Perfect observation, free switching, no hysteresis.

utility
    Soft (YOLO) instances COMPLETED -- not offered, not admitted. A rung gets no
    credit for admitting work it then fails to finish.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# `output_valid_rate` is over every consumer invocation and is therefore capped
# at 28/30 = 0.933 on the canonical workload by the two cold-start invocations no
# policy can serve. `steady_output_valid_rate` excludes exactly those (see
# xpu-rt/freshness.py and tests/test_pipeline_fill.py) and reaches 1.0. Both are
# reported; neither is privileged, because the choice shifts where the feasible
# target range sits and hiding that choice would let a target be picked to suit.
DEFAULT_METRIC = "output_valid_rate"

# Ladder labels in protection order, paired with the policy keys the sweep ran
# under. The ids are the measured keys and deliberately do not encode the level
# (see candidates.py).
LADDER_RUNGS: Tuple[Tuple[str, str], ...] = (
    ("C0", "static_nominal"),
    ("C1", "cand_c1_defer12"),
    ("C2", "cand_c2_defer12_admit2"),
    ("C3", "cand_c2_defer12_admit1"),
)

DEFAULT_TARGETS: Tuple[float, ...] = (
    0.50, 0.60, 0.70, 0.75, 0.80, 0.833, 0.85, 0.867,
    0.90, 0.929, 0.933, 0.95, 1.00,
)


@dataclass(frozen=True)
class Cell:
    admissible: bool
    validity: float
    soft_completed: int
    soft_offered: int


@dataclass(frozen=True)
class BoundResult:
    target: float
    bursts: Tuple[int, ...]
    static_rung: Optional[str]
    static_utility: Optional[int]
    adaptive_utility: Optional[int]
    adaptive_picks: Tuple[str, ...]
    soft_offered: int

    @property
    def gain(self) -> Optional[int]:
        """Soft instances adaptive retains beyond the best safe static rung.

        None when either side is undefined: no rung safe across the range, or no
        rung meeting the target at some burst. Those are distinct outcomes from a
        gain of zero and must not collapse into it.
        """
        if self.static_utility is None or self.adaptive_utility is None:
            return None
        return self.adaptive_utility - self.static_utility

    def describe(self) -> str:
        st = ("NONE (no rung is safe across the whole range)"
              if self.static_rung is None
              else f"{self.static_rung} ({self.static_utility}/{self.soft_offered})")
        ad = ("INFEASIBLE (no rung meets the target at some burst)"
              if self.adaptive_utility is None
              else f"{self.adaptive_utility}/{self.soft_offered}")
        return (f"target={self.target:.3f} bursts={list(self.bursts)}\n"
                f"    best safe static : {st}\n"
                f"    oracle adaptive  : {ad}  picks={list(self.adaptive_picks)}\n"
                f"    gain             : "
                f"{'n/a' if self.gain is None else format(self.gain, '+d')} "
                f"soft instances")


def load_rows(pattern: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise FileNotFoundError(f"no aggregate.csv matched {pattern!r}")
    return rows


def build_table(
    rows: Sequence[Dict[str, str]],
    rungs: Sequence[Tuple[str, str]] = LADDER_RUNGS,
    *,
    delta: float,
    bursts: Sequence[int],
    metric: str = DEFAULT_METRIC,
) -> Dict[Tuple[str, int], Cell]:
    """(rung_label, burst) -> Cell for one freshness window.

    Duplicate rows for the same cell are REJECTED rather than averaged or
    silently first-wins. Every cell in this sweep was measured at one seed with
    schedules verified identical across seeds; if that ever stops being true the
    right response is a visible failure, not a mean that hides the divergence.
    """
    table: Dict[Tuple[str, int], Cell] = {}
    for label, cid in rungs:
        for b in bursts:
            hits = [r for r in rows
                    if r["policy"] == cid
                    and int(float(r["contention_level"])) == b
                    and abs(float(r["delta"]) - delta) < 1e-9]
            if not hits:
                continue
            distinct = {(r["output_valid_rate"], r["soft_instances_completed"],
                         r["fits_in_epoch"]) for r in hits}
            if len(distinct) > 1:
                raise ValueError(
                    f"{cid} at B={b} delta={delta:g} has {len(distinct)} "
                    f"differing measurements across {len(hits)} rows; refusing "
                    f"to collapse them silently: {sorted(distinct)}")
            r = hits[0]
            table[(label, b)] = Cell(
                admissible=r["fits_in_epoch"] == "True",
                validity=float(r[metric]),
                soft_completed=int(r["soft_instances_completed"]),
                soft_offered=int(r["soft_instances_offered"]),
            )
    return table


def bound(
    table: Dict[Tuple[str, int], Cell],
    rungs: Sequence[Tuple[str, str]] = LADDER_RUNGS,
    *,
    target: float,
    bursts: Sequence[int],
) -> BoundResult:
    bursts = tuple(bursts)
    eps = 1e-9
    labels = [lab for lab, _ in rungs]

    def ok(label: str, b: int) -> bool:
        c = table.get((label, b))
        return c is not None and c.admissible and c.validity >= target - eps

    safe = [(sum(table[(lab, b)].soft_completed for b in bursts), lab)
            for lab in labels if all(ok(lab, b) for b in bursts)]
    best = max(safe) if safe else None

    picks: List[str] = []
    adaptive: Optional[int] = 0
    for b in bursts:
        opts = [(table[(lab, b)].soft_completed, lab) for lab in labels if ok(lab, b)]
        if not opts:
            picks.append("--")
            adaptive = None
            continue
        u, lab = max(opts)
        picks.append(lab)
        if adaptive is not None:
            adaptive += u

    offered = sum(max((table[(lab, b)].soft_offered for lab in labels
                       if (lab, b) in table), default=0) for b in bursts)

    return BoundResult(
        target=target, bursts=bursts,
        static_rung=None if best is None else best[1],
        static_utility=None if best is None else best[0],
        adaptive_utility=adaptive,
        adaptive_picks=tuple(picks),
        soft_offered=offered,
    )


def max_gain(
    table: Dict[Tuple[str, int], Cell],
    rungs: Sequence[Tuple[str, str]] = LADDER_RUNGS,
    *,
    bursts: Sequence[int],
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> int:
    """The headline number: the most switching can buy at ANY validity target.

    UNITS MATTER HERE. This is a gain over the burst GRID, where each burst is
    visited exactly once, so the +1 it returns means "one soft instance per visit
    to the burst that carries the gain" -- on this workload, B=3, the single burst
    where the cheap rung both suffices and fits the epoch.

    On a trajectory the gain scales with the number of epochs spent there. The
    `ramp` trajectory visits B=3 twice and measurably gains +2 (see
    benchmarks/freshness_eval/upstream.py), which is this same bound applied per
    visit rather than a violation of it. Quoting "+1 of 10 offered" as a
    trajectory-level result would understate a ramp and overstate a workload that
    never reaches B=3 at all.
    """
    gains = [bound(table, rungs, target=t, bursts=bursts).gain for t in targets]
    real = [g for g in gains if g is not None]
    return max(real) if real else 0


def monotonicity_violations(
    table: Dict[Tuple[str, int], Cell],
    rungs: Sequence[Tuple[str, str]] = LADDER_RUNGS,
    *,
    bursts: Sequence[int],
) -> List[str]:
    """Cells where the ladder is not monotone in protection level.

    Only compares cells where BOTH rungs are admissible: an inadmissible cell has
    no meaningful rate, so reporting it as a monotonicity break would be an
    artifact of the epoch overrun rather than a property of the ladder.
    """
    out: List[str] = []
    labels = [lab for lab, _ in rungs]
    for lo, hi in zip(labels, labels[1:]):
        for b in bursts:
            a, c = table.get((lo, b)), table.get((hi, b))
            if a is None or c is None or not (a.admissible and c.admissible):
                continue
            if c.validity < a.validity - 1e-9:
                out.append(f"validity: {hi} < {lo} at B={b} "
                           f"({c.validity:.3f} < {a.validity:.3f})")
            if c.soft_completed > a.soft_completed:
                out.append(f"utility: {hi} > {lo} at B={b} "
                           f"({c.soft_completed} > {a.soft_completed})")
    return out


def report(
    rows: Sequence[Dict[str, str]],
    rungs: Sequence[Tuple[str, str]] = LADDER_RUNGS,
    *,
    delta: float,
    bursts: Sequence[int] = (0, 1, 2, 3, 4),
    metric: str = DEFAULT_METRIC,
    targets: Sequence[float] = DEFAULT_TARGETS,
) -> str:
    table = build_table(rows, rungs, delta=delta, bursts=bursts, metric=metric)
    lines = [f"switching headroom: delta={delta:g} metric={metric} "
             f"bursts={list(bursts)}"]

    viol = monotonicity_violations(table, rungs, bursts=bursts)
    lines.append("  ladder monotone in protection level: "
                 + ("YES" if not viol else "NO -> " + "; ".join(viol)))
    lines.append("  admissibility (fits epoch): " + ", ".join(
        f"{lab}=" + "".join("." if table[(lab, b)].admissible else "X"
                            for b in bursts if (lab, b) in table)
        for lab, _ in rungs))

    lines.append(f"  {'target':>7} | {'best safe static':>22} | "
                 f"{'adaptive':>9} | gain | picks (one per burst)")
    seen = None
    for t in targets:
        r = bound(table, rungs, target=t, bursts=bursts)
        st = ("NONE" if r.static_rung is None
              else f"{r.static_rung} ({r.static_utility}/{r.soft_offered})")
        ad = ("INFEAS" if r.adaptive_utility is None
              else f"{r.adaptive_utility}/{r.soft_offered}")
        g = "" if r.gain is None else f"{r.gain:+d}"
        key = (st, ad, g, r.adaptive_picks)
        if key != seen:      # collapse runs of identical rows; targets are dense
            lines.append(f"  {t:7.3f} | {st:>22} | {ad:>9} | {g:>4} | "
                         f"{' '.join(r.adaptive_picks)}")
        seen = key

    lines.append(f"  MAX GAIN over all targets: "
                 f"{max_gain(table, rungs, bursts=bursts, targets=targets)} "
                 f"soft instance(s)")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="upper bound on what candidate switching can buy")
    ap.add_argument("--rows", default="results/freshness_cand/*/aggregate.csv",
                    help="glob of aggregate.csv files holding the measured rungs")
    ap.add_argument("--deltas", default="5,20,50")
    ap.add_argument("--metric", default=DEFAULT_METRIC,
                    help=f"validity metric (default {DEFAULT_METRIC}); "
                         f"steady_output_valid_rate excludes the cold start")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pattern = args.rows if os.path.isabs(args.rows) else os.path.join(repo, args.rows)
    rows = load_rows(pattern)

    for d in [float(x) for x in args.deltas.split(",")]:
        # Both ranges are reported because the difference between them IS the
        # finding: restricted to 0..3 a single static rung is optimal at every
        # target, and all of switching's value comes from B=4 disqualifying the
        # otherwise-best rung.
        for bursts in ((0, 1, 2, 3), (0, 1, 2, 3, 4)):
            print(report(rows, delta=d, bursts=bursts, metric=args.metric))
            print()


if __name__ == "__main__":
    main()

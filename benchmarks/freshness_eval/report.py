"""Human-readable Decision Gate A summary from sweep artifacts.

    python -m benchmarks.freshness_eval.report --input results/freshness_eval

Reads only the CSVs and manifest the sweep wrote, and answers the gate
questions directly:

    Does local deadline success hide stale outputs?
    At what contention levels?  For which freshness windows?
    Under which timing assumptions?  How large is the divergence?
    Is the result robust across seeds?

Producer deadline compliance is recomputed here from intervals.csv when the
aggregate lacks it, so summaries of sweeps predating that column stay complete.

Writes `summary.md` next to the inputs and prints it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

ORACLE = "oracle"


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def producer_compliance(
    intervals: List[Dict[str, str]], producer: str
) -> Dict[tuple, Dict[str, float]]:
    """(policy, B, seed) -> producer deadline stats, from intervals.csv."""
    groups: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for r in intervals:
        if r.get("task") != producer or not r.get("deadline"):
            continue
        groups[(r["policy"], int(_f(r["contention_level"])), int(_f(r["seed"])))].append(r)
    out = {}
    for k, rows in groups.items():
        late = [_f(r["end_time"]) - _f(r["deadline"]) for r in rows]
        out[k] = {
            "rate": sum(1 for d in late if d <= 0) / len(late),
            "max_lateness": max(late),
            "n": len(late),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="results/freshness_eval")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    in_dir = args.input if os.path.isabs(args.input) else os.path.join(repo, args.input)

    agg = read_csv(os.path.join(in_dir, "aggregate.csv"))
    intervals = read_csv(os.path.join(in_dir, "intervals.csv"))
    with open(os.path.join(in_dir, "manifest.json")) as f:
        man = json.load(f)

    if not agg:
        print(f"no aggregate.csv in {in_dir}")
        return 1

    A0 = _f(man["A0"]["A0_realized"])
    producer = man["A0"]["producer_task"]
    consumer = man["A0"]["consumer_task"]
    epoch = _f(man.get("epoch_ms"))
    prod = producer_compliance(intervals, producer)

    L: List[str] = []
    w = L.append

    w("# Decision Gate A — does local deadline success hide stale outputs?\n")
    w(f"Workload `{man['config']}`, epoch {epoch:.0f} ms. "
      f"Producer `{producer}` T={man['A0']['producer_period_ms']:.0f} ms "
      f"L={man['A0']['producer_latency_ms']:.3f} ms; "
      f"consumer `{consumer}` T={man['A0']['consumer_period_ms']:.0f} ms "
      f"L={man['A0']['consumer_latency_ms']:.3f} ms "
      f"(on `{man['A0']['fast_cluster_hw']}`).\n")
    w(f"**A0 = {A0:.3f} ms** is the measured uncontended input-age ceiling; the "
      f"uncontended age set is "
      f"`{[round(x, 2) for x in man['A0']['distinct_ages']]}` ms. Windows are "
      f"`phi = A0 + delta`, so every point has real headroom over the sampling "
      f"rate.\n")

    tp = man["timing_provenance"]
    w("## Timing assumptions\n")
    w(f"- source: **{tp['timing_source']}**, {tp['source']}")
    w(f"- backends: {tp['backends']}, target `{tp['target']}`")
    w(f"- derivation: {tp['scaling_factor']} at an assumed "
      f"{tp['clock_mhz_assumed']} MHz")
    w(f"- {tp['clock_caveat']}")
    w(f"- producer instance attribution: {man['producer_instance_provenance']}")
    w(f"- post-passes: compaction={man['post_passes']['compaction_enabled']}, "
      f"automerge={man['post_passes']['automerge_enabled']}\n")

    # ---- the headline table -------------------------------------------------
    policies = [p for p in dict.fromkeys(r["policy"] for r in agg) if p != ORACLE]
    bursts = sorted({int(_f(r["contention_level"])) for r in agg})
    phis = sorted({_f(r["freshness_window"]) for r in agg})
    primary = phis[len(phis) // 2]

    w(f"## Validity vs contention at phi = {primary:.1f} ms "
      f"(= A0 + {primary - A0:.0f})\n")
    w(f"| policy | B | consumer deadline-valid | freshness-valid | output-valid | "
      f"divergence | max input age | producer deadline-valid | producer max late | soft done |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for pol in policies:
        for b in bursts:
            rows = [r for r in agg
                    if r["policy"] == pol
                    and int(_f(r["contention_level"])) == b
                    and abs(_f(r["freshness_window"]) - primary) < 1e-6]
            if not rows:
                w(f"| {pol} | {b} | — | — | — | — | — | — | — | *cell failed* |")
                continue
            r = rows[0]
            pc = prod.get((pol, b, int(_f(r["seed"]))))
            pr = f"{pc['rate']:.3f}" if pc else "—"
            pl = f"{pc['max_lateness']:+.1f} ms" if pc else "—"
            w(f"| {pol} | {b} | {_f(r['deadline_success_rate']):.3f} | "
              f"{_f(r['freshness_success_rate']):.3f} | "
              f"{_f(r['output_valid_rate']):.3f} | "
              f"{_f(r['divergence']):+.3f} | "
              f"{(_f(r['max_input_age']) or 0):.1f} ms | {pr} | {pl} | "
              f"{int(_f(r['soft_instances_completed']))} |")
    w("")

    # ---- flagged operating points ------------------------------------------
    flagged = [r for r in agg
               if r["policy"] != ORACLE
               and _f(r["deadline_success_rate"]) is not None
               and _f(r["deadline_success_rate"]) >= 0.95
               and _f(r["output_valid_rate"]) < _f(r["deadline_success_rate"]) - 0.10]
    # ---- structural floor vs contention-induced loss ------------------------
    # B=0 is the same workload with no soft work at all, so whatever validity is
    # lost there is structural: the first consumer invocations precede any
    # producer completion no matter how idle the machine is. Reporting raw
    # divergence alone would credit contention with that floor.
    w("## Structural floor versus contention-induced loss\n")
    w("B=0 is the same workload with no soft work, so validity lost there is "
      "structural — the first consumer invocations precede any producer "
      "completion however idle the machine is. The contention-induced loss is "
      "the drop below the B=0 control.\n")
    w("| policy | phi (ms) | output-valid at B=0 (floor) | " +
      " | ".join(f"loss at B={b}" for b in bursts if b != 0) + " |")
    w("|---" * (len([b for b in bursts if b != 0]) + 3) + "|")
    for pol in policies:
        for phi in phis:
            base = [_f(r["output_valid_rate"]) for r in agg
                    if r["policy"] == pol and int(_f(r["contention_level"])) == 0
                    and abs(_f(r["freshness_window"]) - phi) < 1e-6
                    and _f(r["output_valid_rate"]) is not None]
            if not base:
                continue
            floor = sum(base) / len(base)
            cells = []
            for b in bursts:
                if b == 0:
                    continue
                vals = [_f(r["output_valid_rate"]) for r in agg
                        if r["policy"] == pol
                        and int(_f(r["contention_level"])) == b
                        and abs(_f(r["freshness_window"]) - phi) < 1e-6
                        and _f(r["output_valid_rate"]) is not None]
                cells.append(f"−{floor - sum(vals)/len(vals):.3f}" if vals else "—")
            w(f"| {pol} | {phi:.1f} | {floor:.3f} | " + " | ".join(cells) + " |")
    w("")

    w(f"## Operating points where deadline success hides invalid output\n")
    w(f"Criterion: `deadline_success_rate >= 0.95` and "
      f"`output_valid_rate < deadline_success_rate - 0.10`.\n")
    w(f"**{len(flagged)} of {len([r for r in agg if r['policy'] != ORACLE])} "
      f"(policy, B, phi, seed) cells qualify.**\n")
    n_b0 = len([r for r in flagged if int(_f(r["contention_level"])) == 0])
    if n_b0:
        w(f"{n_b0} of those {'is' if n_b0 == 1 else 'are'} at **B=0**, i.e. with "
          f"no contention at all — "
          f"that part of the divergence is structural, not contention-induced. "
          f"See the floor table above.\n")
    if flagged:
        by_b = defaultdict(list)
        for r in flagged:
            by_b[int(_f(r["contention_level"]))].append(r)
        w("| B | qualifying cells | phi range (ms) | worst divergence |")
        w("|---|---|---|---|")
        for b in sorted(by_b):
            rs = by_b[b]
            ps = sorted({_f(x["freshness_window"]) for x in rs})
            worst = max(rs, key=lambda x: _f(x["divergence"]))
            w(f"| {b} | {len(rs)} | {ps[0]:.1f}–{ps[-1]:.1f} | "
              f"{_f(worst['divergence']):+.3f} ({worst['policy']}) |")
        w("")
        top = max(flagged, key=lambda r: _f(r["divergence"]))
        w(f"Largest divergence: **{_f(top['divergence']):+.3f}** at "
          f"`{top['policy']}` B={int(_f(top['contention_level']))} "
          f"phi={_f(top['freshness_window']):.1f} ms — consumer deadline-valid "
          f"{_f(top['deadline_success_rate']):.3f} but output-valid "
          f"{_f(top['output_valid_rate']):.3f}.\n")

    # ---- phi sensitivity ----------------------------------------------------
    w("## Sensitivity to the freshness window\n")
    w("output-valid rate; rows are phi, columns are B\n")
    for pol in policies:
        w(f"\n`{pol}`\n")
        w("| phi (ms) | delta | " + " | ".join(f"B={b}" for b in bursts) + " |")
        w("|---" * (len(bursts) + 2) + "|")
        for phi in phis:
            cells = []
            for b in bursts:
                vals = [_f(r["output_valid_rate"]) for r in agg
                        if r["policy"] == pol
                        and int(_f(r["contention_level"])) == b
                        and abs(_f(r["freshness_window"]) - phi) < 1e-6
                        and _f(r["output_valid_rate"]) is not None]
                cells.append(f"{sum(vals)/len(vals):.3f}" if vals else "—")
            w(f"| {phi:.1f} | +{phi - A0:.0f} | " + " | ".join(cells) + " |")
    w("")

    # ---- robustness ---------------------------------------------------------
    det = man.get("determinism", {})
    nondet = [k for k, v in det.items() if v != "identical across seeds"]
    w("## Robustness across seeds\n")
    w(f"- {len(det)} (policy, B) cells checked across seeds "
      f"{man.get('seeds')}.")
    if nondet:
        w(f"- **non-deterministic:** {nondet}")
    else:
        w("- every cell produced an identical schedule across seeds; all four "
          "policies are deterministic, so seeds are a control rather than a "
          "source of variance. Seed-to-seed variability is therefore not "
          "evidence of robustness — the robustness axes here are B and phi.")
    fails = man.get("failures", [])
    if fails:
        w(f"- **{len(fails)} cell(s) failed** and are absent from the grid: "
          + ", ".join(f"{f['policy']} B={f['burst']} ({f['status']})" for f in fails))
    w("")

    # ---- caveats that bound the claim --------------------------------------
    w("## What this does and does not show\n")
    w("- Freshness is **imposed on the schedule and evaluated analytically**, not "
      "observed. No dataflow exists between the two networks, so the consumed "
      "producer instance is inferred from timestamps and never recorded — on "
      "hardware too.")
    w(f"- The consumer is {man['A0']['consumer_latency_ms']:.3f} ms of work in a "
      f"{man['A0']['consumer_period_ms']:.0f} ms window on a two-cluster "
      f"machine, so its deadline success is close to 1.0 **by construction**. "
      f"The divergence is real but is bounded from above only by the "
      f"oversubscribed high-B points; it is not an open-ended gap.")
    w("- Makespan is pinned near the last consumer release at every B and is not "
      "a contention metric here.")
    w("- The oracle row is a post-hoc upper bound, not a deployable policy.")
    w("- These are solver schedules, not hardware traces.")

    text = "\n".join(L) + "\n"
    out = os.path.join(in_dir, "summary.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"wrote {os.path.relpath(out, repo)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

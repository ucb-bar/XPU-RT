#!/usr/bin/env python3
"""The camera->YOLO->nav->control chain latency, from MEASUREMENT rather than from a solve.

WHY THIS EXISTS. The schedule's own makespan (29.93 ms) is a board-RECOST number: profile
costs multiplied by measured multipliers. Executing that schedule on the K1 gives a different
answer, and the difference is not noise:

    stage        predicted    executed     ratio
    yolo           30.59        31.21      1.02x   <- accurate
    fused_full      4.90        11.45      2.34x   <- cold start, and the whole error
    mlp             0.08         0.30
    chain          29.93        42.97

`fused_full` runs ONE instance in this chain, so it pays cold start in full. Pooled over the
eight board traces that DO have several instances of it, its cold/warm ratio is 2.75x and its
warm median is 4.43 ms -- so the 2.34x seen here is cold start, not a modelling error.

WHAT A DEPLOYMENT SEES is the warm chain: a mission runs continuously, so only the first frame
pays cold start. That is the number this reports, and it is built from measurements only:

    yolo        measured in-schedule span on the K1 (its 90 dispatches are warm within the frame)
    fused_full  warm median pooled over the board traces that have >= 3 instances
    mlp         same

Run it to regenerate the constants the figure and the paper quote.
"""
from __future__ import annotations
import collections, csv, glob, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
HZ = 24e6
COLD_TRACE = "results/codesign_feedback/cmp_coupled_cpsat_board_trace.csv"
sys.path.insert(0, os.path.join(REPO, "scripts"))
from measured_timing import CHAIN, ROS      # noqa: E402

ROS_TAX_3HOP = ROS["middleware_tax_3hop_ms"]   # rclcpp, 3 hops, zero compute
ROS_CHAIN_MEASURED = CHAIN["ros_ms"]          # measured directly by the on-board chain sweep
DEADLINE = CHAIN["deadline_ms"]


def warm_medians():
    """{net: warm median ms} pooled over every trace with >= 3 instances of that net."""
    per = collections.defaultdict(list)
    pats = ["ModelBlaster/tmp/*/_gen/*/*_trace.csv",
            "ModelBlaster/scratch/**/_gen/*/*_trace.csv"]
    for pat in pats:
        for T in glob.glob(pat, recursive=True):
            acc = collections.defaultdict(lambda: collections.defaultdict(float))
            for r in csv.DictReader(open(T)):
                try:
                    s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
                except (ValueError, KeyError):
                    continue
                if e <= 0:
                    continue
                acc[r["network"]][r.get("instance", "0")] += (e - s) / HZ * 1000.0
            for net, insts in acc.items():
                if len(insts) < 3:
                    continue
                ks = sorted(insts, key=lambda k: int(k) if str(k).lstrip("-").isdigit() else 0)
                per[net].append(statistics.median([insts[k] for k in ks[1:]]))
    return {n: statistics.median(v) for n, v in per.items()}, {n: len(v) for n, v in per.items()}


def executed_spans(trace):
    """{net: (span_ms, n_dispatch)} from one executed trace; span accounts for intra-net parallelism."""
    acc = collections.defaultdict(lambda: [float("inf"), 0.0, 0])
    for r in csv.DictReader(open(trace)):
        try:
            s, e = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        except (ValueError, KeyError):
            continue
        if e <= 0:
            continue
        d = acc[r["network"]]
        d[0] = min(d[0], s); d[1] = max(d[1], e); d[2] += 1
    return {n: ((e - s) / HZ * 1000.0, c) for n, (s, e, c) in acc.items()}


def main() -> int:
    warm, nwarm = warm_medians()
    spans = executed_spans(COLD_TRACE)
    order = ["yolov8_nano_64x96", "fused_full", "mlp_control"]

    print("CHAIN LATENCY ON THE K1, from measurement\n")
    print(f"{'stage':<22} {'executed span':>14} {'warm median':>13} {'used':>9}  source")
    cold = warmed = 0.0
    for n in order:
        sp = spans.get(n, (0.0, 0))[0]
        wm = warm.get(n)
        # yolo has one instance here but 90 dispatches, and measured 1.02x of predicted, so its
        # executed span already IS warm. The light nets take their pooled warm median.
        use = sp if n == "yolov8_nano_64x96" or wm is None else wm
        src = "executed" if use == sp else f"warm median, {nwarm.get(n,0)} traces"
        cold += sp; warmed += use
        print(f"{n:<22} {sp:>13.2f}ms {(wm if wm else float('nan')):>12.2f}ms {use:>8.2f}ms  {src}")
    print(f"\n{'CHAIN, cold (as executed)':<22} {cold:>13.2f}ms")
    print(f"{'CHAIN, warm (deployment)':<22} {warmed:>13.2f}ms")
    print()
    print(f"{'arm':<40} {'chain ms':>10} {'max rate':>10}  vs {DEADLINE} ms")
    for name, v in (("XPU-RT, global schedule (measured warm)", warmed),
                    ("real ROS 2 (measured on the board)", ROS_CHAIN_MEASURED)):
        print(f"{name:<40} {v:>9.2f}ms {1000.0/v:>9.1f}Hz  {'MEETS' if v <= DEADLINE else 'misses'}")
    print(f"\n  ratio {ROS_CHAIN_MEASURED / warmed:.2f}x")
    print(f"  Both exceed a {DEADLINE} ms budget. The budgets they SEPARATE on are those between")
    print(f"  {warmed:.1f} and {ROS_CHAIN_MEASURED:.1f} ms, i.e. avoidance loops from "
          f"{1000.0/ROS_CHAIN_MEASURED:.0f} to {1000.0/warmed:.0f} Hz: in that band the global")
    print(f"  schedule closes the loop in time and static per-node pinning does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

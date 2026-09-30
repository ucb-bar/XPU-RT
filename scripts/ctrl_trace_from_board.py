#!/usr/bin/env python3
"""Control-output times from a board run, for the flight simulator to replay.

The simulator (`sims/scripts/sweep_rate_demo.py --ctrl_trace`) refreshes the held motor command
only at control steps during which the board produced a control output, looping the series, so
the flight sees the runtime's measured cadence — bursts, gaps and all — rather than one number.

Sources:
  * an XPU-RT trace (`results/codesign_feedback/xpurt_long/trace_<label>.csv`): one output per
    `mlp_control` instance, at the end of its last dispatch;
  * a ROS 2 run (`results/codesign_feedback/ros_traced/<tag>/ctrl_gaps.csv`): one output per
    control fire (`t_fire_ticks`).
The first WARMUP_MS of the run are dropped (cold instances); times are written in ms relative to
the first kept output, with the source and the gap statistics as '#' lines the simulator skips.

    scripts/ctrl_trace_from_board.py <trace.csv | ctrl_gaps.csv> --out <ctrl_trace.csv> [--warmup-ms 100] [--max-s 6]
"""
from __future__ import annotations
import argparse, csv, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKS_PER_MS = 24000.0


def from_xpurt(path):
    """Output times and the run's end: the table repeats after its full length, so a burst of
    outputs followed by silence replays as exactly that."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from make_measured_gantt_pair import read_trace   # noqa: E402
    rows = read_trace(path)
    ends = {}
    for r in rows:
        if r["net"] == "mlp_control":
            ends[r["inst"]] = max(ends.get(r["inst"], 0.0), r["e"])
    return [ends[k] for k in sorted(ends)], max(r["e"] for r in rows)


def from_ros(path):
    ts = sorted(int(r["t_fire_ticks"]) / TICKS_PER_MS for r in csv.DictReader(open(path)))
    return ts, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-ms", type=float, default=100.0, help="dropped from the start of the run")
    ap.add_argument("--max-s", type=float, default=6.0, help="keep at most this much of the run")
    a = ap.parse_args()
    ts, run_end = from_ros(a.src) if os.path.basename(a.src) == "ctrl_gaps.csv" else from_xpurt(a.src)
    t0 = ts[0]
    # the loop the simulator replays: for a table (XPU-RT) it runs from the end of warm-up to the
    # table's end, so silence anywhere in the table counts; for a free-running ROS run it runs
    # from the first kept output to the last plus the mean gap
    loop0 = (t0 + a.warmup_ms) if run_end is not None else None
    ts = [t for t in ts if t - t0 >= a.warmup_ms]
    ts = [t for t in ts if t - ts[0] <= a.max_s * 1000.0]
    gaps = [b - x for x, b in zip(ts, ts[1:])]
    if len(gaps) < 5:
        raise SystemExit(f"{a.src}: only {len(gaps)} gaps after warm-up")
    p95 = sorted(gaps)[int(round(0.95 * (len(gaps) - 1)))]
    if loop0 is None:
        loop0 = ts[0]
    span = (run_end - loop0) if run_end is not None else (ts[-1] - ts[0] + statistics.mean(gaps))
    with open(a.out, "w") as f:
        f.write(f"# source={os.path.relpath(a.src, REPO)}\n# warmup_ms={a.warmup_ms} outputs={len(ts)} span_ms={span:.1f}\n"
                f"# gap_mean_ms={statistics.mean(gaps):.3f} gap_p95_ms={p95:.3f} gap_max_ms={max(gaps):.3f} "
                f"eff_hz={1000.0*len(gaps)/(ts[-1]-ts[0]):.2f}\n")
        f.write("t_ms\n")
        for t in ts:
            f.write(f"{t - loop0:.3f}\n")           # relative to the loop start, not to the first output
    print(f"wrote {a.out}: {len(ts)} outputs, loop span {span:.0f} ms, gap mean {statistics.mean(gaps):.2f} "
          f"p95 {p95:.2f} max {max(gaps):.2f} ms -> {1000.0*len(ts)/span:.1f} outputs/s over the loop")
    return 0


if __name__ == "__main__":
    sys.exit(main())

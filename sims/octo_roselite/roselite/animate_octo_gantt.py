#!/usr/bin/env python3
"""Animate a MEASURED QRB5165 hardware-lane gantt in lockstep with a SIMPLER rollout.

This is the Octo/RoSE-lite counterpart to
``XPU-RT/sims/smolvla_demo/animate_schedule_gantt.py`` (lane rows, playhead,
bars revealed as the playhead passes, per-frame compositing).  Lane layout and
segment colours match ``XPU-RT/qnn_models/octo/plot_octo_gantt.py``; the
per-instance colouring used by ``--schedule pipelined`` matches
``octo_pipe/plot_measured_lanes.py``, which draws the same pipelined traces
statically.

TWO SCHEDULE MODES
------------------
``--schedule serial`` (default)
    One inference in flight.  The trace is a single-instance walk; it is
    redrawn once per issue at steps 0, D, 2D, ...  MEASURED
    ``max_concurrent_lanes = 1``, 0.00 ms of lane overlap.

``--schedule pipelined``
    Several Octo instances in flight at once.  The trace is a MEASURED
    multi-instance pipelined walk executed on the board: each instance's own
    dispatches are rebased to that instance's start and replayed once per
    issue, coloured by instance, so the overlap that the board actually
    produced is drawn rather than asserted.  Overlap and max-concurrent-lanes
    are recomputed from the trace on every run and printed on the frame; they
    are never hardcoded.

    Lane concurrency is counted in DISTINCT LANES, not dispatches (see
    ``lane_concurrency``): the measured traces serialise within a lane so the
    two agree there, but the animation tiles onto an exact issue grid, which
    can double-book a lane and would otherwise report more "lanes" busy than
    the machine has.

WHAT IS MEASURED AND WHAT IS MODELLED
-------------------------------------
MEASURED (on the physical QRB5165, ``root@10.44.120.201``):
  * every gantt bar -- ``actual_start_ms`` / ``actual_end_ms`` of each
    dispatch, from the runtime's AGENTS_QNN_TRACE block.  Header quirk:
    ``kind`` holds the machine id (CPU_X / CPU_E / CPU_P), ``backend_label``
    holds the segment name, ``instance`` the Octo instance, and
    ``actual_backend`` the physical lane (CPU / DSP / HTA).
  * lane busy, lane overlap, max concurrent lanes, max instances in flight,
    per-instance latency and the inter-completion cadence -- all recomputed
    here from the trace being drawn.
  * the end-to-end latency fed to the sweep -- median wall over 20 iterations.
  * the arm's success rate and n, from RESULTS.txt.
  * the rollout frames: a real SIMPLER episode.

MODELLED:
  * the mapping latency -> D = ceil(latency / 200 ms) control steps, and the
    consequent issue/arrival schedule.  ``control_freq = 5`` in
    PutEggplantInBasketScene-v0, so one env step is 200 ms.
  * the repetition of the measured walk once per issue, snapped onto the issue
    grid.  The board ran the walk once, not once per simulator step.

INFERENCE CADENCE AND CONTROL CADENCE ARE DIFFERENT AXES.  The env always
steps at ``--control-period-ms`` (200 ms).  ``--issue-period-ms`` is how often
the *hardware* starts a new inference.  When they differ (the 110 ms walk),
several inferences consume the SAME camera frame -- the env has no newer one
to give -- and those duplicates are drawn hollow and labelled ``dup``, because
they cannot contribute information the controller does not already have.

THE int8 PIPELINE IS A PERFORMANCE VEHICLE.  The 3-way chain measures cosine
0.008 against the JAX golden (OCTO_INT8_QRB5165.md section 6).  The policy in
these rollouts ran at full fp32 on the host GPU; only the *arrival time* of
each action chunk is board-derived.  No frame here claims the int8 model's
outputs drove the robot.

Usage (see make_all_videos.sh for the exact deliverable invocations)::

    T=/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs

    # SERIAL (Task 2)
    python animate_octo_gantt.py \
        --trace "$T/ungated_20260905-112925.log" \
        --video runs_video/lat283_serial/ep00_success_False.mp4 \
        --latency-ms 283.4 --arm-sr 15.8 --arm-n 120 \
        --out videos/gantt_283.4ms_serial_3way-CPU-DSP-HTA.mp4

    # PIPELINED (Task 3) -- --control-trace is the SERIAL walk from the same
    # board session, so the overlap contrast is measured on both sides
    python animate_octo_gantt.py --schedule pipelined \
        --trace "$T/pipe200_gated_20260905-164357.log" \
        --control-trace "$T/ungated_20260905-164735.log" \
        --video runs_video/lat283_pipelined/ep05_success_True.mp4 \
        --latency-ms 260.5 --issue-period-ms 200 \
        --arm-sr 56.9 --arm-n 72 --episode-success True \
        --out videos/gantt_pipelined_200ms_MEASURED.mp4
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Polygon, Rectangle

# ---------------------------------------------------------------- style
# Lane order and segment colours are lifted verbatim from
# qnn_models/octo/plot_octo_gantt.py so the animation and the static figure are
# visibly the same chart.  INSTANCE_COLOUR is lifted from
# octo_pipe/plot_measured_lanes.py for the same reason.
LANES = ["CPU", "DSP", "HTA"]
LANE_MACHINE = {"CPU": "CPU_X#0", "DSP": "CPU_E#0", "HTA": "CPU_P#0"}
KIND_COLOUR = {
    "trunk (monolith)": "#7f8c8d",
    "trunk":            "#7f8c8d",
    "stem":  "#8e44ad",
    "join":  "#95a5a6",
    "pre2":  "#e67e22",
    "posta": "#c0392b",
    "postb": "#e74c3c",
    "mlp":   "#2980b9",
    "tail":  "#16a085",
    "score": "#27ae60",
}
INSTANCE_COLOUR = ["#8e44ad", "#e67e22", "#2980b9", "#27ae60", "#c0392b",
                   "#16a085", "#d35400", "#7d3c98", "#b7950b", "#34495e"]
DEFAULT_COLOUR = "#bdc3c7"

BG          = "#11151c"
PANEL       = "#1b212b"
FG          = "#e8ecf2"
DIM         = "#8a97a8"
HOLD_COLOUR = "#f39c12"
EXEC_COLOUR = "#2ecc71"
LAND_COLOUR = "#2ecc71"
ISSUE_COLOUR = "#e74c3c"
DUP_COLOUR  = "#7f8c8d"
WAIT_COLOUR = "#5d6d7e"
BAND_COLOUR = "#48c9b0"

CHUNK = 4

# Where the "instances in flight" band sits, in gantt data coordinates.
BAND_LO, BAND_HI = -1.80, -0.80


# ---------------------------------------------------------------- trace
def parse_trace(log_path: Path) -> list[dict]:
    """Pull the CSV block between the AGENTS_QNN_TRACE markers out of a log."""
    text = log_path.read_text(errors="replace")
    try:
        body = text.split("AGENTS_QNN_TRACE_BEGIN", 1)[1]
        body = body.split("AGENTS_QNN_TRACE_END", 1)[0]
    except IndexError:
        raise SystemExit(f"{log_path}: no AGENTS_QNN_TRACE block found")
    raw = body.splitlines()
    start = next((i for i, ln in enumerate(raw) if ln.startswith("seg_id,")), None)
    if start is None:
        raise SystemExit(f"{log_path}: trace block has no seg_id header")
    lines = [ln for ln in raw[start:]
             if ln.strip() and not ln.lstrip().startswith("===")]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    if not rows:
        raise SystemExit(f"{log_path}: trace block is empty")
    return rows


def load_dispatches(log_path: Path):
    """-> (dispatches, span_ms, busy_by_lane, kinds_in_order).

    ``dispatches`` is [(lane_row, start_ms, dur_ms, kind, instance)] with time
    rebased so the first dispatch of the whole walk starts at 0.
    """
    rows = parse_trace(log_path)
    t0 = min(float(r["actual_start_ms"]) for r in rows)
    out, busy, kinds = [], {k: 0.0 for k in LANES}, []
    for r in rows:
        lane = (r.get("actual_backend") or "").strip()
        if lane not in LANES:
            continue
        kind = (r.get("backend_label") or "?").strip()
        inst = int(r.get("instance", 0) or 0)
        s = float(r["actual_start_ms"]) - t0
        e = float(r["actual_end_ms"]) - t0
        out.append((LANES.index(lane), s, e - s, kind, inst))
        busy[lane] += e - s
        if kind not in kinds:
            kinds.append(kind)
    span = max(s + w for _r, s, w, _k, _i in out)
    return out, span, busy, kinds


def concurrency(intervals):
    """-> (overlap_ms, max_concurrent, time_at_depth) over [(start, end)].

    ``overlap_ms`` is the wall time during which two or more intervals are
    simultaneously live.  Used for both lane occupancy (an interval per
    dispatch) and instance occupancy (an interval per inference), so the
    "strictly serial" / "genuinely overlapping" claim is checkable from the
    trace itself rather than taken on trust.
    """
    ev = []
    for s, e in intervals:
        ev.append((s, +1))
        ev.append((e, -1))
    ev.sort()
    n, prev, overlap, mx = 0, None, 0.0, 0
    at_depth: dict[int, float] = {}
    for t, d in ev:
        if prev is not None and t > prev:
            at_depth[n] = at_depth.get(n, 0.0) + (t - prev)
            if n >= 2:
                overlap += t - prev
        n += d
        mx = max(mx, n)
        prev = t
    return overlap, mx, at_depth


def lane_concurrency(items):
    """-> (overlap_ms, max_distinct_lanes, edges, counts) over [(lane, s, e)].

    Counts DISTINCT LANES busy, not dispatches.  On the measured traces each
    lane serialises (verified: max same-lane concurrency = 1), so this agrees
    with a naive per-dispatch count -- but the animation tiles the walk onto an
    exact issue grid, which CAN put two dispatches on one lane at once.  Then a
    per-dispatch count reports more "lanes" than the machine has.  Everything
    the frame says about lanes goes through here.
    """
    ev = []
    for lane, s, e in items:
        ev.append((s, +1, lane))
        ev.append((e, -1, lane))
    ev.sort(key=lambda x: (x[0], x[1]))
    live: dict = {}
    prev, ov, mx = None, 0.0, 0
    edges, counts = [ev[0][0]], [0]
    for tt, d, lane in ev:
        n = sum(1 for v in live.values() if v > 0)
        if prev is not None and tt > prev and n >= 2:
            ov += tt - prev
        live[lane] = live.get(lane, 0) + d
        n2 = sum(1 for v in live.values() if v > 0)
        mx = max(mx, n2)
        if tt == edges[-1]:
            counts[-1] = n2
        else:
            edges.append(tt)
            counts.append(n2)
        prev = tt
    import numpy as _np
    return ov, mx, _np.asarray(edges), _np.asarray(counts)


def measured_overlap_ms(dispatches) -> float:
    """Lane overlap of a dispatch list, in ms. Kept for the serial call site."""
    return concurrency([(s, s + w) for _l, s, w, _k, _i in dispatches])[0]


def instance_shapes(dispatches):
    """Split a walk into per-instance shapes rebased to each instance's start.

    -> (shapes, stats).  ``shapes[i]`` is [(lane_row, s_rel, dur, kind)] for
    Octo instance ``i``; ``stats`` carries everything the frame prints about
    the MEASURED walk.
    """
    per: dict[int, list] = {}
    for row, s, w, kind, inst in dispatches:
        per.setdefault(inst, []).append((row, s, w, kind))
    ids = sorted(per)
    shapes, starts, ends = [], [], []
    for i in ids:
        s0 = min(s for _r, s, _w, _k in per[i])
        e1 = max(s + w for _r, s, w, _k in per[i])
        starts.append(s0)
        ends.append(e1)
        shapes.append([(r, s - s0, w, k) for r, s, w, k in per[i]])
    lat = [e - s for s, e in zip(starts, ends)]
    gaps = [b - a for a, b in zip(ends, ends[1:])]
    lane_ov, lane_max, _le, _lc = lane_concurrency(
        [(LANES[l], s, s + w) for l, s, w, _k, _i in dispatches])
    lane_depth = concurrency([(s, s + w) for _l, s, w, _k, _i in dispatches])[2]
    inst_ov, inst_max, _ = concurrency(list(zip(starts, ends)))
    span = max(ends)
    stats = dict(
        ids=ids, n_inst=len(ids), starts=starts, ends=ends, lat=lat, gaps=gaps,
        lat_median=statistics.median(lat),
        gap_median=statistics.median(gaps) if gaps else 0.0,
        gap_min=min(gaps) if gaps else 0.0, gap_max=max(gaps) if gaps else 0.0,
        lane_overlap_ms=lane_ov, lane_overlap_pct=100.0 * lane_ov / span,
        max_lanes=lane_max, lane_depth=lane_depth,
        inst_overlap_ms=inst_ov, max_inst=inst_max, span=span)
    return shapes, stats


# ---------------------------------------------------------------- schedules
def serial_schedule(n_steps: int, D: int):
    """Replay latency_eval.py's SERIAL scheduler.

    One inference in flight: issued at step k, lands at step k + D, and the
    next is issued the instant the previous lands -> issues at 0, D, 2D, ...
    Returns ``(cycles, per_step)`` where ``cycles`` is [(issue_step,
    arrival_step)] and ``per_step[t]`` is ``(phase, chunk_entry, issue_step)``
    with phase in {"hold", "exec", "zoh"}.
    """
    if D <= 0:
        cycles = [(k, k) for k in range(n_steps)]
        return cycles, [("exec", 0, k) for k in range(n_steps)]

    cycles, k = [], 0
    while k < n_steps:
        cycles.append((k, k + D))
        k += D

    per_step = []
    for t in range(n_steps):
        landed = [c for c in cycles if c[1] <= t]
        if not landed:
            per_step.append(("hold", None, None))
            continue
        issue, arrive = landed[-1]
        j = t - arrive
        per_step.append((("exec" if j < CHUNK else "zoh"),
                         min(j, CHUNK - 1), issue))
    return cycles, per_step


def pipelined_schedule(n_steps: int, D: int):
    """Replay latency_eval.py's ``--pipeline`` scheduler.

    A fresh inference is issued on EVERY control step, on that step's own
    observation, and lands D steps later.  The applied action at step t is the
    timestep-aligned mean over every delivered chunk whose entry ``t - issue``
    targets t -- i.e. issues in ``[t - 3, t - D]``.  That is 4 predictions at
    D = 0, 2 at D = 2, 1 at D = 3 and NONE at D >= 4 (RESULTS.txt section 2).

    Returns ``(cycles, per_step)`` with ``per_step[t] = (phase, entries,
    issues)``; ``entries[j]`` is the chunk entry taken from the inference
    issued at step ``issues[j]``, so the pair is the whole sensor -> actuation
    path for that step.
    """
    cycles = [(k, k + D) for k in range(n_steps)]
    per_step = []
    for t in range(n_steps):
        lo, hi = max(0, t - CHUNK + 1), t - D
        issues = [i for i in range(lo, hi + 1)]
        if not issues:
            per_step.append(("hold", [], []))
        else:
            per_step.append(("avg", [t - i for i in issues], issues))
    return cycles, per_step


# ---------------------------------------------------------------- video
def read_frames(video: Path) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")
    return frames


def step_fn(intervals):
    """-> (edges, counts) so counts[searchsorted(edges, t, 'right') - 1] is the
    number of live intervals at t.  Used for the live 'in flight now' readout
    and for the instances-in-flight band."""
    ev = sorted([(s, +1) for s, _e in intervals] + [(e, -1) for _s, e in intervals])
    edges, counts, n = [ev[0][0]], [0], 0
    for t, d in ev:
        n += d
        if t == edges[-1]:
            counts[-1] = n
        else:
            edges.append(t)
            counts.append(n)
    return np.asarray(edges), np.asarray(counts)


def at(edges, counts, t):
    i = int(np.searchsorted(edges, t, side="right")) - 1
    return int(counts[i]) if 0 <= i < len(counts) else 0


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True,
                   help="runtime log carrying an AGENTS_QNN_TRACE block (MEASURED)")
    p.add_argument("--video", type=Path, required=True,
                   help="SIMPLER rollout mp4; frame i is the state at env step i")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--schedule", choices=["serial", "pipelined"], default="serial")
    p.add_argument("--latency-ms", type=float, required=True,
                   help="MEASURED end-to-end latency that sets D (per-inference, "
                        "sensor -> result)")
    p.add_argument("--latency-note", default="",
                   help="provenance string shown next to the latency")
    p.add_argument("--config-label", default="3-way CPU+DSP+HTA",
                   help="hardware configuration this trace came from")
    p.add_argument("--control-period-ms", type=float, default=200.0,
                   help="env control period; control_freq = 5 -> 200 ms")
    p.add_argument("--issue-period-ms", type=float, default=0.0,
                   help="pipelined only: MEASURED hardware issue cadence. "
                        "0 = same as the control period")
    p.add_argument("--throughput-note", default="",
                   help="pipelined only: extra MEASURED throughput provenance")
    p.add_argument("--control-trace", type=Path, default=None,
                   help="pipelined only: a SERIAL trace from the same board "
                        "session. Its overlap and max concurrent lanes are "
                        "recomputed here and shown as the non-overlapping "
                        "reference, so the contrast is measured, not asserted")
    p.add_argument("--arm-sr", type=float, required=True,
                   help="MEASURED success rate of this arm, percent")
    p.add_argument("--arm-n", type=int, required=True)
    p.add_argument("--arm-note", default="",
                   help="extra line under the arm block")
    p.add_argument("--episode-success", default="False")
    p.add_argument("--episode-label", default="")
    p.add_argument("--headline", default="",
                   help="one-line claim printed under the title")
    p.add_argument("--window-ms", type=float, default=1600.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--slow-steps", type=int, default=9,
                   help="env steps played in slow motion before real time")
    p.add_argument("--slow-speed", type=float, default=0.22)
    p.add_argument("--fast-speed", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=0, help="0 = whole episode")
    p.add_argument("--title-card-s", type=float, default=3.0)
    p.add_argument("--dpi", type=int, default=110)
    args = p.parse_args()

    pipelined = args.schedule == "pipelined"
    ep_success = str(args.episode_success).lower() in ("1", "true", "yes")
    period = args.control_period_ms
    issue_period = args.issue_period_ms or period
    D = max(int(np.ceil(args.latency_ms / period - 1e-9)), 0)

    dispatches, span, busy, kinds = load_dispatches(args.trace)
    shapes, mst = instance_shapes(dispatches)
    overlap = mst["lane_overlap_ms"]

    # Same-session SERIAL control, recomputed the same way, so "overlapping"
    # is a measured contrast against a measured baseline rather than a claim.
    ctl = None
    if args.control_trace is not None:
        c_disp, c_span, _c_busy, _c_kinds = load_dispatches(args.control_trace)
        _c_shapes, c_mst = instance_shapes(c_disp)
        ctl = dict(name=args.control_trace.name, n=len(c_disp), span=c_span,
                   ov=c_mst["lane_overlap_ms"], pct=c_mst["lane_overlap_pct"],
                   lanes=c_mst["max_lanes"], lat=c_mst["lat_median"])
    frames = read_frames(args.video)

    n_steps = len(frames) - 1          # frame 0 is the reset state
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    total_ms = n_steps * period

    if pipelined:
        cycles, per_step = pipelined_schedule(n_steps, D)
    else:
        cycles, per_step = serial_schedule(n_steps, D)

    # Cross-check the latency the caller supplied against what the drawn trace
    # actually did, so a mismatched --latency-ms cannot pass silently.
    if abs(args.latency_ms - mst["lat_median"]) > 0.12 * mst["lat_median"]:
        print(f"[warn] --latency-ms {args.latency_ms:.1f} differs by "
              f">12% from this trace's median per-instance latency "
              f"{mst['lat_median']:.1f} ms")

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.00, 0.95, 0.26],
                          width_ratios=[1.0, 1.12],
                          hspace=0.30, wspace=0.05,
                          left=0.108, right=0.982,
                          top=0.900 if pipelined else 0.912,
                          bottom=0.238 if pipelined else 0.180)
    ax_vid = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_g = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :], sharex=ax_g)

    ax_vid.set_facecolor(BG)
    ax_vid.axis("off")
    im = ax_vid.imshow(frames[0])

    # ------------------------------------------------------------ status panel
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")
    ax_txt.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, facecolor=PANEL,
                               edgecolor="#2b3441", lw=1.0, zorder=0))
    banner = ax_txt.text(0.5, 0.985 if pipelined else 0.975, "", ha="center",
                         va="top", fontsize=18 if pipelined else 21,
                         color=HOLD_COLOUR, family="DejaVu Sans Mono",
                         fontweight="bold", zorder=3)
    status = ax_txt.text(0.045, 0.885 if pipelined else 0.862, "", ha="left",
                         va="top", fontsize=10.4 if pipelined else 11.5,
                         color=FG, family="DejaVu Sans Mono",
                         linespacing=1.45 if pipelined else 1.56, zorder=3)

    ep_txt = "SUCCEEDED" if ep_success else "FAILED"
    if pipelined:
        # How many MEASURED inferences actually fit inside the D that the
        # median implies.  D = ceil(median / 200) can be true of the median and
        # false of a large minority, and at D+1 this arm's SR collapses, so the
        # count is computed and shown rather than glossed.
        n_over = sum(1 for x in mst["lat"] if x > D * period)
        d_line = (f" {n_over} of {mst['n_inst']} exceeded {D*period:.0f} ms → "
                  f"those land at D = {D+1}"
                  if n_over else
                  f" all {mst['n_inst']} landed inside {D*period:.0f} ms → "
                  f"D = {D} holds")
        # One contiguous block anchored to the BOTTOM of the panel, so it grows
        # upward and cannot be pushed off by a long live-status block above it.
        arm_block = (
            f"── THIS WALK ON THE BOARD (MEASURED) ──\n"
            f" {mst['n_inst']} instances · {len(dispatches)} dispatches · "
            f"{issue_period:.0f} ms issue cadence\n"
            f" overlap {overlap:.1f} ms = {mst['lane_overlap_pct']:.1f}% · "
            f"max {mst['max_lanes']} lanes · ≤{mst['max_inst']} in flight\n"
            f" per-inference {mst['lat_median']:.0f} ms · "
            f"completes every {mst['gap_median']:.0f} ms\n"
            f"{d_line}"
            + (f"\n serial control same session: {ctl['pct']:.1f}% overlap, "
               f"{ctl['lanes']} lane" if ctl else "")
            + (f"\n{args.throughput_note}" if args.throughput_note else "")
            + f"\n── THIS ARM (MEASURED) ────────────────\n"
            f" arm SR {args.arm_sr:.1f}% (n = {args.arm_n}) · this ep {ep_txt}\n"
            f" policy octo-small-1.0, fp32 on host"
            + (f" · {args.episode_label}" if args.episode_label else "")
            + (f"\n{args.arm_note}" if args.arm_note else ""))
        ax_txt.text(0.045, 0.028, arm_block, ha="left", va="bottom",
                    fontsize=8.2, color=DIM, family="DejaVu Sans Mono",
                    linespacing=1.34, zorder=3)
    else:
        arm_block = (
            f"── THIS ARM (MEASURED) ─────────────\n"
            f" arm SR    {args.arm_sr:.1f}%   (n = {args.arm_n})\n"
            f" this ep   {ep_txt}\n"
            f" policy    octo-small-1.0, fp32 on host"
            + (f"\n episode   {args.episode_label}" if args.episode_label else ""))
        ax_txt.text(0.045, 0.40, arm_block, ha="left", va="top", fontsize=10.8,
                    color=DIM, family="DejaVu Sans Mono", linespacing=1.56,
                    zorder=3)

    # ------------------------------------------------------------ gantt
    ax_g.set_facecolor(PANEL)
    for sp in ax_g.spines.values():
        sp.set_color("#2b3441")
    ax_g.tick_params(colors=DIM, labelsize=9, labelbottom=False)
    y_top = len(LANES) + (0.98 if pipelined else 0.55)
    ax_g.set_ylim(BAND_LO - 0.16 if pipelined else -1.05, y_top)
    yt = list(range(len(LANES)))
    ytl = [f"{l} ({LANE_MACHINE[l]})\nbusy {busy[l]:.1f} ms / {100*busy[l]/span:.1f}%"
           for l in LANES]
    if pipelined:
        yt.append((BAND_LO + BAND_HI) / 2)
        ytl.append(f"instances\nin flight\n(0–{mst['max_inst']})")
    ax_g.set_yticks(yt)
    ax_g.set_yticklabels(ytl, fontsize=8.4, color=FG)
    ax_g.grid(True, axis="x", alpha=0.15, color=FG, zorder=0)

    if pipelined:
        title = textwrap.fill(
            f"MEASURED pipelined per-dispatch trace — {args.config_label} · "
            f"{len(dispatches)} dispatches · {mst['n_inst']} Octo instances · "
            f"walk {span:.1f} ms · overlap {overlap:.1f} ms ({mst['lane_overlap_pct']:.1f}%) · "
            f"max {mst['max_lanes']} concurrent lanes · up to {mst['max_inst']} instances in flight"
            f"{('  ' + args.latency_note) if args.latency_note else ''}", width=168)
    else:
        title = (f"MEASURED per-dispatch trace — {args.config_label}, "
                 f"{len(dispatches)} dispatches, {span:.1f} ms in this iteration"
                 f"{('   ' + args.latency_note) if args.latency_note else ''}")
    ax_g.set_title(title, fontsize=8.8 if pipelined else 9.8, color=DIM, pad=4,
                   linespacing=1.35)

    # Every bar of every inference cycle, precomputed once. Drawn twice: a
    # ghost collection (always visible) and a solid one clipped to the left of
    # the playhead, so the schedule "fills in" without rebuilding artists.
    verts, colours, drawn_lanes = [], [], []
    issues = []          # (j, base_ms, frame_step, is_fresh, inst_idx, shape_span)
    if pipelined:
        # An inference is started every issue_period ms.  It consumes the most
        # recent camera frame, i.e. the observation of env step
        # floor(base / control_period).  When the issue cadence is faster than
        # the control cadence, consecutive issues consume the SAME frame; those
        # duplicates are marked, because no new observation exists for them.
        n_issue = int(np.ceil(total_ms / issue_period)) + 1
        prev_frame = None
        for j in range(n_issue):
            base = j * issue_period
            k = int(base // period)
            fresh = k != prev_frame
            prev_frame = k
            si = j % mst["n_inst"]
            shape = shapes[si]
            sspan = max(s + w for _r, s, w, _k in shape)
            issues.append((j, base, k, fresh, si, sspan))
            for row, s, w, _kind in shape:
                x0, x1 = base + s, base + s + w
                y0, y1 = row - 0.33, row + 0.33
                verts.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                colours.append(INSTANCE_COLOUR[si % len(INSTANCE_COLOUR)])
                drawn_lanes.append((row, x0, x1))
    else:
        for issue_step, _arrive in cycles:
            base = issue_step * period
            for row, s, w, kind, _i in dispatches:
                x0, x1 = base + s, base + s + w
                y0, y1 = row - 0.33, row + 0.33
                verts.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                colours.append(KIND_COLOUR.get(kind, DEFAULT_COLOUR))
                drawn_lanes.append((row, x0, x1))

    ghost = PolyCollection(verts, facecolors=colours, edgecolors="none",
                           alpha=0.15, zorder=2)
    solid = PolyCollection(verts, facecolors=colours, edgecolors="black",
                           linewidths=0.25, zorder=3)
    ax_g.add_collection(ghost)
    ax_g.add_collection(solid)

    # Live occupancy of the DRAWN schedule, for the per-frame readout and for
    # the instances-in-flight band.  Computed from the bars actually on screen,
    # so what the status line says is what the figure shows.
    drawn_span = max(v[1][0] for v in verts)
    drawn_overlap, drawn_max_lanes, lane_edges, lane_counts = lane_concurrency(
        [(LANES[lr], x0, x1) for lr, x0, x1 in drawn_lanes])

    if pipelined:
        inf_iv = [(b, b + ss) for _j, b, _k, _f, _si, ss in issues]
        inf_edges, inf_counts = step_fn(inf_iv)
        drawn_inst_max = concurrency(inf_iv)[1]
        # the band: instances in flight over the whole timeline
        bx = np.append(inf_edges, total_ms + issue_period)
        by = BAND_LO + (inf_counts / max(drawn_inst_max, 1)) * (BAND_HI - BAND_LO)
        by = np.append(by, BAND_LO)
        ax_g.fill_between(bx, BAND_LO, by, step="post", facecolor=BAND_COLOUR,
                          alpha=0.30, edgecolor=BAND_COLOUR, linewidth=0.8,
                          zorder=2)
        ax_g.axhline(BAND_HI, color="#2b3441", lw=0.6, zorder=1)
        ytl[-1] = f"instances\nin flight\n(0–{drawn_inst_max})"
        ax_g.set_yticklabels(ytl, fontsize=8.4, color=FG)
    else:
        inf_edges = inf_counts = None
        drawn_inst_max = 1

    # The gap between "compute finished" and "the env can use it": D is a
    # ceil() of the latency onto the 200 ms control tick, so a result that is
    # ready early still waits. Drawn, because it is a real part of the delay.
    wait_verts = []
    y_lo = BAND_LO - 0.16 if pipelined else -1.05
    for issue_step, arrive_step in cycles:
        base = issue_step * period
        if base + args.latency_ms >= arrive_step * period:
            continue
        wait_verts.append([(base + args.latency_ms, y_lo),
                           (arrive_step * period, y_lo),
                           (arrive_step * period, y_top),
                           (base + args.latency_ms, y_top)])
    if wait_verts:
        ax_g.add_collection(PolyCollection(
            wait_verts, facecolors=WAIT_COLOUR, edgecolors="none",
            alpha=0.10 if pipelined else 0.22, zorder=1))

    marker_y = len(LANES) + 0.20
    if pipelined:
        # Two marker rows, because these are two different clocks:
        #   lower row  = the HARDWARE axis, one mark per inference the board
        #                starts (every issue_period ms), labelled with the
        #                camera frame that inference consumed;
        #   upper row  = the CONTROL axis, one mark per env step (every 200 ms),
        #                labelled sK -> sK+D, the frame -> actuation hop.
        land_y = len(LANES) + 0.62
        _bb = dict(facecolor=PANEL, edgecolor="none", pad=1.2, alpha=0.92)
        ax_g.text(0.0015, (marker_y + 0.02 - (BAND_LO - 0.16)) / (y_top - (BAND_LO - 0.16)),
                  "hardware issues", transform=ax_g.transAxes, fontsize=6.8,
                  color=ISSUE_COLOUR, ha="left", va="center", zorder=9, bbox=_bb)
        ax_g.text(0.0015, (land_y + 0.02 - (BAND_LO - 0.16)) / (y_top - (BAND_LO - 0.16)),
                  "control steps", transform=ax_g.transAxes, fontsize=6.8,
                  color=LAND_COLOUR, ha="left", va="center", zorder=9, bbox=_bb)
        # issue markers: one per hardware inference, tied to the FRAME it read
        for _j, base, k, fresh, si, _ss in issues:
            if base > total_ms:
                continue
            col = ISSUE_COLOUR if fresh else DUP_COLOUR
            ax_g.axvline(base, color=col, lw=1.0, ls=":",
                         alpha=0.50 if fresh else 0.22, zorder=4)
            ax_g.plot([base], [marker_y], marker="v", ms=6 if fresh else 5,
                      color=col, mfc=col if fresh else "none", mew=1.0,
                      zorder=6, clip_on=True)
            ax_g.text(base + 5, marker_y + 0.11,
                      f"reads s{k}" if fresh else f"dup s{k}",
                      fontsize=6.9, color=col, va="bottom", ha="left",
                      zorder=6, clip_on=True)
        # landing markers: one per CONTROL step, D steps after the frame
        for issue_step, arrive_step in cycles:
            land = arrive_step * period
            if land > total_ms + period:
                continue
            ax_g.axvline(land, color=LAND_COLOUR, lw=1.1, alpha=0.35, zorder=4)
            ax_g.plot([land], [land_y], marker="^", ms=6, color=LAND_COLOUR,
                      zorder=6, clip_on=True)
            ax_g.text(land + 5, land_y + 0.11, f"s{issue_step}→s{arrive_step}",
                      fontsize=6.9, color=LAND_COLOUR, va="bottom", ha="left",
                      zorder=6, clip_on=True)
    else:
        for issue_step, arrive_step in cycles:
            base = issue_step * period
            land = arrive_step * period
            ax_g.axvline(base, color=ISSUE_COLOUR, lw=1.1, ls=":", alpha=0.6, zorder=4)
            ax_g.axvline(base + args.latency_ms, color="#ff8a80", lw=1.0, ls="--",
                         alpha=0.45, zorder=4)
            ax_g.axvline(land, color=LAND_COLOUR, lw=1.5, alpha=0.65, zorder=4)
            ax_g.plot([base + 10, land - 10], [marker_y, marker_y],
                      color=LAND_COLOUR, lw=1.0, alpha=0.5, zorder=5, clip_on=True)
            ax_g.plot([base], [marker_y], marker="v", ms=6, color=ISSUE_COLOUR,
                      zorder=6, clip_on=True)
            ax_g.plot([land], [marker_y], marker="^", ms=6, color=LAND_COLOUR,
                      zorder=6, clip_on=True)
            ax_g.text(base + 7, marker_y + 0.14, f"issue s{issue_step}", fontsize=7.4,
                      color=ISSUE_COLOUR, va="bottom", ha="left", zorder=6,
                      clip_on=True)
            ax_g.text(land - 7, marker_y + 0.14, f"lands s{arrive_step}", fontsize=7.4,
                      color=LAND_COLOUR, va="bottom", ha="right", zorder=6,
                      clip_on=True)
            ax_g.text((base + args.latency_ms + land) / 2, -0.95,
                      "ready, waiting\nfor the 200 ms tick", fontsize=6.6,
                      color="#9fb0c0", ha="center", va="bottom", zorder=6,
                      clip_on=True)

    playhead_g = ax_g.axvline(0, color="#ff4d4d", lw=2.2, zorder=8)

    # ------------------------------------------------------------ control strip
    ax_c.set_facecolor(PANEL)
    for sp in ax_c.spines.values():
        sp.set_color("#2b3441")
    ax_c.tick_params(colors=DIM, labelsize=9)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([])
    ax_c.set_xlabel("env wall-clock time since episode start (ms)  —  "
                    "one control step = 200 ms  (control_freq = 5)"
                    + (f"   —  hardware issues an inference every "
                       f"{issue_period:.0f} ms" if pipelined else ""),
                    fontsize=9.8, color=DIM, labelpad=3)
    ax_c.text(-0.012, 0.5, "applied\naction", transform=ax_c.transAxes,
              fontsize=9, color=FG, rotation=0, ha="right", va="center")

    cell_v, cell_c, cell_t = [], [], []
    for t, item in enumerate(per_step):
        x0, x1 = t * period, (t + 1) * period - 4
        cell_v.append([(x0, 0.10), (x1, 0.10), (x1, 0.90), (x0, 0.90)])
        if pipelined:
            phase, entries, _iss = item
            cell_c.append(HOLD_COLOUR if phase == "hold" else
                          (EXEC_COLOUR if len(entries) >= 2 else "#f1c40f"))
            cell_t.append("HOLD" if phase == "hold"
                          else "+".join(f"e{e}" for e in entries))
        else:
            phase, entry, _iss = item
            cell_c.append({"hold": HOLD_COLOUR, "exec": EXEC_COLOUR,
                           "zoh": "#7f8c8d"}[phase])
            cell_t.append("HOLD" if phase == "hold"
                          else (f"e{entry}" if phase == "exec" else "ZOH"))
    ax_c.add_collection(PolyCollection(cell_v, facecolors=cell_c,
                                       edgecolors="none", alpha=0.16, zorder=2))
    cells_solid = PolyCollection(cell_v, facecolors=cell_c, edgecolors="#0d1117",
                                 linewidths=0.6, zorder=3)
    ax_c.add_collection(cells_solid)
    for t, lab in enumerate(cell_t):
        ax_c.text(t * period + period / 2 - 2, 0.5, lab, ha="center",
                  va="center", fontsize=7.4 if pipelined else 8.2,
                  color="#0d1117", fontweight="bold", zorder=4, clip_on=True)
    playhead_c = ax_c.axvline(0, color="#ff4d4d", lw=2.2, zorder=8)

    # ------------------------------------------------------------ legend + footer
    if pipelined:
        handles = [Patch(facecolor=INSTANCE_COLOUR[i % len(INSTANCE_COLOUR)],
                         edgecolor="black", label=f"octo{i}")
                   for i in range(mst["n_inst"])]
        handles += [Patch(facecolor=BAND_COLOUR, alpha=0.4, edgecolor="none",
                          label="instances in flight"),
                    Patch(facecolor=EXEC_COLOUR, edgecolor="#0d1117",
                          label="applying a timestep-aligned mean"),
                    Patch(facecolor=HOLD_COLOUR, edgecolor="#0d1117",
                          label="robot holding (nothing has landed)")]
        ncol = min(len(handles), 8)
    else:
        handles = [Patch(facecolor=KIND_COLOUR.get(k, DEFAULT_COLOUR),
                         edgecolor="black", label=k) for k in kinds]
        handles += [Patch(facecolor=WAIT_COLOUR, alpha=0.4, edgecolor="none",
                          label="result ready, waiting for the control tick"),
                    Patch(facecolor=EXEC_COLOUR, edgecolor="#0d1117",
                          label="applying a chunk entry"),
                    Patch(facecolor=HOLD_COLOUR, edgecolor="#0d1117",
                          label="robot holding (nothing has landed)")]
        ncol = 6
    fig.legend(handles=handles, loc="lower left",
               bbox_to_anchor=(0.108, 0.132 if pipelined else 0.088),
               ncol=ncol, fontsize=8.2, framealpha=0.0, labelcolor=DIM,
               handlelength=1.5, columnspacing=1.3, handletextpad=0.5)

    if pipelined:
        n_avg = sorted({len(e) for ph, e, _i in per_step if ph == "avg"}) or [0]
        stale = [e * period for ph, es, _i in per_step if ph == "avg" for e in es]
        st_lo = min(stale) if stale else 0.0
        st_hi = max(stale) if stale else 0.0
        surplus = sum(1 for _j, b, _k, f, _s, _ss in issues
                      if not f and b <= total_ms)
        total_iss = sum(1 for _j, b, _k, _f, _s, _ss in issues if b <= total_ms)
        dup_note = (
            f"{surplus} of {total_iss} inferences re-read a frame an earlier one had "
            f"already read — the camera yields one frame per 200 ms step, so they add "
            f"throughput, not information"
            if surplus else
            "one inference per control step, every one reading a fresh camera frame")
        tighter = drawn_overlap / drawn_span > mst["lane_overlap_pct"] / 100
        left_col = (
            f"MEASURED — recomputed from this trace on every run, never hardcoded: "
            f"every gantt bar; lane busy; lane overlap {overlap:.1f} ms = "
            f"{mst['lane_overlap_pct']:.1f}% of the walk; max {mst['max_lanes']} concurrent "
            f"lanes; ≤{mst['max_inst']} instances in flight; per-inference latency "
            f"{mst['lat_median']:.0f} ms; completion cadence {mst['gap_median']:.0f} ms.\n"
            f"MODELLED — D = ceil({args.latency_ms:.0f}/200) = {D} control steps, and the measured "
            f"walk redrawn once per issue on an exact {issue_period:.0f} ms grid. That grid packs "
            f"{'tighter' if tighter else 'looser'} than the board's own releases "
            f"({', '.join(f'{g:.0f}' for g in np.diff(mst['starts']))} ms apart), so the DRAWN "
            f"composite runs at {100*drawn_overlap/drawn_span:.1f}% overlap against the walk's "
            f"{mst['lane_overlap_pct']:.1f}%. Neither is rescaled to match the other.")
        ctl_txt = (
            f" The same-session SERIAL control ({ctl['name']}, {ctl['n']} dispatches) "
            f"recomputes to {ctl['ov']:.1f} ms overlap = {ctl['pct']:.1f}%, max {ctl['lanes']} lane — "
            f"so the contrast is measured on both sides, not asserted." if ctl else "")
        right_col = (
            f"GENUINELY OVERLAPPING, AND MEASURED — not a serial trace, not a modelled "
            f"strategy. Inference cadence ({issue_period:.0f} ms, hardware) and control cadence "
            f"(200 ms, control_freq = 5) are separate axes: {dup_note}.{ctl_txt}\n"
            f"SENSOR → ACTUATION — the action applied at step k is the mean of "
            f"{'/'.join(str(x) for x in n_avg)} chunk entr{'y' if n_avg == [1] else 'ies'}, computed "
            f"from camera frames captured {st_lo:.0f}–{st_hi:.0f} ms earlier. The int8 pipeline is a "
            f"PERFORMANCE VEHICLE (3-way chain cos 0.008 vs the JAX golden): the policy ran at fp32 "
            f"on the host, and only each output's ARRIVAL TIME is board-derived.")
        for x, blk in ((0.108, left_col), (0.552, right_col)):
            fig.text(x, 0.010,
                     "\n".join(textwrap.fill(p, width=112) for p in blk.split("\n")),
                     fontsize=7.4, color=DIM, family="DejaVu Sans", va="bottom",
                     linespacing=1.52)
    else:
        used_entries = sorted({e for ph, e, _i in per_step if ph == "exec"})
        dead = [e for e in range(CHUNK) if e not in used_entries]
        dead_note = (f"chunk entries {', '.join('e%d' % e for e in dead)} are never reached "
                     f"— a new chunk supersedes the old one after D = {D} steps"
                     if dead else
                     f"all {CHUNK} chunk entries are consumed before the next chunk lands")
        fig.text(0.108, 0.011,
                 "MEASURED  gantt bars, lane busy, end-to-end latency, arm success rate, rollout frames.        "
                 "MODELLED  latency → D = ceil(latency / 200 ms) steps; one measured trace redrawn per issue.\n"
                 "The int8 pipeline is a PERFORMANCE VEHICLE — the 3-way chain measures cos 0.008 against the JAX golden. "
                 "The policy here ran at full fp32 on the host; only each output's ARRIVAL TIME is board-derived.\n"
                 f"Strictly serial: MEASURED max_concurrent_lanes = {mst['max_lanes']}, {overlap:.2f} ms lane overlap. "
                 f"This trace draws one inference at a time.        {dead_note}.",
                 fontsize=8.3, color=DIM, family="DejaVu Sans", va="bottom",
                 linespacing=1.65)

    if pipelined:
        sup = (f"Octo-small on QRB5165 — MEASURED PIPELINED execution, "
               f"{issue_period:.0f} ms issue cadence (D = {D} control steps)")
    else:
        sup = (f"Octo-small on QRB5165 — MEASURED {args.latency_ms:.1f} ms inference "
               f"latency, SERIAL schedule (D = {D} control steps)")
    fig.suptitle(sup, fontsize=16, color=FG,
                 y=0.978 if pipelined else 0.972, fontweight="bold")
    if args.headline:
        fig.text(0.5, 0.944, textwrap.fill(args.headline, width=155),
                 ha="center", va="top", fontsize=9.6, color=BAND_COLOUR,
                 family="DejaVu Sans", linespacing=1.4)

    # ------------------------------------------------------------ time warp
    ramp = period
    times, t = [], 0.0
    while t <= total_ms:
        times.append(t)
        s0 = args.slow_steps * period
        if t < s0:
            sp = args.slow_speed
        elif t < s0 + ramp:
            f = (t - s0) / ramp
            sp = args.slow_speed + f * (args.fast_speed - args.slow_speed)
        else:
            sp = args.fast_speed
        t += sp * 1000.0 / args.fps
    times.append(total_ms)

    import imageio.v2 as imageio
    writer = imageio.get_writer(str(args.out), fps=args.fps, codec="libx264",
                                quality=9, macro_block_size=None,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])

    def _clip(t: float) -> None:
        """Reveal everything left of the playhead, and nothing outside the axes.

        Two matplotlib traps here, both of which silently produce a wrong
        figure rather than an error:

        * ``set_clip_path`` REPLACES an artist's default axes clipping, so
          without a clip box the off-window bars paint over the whole figure.
        * ``set_clip_path`` given a ``Rectangle`` does not set a clip *path* at
          all -- it converts it to a clip *box*, which the ``set_clip_box``
          call below would then overwrite, silently disabling the reveal. A
          ``Polygon`` takes the real clip-path route, so the two compose.
        """
        for art, ax in ((solid, ax_g), (cells_solid, ax_c)):
            art.set_clip_path(Polygon([(-1e6, -1e3), (t, -1e3),
                                       (t, 1e3), (-1e6, 1e3)],
                                      transform=ax.transData))
            art.set_clip_box(ax.bbox)

    def render() -> np.ndarray:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    try:
        # ---- title card: the same figure, playhead parked at 0 ----
        left0 = -0.06 * args.window_ms
        ax_g.set_xlim(left0, left0 + args.window_ms)
        _clip(0.0)
        banner.set_text("t = 0 ms")
        banner.set_color(HOLD_COLOUR)
        if pipelined:
            status.set_text(
                f"env step     0 / {n_steps}\n"
                f"\n"
                f"the board starts an inference every\n"
                f"{issue_period:.0f} ms and keeps up to {mst['max_inst']} of them in\n"
                f"flight at once (MEASURED). the first\n"
                f"result the env can use lands at step\n"
                f"{D}, {D * period:.0f} ms from now. until then: HOLD.")
        else:
            status.set_text(
                f"env step     0 / {n_steps}\n"
                f"\n"
                f"nothing has landed yet. the first\n"
                f"inference is issued now and lands\n"
                f"at step {D}, {D * period:.0f} ms from now.\n"
                f"until then the robot HOLDS.")
        card = render()
        for _ in range(int(args.title_card_s * args.fps)):
            writer.append_data(card)

        # ---- the animation ----
        for t in times:
            step = min(int(t / period), n_steps - 1)
            item = per_step[step]

            left = max(left0, t - 0.62 * args.window_ms)
            ax_g.set_xlim(left, left + args.window_ms)
            _clip(t)
            playhead_g.set_xdata([t, t])
            playhead_c.set_xdata([t, t])
            im.set_data(frames[step])

            if pipelined:
                phase, entries, iss = item
                n_lanes = at(lane_edges, lane_counts, t)
                n_inf = at(inf_edges, inf_counts, t)
                live = [x for x in issues if x[1] <= t < x[1] + x[5]]
                names = [f"octo{x[4]}" for x in live[-4:]]
                if len(live) > 4:
                    names.append(f"+{len(live) - 4}")
                who = ",".join(names) or "—"
                inflight = (f"in flight  {n_inf} on the board ({who}), "
                            f"{n_lanes} of 3 lanes busy")
                if phase == "hold":
                    banner.set_text("ROBOT HOLDING")
                    banner.set_color(HOLD_COLOUR)
                    applied = ("applying   HOLD — zero end-effector delta, gripper\n"
                               "           open. The arm does not move.")
                else:
                    banner.set_text("EXECUTING MEAN")
                    banner.set_color(EXEC_COLOUR)
                    parts = "\n".join(
                        f"           e{e} ← frame s{i}  "
                        f"({(step - i) * period:.0f} ms old)"
                        for e, i in zip(entries[:3], iss[:3]))
                    applied = (f"applying   mean of {len(entries)} prediction(s), "
                               f"sensor → actuation:\n{parts}")
            else:
                phase, entry, issue = item
                live = [c for c in cycles if c[0] * period <= t < c[1] * period]
                if live:
                    isx = live[-1][0]
                    el = t - isx * period
                    if el < span:
                        state = f"COMPUTING  ({el:5.0f} / {span:.0f} ms this trace)"
                    elif el < args.latency_ms:
                        state = f"this trace done at {span:.0f} ms; median {args.latency_ms:.0f}"
                    else:
                        state = "done — held for the 200 ms control tick"
                    inflight = (f"in flight    issued at step {isx}, "
                                f"{el:6.1f} ms ago\n"
                                f"             {state}")
                else:
                    inflight = "in flight    — (no request outstanding)\n "

                if phase == "hold":
                    banner.set_text("ROBOT HOLDING")
                    banner.set_color(HOLD_COLOUR)
                    applied = ("applying     HOLD — zero end-effector delta,\n"
                               "             gripper open. The arm does not move.")
                else:
                    banner.set_text("EXECUTING CHUNK")
                    banner.set_color(EXEC_COLOUR)
                    applied = (f"applying     chunk entry e{entry}"
                               f"{'  (ZOH, chunk exhausted)' if phase == 'zoh' else ''}\n"
                               f"             from the inference issued at step {issue}\n"
                               f"             — that observation is "
                               f"{(step - issue) * period:.0f} ms old")

            status.set_text(
                f"t = {t:8.1f} ms    env step {step:3d} / {n_steps}\n"
                f"{inflight}\n{applied}")
            writer.append_data(render())

        # ---- hold the last frame ----
        last = render()
        for _ in range(int(1.6 * args.fps)):
            writer.append_data(last)
    finally:
        writer.close()
        plt.close(fig)

    print(f"[ok] {args.out}")
    print(f"     schedule={args.schedule}  D={D}  control_period={period:g} ms  "
          f"issue_period={issue_period:g} ms")
    print(f"     MEASURED trace {args.trace.name}: {len(dispatches)} dispatches, "
          f"{mst['n_inst']} instance(s), walk {span:.2f} ms")
    print(f"     MEASURED lane overlap {overlap:.2f} ms ({mst['lane_overlap_pct']:.1f}%), "
          f"max concurrent lanes {mst['max_lanes']}, "
          f"max instances in flight {mst['max_inst']}")
    print(f"     MEASURED per-instance latency: median {mst['lat_median']:.2f} ms  "
          f"[{min(mst['lat']):.1f} .. {max(mst['lat']):.1f}]")
    if mst["gaps"]:
        print(f"     MEASURED completion cadence: median {mst['gap_median']:.2f} ms "
              f"[{mst['gap_min']:.2f} .. {mst['gap_max']:.2f}] over "
              f"{len(mst['gaps'])} gaps -> "
              + ", ".join(f"{g:.2f}" for g in mst["gaps"]))
    print(f"     lanes: " + " | ".join(
        f"{l} {busy[l]:.1f} ms ({100*busy[l]/span:.1f}%)" for l in LANES))
    if ctl:
        print(f"     MEASURED serial control {ctl['name']}: {ctl['n']} dispatches, "
              f"overlap {ctl['ov']:.2f} ms ({ctl['pct']:.1f}%), "
              f"max concurrent lanes {ctl['lanes']}, span {ctl['span']:.2f} ms")
    print(f"     DRAWN composite: overlap {drawn_overlap:.1f} ms "
          f"({100*drawn_overlap/drawn_span:.1f}% of {drawn_span:.0f} ms), "
          f"max concurrent lanes {drawn_max_lanes} (distinct lanes busy)")
    print(f"     {len(times)} animated frames @ {args.fps:g} fps, "
          f"env horizon {total_ms:.0f} ms over {n_steps} steps")


if __name__ == "__main__":
    main()

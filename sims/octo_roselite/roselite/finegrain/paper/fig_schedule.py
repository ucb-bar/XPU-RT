#!/usr/bin/env python3
"""Hardware-lane occupancy: what the three schedules actually do with the silicon.

One MEASURED figure, three panels, ONE SHARED WINDOW whose width is the CPU-only
baseline's own inference time. The question the panel answers is therefore always
the same -- "in the time the CPU-only baseline needs for one inference, how many
actions did this schedule deliver, and how much of the SoC was moving?" -- and the
answer is read off the same x-axis three times.

  1  CPU-only int8 monolith   one lane for ~677 ms; the DSP and the HTA are idle
                              for 100% of the window. Nothing to schedule.
  2  3-way CPU+DSP+HTA        all three lanes are used and NONE of them overlap.
                              Every dispatch waits for its predecessor (the grey
                              staircase is the k -> k+1 dependency chain, drawn
                              from the trace's own ordering). The speedup here is
                              backend SPECIALISATION, not concurrency.
  3  CP-SAT p150/w300         the same per-inference work, packed so that a lane's
                              idle is filled by a DIFFERENT inference. Up to three
                              are in flight. That is where the throughput comes from.

Everything is MEASURED: every bar is an `actual_start_ms` / `actual_end_ms` pair
from the board's own AGENTS_QNN_TRACE block, so the gaps are real idle rather than
the difference between summed durations.

Two honesty notes that are drawn, not just claimed:

  TILING.  The 3-way trace contains ONE inference (71 dispatches, 1 instance),
  because a serial schedule cannot start the next one until this one ends -- which
  is the very fact the panel exists to show. The window is therefore filled by
  repeating the measured template at its own measured period. The first repeat is
  drawn solid and is real; the tiled repeats are hatched and translucent and are a
  reconstruction. They are exact up to run-to-run noise (~15%, section 5.4 of
  OCTO_INT8_QRB5165.md), and the 4 measured replicates spanning 239-270 ms are
  printed so the reader can size that noise.

  THE GATE.  `runtime_main.cpp` busy-waits each dispatch to its SCHEDULED start
  unless XPURT_NO_GATE=1, which would make "measured tracks predicted" self-
  fulfilling. This is checked per trace from the gate_done_ms / dep_wait_done_ms
  columns and the count of gate-held dispatches is printed on the figure.

Trace column quirk: `kind` holds the MACHINE id (CPU_X/CPU_E/CPU_P) and
`backend_label` holds the SEGMENT name; `actual_backend` is the resolved lane.
"""
from __future__ import annotations
import csv, io, os, statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "fig_schedule.png"
RUNS = Path(os.environ.get(
    "OCTO_REPRO_RUNS", "/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs"))

LANES = ["CPU", "DSP", "HTA"]
LANE_MACHINE = {"CPU": "CPU_X  (Cortex-X1)", "DSP": "CPU_E  (Hexagon DSP)",
                "HTA": "CPU_P  (Hexagon HTA)"}
FAM = {"ideal": "#7f8c8d", "pipelined": "#16a085",
       "serial": "#e67e22", "CPU only": "#c0392b"}
# Shade cycle for the pipelined panel. At most 3 inferences are ever in flight
# (asserted below from the trace), so a 4-cycle never puts two concurrent
# inferences in the same shade -- and the panel keeps its family colour.
PIPE_SHADES = ["#16a085", "#0b5345", "#5ecfba", "#117a65"]

GROUPS = {  # label -> glob of interchangeable replicates of the SAME config
    "mono":     "mono_2*.log",          # mono_gated_* excluded: gate held 16/21
    "serial":   "ungated_*.log",
    "p150w300": "p150w300*.log",
}


# ----------------------------------------------------------------- trace I/O
def parse(path: Path) -> list[dict]:
    body = path.read_text(errors="replace")
    if "AGENTS_QNN_TRACE_BEGIN" not in body:
        raise SystemExit(f"{path}: no AGENTS_QNN_TRACE block")
    body = body.split("AGENTS_QNN_TRACE_BEGIN", 1)[1].split("AGENTS_QNN_TRACE_END", 1)[0]
    raw = body.splitlines()
    i = next((j for j, l in enumerate(raw) if l.startswith("seg_id,")), None)
    if i is None:
        raise SystemExit(f"{path}: no seg_id header")
    rows = list(csv.DictReader(io.StringIO("\n".join(
        l for l in raw[i:] if l.strip() and not l.lstrip().startswith("===")))))
    if not rows:
        raise SystemExit(f"{path}: empty trace")
    t0 = min(float(r["actual_start_ms"]) for r in rows)   # re-zero: a trace's
    for r in rows:                                        # own first dispatch
        r["_s"] = float(r["actual_start_ms"]) - t0
        r["_e"] = float(r["actual_end_ms"]) - t0
        r["_lane"] = (r.get("actual_backend") or "").strip()
        r["_inst"] = int(r.get("instance", 0) or 0)
        r["_gate"] = float(r["gate_done_ms"]) - float(r["dep_wait_done_ms"])
    return rows


def instances(rows):
    """{instance: (first start, last end)} -- an inference's real wall span."""
    out = {}
    for r in rows:
        s, e = out.get(r["_inst"], (r["_s"], r["_e"]))
        out[r["_inst"]] = (min(s, r["_s"]), max(e, r["_e"]))
    return dict(sorted(out.items()))


def span_of(rows):
    iv = instances(rows)
    return st.median([e - s for s, e in iv.values()])


def cadence_of(rows):
    st_ = sorted(s for s, _ in instances(rows).values())
    if len(st_) < 2:
        return float("nan")
    return st.median([st_[i + 1] - st_[i] for i in range(len(st_) - 1)])


def pick(group: str):
    """The replicate whose per-inference span is closest to the group median.

    Never "the newest file", which is what the draft did and which silently
    re-draws the figure when a run is added.
    """
    files = sorted(RUNS.glob(GROUPS[group]))
    if not files:
        raise SystemExit(f"no traces matching {GROUPS[group]} in {RUNS}")
    traces = [(f, parse(f)) for f in files]
    spans = [span_of(t) for _, t in traces]
    med = st.median(spans)
    k = int(np.argmin([abs(s - med) for s in spans]))
    return traces[k][0], traces[k][1], spans, med


def lane_busy(rows, lo=None, hi=None):
    """Busy ms per lane, clipped to [lo, hi] so a window measures the window."""
    b = {k: 0.0 for k in LANES}
    for r in rows:
        if r["_lane"] not in b:
            continue
        s, e = r["_s"], r["_e"]
        if lo is not None:
            s, e = max(s, lo), max(min(e, hi), lo)
        b[r["_lane"]] += max(0.0, e - s)
    return b


def max_inflight(rows):
    ev = sorted([(s, 1) for s, _ in instances(rows).values()] +
                [(e, -1) for _, e in instances(rows).values()])
    c = m = 0
    for _, d in ev:
        c += d
        m = max(m, c)
    return m


# ------------------------------------------------------------------ drawing
def lane_frame(ax, W):
    for row in range(len(LANES)):
        ax.add_patch(Rectangle((0, row - 0.40), W, 0.80, facecolor="#eef0f1",
                               edgecolor="none", zorder=1))
    ax.set_yticks(range(len(LANES)))
    ax.set_yticklabels([f"{l}\n{LANE_MACHINE[l]}" for l in LANES], fontsize=7.6)
    ax.set_ylim(-1.15, len(LANES) - 0.30)
    ax.set_xlim(-W * 0.012, W * 1.20)
    ax.grid(True, axis="x", alpha=0.28, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bars(ax, rows, colour_of, W, off=0.0, alpha=1.0, hatch=None, edge=0.0):
    for r in rows:
        if r["_lane"] not in LANES:
            continue
        s, e = r["_s"] - off, r["_e"] - off
        if e <= 0 or s >= W:
            continue
        s, e = max(s, 0.0), min(e, W)
        ax.add_patch(Rectangle((s, LANES.index(r["_lane"]) - 0.34), max(e - s, 0.30),
                               0.68, facecolor=colour_of(r), alpha=alpha,
                               hatch=hatch, edgecolor="black" if edge else "none",
                               linewidth=edge, zorder=3))


def chain(ax, rows, W, off=0.0):
    """The k -> k+1 dependency staircase, from the trace's own ordering."""
    seq = sorted((r for r in rows if r["_lane"] in LANES), key=lambda r: r["_s"])
    for a, b in zip(seq, seq[1:]):
        x0, x1 = a["_e"] - off, b["_s"] - off
        if x1 <= 0 or x0 >= W:
            continue
        ax.plot([x0, x1], [LANES.index(a["_lane"]), LANES.index(b["_lane"])],
                color="#6d7b7b", lw=0.6, alpha=0.75, zorder=2, solid_capstyle="butt")


def deliveries(ax, times, W, colour):
    """A marker per inference COMPLETION -- the instant a fresh action exists."""
    for t in times:
        if 0 <= t <= W:
            ax.plot([t], [len(LANES) - 0.34], marker="v", ms=6.5, color=colour,
                    mec="white", mew=0.8, clip_on=False, zorder=8)


def annotate_lanes(ax, busy, W, denom=None, extra=None):
    denom = denom or W
    for row, lane in enumerate(LANES):
        b = busy[lane]
        pct = 100.0 * b / denom
        col = "#c0392b" if b == 0 else ("#7d3c98" if pct < 30 else "#2c3e50")
        ax.text(W * 1.025, row + 0.10, f"{b:6.1f} ms busy", va="center", ha="left",
                fontsize=7.8, family="monospace", color=col)
        ax.text(W * 1.025, row - 0.20,
                f"{pct:5.1f}% busy" if b else "      IDLE  100%",
                va="center", ha="left", fontsize=7.4, family="monospace", color=col)
    if extra:
        ax.text(W * 1.025, -0.47, extra, va="top", ha="left", fontsize=7.2,
                family="monospace", color="#555")


# --------------------------------------------------------------------- data
f_mono, mono, mono_sp, _ = pick("mono")
f_ser, ser, ser_sp, ser_med = pick("serial")
f_pipe, pipe, pipe_sp, _ = pick("p150w300")

W = span_of(mono)                      # the window: one CPU-only inference
ser_period = span_of(ser)              # serial: 1 in flight, so period == span
n_tiles = int(np.ceil(W / ser_period))
pipe_cad = cadence_of(pipe)
pipe_inflight = max_inflight(pipe)
assert pipe_inflight <= len(PIPE_SHADES), "shade cycle too short for the concurrency"

# Pipelined steady-state window: start when the pipeline is already full.
pipe_iv = instances(pipe)
t0 = sorted(s for s, _ in pipe_iv.values())[pipe_inflight - 1]
if t0 + W > max(e for _, e in pipe_iv.values()):
    t0 = max(0.0, max(e for _, e in pipe_iv.values()) - W)

gate_held = {n: sum(1 for r in t if r["_gate"] > 0.5)
             for n, t in (("mono", mono), ("serial", ser), ("pipe", pipe))}

# Inferences delivered inside the window, counted the same way in all three.
del_mono = [e for _, e in instances(mono).values() if e <= W + 1e-6]
del_ser = [ser_period * (k + 1) for k in range(n_tiles) if ser_period * (k + 1) <= W + 1e-6]
del_pipe = [e - t0 for _, e in pipe_iv.values() if t0 <= e <= t0 + W]

# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.4, 9.4), dpi=150)
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 1.0, 1.0, 0.60], hspace=0.52,
                      left=0.085, right=0.995, top=0.895, bottom=0.175)
ax0, ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(4))

# --- panel 1: CPU-only monolith --------------------------------------------
lane_frame(ax0, W)
bars(ax0, mono, lambda r: FAM["CPU only"], W)
annotate_lanes(ax0, lane_busy(mono, 0, W), W,
               extra=f"1 action / {W:.0f} ms   (1.0x)")
deliveries(ax0, del_mono, W, FAM["CPU only"])
ax0.set_title(f"1 · CPU-ONLY int8 monolith — {len(del_mono)} inference in the window."
              "  Two accelerators sit idle for the whole of it; there is nothing to schedule.",
              fontsize=10.6, color=FAM["CPU only"], fontweight="bold", loc="left", pad=6)

# --- panel 2: 3-way serial, tiled ------------------------------------------
lane_frame(ax1, W)
ser_busy = {k: 0.0 for k in LANES}
for k in range(n_tiles):
    off = -ser_period * k
    first = (k == 0)
    bars(ax1, ser, lambda r: FAM["serial"], W, off=off,
         alpha=1.0 if first else 0.40, hatch=None if first else "///")
    if first:
        chain(ax1, ser, W)
    for lane, v in lane_busy(ser, 0, min(ser_period, W + off * -1)).items():
        ser_busy[lane] += v if first else v * min(1.0, (W - ser_period * k) / ser_period)
annotate_lanes(ax1, lane_busy(ser, 0, ser_period), W, denom=ser_period,
               extra=f"% is per {ser_period:.0f} ms inference\n"
                     f"1 action / {ser_period:.0f} ms   ({W / ser_period:.1f}x)")
deliveries(ax1, del_ser, W, FAM["serial"])
for k in range(1, n_tiles):
    ax1.axvline(ser_period * k, color="#8d6e3a", lw=0.9, ls=(0, (4, 3)), zorder=6)
idle = {l: ser_period - lane_busy(ser, 0, ser_period)[l] for l in LANES}
ax1.set_title(
    f"2 · 3-WAY CPU+DSP+HTA, ONE inference in flight — {len(del_ser)} inferences in the window. "
    f"All three lanes are used and none of them OVERLAP:\n     every lane is idle "
    f"{min(idle.values()):.0f}-{max(idle.values()):.0f} ms of each {ser_period:.0f} ms inference. "
    f"The grey staircase is the k → k+1 dependency chain — the makespan is its sum.",
    fontsize=10.6, color=FAM["serial"], fontweight="bold", loc="left", pad=6)

# --- panel 3: pipelined CP-SAT ---------------------------------------------
lane_frame(ax2, W)
bars(ax2, pipe, lambda r: PIPE_SHADES[r["_inst"] % len(PIPE_SHADES)], W, off=t0)
annotate_lanes(ax2, lane_busy(pipe, t0, t0 + W), W,
               extra=f"1 action / {pipe_cad:.0f} ms   ({W / pipe_cad:.1f}x)")
deliveries(ax2, del_pipe, W, FAM["pipelined"])
ax2.set_title(
    f"3 · PIPELINED CP-SAT p150/w300, up to {pipe_inflight} in flight — {len(del_pipe)} inferences "
    f"in the window, one every {pipe_cad:.0f} ms MEASURED.\n     The SAME per-inference work as "
    f"panel 2; the idle is now filled by a DIFFERENT inference. Shade cycles with inference index.",
    fontsize=10.6, color=FAM["pipelined"], fontweight="bold", loc="left", pad=6)

_pb = lane_busy(pipe, t0, t0 + W)
ax2.text(W * 0.5, -0.96,
         f"HTA is loaded to {100 * _pb['HTA'] / W:.0f}% while CPU and DSP run at "
         f"{100 * _pb['CPU'] / W:.0f}% and {100 * _pb['DSP'] / W:.0f}% — that gap is the "
         f"headroom a better placement could still take (fig_lane_balance).",
         ha="center", va="center", fontsize=8.2, color="#7d3c98", style="italic")

# --- panel 4: the pipeline ladder ------------------------------------------
shown = [(i, s, e) for i, (s, e) in pipe_iv.items() if e > t0 and s < t0 + W]
for k, (i, s, e) in enumerate(shown):
    lo, hi = max(s - t0, 0.0), min(e - t0, W)
    ax3.add_patch(Rectangle((lo, k - 0.33), hi - lo, 0.66,
                            facecolor=PIPE_SHADES[i % len(PIPE_SHADES)],
                            edgecolor="none", zorder=3))
    ax3.text(hi + W * 0.006, k, f"{e - s:.0f} ms", va="center", fontsize=6.8,
             family="monospace", color="#555")
ax3.set_ylim(-0.8, len(shown) - 0.2)
ax3.set_xlim(-W * 0.012, W * 1.20)
ax3.set_yticks(range(len(shown)))
ax3.set_yticklabels([f"inf {i}" for i, _, _ in shown], fontsize=7.0)
ax3.grid(True, axis="x", alpha=0.28, zorder=0)
for s_ in ("top", "right"):
    ax3.spines[s_].set_visible(False)
ax3.set_title(f"     the same window as panel 3, one row per inference — "
              f"MEASURED release cadence {pipe_cad:.0f} ms, span {span_of(pipe):.0f} ms, "
              f"so {pipe_inflight} overlap",
              fontsize=8.8, color="#555", loc="left", pad=4)
ax3.set_xlabel(f"time within one CPU-only inference ({W:.0f} ms) — MEASURED per-dispatch trace, "
               f"▼ = an inference completes and a fresh action becomes available", fontsize=9.5)

TICKS = list(range(0, int(W) + 1, 100))
for ax in (ax0, ax1, ax2, ax3):
    ax.set_xticks(TICKS)
for ax in (ax0, ax1, ax2):
    ax.set_xticklabels([])

# ------------------------------------------------------------------- legend
h = [Patch(facecolor=FAM["CPU only"], label="CPU-only monolith"),
     Patch(facecolor=FAM["serial"], label="3-way serial, MEASURED template"),
     Patch(facecolor=FAM["serial"], alpha=0.40, hatch="///",
           label="3-way serial, TILED (translucent) at that period"),
     Patch(facecolor=PIPE_SHADES[0], label="pipelined (shade = inference index)"),
     Line2D([], [], color="#6d7b7b", lw=1.0, label="k → k+1 dependency"),
     Line2D([], [], ls="", marker="v", ms=7, color="#444", label="inference completes")]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.093), ncol=6,
           fontsize=8.4, framealpha=0.95, handlelength=1.7, columnspacing=1.5)

fig.suptitle("The accelerated 3-way split uses all three lanes and still overlaps NONE of them. "
             "Pipelining is what converts specialisation into throughput.",
             fontsize=13.2, y=0.975)

held = sum(gate_held.values())
CAPTION = "\n".join([
    "Every bar is a MEASURED per-dispatch interval (actual_start_ms / actual_end_ms) from the board's own AGENTS_QNN_TRACE, so the gaps are real idle, not the difference of summed durations.",
    f"THE GATE: the runtime busy-waits each dispatch to its scheduled start unless XPURT_NO_GATE=1, which would make timing self-fulfilling. Checked per trace here — {held} of {len(mono) + len(ser) + len(pipe)} dispatches were gate-held.",
    f"TILING: panel 2 holds ONE measured inference, because a serial schedule cannot start the next until this one ends. The hatched repeats are that template tiled at its own {ser_period:.0f} ms period — exact up to the ~15% between-process noise",
    f"floor (n={len(ser_sp)} replicates of this config span {min(ser_sp):.0f}-{max(ser_sp):.0f} ms, median {ser_med:.0f} ms). Panel 3 is a steady-state window from a {len(pipe)}-dispatch, {len(pipe_iv)}-inference trace (n={len(pipe_sp)} replicates, median cadence {pipe_cad:.0f} ms).",
    "In each group the replicate drawn is the one whose per-inference span is CLOSEST TO THAT GROUP'S MEDIAN — not the newest file, which is what silently re-draws a figure when a run is added.",
])
fig.text(0.5, 0.010, CAPTION, ha="center", va="bottom", fontsize=7.8,
         style="italic", color="#555", linespacing=1.45)

fig.savefig(OUT)
print(f"[ok] {OUT}")
print(f"  window W = {W:.1f} ms (CPU-only inference span, {f_mono.name})")
print(f"  serial   {f_ser.name}: period {ser_period:.1f} ms, replicates {[round(x,1) for x in ser_sp]}")
print(f"  pipelined {f_pipe.name}: cadence {pipe_cad:.1f} ms, span {span_of(pipe):.1f} ms, "
      f"replicates {[round(x, 1) for x in pipe_sp]}")
print(f"  inferences in window: mono {len(del_mono)}, serial {len(del_ser)} (tiled), "
      f"pipelined {len(del_pipe)}")
for name, t, lo, hi in (("mono", mono, 0, W), ("serial", ser, 0, ser_period),
                        ("pipe", pipe, t0, t0 + W)):
    b = lane_busy(t, lo, hi)
    print(f"  {name:7s} lane busy: " + "  ".join(f"{k} {v:6.1f}" for k, v in b.items())
          + f"   gate-held {gate_held[name if name != 'pipe' else 'pipe']}")

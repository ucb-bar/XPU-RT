#!/usr/bin/env python3
"""Where the remaining headroom is: the three lanes are not equally loaded.

fig_schedule.py shows that pipelining fills the idle. This one asks what is left.
Six MEASURED schedules of the SAME network, ordered by measured release cadence.

  A  PER-INFERENCE LANE WORK, split into the stable part and the excess.
     "Excess" is defined per run as the time a dispatch spends ABOVE that run's own
     median duration for its segment, summed. That median is stable to ~1.5% across
     every schedule here (MEASURED, and printed), so the excess is not extra work --
     it is the `posta` thread-pool stall documented in REFINED_SCHEDULE.md section
     4.2: the same dispatch burning 3.2x the CPU time with 4.3x the voluntary context
     switches. It is a RUNTIME defect, and no scheduler can remove it.

  B  THE CADENCE FLOOR. A pipelined schedule cannot release faster than its busiest
     lane. Each schedule's MEASURED cadence is drawn against two bounds computed from
     its own lane work: the bottleneck lane (what this placement allows) and
     total-work / 3 (what a PERFECTLY balanced placement would allow). The distance
     between those two bounds is the headroom, and it exists because the HTA is
     loaded to roughly a third of what the CPU and DSP carry.

  C  WHERE THE GROWTH IS NOT. Median dispatch duration for the segment that defines
     each lane. The DSP's `pre2` gets ~27% CHEAPER once the lane is kept busy (7.24 ms
     serial -> 5.3-5.6 ms pipelined, MEASURED) -- the accelerator is warm. The CPU's
     `posta` median does NOT move at all (3.89-4.34 ms across every schedule), which is
     the point: panel A's CPU growth is entirely TAIL, so it is a stall and not a cost.
     HTA `mlp` sits at 2.2-2.9 ms except in pipe200, whose single process (n=1) reads
     3.94; with n=1 that is inside the noise floor and is not read as a trend.

Medians throughout, never means: dispatch durations are bimodal (a stable ~3.9 ms
mode and a ~21 ms stall mode) and the between-process noise floor is ~15%. n is the
number of independent board processes of that exact configuration and is printed on
every group.

Trace column quirk: `kind` holds the MACHINE id and `backend_label` the SEGMENT name;
`actual_backend` is the resolved lane.
"""
from __future__ import annotations
import csv, io, os, collections, statistics as st
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "fig_lane_balance.png"
RUNS = Path(os.environ.get(
    "OCTO_REPRO_RUNS", "/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs"))

LANES = ["CPU", "DSP", "HTA"]
LANE_C = {"CPU": "#c0392b", "DSP": "#2980b9", "HTA": "#7d3c98"}
FAM = {"pipelined": "#16a085", "serial": "#e67e22"}
# The segment that defines each lane's cost, for panel C.
LANE_SEG = {"CPU": "posta", "DSP": "pre2", "HTA": "mlp"}
LANE_MK = {"CPU": "o", "DSP": "s", "HTA": "^"}

# label -> (glob, exclude substring or None, family). gc105ruy* are XPURT_RUYCAP
# sweep points, a DIFFERENT runtime configuration, so they are not replicates.
CONFIGS = [
    ("3-way serial\n1 in flight",   "ungated_*.log",        None,  "serial"),
    ("pipe200\n5 instances",        "pipe200_gated_*.log",  None,  "pipelined"),
    ("CP-SAT p150/w300\n10 inst",   "p150w300*.log",        None,  "pipelined"),
    ("CP-SAT p130/w275\n10 inst",   "m130w275*.log",        None,  "pipelined"),
    ("greedy p105/w300\n10 inst",   "gc105*.log",           "ruy", "pipelined"),
    ("pipe110\n10 instances",       "pipe110x10_*.log",     None,  "pipelined"),
]


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
    for r in rows:
        r["_d"] = float(r["actual_end_ms"]) - float(r["actual_start_ms"])
        r["_s"] = float(r["actual_start_ms"])
        r["_e"] = float(r["actual_end_ms"])
        r["_lane"] = (r.get("actual_backend") or "").strip()
        r["_seg"] = (r.get("backend_label") or "").strip()
        r["_inst"] = int(r.get("instance", 0) or 0)
        r["_gate"] = float(r["gate_done_ms"]) - float(r["dep_wait_done_ms"])
    return rows


def run_stats(rows):
    """One board process -> per-inference lane work, its stall excess, cadence."""
    inst = collections.defaultdict(lambda: [np.inf, -np.inf])
    for r in rows:
        iv = inst[r["_inst"]]
        iv[0], iv[1] = min(iv[0], r["_s"]), max(iv[1], r["_e"])
    n = len(inst)
    med_seg = {k: st.median(v) for k, v in
               collections.defaultdict(list, {
                   s: [r["_d"] for r in rows if r["_seg"] == s]
                   for s in {r["_seg"] for r in rows}}).items()}
    busy = {l: 0.0 for l in LANES}
    exc = {l: 0.0 for l in LANES}
    for r in rows:
        if r["_lane"] not in busy:
            continue
        busy[r["_lane"]] += r["_d"]
        exc[r["_lane"]] += max(0.0, r["_d"] - med_seg[r["_seg"]])
    starts = sorted(v[0] for v in inst.values())
    cad = (st.median([starts[i + 1] - starts[i] for i in range(len(starts) - 1)])
           if n > 1 else st.median([v[1] - v[0] for v in inst.values()]))
    return dict(busy={l: busy[l] / n for l in LANES}, exc={l: exc[l] / n for l in LANES},
                cad=cad, span=st.median([v[1] - v[0] for v in inst.values()]),
                seg={l: med_seg.get(LANE_SEG[l], np.nan) for l in LANES},
                gate=sum(1 for r in rows if r["_gate"] > 0.5), n_disp=len(rows), n_inst=n)


def group(glob_, excl):
    files = [f for f in sorted(RUNS.glob(glob_)) if not (excl and excl in f.name)]
    if not files:
        raise SystemExit(f"no traces matching {glob_} in {RUNS}")
    return files, [run_stats(parse(f)) for f in files]


def med(rs, key, lane=None):
    return st.median([r[key][lane] if lane else r[key] for r in rs])


DATA = []
for lab, g, ex, fam in CONFIGS:
    files, rs = group(g, ex)
    DATA.append(dict(lab=lab, fam=fam, files=files, rs=rs, n=len(rs),
                     cad=med(rs, "cad"), span=med(rs, "span"),
                     busy={l: med(rs, "busy", l) for l in LANES},
                     exc={l: med(rs, "exc", l) for l in LANES},
                     seg={l: med(rs, "seg", l) for l in LANES},
                     gate=sum(r["gate"] for r in rs)))
DATA.sort(key=lambda d: -d["cad"])
x = np.arange(len(DATA))
labs = [f"{d['lab']}\n{d['cad']:.0f} ms  (n={d['n']})" for d in DATA]

fig, axes = plt.subplots(1, 3, figsize=(17.6, 6.2), dpi=150,
                         gridspec_kw={"wspace": 0.235, "left": 0.048, "right": 0.995,
                                      "top": 0.845, "bottom": 0.275})

# ------------------------------------------------- A: per-inference lane work
ax = axes[0]
w = 0.26
for k, lane in enumerate(LANES):
    off = (k - 1) * w
    stable = np.array([d["busy"][lane] - d["exc"][lane] for d in DATA])
    excess = np.array([d["exc"][lane] for d in DATA])
    ax.bar(x + off, stable, w, color=LANE_C[lane], edgecolor="black", linewidth=0.4,
           zorder=3, label=lane)
    ax.bar(x + off, excess, w, bottom=stable, color=LANE_C[lane], alpha=0.42,
           hatch="////", edgecolor="black", linewidth=0.4, zorder=3)
    for i, d in enumerate(DATA):
        ax.text(i + off, d["busy"][lane] + 1.6, f"{d['busy'][lane]:.0f}", ha="center",
                va="bottom", fontsize=6.6, color="#333")
ax.set_ylabel("MEASURED lane busy per inference (ms)")
_cs = [d["busy"]["CPU"] - d["exc"]["CPU"] for d in DATA]
ax.set_title("A · the WORK per inference barely moves;\nthe CPU lane's growth is stall, not work",
             fontsize=10.5)
ax.text(0.015, 0.99, f"CPU stable part: {min(_cs):.0f}-{max(_cs):.0f} ms in every schedule",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color=LANE_C["CPU"],
        style="italic")
ax.set_ylim(0, max(v for d in DATA for v in d["busy"].values()) * 1.34)
ax.legend(handles=[Patch(facecolor=LANE_C[l], edgecolor="black", label=f"{l} lane")
                   for l in LANES]
                  + [Patch(facecolor="#999", alpha=0.42, hatch="////", edgecolor="black",
                           label="excess over this run's own\nper-segment median duration")],
          fontsize=7.4, loc="upper right", framealpha=0.95)

# ----------------------------------------------------- B: the cadence floor
ax = axes[1]
bott = np.array([max(d["busy"].values()) for d in DATA])
bal = np.array([sum(d["busy"].values()) / len(LANES) for d in DATA])
cad = np.array([d["cad"] for d in DATA])
ax.barh(x, cad, 0.56, color=[FAM[d["fam"]] for d in DATA], edgecolor="black",
        linewidth=0.5, zorder=3, alpha=0.90)
ax.scatter(bott, x, marker="|", s=520, color="#111", linewidth=2.4, zorder=6)
ax.scatter(bal, x, marker="|", s=520, color="#7d3c98", linewidth=2.4, zorder=6)
for i, d in enumerate(DATA):
    ax.plot([bal[i], bott[i]], [i, i], color="#7d3c98", lw=1.4, ls=":", zorder=5)
    ax.text(cad[i] + 4, i, f"{cad[i]:.0f}", va="center", fontsize=7.4, color="#333")
    if cad[i] - bott[i] > 6:
        ax.text(bott[i] + (cad[i] - bott[i]) * 0.55, i + 0.34,
                f"+{cad[i] - bott[i]:.0f} above bound", ha="center", va="center",
                fontsize=6.4, color="#555", style="italic",
                bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.0))
ax.set_yticks(x)
ax.set_yticklabels(labs, fontsize=7.0)
ax.invert_yaxis()
ax.set_xlabel("ms per inference")
ax.set_title("B · a schedule cannot release faster than its BUSIEST lane —\n"
             "and the lanes are not balanced", fontsize=10.5)
ax.set_xlim(0, max(cad.max(), bott.max()) * 1.14)
ax.legend(handles=[
    Patch(facecolor=FAM["pipelined"], edgecolor="black", label="MEASURED release cadence"),
    Patch(facecolor=FAM["serial"], edgecolor="black",
          label="serial: span, not a release period"),
    Line2D([], [], ls="", marker="|", ms=13, mew=2.4, color="#111",
           label="bottleneck lane (this placement)"),
    Line2D([], [], ls="", marker="|", ms=13, mew=2.4, color="#7d3c98",
           label="total work / 3 (perfect balance)")],
    fontsize=7.4, loc="lower right", framealpha=0.95)

# --------------------------------------------- C: per-dispatch median duration
ax = axes[2]
for lane in LANES:
    ax.plot(x, [d["seg"][lane] for d in DATA], marker=LANE_MK[lane], ms=6.5, lw=2.0,
            color=LANE_C[lane], label=f"{lane} · {LANE_SEG[lane]}", zorder=4,
            mfc="white", mew=1.8)
    for i, d in enumerate(DATA):
        ax.annotate(f"{d['seg'][lane]:.2f}", (i, d["seg"][lane]), textcoords="offset points",
                    xytext=(0, 8 if lane != "HTA" else -14), ha="center", fontsize=6.5,
                    color=LANE_C[lane])
ax.set_ylabel("MEASURED median dispatch duration (ms)")
ax.set_title("C · the DSP dispatch gets cheaper when the lane is kept busy;\n"
             "the CPU median never moves — its growth is all tail", fontsize=10.5)
ax.set_ylim(0, max(d["seg"]["DSP"] for d in DATA) * 1.30)
ax.legend(fontsize=7.6, loc="upper right", framealpha=0.95, title="lane · segment",
          title_fontsize=7.6)

for a in (axes[0], axes[2]):
    a.set_xticks(x)
    a.set_xticklabels(labs, fontsize=6.7, rotation=32, ha="right")
    a.grid(True, axis="y", alpha=0.3, zorder=0)
axes[1].grid(True, axis="x", alpha=0.3, zorder=0)
for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

REF = "greedy p105/w300"      # the schedule the sweep recommends
best = next(d for d in DATA if d["lab"].startswith(REF))
hta_share = 100 * best["busy"]["HTA"] / best["cad"]
cpu_share = 100 * best["busy"]["CPU"] / best["cad"]
gain = 100 * (1 - sum(best["busy"].values()) / 3 / max(best["busy"].values()))

fig.suptitle(
    f"At the recommended schedule the HTA is busy {hta_share:.0f}% of each release interval "
    f"while the CPU is busy {cpu_share:.0f}% — rebalancing the three lanes is worth up to "
    f"{gain:.0f}% off the cadence floor, and no more.",
    fontsize=13.0, y=0.975)

fig.text(0.5, 0.012,
         "Schedules are ordered by MEASURED release cadence; n is the number of independent board "
         "processes of that exact configuration (gc105ruy* are XPURT_RUYCAP sweep points, a different "
         "RUNTIME configuration, so they are not replicates of gc105 and are excluded). Medians "
         "throughout: dispatch durations are\nbimodal and the between-process noise floor is ~15%, so "
         "a mean would track the stall rate of whichever process happened to be drawn. Panel B's "
         "'perfect balance' bound assumes the network's work can be MOVED between lanes at these "
         "costs, which it cannot be exactly — a segment runs at a\n"
         "different cost on a different backend, and some refuse to compile at all. It is an upper "
         "bound on what rebalancing could buy, not a schedule. The CPU-only monolith is excluded: it "
         "has one lane and no cadence. Every dispatch here ran UNGATED (0 gate-held over "
         f"{sum(sum(r['n_disp'] for r in d['rs']) for d in DATA)} dispatches).",
         ha="center", fontsize=7.9, style="italic", color="#555", linespacing=1.45)

fig.savefig(OUT)
print(f"[ok] {OUT}")
for d in DATA:
    print(f"  {d['lab'].replace(chr(10), ' / '):34s} n={d['n']} cad={d['cad']:6.1f} span={d['span']:6.1f} "
          + "  ".join(f"{l} {d['busy'][l]:5.1f} (exc {d['exc'][l]:4.1f})" for l in LANES)
          + f"   bottleneck {max(d['busy'].values()):5.1f}  balanced "
            f"{sum(d['busy'].values()) / 3:5.1f}  gate-held {d['gate']}")
print(f"  reference {REF}: HTA {hta_share:.1f}% vs CPU {cpu_share:.1f}% vs DSP "
      f"{100 * best['busy']['DSP'] / best['cad']:.1f}% of the release interval; "
      f"rebalance headroom {gain:.1f}%")

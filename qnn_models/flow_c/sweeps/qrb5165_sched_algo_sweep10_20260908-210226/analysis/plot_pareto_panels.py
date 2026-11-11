#!/usr/bin/env python3
"""Two panels: Pareto optimality on the left, distributions on the right.

The single-panel version had to carry twelve floating labels on top of twelve
overlapping interquartile rectangles, and the labels and the shading fought
each other. Splitting the two questions fixes that structurally rather than by
tuning offsets:

  LEFT   which solvers are Pareto optimal in quality against solve time. Only
         the frontier is labelled -- four names instead of twelve -- because
         the right panel already names everything in reading order.
  RIGHT  what the spread actually is. Solver names live on a CATEGORICAL AXIS,
         so they are tick labels rather than annotations and cannot collide by
         construction. Rows are sorted by mean gain, so the reader maps left to
         right by colour and by rank.

Deadline misses ride along the right panel as a trailing column, which keeps
the "is it even usable" question attached to the solver it belongs to instead
of in a separate strip.

  plot_pareto_panels.py [--wide]
"""
import csv, math, os, statistics, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
# Point at the CORRECTED rerun data rather than the original campaign's
# results/. Same schema, regenerated from the emit .meta.json files, so the
# figure is directly comparable to the one it replaces.
R = os.environ.get("PARETO_DATA", HERE)
WIDE = "--wide" in sys.argv

FAMILY = {"greedy": "#2a78d6", "greedy_periodic": "#2a78d6",
          "greedy_reserved": "#2a78d6", "decomposed": "#2a78d6",
          "heft": "#eb6834", "heft_edf": "#eb6834", "best-of-fast": "#eb6834",
          "pso": "#1baf7a", "sa": "#1baf7a",
          "cpsat": "#eda100", "cpsat:warm": "#eda100", "cpsat:warmbest": "#eda100"}
FAMNAME = {"#2a78d6": "greedy family", "#eb6834": "list heuristic",
           "#1baf7a": "metaheuristic", "#eda100": "CP-SAT"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
FIG = (13.0, 4.8) if WIDE else (9.4, 4.2)
FT, FA, FK, FL, FG = (15, 13, 12, 12, 11) if WIDE else (12.5, 11, 10, 10, 9.5)

# Exclude cells that CANNOT rank a scheduler, read from the spec rather than
# hardcoded by family name. A taskset with no aperiodic work has no makespan to
# minimise: schedule_decoder.evaluate scores (num_instances-1)*period + critical
# path, a property of the SPEC, so every solver that finds a feasible schedule
# returns the identical objective. Averaging those cells into a ranking dilutes
# every solver toward zero difference.
#
# This was `family != "tight_loop"`, which caught 8 of the 16 such cells and
# silently averaged in the other 8 (depth_chain, all-periodic for the same
# reason). The specs now carry discriminates=false / role=calibration, so read
# that and stay correct as the workload set changes.
import json as _json
# PORT DIFF vs RoSE/experiments/xpurt_rerun/analysis/plot_pareto_panels.py.
# The reference globs the RoSE spec tree for `discriminates: false`. This
# sweep's ported specs do not carry that annotation, and the reference's spec
# tree describes the FireSim workloads, not these. `mk_qrb5165_csv.py` computes
# the same exclusion structurally instead -- a family every one of whose cells
# is all-periodic has no makespan to minimise -- and writes it next to the CSV.
_NONDISC = set(_json.load(open(f"{R}/nondiscriminating.json")))
rows = [r for r in csv.DictReader(open(f"{R}/results.csv")) if r["family"] not in _NONDISC]
print("excluded non-discriminating families:", sorted(_NONDISC))
base = {}
for r in rows:
    if r["solver"] == "greedy":
        try: base[(r["arm"], r["workload"])] = float(r["objective"])
        except (TypeError, ValueError): pass
imp, wall, mp = (collections.defaultdict(list) for _ in range(3))
CELLS = set()      # workload-arms that actually contribute a comparison
for r in rows:
    k, s = (r["arm"], r["workload"]), r["solver"]
    b = base.get(k)
    try: obj = float(r["objective"])
    except (TypeError, ValueError): continue
    if not b or obj <= 0: continue
    imp[s].append((b - obj) / b * 100.0)
    CELLS.add(k)
    try: wall[s].append(float(r["wall_s"]))
    except (TypeError, ValueError): pass
    po = int(float(r["periodic_ops"] or 0))
    if po: mp[s].append(int(float(r["misses"] or 0)) / po * 100.0)

S = sorted(imp, key=lambda s: -statistics.mean(imp[s]))
# PORT DIFF: clamp the x axis at the recording floor.
# `wall_s` is recorded as round(perf_counter_delta, 3), so a solve faster than
# 0.5 ms is stored as exactly 0.000 -- and this panel's x axis is LOG, where
# zero is undefined and the marker silently disappears. On the reference's
# 126-801 op workloads no solver was ever that fast; on these 4-33 op cells
# greedy, greedy_periodic, greedy_reserved and decomposed all are, so four of
# twelve solvers vanished from the panel with no error. Clamping to the floor
# keeps them visible and states the true fact -- they are at or below the
# resolution of the measurement and this data cannot separate them -- instead
# of inventing a time for them. WALL_FLOOR is annotated on the figure.
WALL_FLOOR = 0.001
X = {s: max(statistics.median(wall[s]), WALL_FLOOR) for s in S}
_at_floor = sorted(s for s in S if statistics.median(wall[s]) < WALL_FLOOR)
Y = {s: statistics.mean(imp[s]) for s in S}
M = {s: statistics.mean(mp[s]) for s in S}
Q = {s: statistics.quantiles(sorted(imp[s]), n=4) for s in S}   # p25, p50, p75
# 10th-90th percentile, NOT min-max. The extremes here are single cells --
# heft, heft_edf and decomposed each have one workload where they land near
# -119% -- and letting those set the axis compresses every interquartile box
# into an unreadable smear around zero. The outliers are real and belong in the
# text, not in charge of the scale.
W = {s: (statistics.quantiles(sorted(imp[s]), n=10)[0],
         statistics.quantiles(sorted(imp[s]), n=10)[8]) for s in S}

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIG,
                             gridspec_kw=dict(width_ratios=[1.62, 1.0], wspace=0.46))

# ---------------- left: Pareto ----------------
clean = [s for s in S if M[s] < 0.05]
pts = sorted(((X[s], Y[s], s) for s in clean), key=lambda p: (p[0], -p[1]))
best, front = -1e9, []
for x, y, n in pts:
    if y > best: front.append((x, y, n)); best = y
FRONT = {p[2] for p in front}
# Shade the DOMINATED region: everything slower and no better than some point
# on the frontier. Drawn as a staircase because dominance is a step relation --
# between two frontier points the best achievable quality is still the left
# one's, so a straight line would claim performance nothing achieves.
_fx = [p[0] for p in front]
_fy = [p[1] for p in front]
ax.plot(_fx, _fy, "-", color=INK2, lw=1.3, alpha=0.5, zorder=2)
for s in S:
    on, ok = s in FRONT, M[s] < 0.05
    # Hollow marks the solvers that miss deadlines. Without this cue the
    # frontier looks wrong: heft sits at 0.04 s / 3.81%, above AND left of
    # heft_edf, so the eye says the line should start there. It is excluded
    # because it misses 22.9% of periodic operations, and the reader needs to
    # be able to see that from this panel rather than infer it from the other.
    # Fill carries ONE thing (feasible or not) and size carries the other (on
    # the frontier or not). Using opacity for the second made a faded fill read
    # as hollow, which is the very distinction the panel turns on.
    ax.scatter([X[s]], [Y[s]], s=(95 if WIDE else 66) if on else (42 if WIDE else 30),
               facecolor=FAMILY[s] if ok else "none",
               edgecolor="white" if (on and ok) else FAMILY[s],
               linewidth=1.2 if ok else 1.4, zorder=3 if on else 2)
# Label the frontier (bold) PLUS two groups the reader specifically needs to
# find and cannot, because they are off the frontier and so were unlabelled:
#   * cold cpsat -- the point of the panel is that it is DOMINATED (same ~65 s
#     budget as cpsat:warmbest for far less gain), which is invisible if the
#     marker is anonymous.
#   * every solver that MISSES DEADLINES -- drawn hollow, and "which hollow one
#     is that" is exactly the question the hollow marker provokes.
# Non-frontier labels are lighter and italic so the frontier still reads first.
# Bottom-up. Labels prefer to sit ABOVE their marker and get pushed further
# up when that slot is taken, so placing the LOWEST point first keeps stacked
# labels in the same vertical order as the points they name. Top-down inverted
# them: cpsat:warmbest (21.7%) took the near slot and cpsat:warm (20.9%) was
# pushed above it, i.e. the upper label named the lower point.
LABELS = [(s, dict(fontsize=FL, color=INK, fontweight="bold"))
          for s in sorted(FRONT, key=lambda s: Y[s])]

extra = [s for s in S if s not in FRONT and (s == "cpsat" or M.get(s, 0) > 0)]
# Place these by walking DOWN the y axis and alternating which SIDE of the
# marker the text sits on. Alternating only the vertical offset was not enough:
# greedy (0.16 s, 0.0%) and greedy_reserved (0.28 s, -2.5%) are close on both
# axes, so both labels landed in the same strip and overprinted.
# All to the RIGHT -- putting some on the left pushed them off the axis
# (greedy, decomposed and greedy_periodic all clipped). Stagger vertically only
# where two points are actually close in y, which is just greedy (0.0%) and
# greedy_reserved (-2.5%); everything else is far enough apart to sit flat.
# Name only: the miss RATE is already a column on the right panel, and
# repeating it here doubled the label width for no new information. The hollow
# marker plus the legend already say "this one misses".
LABELS += [(s, dict(fontsize=FG, color=INK2, style="italic"))
           for s in sorted(extra, key=lambda z: Y[z])]
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.margins(x=0.22, y=0.20)
ax.axhline(0, color=GRID, lw=1.1, zorder=0)
ax.grid(alpha=0.22, lw=0.6, color=GRID); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=FK)
# Fill AFTER the scale and margins are set, then restore the limits: extending
# the staircase to the right edge as DATA stretched the log axis to 10,000 s.
_xl, _yl = ax.get_xlim(), ax.get_ylim()
ax.fill_between(_fx + [_xl[1]], _yl[0], _fy + [_fy[-1]], step="post",
                color=INK2, alpha=0.055, linewidth=0, zorder=0)
ax.set_xlim(*_xl); ax.set_ylim(*_yl)
ax.set_xlabel("median solve time (s)", fontsize=FA, color=INK2)
ax.set_ylabel("mean makespan gain over greedy (%)", fontsize=FA, color=INK2)
ax.set_title("Pareto: quality vs solve time", fontsize=FT, color=INK, loc="left")
_h = [Line2D([], [], marker="o", ls="", color=c, markersize=5, label=n)
      for c, n in FAMNAME.items()]
_h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none",
                 markeredgecolor=INK2, markersize=5, label="misses deadlines"))
from matplotlib.patches import Patch
_h.append(Patch(facecolor=INK2, alpha=0.10, edgecolor="none", label="dominated"))
ax.legend(handles=_h,
          loc="lower right", frameon=False, fontsize=FG, labelcolor=INK2,
          handletextpad=0.3, borderpad=0.15, labelspacing=0.25)

# Place every label only once the axes, the frontier shading and the legend are
# final, trying a few offsets each and keeping the first that collides with
# nothing already down. The previous scheme alternated the offset by rank,
# which quietly assumes consecutive points are far apart in x -- on measured
# data they are not: pso (21 s, 16.0%) and cpsat:warm (61 s, 20.9%) sat one
# above and one below their markers at the same visual height and the two words
# overprinted. Frontier labels go first so they win the good positions.
_CAND = [(0, 11), (0, -17), (12, -4), (-12, -4), (12, 8), (-12, 8),
         (0, 22), (0, -28), (12, 20), (-12, 20)]
fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_taken = [ax.get_legend().get_window_extent(_rend).expanded(1.02, 1.02)]
# Reserve the markers too. Without this a label is free to sit ON a point:
# "greedy_periodic" printed straight through heft's hollow marker.
from matplotlib.transforms import Bbox as _Bbox
_msz = 7.5 * fig.dpi / 72.0
for _s in S:
    _px, _py = ax.transData.transform((X[_s], Y[_s]))
    _taken.append(_Bbox.from_bounds(_px - _msz, _py - _msz, 2 * _msz, 2 * _msz))
_frame = ax.bbox.expanded(0.99, 0.99)
for _s, _kw in LABELS:
    _got = None
    for _dx, _dy in _CAND:
        _ha = "center" if _dx == 0 else ("left" if _dx > 0 else "right")
        # A label shoved well clear of its point gets a leader, otherwise the
        # reader cannot tell which marker it belongs to: cpsat:warm (20.9%) and
        # cpsat:warmbest (21.7%) sit almost on top of each other, and the
        # second label to be placed ends up ABOVE the first -- i.e. above the
        # higher-scoring point it does not name.
        _ar = (dict(arrowprops=dict(arrowstyle="-", lw=0.7, color=INK2,
                                    alpha=0.5, shrinkA=1, shrinkB=5))
               if abs(_dy) >= 20 else {})
        _an = ax.annotate(_s, (X[_s], Y[_s]), textcoords="offset points",
                          xytext=(_dx, _dy), ha=_ha, zorder=4, **_ar, **_kw)
        _bb = _an.get_window_extent(_rend).expanded(1.05, 1.25)
        if (not any(_bb.overlaps(t) for t in _taken)
                and _frame.contains(_bb.x0, _bb.y0)
                and _frame.contains(_bb.x1, _bb.y1)):
            _got = _bb
            break
        _an.remove()
    if _got is None:   # nothing clean -- put it above the marker anyway
        _an = ax.annotate(_s, (X[_s], Y[_s]), textcoords="offset points",
                          xytext=(0, 11), ha="center", zorder=4, **_kw)
        _got = _an.get_window_extent(_rend).expanded(1.05, 1.25)
    _taken.append(_got)

# ---------------- right: distributions ----------------
ypos = list(range(len(S)))[::-1]
for yi, s in zip(ypos, S):
    col = FAMILY[s]
    lo, hi = W[s]
    bx.plot([lo, hi], [yi, yi], color=col, lw=0.9, alpha=0.35, zorder=1)   # p10-p90
    bx.plot([Q[s][0], Q[s][2]], [yi, yi], color=col, lw=5.0, alpha=0.42,
            solid_capstyle="butt", zorder=2)                                # IQR
    bx.plot([Q[s][1]], [yi], marker="|", color=col, ms=9, mew=1.6, zorder=3)  # median
    bx.scatter([Y[s]], [yi], s=30, color=col, edgecolor="white", lw=0.8, zorder=4)
bx.axvline(0, color=GRID, lw=1.1, zorder=0)
bx.set_yticks(ypos)
bx.set_yticklabels(S, fontsize=FL, color=INK2)
for t, s in zip(bx.get_yticklabels(), S):
    if s in FRONT: t.set_color(INK); t.set_fontweight("bold")
bx.set_ylim(-0.8, len(S) - 0.2)
bx.grid(axis="x", alpha=0.22, lw=0.6, color=GRID); bx.set_axisbelow(True)
for sp in ("top", "right", "left"): bx.spines[sp].set_visible(False)
bx.spines["bottom"].set_color(GRID)
bx.tick_params(colors=INK2, labelsize=FK, length=2)
bx.set_xlabel("makespan gain over greedy (%)", fontsize=FA, color=INK2)
bx.set_title("Distribution across workloads", fontsize=FT, color=INK, loc="left")


# trailing column: the miss rate, attached to its own solver
xr = bx.get_xlim()[1]
bx.text(xr * 1.02, len(S) - 0.35, "misses", fontsize=FG, color=INK2, ha="left",
        va="center", fontweight="bold")
for yi, s in zip(ypos, S):
    txt = "—" if M[s] < 0.05 else (f"{M[s]:.1f}%" if M[s] < 1 else f"{M[s]:.0f}%")
    bx.text(xr * 1.02, yi, txt, fontsize=FG, ha="left", va="center",
            color=INK2 if M[s] < 0.05 else "#b3452a")
x0 = bx.get_xlim()[0]
bx.set_xlim(x0, xr * 1.34)

# One key line for the whole figure: per-panel subtitles collided with the
# titles and with each other at this width.
# PORT DIFF: PARETO_NOTE appends the data-provenance caveat. The measured
# panel needs one (its feasibility cue is predicted, not traced), and a figure
# that carries a caveat only in a README travels without it.
_note = os.environ.get("PARETO_NOTE", "")
if _at_floor:
    _note = (f"solve time for {', '.join(_at_floor)} is below the {WALL_FLOOR*1000:.0f} ms "
             f"recording floor and is drawn at it" + (f"\n{_note}" if _note else ""))
fig.text(0.5, 0.012,
         f"over {len(CELLS)} workload-arms:   thin bar = p10–p90    block = interquartile"
         "    | median    ● mean" + (f"\n{_note}" if _note else ""),
         fontsize=FG - 0.5, color=INK2, ha="center")
fig.subplots_adjust(left=0.068 if WIDE else 0.085, right=0.965,
                    top=0.90, bottom=0.175 + 0.03 * _note.count(chr(10)) + (0.03 if _note else 0))
# PORT DIFF: PARETO_TAG separates the predicted and measured figures, which
# the reference did not need -- it only ever plotted one of the two.
_tag = os.environ.get("PARETO_TAG", "")
out = os.path.join(os.path.dirname(HERE), "plots",
                   f"solver_pareto_panels{_tag}{'_wide' if WIDE else ''}.png")
fig.savefig(out, dpi=200, facecolor="#fcfcfb")
print("wrote", out, f"({FIG[0]}x{FIG[1]} in)")
print("  frontier:", " → ".join(p[2] for p in front))

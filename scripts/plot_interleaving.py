"""Is the schedule doing anything non-trivial? Concurrency and interleaving.

A metrics table can say "0 deadline misses" about a schedule that simply runs
everything one-at-a-time in period order. These two panels are the check: how
many dispatches are actually in flight, and whether a core is genuinely
time-sharing between models rather than draining one queue at a time.
"""
import csv, collections
import os

import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle  # noqa: E402

# The print rcParams and the palette live in `figstyle` because they were
# copy-pasted into five renderers and drifted: DroNet was blue in one figure
# and orange in another, and yolov8_nano was blue in that one. Colour is an
# identity claim, so it is made once.
figstyle.use()
MM = figstyle.MM
SINGLE_COL = figstyle.SINGLE_COL
DOUBLE_COL = figstyle.DOUBLE_COL
DOUBLE = DOUBLE_COL   # this file's older spelling
SINGLE = SINGLE_COL
HZ = 24e6
# Okabe-Ito, assigned by descending period so colours match the other figures.
COL = {m: figstyle.model_color(m)
       for m in ("yolov8_nano", "dronet", "mlp_control")}

rows = [r for r in csv.DictReader(
    open("results/k1_ladder_mb/trace_3model_4hz.csv"))
    if int(r["actual_end_cycles"])]
def hart(r): return f'{r["core_kind"]}#{r.get("worker_hart") or r["hart"]}'
t0 = min(int(r["actual_start_cycles"]) for r in rows)

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 54*MM),
                         gridspec_kw={"width_ratios": [1, 1.9]})

# --- a: concurrency distribution -------------------------------------------
ev = []
for r in rows:
    ev.append((int(r["actual_start_cycles"]), 1))
    ev.append((int(r["actual_end_cycles"]), -1))
ev.sort()
cur, last, at = 0, ev[0][0], collections.Counter()
for t, d in ev:
    if t > last:
        at[cur] += t - last
    cur += d
    last = t
tot = sum(at.values())
ks = sorted(k for k in at if at[k])
ax = axes[0]
ax.bar(ks, [100*at[k]/tot for k in ks],
       color=["#BBBBBB" if k == 0 else figstyle.BLUE for k in ks], width=0.72)
mean = sum(k*v for k, v in at.items())/tot
ax.axvline(mean, color=figstyle.C_DEADLINE, ls="--", lw=0.8)
ax.text(mean+0.18, 40, f"mean {mean:.2f}", fontsize=5, color=figstyle.C_DEADLINE)
ax.set_xlabel("Dispatches in flight")
ax.set_ylabel("Share of wall time (%)")
ax.set_xticks(ks)
ax.spines[["top", "right"]].set_visible(False)

# --- b: interleaving on the busiest core -----------------------------------
byh = collections.Counter(hart(r) for r in rows)
hot = byh.most_common(1)[0][0]
ax = axes[1]
WIN = 34.0
lanes = sorted({hart(r) for r in rows},
               key=lambda h: (0 if h.startswith("rvv#") else 1, h))
idx = {h: i for i, h in enumerate(lanes)}
for r in rows:
    s = (int(r["actual_start_cycles"]) - t0)/HZ*1e3
    if s > WIN:
        continue
    d = (int(r["actual_end_cycles"]) - int(r["actual_start_cycles"]))/HZ*1e3
    ax.broken_barh([(s, max(d, 0.045))], (idx[hart(r)]-0.40, 0.80),
                   facecolors=COL.get(r["network"], "#777"),
                   edgecolors="white", linewidth=0.12)
ax.set_yticks(range(len(lanes)))
ax.set_yticklabels(lanes)
ax.invert_yaxis()
ax.set_xlim(0, WIN)
ax.set_xlabel("Time on the K1 (ms)")
ax.spines[["top", "right"]].set_visible(False)
sw = 0
v = sorted((int(r["actual_start_cycles"]), r["network"])
           for r in rows if hart(r) == hot)
sw = sum(1 for a, b in zip(v, v[1:]) if a[1] != b[1])
# Annotated INSIDE the axes: the lower half is empty (cluster 1 is barely
# used) and the top strip is taken by the legend.
ax.text(0.985, 0.30,
        f"{hot} alone switches model {sw}x\nacross {len(v)} dispatches",
        transform=ax.transAxes, fontsize=5.5, color="#444444",
        ha="right", va="top")
handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[m]) for m in
           ("yolov8_nano", "dronet", "mlp_control")]
ax.legend(handles, ("yolov8_nano 4 Hz", "dronet 30 Hz", "mlp_control 100 Hz"),
          frameon=False, ncol=3, loc="lower right",
          bbox_to_anchor=(1.0, 1.005), borderaxespad=0.0)

for a, lab in zip(axes, "ab"):
    a.text(-0.13, 1.06, lab, transform=a.transAxes, fontsize=8,
           fontweight="bold", va="bottom")
fig.tight_layout(rect=(0.01, 0, 1, 0.99))
for ext in ("png", "pdf"):
    fig.savefig(f"results/frontier/interleaving.{ext}",
                bbox_inches="tight", pad_inches=0.03)
    print(f"wrote results/frontier/interleaving.{ext}")

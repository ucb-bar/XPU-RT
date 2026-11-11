#!/usr/bin/env python3
"""C10v5 -- SIM STRIP normalised to EPISODE PROGRESS, with a true-scale footer.

C10v3 and C10v4 both put the two rollouts on one absolute clock, which is
honest but spends most of the canvas showing that one row is short. This one
asks the other question -- WHAT DID THE ARM DO, stage for stage -- by giving
each row its own 0-100% progress axis so column k is the same fraction of each
episode. Frames align, and the behaviours can be compared directly.

Normalising hides the duration, so the duration comes back explicitly as a
true-scale bar at the foot: the reader sees 5.6 s against 24.0 s as lengths,
not as a number they have to hold in their head. Each frame still carries its
own absolute instant.

The failure mode this makes legible: the baseline reaches the eggplant at the
same PROGRESS fraction, then spends the back half of its episode losing it.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vocab as V

V.style()
TICK = 40.0
ARMS = [("pipe110", "XPU-RT pipelined", 117.6, 1),
        ("cpu685", "QNN baseline", 684.8, 1)]
NF = 8

fig = plt.figure(figsize=(7.16, 3.92))
X0, X1 = 0.052, 0.988
SPAN = X1 - X0
FW = SPAN / NF
durs = []

for k, (arm, lab, per, ep) in enumerate(ARMS):
    age, act, ok = V.age_trace(arm, ep)
    dur = len(age) * TICK / 1000.0
    durs.append((lab, dur, ok))
    V.video_frames(arm, ep, n=2, src="runs")
    fs = sorted((V.HERE / ".frames" / f"runs_egg_{arm}_{ep:02d}").glob("f_*.jpg"))
    top = 0.842 - k * 0.372
    fr = np.linspace(0, 1, NF)
    for j, u in enumerate(fr):
        a = fig.add_axes([X0 + j * FW, top - 0.210, FW * 0.972, 0.210])
        a.axis("off")
        a.imshow(mpimg.imread(fs[min(int(round(u * (len(fs) - 1))), len(fs) - 1)]))
        a.set_title(f"{u * dur:.1f} s", fontsize=6.2, color=V.INK2, pad=1.2)

    ax = fig.add_axes([X0, top - 0.278, SPAN, 0.055])
    ax.set_xlim(0, 1); ax.set_ylim(1.0, 0.0); ax.axis("off")
    rel = np.arange(0, dur, per / 1000.0) / dur          # fraction of the episode
    ax.plot([0, 1], [0.5, 0.5], color=V.INK2, lw=1.2, zorder=2)
    V.ev(ax, rel, 0.86, "release", ms=4.0)
    don = rel + (per / 1000.0) / dur
    V.ev(ax, don[don <= 1.0], 0.5, "complete", ms=4.6)
    for t in V.gripper_events(act):
        V.ev(ax, t * TICK / 1000.0 / dur, 0.5, "grip", ms=6.0)
    V.ev(ax, 1.0, 0.5, "success" if ok else "failure", ms=11 if ok else 8)
    # Row label as figure text ABOVE the per-frame time captions -- placing it in
    # the rail axes puts it straight through them.
    fig.text(X0, top + 0.048, f"{lab}   ·   period {per:.0f} ms   ·   "
                              f"{len(rel)} commands   ·   mean age {np.nanmean(age):.0f} ms",
             fontsize=7.2, color=V.INK, va="center", ha="left", fontweight="bold")

# --- the footer: what normalising took away, handed back at true scale -------
fax = fig.add_axes([X0, 0.118, SPAN, 0.080])
tmax = max(d for _, d, _ in durs)
for i, (lab, d, ok) in enumerate(durs):
    fax.barh(i, d, height=0.5, color=V.INK2 if i else V.LANE["CPU"],
             linewidth=0, zorder=3)
    fax.text(d + tmax * 0.012, i, f"{d:.1f} s", va="center", ha="left",
             fontsize=6.8, color=V.INK, fontweight="bold")
    V.ev(fax, d, i, "success" if ok else "failure", ms=9 if ok else 7)
fax.set_xlim(0, tmax * 1.13); fax.set_ylim(1.7, -0.7)
fax.set_yticks([])          # the bars are already named by the rows above
fax.set_xticks(range(0, 25, 4))
fax.tick_params(labelsize=6.4, colors=V.MUTED, labelcolor=V.INK2, pad=1.2)
for sp in ("left", "right", "top"):
    fax.spines[sp].set_visible(False)
fax.spines["bottom"].set_color(V.AXIS)
fax.set_xlabel("real episode length (s)  ·  the scale the rows above normalise away",
               fontsize=7.0, labelpad=1.2)

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"]),
           loc="upper center", bbox_to_anchor=(0.5, 1.006), ncol=5, fontsize=6.8)
V.save(fig, "cand_c10v5_sim_strip")

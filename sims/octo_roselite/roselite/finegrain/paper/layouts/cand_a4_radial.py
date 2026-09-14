#!/usr/bin/env python3
"""A4 -- RADIAL RELEASE CLOCK. A periodic schedule drawn as a periodic object.

theta = phase within one release period; radius = release index (the spiral).
An inference that outlives its own period wraps past theta = 2*pi and therefore
overlaps the NEXT release -- which is exactly what pipelining is. The baseline
never wraps because its latency IS its period.

Honest about what it costs: a polar axis makes durations hard to compare, so
every arc carries its ms value and the linear cascade stays the primary view.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import vocab as V

V.style()
CFG = [("QNN CPU baseline", 684.8, 684.8, 3),
       ("XPU-RT serial", 283.4, 283.4, 4),
       ("XPU-RT pipelined", 283.3, 124.8, 6)]

fig = plt.figure(figsize=(7.16, 2.75))
gs = fig.add_gridspec(1, 3, left=0.02, right=0.985, top=0.855, bottom=0.02, wspace=0.06)

for i, (title, lat, per, nrel) in enumerate(CFG):
    ax = fig.add_subplot(gs[i], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    inflight = int(np.ceil(lat / per))
    for k in range(nrel):
        r0 = 0.42 + 0.088 * k
        th0 = 0.0
        th1 = 2 * np.pi * lat / per
        th = np.linspace(th0, th1, 220)
        col = V.CMAP_PERIOD(0.35 + 0.10 * (k % 3))
        ax.plot(th, np.full_like(th, r0), lw=4.6, color=col,
                solid_capstyle="butt", zorder=3, alpha=0.95)
        ax.plot([0], [r0], **{**V.GLYPH["release"], "ms": 6, "clip_on": False}, zorder=6)
        ax.plot([th1 % (2 * np.pi)], [r0], **{**V.GLYPH["complete"], "ms": 5.4,
                                              "clip_on": False}, zorder=6)
    ax.set_rlim(0, 0.42 + 0.088 * nrel + 0.10)
    ax.set_rticks([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax.set_xticklabels(["0", f"{per * .25:.0f}", f"{per * .5:.0f}", f"{per * .75:.0f}"],
                       fontsize=6.2, color=V.MUTED)
    ax.grid(color=V.GRID, lw=0.5)
    ax.spines["polar"].set_color(V.AXIS)
    ax.set_title(f"{title}\nperiod {per:.0f} ms · latency {lat:.0f} ms · "
                 f"{inflight} in flight", pad=6, fontsize=7.6)
    ax.text(0, 0, f"{inflight}", ha="center", va="center", fontsize=15,
            color=V.INK, fontweight="bold")
    ax.text(0, 0.20, "wraps" if inflight > 1 else "no wrap", ha="center", va="center",
            fontsize=6.2, color=V.MUTED)

fig.text(0.5, 0.012, "angle = phase within one release period · one ring per release · "
         "an arc past 360 deg overlaps the next release",
         ha="center", fontsize=6.6, color=V.MUTED)
V.save(fig, "cand_a4_radial")

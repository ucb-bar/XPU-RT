#!/usr/bin/env python3
"""C9v2 -- EVENT-MARKED EPISODE TIMELINE, refined.

Changes from C9:
  * episode selection is stated and is not cherry-picked to succeed. For each
    arm the drawn episode is the one whose observation-age mean is closest to
    that arm's own median over its available episodes; the outcome is whatever
    that episode did. The aggregate rate over the full 10-seed sweep is printed
    beside it so a single rollout is never mistaken for the result.
  * left margin widened until no label is clipped (checked by arithmetic).
  * releases hang below the rail, completions sit on it, so the baseline row
    shows both instead of one glyph on top of the other.
  * the age band is drawn on ONE shared 0-1400 ms scale across rows, so row
    height is comparable rather than per-row normalised.
"""
from __future__ import annotations
import glob, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
ARMS = [("ideal", "ideal", 0.0, 200.0, "lat0"),
        ("pipe110", "XPU-RT pipe110", 117.7, 117.6, "pipe110fix"),
        ("serial283", "XPU-RT serial", 283.4, 283.4, "serial283"),
        ("cpu685", "QNN baseline", 684.8, 684.8, "cpu685")]
TICK, AGE_FULL = 40.0, 1400.0
CUR = V.curated()


def pick(arm):
    """the episode whose mean age is closest to this arm's own median"""
    d = V.AGE_RUNS[arm][0]
    eps = sorted(int(p.split("ep")[-1][:2]) for p in
                 glob.glob(str(d / "ep*_action_age_ms.npy")))
    means = {e: float(np.nanmean(np.load(d / f"ep{e:02d}_action_age_ms.npy")))
             for e in eps}
    med = np.median(list(means.values()))
    return min(means, key=lambda e: abs(means[e] - med)), len(eps)


fig = plt.figure(figsize=(7.16, 3.30))
ax = fig.add_axes([0.180, 0.115, 0.640, 0.760])
GUT = fig.add_axes([0.828, 0.115, 0.168, 0.760]); GUT.axis("off")
ROW, tmax = 1.0, 0.0
picks = []

for k, (arm, lab, lat, per, curated_arm) in enumerate(ARMS):
    ep, nep = pick(arm)
    picks.append((arm, ep, nep))
    age, act, ok = V.age_trace(arm, ep)
    t = np.arange(len(age)) * TICK / 1000.0
    tmax = max(tmax, t[-1])
    y0 = -k * ROW

    h = 0.50 * np.nan_to_num(age, nan=0.0) / AGE_FULL
    ax.fill_between(t, y0, y0 + h, step="post", color=V.c_period(per), lw=0, zorder=3)

    on, off, closed = V.gripper_events(act)
    seg, s0 = [], None
    for i, c in enumerate(closed):
        if c and s0 is None:
            s0 = i
        elif not c and s0 is not None:
            seg.append((s0, i)); s0 = None
    if s0 is not None:
        seg.append((s0, len(closed) - 1))
    for a0, a1 in seg:
        ax.add_patch(Rectangle((t[a0], y0 - 0.235), t[a1] - t[a0], 0.10,
                               facecolor=V.MUTED, alpha=0.34, linewidth=0, zorder=2))
    if len(on):
        V.ev(ax, t[on], y0 - 0.185, "grip", ms=5.4)

    yr = y0 + 0.68
    ax.plot([0, t[-1]], [yr, yr], color=V.AXIS, lw=0.9, zorder=2)
    rel = np.arange(0, t[-1] * 1000, per) / 1000.0
    V.ev(ax, rel, yr - 0.085, "release", ms=4.6)
    V.ev(ax, t[V.completions(age)], yr, "complete", ms=4.4)
    V.ev(ax, [t[-1] + 0.16], [y0 + 0.22], "success" if ok else "failure", ms=9.5)

    ax.text(-0.14, y0 + 0.44, lab, ha="right", va="center", fontsize=7.4,
            color=V.INK, fontweight="bold")
    ax.text(-0.14, y0 + 0.19, f"period {per:.0f} ms", ha="right", va="center",
            fontsize=6.2, color=V.INK2)
    ax.text(-0.14, y0 - 0.03, f"age {lat:.0f} ms", ha="right", va="center",
            fontsize=6.2, color=V.MUTED)

    s = json.load(open(V.AGE_RUNS[arm][0] / "summary.json"))["episodes"][ep]
    rate = CUR["egg"][curated_arm]["success"]
    GUT.text(0.0, y0 + 0.60, f"{s['n_inferences']}", fontsize=11.5, color=V.INK,
             fontweight="bold", va="center")
    GUT.text(0.30, 0.0 + y0 + 0.60, "commands", fontsize=6.1, color=V.INK2, va="center")
    GUT.text(0.0, y0 + 0.32, f"{np.nanmean(age):.0f}", fontsize=8.4, color=V.INK2,
             va="center")
    GUT.text(0.30, y0 + 0.32, "ms mean age", fontsize=6.1, color=V.MUTED, va="center")
    GUT.text(0.0, y0 + 0.06, f"{rate:.0f}%", fontsize=8.4, color=V.INK2, va="center")
    GUT.text(0.30, y0 + 0.06, "success, n=240", fontsize=6.1, color=V.MUTED, va="center")

for a in (ax, GUT):
    a.set_ylim(-len(ARMS) * ROW + 0.42, 0.92)
GUT.set_xlim(0, 1)
ax.set_xlim(-0.05, tmax + 0.60)
ax.set_yticks([])
ax.set_xlabel("episode time (s)   ·   40 ms control tick   ·   MEASURED rollout, "
              "eggplant in basket")
for s_ in ("top", "right", "left"):
    ax.spines[s_].set_visible(False)
ax.grid(True, axis="x", color=V.GRID, lw=0.6)
ax.set_axisbelow(True)
ax.text(tmax + 0.55, 0.86, "band height = observation age, shared 0-1400 ms scale",
        fontsize=6.1, color=V.MUTED, ha="right", va="top")

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"])
           + [Rectangle((0, 0), 1, 1, facecolor=V.MUTED, alpha=0.34,
                        label="gripper closed")],
           loc="upper left", bbox_to_anchor=(0.180, 1.006), ncol=6, handlelength=1.1,
           columnspacing=1.1)
V.save(fig, "cand_c9v2_episode_timeline", f"episodes drawn: {picks}")

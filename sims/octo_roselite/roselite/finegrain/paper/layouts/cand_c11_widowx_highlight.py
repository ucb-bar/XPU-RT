#!/usr/bin/env python3
"""C11 -- HIGHLIGHT: success and actuator energy on the two widowx scenes.

Four panels, scene-major (eggplant success, eggplant energy, spoon success,
spoon energy) so each scene's pair reads as one statement: the same move up the
ladder that raises success also lowers energy. They are not traded.

COLOUR is the original curated-sweep palette, verbatim from fig_metrics3.py, so
this sits beside fig_metrics3 / fig_pareto3 / fig_energy without a reader
relearning which colour is which arm: grey ideal, teal-green pipelined (dark ->
light as cadence slows), amber/orange serial, red CPU-only.

LABELS carry period/latency, both MEASURED, because an arm's name encodes only
its period (or its period/window request) -- latency and period coincide just
for the serial and CPU-only arms, where nothing overlaps. Printed this way the
ideal arm reads honestly as 200/0: zero latency, but a 200 ms cadence, so it is
a reference and not a ceiling. Ticks stagger over two rows so they can stay
horizontal, which is shorter than rotating them.

SIX arms, not the sweep's nine: four pipelined points on one ladder is
repetition, not evidence. Energy is `t2_drive_arm_sus` as a fraction of the QNN
baseline, unconditioned on success -- conditioning keeps only the episodes an arm
won, which on eggplant cpu685 is 1 in 60, and hides the flailing that costs the
energy. n = 30 seeds x 24 episodes per bar, bootstrap 95% CI.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import vocab as V

V.style()
SRC = V.BASE / "traces_torque3"
ECH = "t2_drive_arm_sus"
ARMS = ["lat0", "p105w300", "p150w300", "pipe200fix", "serial283", "cpu685"]
COL = {"lat0": "#7f8c8d", "p105w300": "#148f77", "p150w300": "#45b39d",
       "pipe200fix": "#d68910", "serial283": "#e67e22", "cpu685": "#7b241c"}
NAME = {"lat0": "ideal", "p105w300": "pipe", "p150w300": "pipe",
        "pipe200fix": "pipe", "serial283": "serial", "cpu685": "cpu int8"}
SCENES = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]
BASE_ARM, IDEAL_ARM = "cpu685", "lat0"
# End-state frames already cached locally by the strip figures. Success first.
THUMB = {"egg": ("runs_egg_pipe110_01", "runs_egg_cpu685_01"),
         "spoon": ("ladder_spoon_pipe110_21", "ladder_spoon_cpu685_21")}
# Which corner the outcome badge sits in, per scene -- whichever corner the
# rollout leaves empty. On eggplant the sink and basket fill the left, on spoon
# the table runs to the right.
BADGE = {"egg": "right", "spoon": "left"}

D = collections.defaultdict(dict)
for f in glob.glob(str(SRC / "*" / "energy2.json")):
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", Path(f).parent.name)
    D[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))


def ci(v, B=4000, seed=0):
    r = np.random.default_rng(seed); a = np.asarray(v, float)
    return np.percentile([r.choice(a, a.size, replace=True).mean() for _ in range(B)],
                         [2.5, 97.5])


fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.16), dpi=170)
fig.subplots_adjust(left=0.062, right=0.995, top=0.845, bottom=0.255, wspace=0.30)

for si, (task, tlab) in enumerate(SCENES):
    one = {a: next(iter(D[(task, a)].values())) for a in ARMS}
    per = {a: one[a]["issue_period_ms"] for a in ARMS}
    lat = {a: one[a]["latency_ms"] for a in ARMS}
    order = [IDEAL_ARM] + sorted([a for a in ARMS if a != IDEAL_ARM], key=lambda a: per[a])
    x = np.arange(len(order))
    cols = [COL[a] for a in order]

    sr = {a: [100 * d["n_success"] / d["n_episodes"] for d in D[(task, a)].values()]
          for a in ARMS}
    en = {a: [float(np.median([e[ECH] for e in d["episodes"]]))
              for d in D[(task, a)].values()] for a in ARMS}
    ep_en = {a: [float(e[ECH]) for d in D[(task, a)].values() for e in d["episodes"]
                 if e[ECH] > 0] for a in ARMS}
    b = float(np.mean(en[BASE_ARM]))

    for mi, metric in enumerate(("success", "energy")):
        ax = axes[si * 2 + mi]
        if metric == "success":
            # PER SEED (n = 10). A per-episode success point is 0 or 1, so there
            # is no episode-level cloud to draw here -- the seed rate is the
            # finest unit that carries information.
            pts = [np.asarray(sr[a], float) for a in order]
            ttl, ylab = "success rate", "success (%)"
        else:
            # PER SEED (n = 10), deliberately NOT per episode. fig_energy.py
            # originally drew one point per episode at a single seed -- 24 per arm,
            # which across nine arms is the dense strip this figure is compared
            # against. The same unit is available here at ten times the density
            # (240/arm), and cand_c11b_widowx_episodes.py draws it. It is not used
            # here because the per-episode integral spans 4,600x-9,400x from p5 to
            # p95 on a scheduled arm, so the cloud needs a log axis and on a log
            # axis every arm's box overlaps every other -- the arm-level effect
            # this figure exists to show disappears into within-arm spread.
            pts = [np.asarray(en[a], float) / b for a in order]
            ttl, ylab = "actuator energy", "energy / QNN baseline"
            ax.axhline(1.0, color=V.STATUS["critical"], lw=0.9, ls="--", alpha=0.75,
                       zorder=2)

        # BOX + STRIP, as fig_energy.py drew it originally: the box carries the
        # median and quartiles, the dots are the individual seeds behind it. The
        # unit is the per-SEED value (n = 10), not the per-episode one -- that is
        # the unit the CI was taken over, and a 240-point episode cloud would show
        # a much wider spread than the statistic the panel actually reports.
        bp = ax.boxplot(pts, positions=x, widths=0.62, patch_artist=True,
                        showmeans=True, showfliers=False, zorder=3,
                        medianprops=dict(color=V.INK, lw=1.0),
                        whiskerprops=dict(color=V.INK2, lw=0.7),
                        capprops=dict(color=V.INK2, lw=0.7),
                        meanprops=dict(marker="D", ms=2.6, mfc="white",
                                       mec=V.INK, mew=0.7))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("white"); patch.set_linewidth(0.45)
        rng = np.random.default_rng(0)
        for i, v in enumerate(pts):
            ax.scatter(rng.normal(i, 0.075, len(v)), v,
                       s=3.2, c=V.INK, alpha=0.30,
                       linewidths=0, zorder=5, clip_on=True)

        # Ticks STAGGERED over two rows so they stay horizontal: rotating them
        # upright costs more vertical space than a second row does.
        ax.set_xlim(-0.62, len(order) - 0.38)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        for i, a in enumerate(order):
            ax.text(i, -0.045 - 0.145 * (i % 2), f"{NAME[a]}\n{per[a]:.0f}/{lat[a]:.0f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5.0, color=V.INK2, linespacing=1.18)
        ax.set_ylabel(ylab, fontsize=6.6, labelpad=1.5)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_title(ttl, fontsize=7.0, pad=2.6)
        # Explicit limits, not margins: both quantities are bounded below at zero
        # and a success rate is bounded above at 100, so autoscaling a box plot
        # into negative success is a nonsense axis. The success panels get the
        # extra headroom because they carry the thumbnails.
        if metric == "success":
            ax.set_ylim(0, min(100.0, max(v.max() for v in pts) * 1.62))
        else:
            ax.set_ylim(0, max(v.max() for v in pts) * 1.10)
        V.tidy(ax, grid="y")

        # Two end-state frames, overlaid top-right of the SUCCESS panel: what the
        # bars are actually counting. Failure behind, success in front.
        if metric == "success":
            bb = ax.get_position()
            aw, ah = bb.width * 7.16, bb.height * 2.16      # axes size in inches
            wf = 0.362                                       # thumb width, axes frac
            hf = wf * (aw / ah) * 0.75                       # keep the frames 4:3
            for j, key in enumerate(THUMB[task][::-1]):      # failure first = behind
                fs = sorted((V.HERE / ".frames" / key).glob("f_*.jpg"))
                # STACKED, not overlapped: two wider cards one above the other
                # keep both end states fully visible and leave the bars alone.
                # Success on top, failure below it.
                x0 = 1.0 - wf
                y0 = 0.995 - hf - (1 - j) * (hf + 0.020)
                ib = ax.inset_axes([x0, y0, wf, hf], zorder=6 + j)
                ib.imshow(mpimg.imread(fs[-1]), aspect="auto")
                ib.set_xticks([]); ib.set_yticks([])
                for sp in ib.spines.values():
                    sp.set_edgecolor("white"); sp.set_linewidth(1.2)
                bx = 0.945 if BADGE[task] == "right" else 0.055
                ib.text(bx, 0.93, "\u2713" if j else "\u2717",
                        transform=ib.transAxes, ha=BADGE[task], va="top", fontsize=6.4,
                        fontweight="bold", zorder=9,
                        color=V.STATUS["good"] if j else V.STATUS["critical"],
                        bbox=dict(fc="white", ec="none", alpha=0.82, pad=0.9))

    a0 = axes[si * 2].get_position(); a1 = axes[si * 2 + 1].get_position()
    fig.text((a0.x0 + a1.x1) / 2, 0.995, tlab, ha="center", va="top",
             fontsize=7.8, fontweight="bold", color=V.INK)

fig.text(0.53, 0.006, "arm  ·  release period / observation latency (ms)  ·  "
                      "reference first, then by period",
         ha="center", va="bottom", fontsize=6.2, color=V.INK2)
V.save(fig, "cand_c11_widowx_highlight")

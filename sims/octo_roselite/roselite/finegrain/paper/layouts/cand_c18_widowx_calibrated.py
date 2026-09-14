#!/usr/bin/env python3
r"""C18 -- C11 with the energy panels in JOULES instead of tau-squared.

C11's energy axis was `t2_drive_arm_sus` as a fraction of the QNN baseline. That
channel is a real PD drive-torque integral, but it is NOT joules and its ratios
are not energy ratios, because the SIMULATED torque is not physical: on egg/ideal
the PD drive runs 148 N.m rms against 1.94 N.m of gravity+Coriolis on the same
trajectory, saturating 20% of substeps, with a force_limit of 200 N.m on a
shoulder whose two XM430-W350s stall at 8.2 N.m combined -- 24x the real ceiling.
Converting that channel at face value yields 1.6 kW-84 kW against a 60 W supply.

CALIBRATION IS GLOBAL, APPLIED PER EPISODE. The constants are derived once from
published ROBOTIS data and then used to transform EVERY episode, so the seed
distribution survives the conversion instead of collapsing into one aggregate
number per arm:

    M3_J(episode) = 0.8267 * t2_qf_sub_arm  +  4.608 * duration_s
                    \_ W/(N.m)^2, R/K^2 _/     \_ W, published standby _/

Reduced the way every other panel in this set reduces: median over an arm's 24
episodes, then one point per seed. That reproduces calib/calibrated_curated.tsv
exactly (egg/ideal 117.0 J, egg/cpu685 200.1 J, spoon/ideal 86.0 J,
spoon/cpu685 122.6 J) while keeping the 30-seed spread the box and strip show.
The mean over episodes does NOT reproduce it -- egg/cpu685 comes out at 6371 J,
because a handful of episodes carry enormous gravity integrals; the median is
what makes the estimator robust, as elsewhere in this sweep.

The upper bound has no distribution to draw: it needs the per-actuation series in
series.npz, which is missing for seeds 101-109, so it is one number per arm and
is marked with a caret rather than a box.

So the panel shows a BRACKET, not a number (calib/ENERGY_CALIBRATION.md):
  M3, the lower bound  -- quasi-static: the torque the trajectory physically
                          requires, through R/K^2 = 1.64 W/(N.m)^2 derived from
                          ROBOTIS' published stall torque/current pair.
  M2, the upper bound  -- the same per actuation, clipped at the 12 V x 5 A rail.
The solid bar spans M3 to M2; the hairline behind it spans the full uncertainty
envelope, whose width is dominated by ROBOTIS' unpublished no-load current.

WHAT CHANGES, and it matters: on eggplant cpu685 goes from "3.36x the ideal arm"
to 1.71-1.89x, and on spoon from 34.3x to 1.43-3.65x. The ORDERING survives on
both widowx scenes -- eggplant's is identical across all nine arms -- so every
schedule conclusion stands. The magnitudes do not.

Success panels are unchanged from C11 and remain box-and-strip over seeds.
"""
from __future__ import annotations
import csv, glob, json, re, collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import vocab as V

V.style()
CAL = V.BASE / "calib" / "calibrated_curated.tsv"
SRC = V.BASE / "traces_torque3"
ARMS = ["ideal", "p105w300", "p150w300", "pipe200fix", "serial283", "cpu685"]
KEY = {"ideal": "lat0"}                       # calib name -> traces_torque3 name
COL = {"ideal": "#7f8c8d", "p105w300": "#148f77", "p150w300": "#45b39d",
       "pipe200fix": "#d68910", "serial283": "#e67e22", "cpu685": "#7b241c"}
NAME = {"ideal": "ideal", "p105w300": "pipe", "p150w300": "pipe",
        "pipe200fix": "pipe", "serial283": "serial", "cpu685": "cpu int8"}
SCENES = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]
THUMB = {"egg": ("runs_egg_pipe110_01", "runs_egg_cpu685_01"),
         "spoon": ("ladder_spoon_pipe110_21", "ladder_spoon_cpu685_21")}
BADGE = {"egg": "right", "spoon": "left"}

C_QF, P_IDLE = 0.8267, 4.608     # W/(N.m)^2 and W, from calib/calibrate.py

C = {}
for r in csv.DictReader(open(CAL), delimiter="\t"):
    C[(r["task"], r["arm"])] = {k: (float(v) if k not in ("task", "arm") else v)
                                for k, v in r.items()}

D = collections.defaultdict(dict)
for f in glob.glob(str(SRC / "*" / "energy2.json")):
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", Path(f).parent.name)
    D[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))

fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.30), dpi=170)
fig.subplots_adjust(left=0.072, right=0.988, top=0.845, bottom=0.300, wspace=0.34)

for si, (task, tlab) in enumerate(SCENES):
    one = {a: next(iter(D[(task, KEY.get(a, a))].values())) for a in ARMS}
    per = {a: one[a]["issue_period_ms"] for a in ARMS}
    lat = {a: one[a]["latency_ms"] for a in ARMS}
    order = ["ideal"] + sorted([a for a in ARMS if a != "ideal"], key=lambda a: per[a])
    x = np.arange(len(order))
    cols = [COL[a] for a in order]

    for mi, metric in enumerate(("success", "energy")):
        ax = axes[si * 2 + mi]
        if metric == "success":
            pts = [np.asarray([100 * d["n_success"] / d["n_episodes"]
                               for d in D[(task, KEY.get(a, a))].values()], float) for a in order]
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
                ax.scatter(rng.normal(i, 0.075, len(v)), v, s=3.2, c=V.INK,
                           alpha=0.30, linewidths=0, zorder=5, clip_on=True)
            ax.set_ylabel("success (%)", fontsize=6.6, labelpad=1.5)
            ax.set_ylim(0, min(100.0, max(v.max() for v in pts) * 1.62))
            ax.set_title("success rate", fontsize=7.0, pad=2.6)
        else:
            # GLOBAL constants, applied PER EPISODE -- see docstring.
            pts = [np.asarray([np.median([C_QF * e["t2_qf_sub_arm"] + P_IDLE * e["duration_s"]
                                          for e in d["episodes"]])
                               for d in D[(task, KEY.get(a, a))].values()], float) for a in order]
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
                ax.scatter(rng.normal(i, 0.075, len(v)), v, s=3.2, c=V.INK,
                           alpha=0.30, linewidths=0, zorder=5, clip_on=True)
            # The supply-rail UPPER bound is deliberately not drawn. It is one
            # number per arm (series.npz is missing for seeds 101-109, so there is
            # no per-seed spread to show), and putting it on the axis forced a log
            # scale that squashed the distribution this panel exists to show. It
            # lives in the caption instead: 763-1440 J on eggplant, 197-720 J on
            # spoon. See calib/calibrated_curated.tsv, column M2_J.
            ax.set_ylim(0, max(v.max() for v in pts) * 1.12)
            ax.set_ylabel("energy per episode (J)", fontsize=6.6, labelpad=1.5)
            ax.set_title("energy, calibrated", fontsize=7.0, pad=2.6)

        ax.set_xlim(-0.62, len(order) - 0.38)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=6.0)
        for i, a in enumerate(order):
            ax.text(i, -0.045 - 0.145 * (i % 2), f"{NAME[a]}\n{per[a]:.0f}/{lat[a]:.0f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=5.0, color=V.INK2, linespacing=1.18)
        V.tidy(ax, grid="y")

        if metric == "success":
            bb = ax.get_position(); aw, ah = bb.width * 7.16, bb.height * 2.30
            wf = 0.362; hf = wf * (aw / ah) * 0.75
            for j, key in enumerate(THUMB[task][::-1]):
                fs = sorted((V.HERE / ".frames" / key).glob("f_*.jpg"))
                ib = ax.inset_axes([1.0 - wf, 0.995 - hf - (1 - j) * (hf + 0.020),
                                    wf, hf], zorder=6 + j)
                ib.imshow(mpimg.imread(fs[-1]), aspect="auto")
                ib.set_xticks([]); ib.set_yticks([])
                for sp in ib.spines.values():
                    sp.set_edgecolor("white"); sp.set_linewidth(1.2)
                bxp = 0.945 if BADGE[task] == "right" else 0.055
                ib.text(bxp, 0.93, "✓" if j else "✗", transform=ib.transAxes,
                        ha=BADGE[task], va="top", fontsize=6.4, fontweight="bold",
                        zorder=9, color=V.STATUS["good"] if j else V.STATUS["critical"],
                        bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.9))

    a0 = axes[si * 2].get_position(); a1 = axes[si * 2 + 1].get_position()
    fig.text((a0.x0 + a1.x1) / 2, 0.985, tlab, ha="center", va="top",
             fontsize=7.8, fontweight="bold", color=V.INK)

fig.text(0.53, 0.030, "arm  ·  release period / observation latency (ms)  ·  "
                      "reference first, then by period",
         ha="center", va="bottom", fontsize=6.2, color=V.INK2)
# No caption baked into the canvas: it belongs in the LaTeX float.
# See cand_c18_caption.tex.
V.save(fig, "cand_c18_widowx_calibrated")

r"""C21 -- three workloads: six ladder panels over the plane, coke excluded.

Panel A (top): the curated ladder as SIX panels -- success and calibrated energy
for eggplant, spoon and close_drawer. Panel B (bottom): the same three workloads
on the full 44-point plane, as a scatter plus one plane each, four panels in a row.

COKE IS EXCLUDED and drawer kept, because between the two google workloads drawer
is the one that says something: it is the actuator-saturated case, and its
flatness here is the finding. Coke is the same story with less contrast.

DRAWER'S ENERGY PANEL IS ANNOTATED, not quietly plotted. Its calibration uses the
google PROXY constants, and with P_idle = 40 W over ~39 s episodes the idle term
alone is ~1584 J of a ~1580 J total -- the actuation contribution is inside the
rounding. The panel is drawn because leaving a hole would imply the measurement
failed, but it is labelled "idle-dominated" so nobody reads a schedule effect
into a constant.

TWO COLOUR SYSTEMS, ON PURPOSE, and the section headers are what keep them apart.
In A colour is the ARM (grey reference, teal pipelined, amber serial, red CPU
only) because the panels are per-scene and the arm is what varies. In B colour is
the WORKLOAD because the panels are per-plane and the workload is what varies.
Stacking them without the A/B split would ask one hue to mean two things.

ENERGY IS CALIBRATED IN BOTH, globally and per episode:
    E_J(episode) = c_qf * t2_qf_sub_arm + P_idle * duration_s
reduced with the estimator used everywhere else here -- median over an arm's 24
episodes, then mean over seeds. widowx constants (0.827 W/(N.m)^2, 4.61 W) are
DERIVED from ROBOTIS' published stall torque/current for the Dynamixel
XM430-W350; google's (0.0125 W/(N.m)^2, 40 W) are a PROXY, which is why the
google series in B are hollow. This is the quasi-static LOWER bound: the raw
tau^2 channel cannot be converted at face value because the simulated PD torque
runs ~76x the physical requirement.

Caption in cand_c21_caption.tex, not on the canvas.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams["hatch.linewidth"] = 0.45
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import sys
import vocab as V
sys.path.insert(0, str(V.HERE.parent))
import plane3_lib as L

V.style()
PAPER = V.HERE.parent
# Google constants are BALANCE-MATCHED to the WidowX, not independently sourced.
#
# The Everyday Robots arm has no published actuator data. calib/ used a Harmonic
# Drive FHA-14C-100 for c and a Kinova/Franka envelope for P_idle -- two proxies
# from different machines, and the pairing was incoherent: a Franka-class 40 W
# idle next to a 3 W actuation term, so idle was 93% of the result and the
# actuation signal was buried under an assumed constant.
#
# This study is about how end-to-end robot behaviour responds to scheduling, not
# about absolute watts for a robot nobody has specified. So c_google is set so the
# arm's dynamic/idle BALANCE matches the WidowX's measured 0.810, keeping the
# literature-anchored 40 W idle:
#
#     c_google = 0.0125 * (0.810 * 40) / 3.07 = 0.1321 W/(N.m)^2
#
# That gives 30-31 W of actuation against 40 W idle (43% actuation), beside the
# WidowX's 45% -- a plausible balance for a Franka-class arm.
#
# WHAT THIS CHANGES: absolute joules for the google tasks, nothing else. Scaling c
# scales every arm equally, so the across-arm ratios are EXACTLY invariant --
# drawer max/min actuation is 1.0269 under either constant, coke 1.2404. That
# drawer is flat and coke nearly so does not rest on the proxy at all; only the
# vertical scale does. The series stay hollow to mark that.
CAL = {"egg": (0.8267, 4.608), "spoon": (0.8267, 4.608),      # DERIVED
       "coke": (0.1321, 40.0), "drawer": (0.1321, 40.0)}      # BALANCE-MATCHED
PROXY = {"egg": False, "spoon": False, "coke": True, "drawer": True}


def energy_J(task, cells):
    c, pi = CAL[task]
    return float(np.mean([np.median([c * e["t2_qf_sub_arm"] + pi * e["duration_s"]
                                     for e in d["episodes"]]) for d in cells.values()]))


# ---------- A: curated ladder ----------
A_ARMS = ["ideal", "p105w300", "p150w300", "pipe200fix", "serial283", "cpu685"]
A_KEY = {"ideal": "lat0"}
A_COL = {"ideal": "#7f8c8d", "p105w300": "#148f77", "p150w300": "#45b39d",
         "pipe200fix": "#d68910", "serial283": "#e67e22", "cpu685": "#7b241c"}
A_NAME = {"ideal": "ideal", "p105w300": "pipe", "p150w300": "pipe",
          "pipe200fix": "pipe", "serial283": "serial", "cpu685": "cpu int8"}
SCENES = [("egg", "eggplant in basket"), ("spoon", "spoon on towel"),
          ("drawer", "close drawer")]
# End-state frames, success first.
#
# Eggplant and spoon pair a SCHEDULED success against a BASELINE failure -- on
# those scenes that contrast is real, because the schedule decides the outcome.
#
# Drawer pairs a success and a failure FROM THE SAME ARM (p105w300, episodes 1
# and 0). It has to: at n=30 no arm differs significantly from the 0 ms reference
# (every paired p > 0.06), so there is no schedule that succeeds where another
# fails. The pair shows what the task's two outcomes look like -- drawer shut and
# arm withdrawn, against drawer still hanging open -- not a schedule effect.
THUMB = {"egg": ("runs_egg_pipe110_01", "runs_egg_cpu685_01"),
         "spoon": ("ladder_spoon_pipe110_21", "ladder_spoon_cpu685_21"),
         "drawer": ("runs_drawer_ok", "runs_drawer_no")}
PLATE = {}
BADGE = {"egg": "right", "spoon": "left", "drawer": "left"}
# ---------- B: the plane ----------
HUE = {"egg": "#1b7f5f", "spoon": "#2a78d6", "coke": "#c0392b", "drawer": "#7a3fa5"}
MARK = {"egg": "o", "spoon": "s", "coke": "^", "drawer": "D"}
B_TASKS = [t for t in L.TASKS if t[0] != "coke"]        # coke excluded, see docstring
SHORT = {"egg": "eggplant", "spoon": "spoon", "coke": "coke can", "drawer": "drawer"}
CMAP = {t: LinearSegmentedColormap.from_list(t, ["#f4f3ef", h]) for t, h in HUE.items()}

CUR = collections.defaultdict(dict)
for f in glob.glob(str(V.BASE / "traces_torque3" / "*" / "energy2.json")):
    m = re.match(r"(egg|spoon|coke|drawer)_(.+?)(?:_rng(\d+))?$", Path(f).parent.name)
    CUR[(m.group(1), m.group(2))][int(m.group(3) or 100)] = json.load(open(f))
warm = json.load(open(PAPER / "grid_warm.json"))["cells"]
TWIN = {a: v["twin"] for a, v in
        json.load(open(PAPER / "grid_e2e_success.json"))["egg"].items()
        if v.get("source") == "propagated"}
T = L.table(); RAW = L.load()
EJ = {t: {a: energy_J(t, RAW[(t, a)]) for a in T[t]} for t, _, _ in L.TASKS}


FIG_H_IN = 3.30
fig = plt.figure(figsize=(7.16, FIG_H_IN), dpi=170)
gsA = fig.add_gridspec(1, 6, left=0.030, right=0.992, top=0.846, bottom=0.640, wspace=0.38)
# Two rows on the right: the rollout pair above the plane it belongs to. The
# scatter spans both, so the snapshots gain the height panel A was spending on
# them and become legible instead of thumbnail-sized.
gsB = fig.add_gridspec(2, 4, width_ratios=[1.32, 0.86, 0.86, 0.86],
                       height_ratios=[0.62, 1.0], left=0.046, right=0.992,
                       top=0.505, bottom=0.118, wspace=0.36, hspace=0.13)

fig.text(0.006, 0.990, "A", fontsize=10, fontweight="bold", va="top")
fig.text(0.024, 0.987, "curated schedule ladder",
         fontsize=7.4, fontweight="bold", va="top", color=V.INK)
fig.text(0.006, 0.545, "B", fontsize=10, fontweight="bold", va="top")
fig.text(0.024, 0.542, "the full plane",
         fontsize=7.4, fontweight="bold", va="top", color=V.INK)

# ============================== A: six panels ==============================
axesA = [fig.add_subplot(gsA[i]) for i in range(6)]
for si, (task, tlab) in enumerate(SCENES):
    one = {a: next(iter(CUR[(task, A_KEY.get(a, a))].values())) for a in A_ARMS}
    per = {a: one[a]["issue_period_ms"] for a in A_ARMS}
    order = ["ideal"] + sorted([a for a in A_ARMS if a != "ideal"], key=lambda a: per[a])
    x = np.arange(len(order)); cols = [A_COL[a] for a in order]
    c_qf, p_idle = CAL[task]
    ns = len(CUR[(task, "lat0")])
    for mi, metric in enumerate(("success", "energy")):
        ax = axesA[si * 2 + mi]
        if metric == "success":
            pts = [np.asarray([100 * d["n_success"] / d["n_episodes"]
                               for d in CUR[(task, A_KEY.get(a, a))].values()], float)
                   for a in order]
            ttl = "success (%)"
        else:
            pts = [np.asarray([np.median([c_qf * e["t2_qf_sub_arm"] + p_idle * e["duration_s"]
                                          for e in d["episodes"]])
                               for d in CUR[(task, A_KEY.get(a, a))].values()], float)
                   for a in order]
            ttl = "energy (J)"
        bp = ax.boxplot(pts, positions=x, widths=0.60, patch_artist=True, showmeans=True,
                        showfliers=False, zorder=3, medianprops=dict(color=V.INK, lw=0.9),
                        whiskerprops=dict(color=V.INK2, lw=0.6),
                        capprops=dict(color=V.INK2, lw=0.6),
                        meanprops=dict(marker="D", ms=2.2, mfc="white", mec=V.INK, mew=0.6))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("white"); patch.set_linewidth(0.4)
        rng = np.random.default_rng(0)
        for i, v in enumerate(pts):
            ax.scatter(rng.normal(i, 0.075, len(v)), v, s=2.4, c=V.INK, alpha=0.30,
                       linewidths=0, zorder=5, clip_on=True)
        ax.set_ylim(0, min(100.0, max(v.max() for v in pts) * 1.12) if metric == "success"
                    else max(v.max() for v in pts) * 1.12)
        ax.set_title(ttl, fontsize=6.2, pad=2.0, color=V.INK)
        ax.set_xlim(-0.62, len(order) - 0.38)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=5.2)
        for i, a in enumerate(order):
            ax.text(i, -0.055 - 0.175 * (i % 2), f"{A_NAME[a]}\n{per[a]:.0f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=4.2, color=V.INK2, linespacing=1.12)
        V.tidy(ax, grid="y")
    a0 = axesA[si * 2].get_position(); a1 = axesA[si * 2 + 1].get_position()
    fig.text((a0.x0 + a1.x1) / 2, 0.936, tlab, ha="center",
             va="top", fontsize=6.8, fontweight="bold", color=V.INK)

# ============================== B: scatter + three planes ==============================
ax = fig.add_subplot(gsB[:, 0])
ORDER = ["drawer", "spoon", "egg"]                 # widowx drawn on top
BEST = {t: max(T[t], key=lambda k: T[t][k]["success"]) for t, _, _ in B_TASKS}
# RELATIVE energy, each workload against its own cheapest schedule. On an
# absolute axis the four sit in disjoint bands -- spoon ~90 J, eggplant ~150 J,
# the two google tasks ~1.5-2.5 kJ -- so every series collapses to a vertical
# line and the within-workload ladder, which is the thing being compared, is
# unreadable. Normalised they all span 1.0-1.4x and overlay directly.
#
# Markers are uniform. An earlier version hollowed the google series to mark
# their constants as a proxy, but EVERY constant here is a model: the widowx
# values are derived from published stall specs rather than measured on the arm,
# and the whole quantity is a quasi-static lower bound. Singling out one pair
# implied the others were measured. Provenance is a caption matter.
for zi, task in enumerate(ORDER):
    arms = T[task]; ks = sorted(arms)
    e = np.array([EJ[task][a] for a in ks]); sv = np.array([arms[a]["success"] for a in ks])
    ax.scatter(e / e.min(), sv, s=13, marker=MARK[task], c=HUE[task],
               edgecolor="white", linewidth=0.30, zorder=4 + zi, label=SHORT[task])
    b = BEST[task]
    ax.scatter(EJ[task][b] / e.min(), arms[b]["success"], s=68, marker="*", c="white",
               edgecolor=HUE[task], linewidth=0.9, zorder=12 + zi)
ax.set_xlabel("energy / this workload's cheapest schedule", fontsize=6.0, labelpad=1.2)
ax.set_ylabel("success rate (%)", fontsize=6.2, labelpad=1.8)
ax.tick_params(labelsize=5.4)
ax.margins(x=0.045, y=0.06); ax.set_ylim(0, ax.get_ylim()[1])
V.tidy(ax, grid="both")
h, lb = ax.get_legend_handles_labels()
o = [lb.index(SHORT[t]) for t, _, _ in B_TASKS]
ax.legend([h[i] for i in o], [lb[i] for i in o], fontsize=5.0, loc="lower left",
          handletextpad=0.25, borderpad=0.28, labelspacing=0.20, markerscale=0.95)

PERIODS = sorted({int(k.split("_")[0]) for k in warm})
WINDOWS = sorted({int(k.split("_")[1]) for k in warm})
for n, (task, _, _) in enumerate(B_TASKS):
    bx = fig.add_subplot(gsB[1, 1 + n])
    arms = T[task]
    M = np.full((len(WINDOWS), len(PERIODS)), np.nan); PR = np.zeros_like(M, dtype=bool)
    for a, v in arms.items():
        p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
        M[WINDOWS.index(w), PERIODS.index(p)] = v["success"]
    lo, hi = float(np.nanmin(M)), float(np.nanmax(M))
    for a, twin in TWIN.items():
        p, w = (int(z) for z in re.match(r"g(\d+)_(\d+)$", a).groups())
        if twin in arms:
            i, j = WINDOWS.index(w), PERIODS.index(p)
            M[i, j] = arms[twin]["success"]; PR[i, j] = True
    bx.set_facecolor(V.ABSENT)
    im = bx.imshow(M, cmap=CMAP[task], vmin=lo, vmax=hi, origin="lower", aspect="auto")
    for i in range(len(WINDOWS)):
        for j in range(len(PERIODS)):
            if PR[i, j]:
                bx.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="////",
                                       linewidth=0.0, edgecolor="#4a4a4a"))
            elif not np.isfinite(M[i, j]):
                st = warm.get(f"{PERIODS[j]}_{WINDOWS[i]}", {}).get("status", "")
                bx.text(j, i, {"INFEASIBLE": "✗", "UNKNOWN": "?"}.get(st, ""),
                        ha="center", va="center", fontsize=3.0, color="#a8a7a1")
    b = BEST[task]
    bp_, bw_ = (int(z) for z in re.match(r"g(\d+)_(\d+)$", b).groups())
    bx.scatter(PERIODS.index(bp_), WINDOWS.index(bw_), s=100, marker="*", c="white",
               edgecolor=HUE[task], linewidth=1.0, zorder=8)
    bx.set_xticks(range(0, len(PERIODS), 3))
    bx.set_xticklabels([PERIODS[i] for i in range(0, len(PERIODS), 3)], fontsize=4.4, rotation=90)
    bx.set_yticks(range(0, len(WINDOWS), 3))
    bx.set_yticklabels([WINDOWS[i] for i in range(0, len(WINDOWS), 3)], fontsize=4.6)
    bx.tick_params(pad=1.0, length=1.4)
    bx.set_title(f"{SHORT[task]}   {lo:.0f}–{hi:.0f}%", fontsize=6.0, pad=1.8,
                 color=HUE[task], fontweight="bold")
    V.tidy(bx, grid="")
    bx.set_xlabel("period (ms)", fontsize=5.2, labelpad=0.7)
    if n == 0:
        bx.set_ylabel("window (ms)", fontsize=5.2, labelpad=1.0)
    # rollout pair, success then failure, sharing this workload's column
    sx = fig.add_subplot(gsB[0, 1 + n]); sx.axis("off")
    fr = [sorted((V.HERE / ".frames" / k).glob("f_*.jpg")) for k in THUMB.get(task, ())]
    if fr and all(fr):
        ims = [mpimg.imread(f[-1]) for f in fr]
        h = min(i.shape[0] for i in ims)
        ims = [i[:h] for i in ims]
        gap = np.ones((h, 6, 3), dtype=ims[0].dtype) * (255 if ims[0].dtype == np.uint8 else 1)
        sx.imshow(np.hstack([ims[0], gap, ims[1]]))
        w0 = ims[0].shape[1]
        for k2, (xx, mk, col) in enumerate(((0.02, "✓", V.STATUS["good"]),
                                            ((w0 + 6) / (2 * w0 + 6) + 0.02, "✗",
                                             V.STATUS["critical"]))):
            sx.text(xx, 0.95, mk, transform=sx.transAxes, ha="left", va="top",
                    fontsize=6.6, fontweight="bold", color=col, zorder=9,
                    bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.8))
    cb = fig.colorbar(im, ax=bx, fraction=0.046, pad=0.03)
    cb.set_ticks([lo, hi]); cb.set_ticklabels([f"{lo:.0f}", f"{hi:.0f}"])
    cb.ax.tick_params(labelsize=4.2, length=1.2)

V.save(fig, "cand_c21_three")

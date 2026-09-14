r"""C20 -- C18 over C19 as one page: the curated ladder, then the whole plane.

Panel A (top) is the curated widowx ladder -- nine schedules pruned to six, 30
seeds, success and calibrated energy per scene. Panel B (bottom) opens it out to
all 44 operating points and all four workloads.

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

Caption in cand_c20_caption.tex, not on the canvas.
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
SCENES = [("egg", "eggplant in basket"), ("spoon", "spoon on towel")]
THUMB = {"egg": ("runs_egg_pipe110_01", "runs_egg_cpu685_01"),
         "spoon": ("ladder_spoon_pipe110_21", "ladder_spoon_cpu685_21")}
BADGE = {"egg": "right", "spoon": "left"}
# ---------- B: the plane ----------
HUE = {"egg": "#1b7f5f", "spoon": "#2a78d6", "coke": "#c0392b", "drawer": "#7a3fa5"}
MARK = {"egg": "o", "spoon": "s", "coke": "^", "drawer": "D"}
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

fig = plt.figure(figsize=(7.16, 5.62), dpi=170)
gsA = fig.add_gridspec(1, 4, left=0.072, right=0.988, top=0.905, bottom=0.700, wspace=0.34)
gsB = fig.add_gridspec(2, 3, width_ratios=[1.42, 0.79, 0.79], left=0.068,
                       right=0.972, top=0.542, bottom=0.070, wspace=0.52, hspace=0.54)

fig.text(0.010, 0.983, "A", fontsize=10, fontweight="bold", va="top")
fig.text(0.030, 0.980, "curated schedule ladder — 6 arms × 30 seeds, widowx",
         fontsize=7.6, fontweight="bold", va="top", color=V.INK)
fig.text(0.010, 0.622, "B", fontsize=10, fontweight="bold", va="top")
fig.text(0.030, 0.619, "the full plane — 44 operating points × 10 seeds, all four workloads",
         fontsize=7.6, fontweight="bold", va="top", color=V.INK)

# ============================== A ==============================
axesA = [fig.add_subplot(gsA[i]) for i in range(4)]
for si, (task, tlab) in enumerate(SCENES):
    one = {a: next(iter(CUR[(task, A_KEY.get(a, a))].values())) for a in A_ARMS}
    per = {a: one[a]["issue_period_ms"] for a in A_ARMS}
    lat = {a: one[a]["latency_ms"] for a in A_ARMS}
    order = ["ideal"] + sorted([a for a in A_ARMS if a != "ideal"], key=lambda a: per[a])
    x = np.arange(len(order)); cols = [A_COL[a] for a in order]
    c_qf, p_idle = CAL[task]
    for mi, metric in enumerate(("success", "energy")):
        ax = axesA[si * 2 + mi]
        if metric == "success":
            pts = [np.asarray([100 * d["n_success"] / d["n_episodes"]
                               for d in CUR[(task, A_KEY.get(a, a))].values()], float)
                   for a in order]
            ylab, ttl = "success (%)", "success rate"
        else:
            pts = [np.asarray([np.median([c_qf * e["t2_qf_sub_arm"] + p_idle * e["duration_s"]
                                          for e in d["episodes"]])
                               for d in CUR[(task, A_KEY.get(a, a))].values()], float)
                   for a in order]
            ylab, ttl = "energy per episode (J)", "energy, calibrated"
        bp = ax.boxplot(pts, positions=x, widths=0.62, patch_artist=True, showmeans=True,
                        showfliers=False, zorder=3, medianprops=dict(color=V.INK, lw=1.0),
                        whiskerprops=dict(color=V.INK2, lw=0.7),
                        capprops=dict(color=V.INK2, lw=0.7),
                        meanprops=dict(marker="D", ms=2.6, mfc="white", mec=V.INK, mew=0.7))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor("white"); patch.set_linewidth(0.45)
        rng = np.random.default_rng(0)
        for i, v in enumerate(pts):
            ax.scatter(rng.normal(i, 0.075, len(v)), v, s=3.0, c=V.INK, alpha=0.30,
                       linewidths=0, zorder=5, clip_on=True)
        ax.set_ylim(0, min(100.0, max(v.max() for v in pts) * 1.62) if metric == "success"
                    else max(v.max() for v in pts) * 1.12)
        ax.set_ylabel(ylab, fontsize=6.4, labelpad=1.5)
        ax.set_title(ttl, fontsize=6.9, pad=2.4)
        ax.set_xlim(-0.62, len(order) - 0.38)
        ax.set_xticks(x); ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", labelsize=5.8)
        for i, a in enumerate(order):
            ax.text(i, -0.05 - 0.16 * (i % 2), f"{A_NAME[a]}\n{per[a]:.0f}/{lat[a]:.0f}",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=4.8, color=V.INK2, linespacing=1.16)
        V.tidy(ax, grid="y")
        if metric == "success":
            bb = ax.get_position(); aw, ah = bb.width * 7.16, bb.height * 5.62
            wf = 0.362; hf = wf * (aw / ah) * 0.75
            for j, key in enumerate(THUMB[task][::-1]):
                fs = sorted((V.HERE / ".frames" / key).glob("f_*.jpg"))
                ib = ax.inset_axes([1.0 - wf, 0.995 - hf - (1 - j) * (hf + 0.020), wf, hf],
                                   zorder=6 + j)
                ib.imshow(mpimg.imread(fs[-1]), aspect="auto")
                ib.set_xticks([]); ib.set_yticks([])
                for sp in ib.spines.values():
                    sp.set_edgecolor("white"); sp.set_linewidth(1.1)
                bxp = 0.945 if BADGE[task] == "right" else 0.055
                ib.text(bxp, 0.93, "✓" if j else "✗", transform=ib.transAxes,
                        ha=BADGE[task], va="top", fontsize=5.8, fontweight="bold", zorder=9,
                        color=V.STATUS["good"] if j else V.STATUS["critical"],
                        bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.8))
    a0 = axesA[si * 2].get_position(); a1 = axesA[si * 2 + 1].get_position()
    fig.text((a0.x0 + a1.x1) / 2, 0.940, tlab, ha="center", va="top",
             fontsize=7.2, fontweight="bold", color=V.INK)

# ============================== B ==============================
ax = fig.add_subplot(gsB[:, 0])
ORDER = ["coke", "drawer", "spoon", "egg"]
BEST = {t: max(T[t], key=lambda k: T[t][k]["success"]) for t in ORDER}
for zi, task in enumerate(ORDER):
    arms = T[task]; ks = sorted(arms)
    e = np.array([EJ[task][a] for a in ks]); sv = np.array([arms[a]["success"] for a in ks])
    hollow = PROXY[task]
    ax.scatter(e, sv, s=14, marker=MARK[task], c="none" if hollow else HUE[task],
               edgecolor=HUE[task], linewidth=0.7 if hollow else 0.30, zorder=4 + zi,
               label=SHORT[task] + (" (proxy)" if hollow else ""))
    b = BEST[task]
    ax.scatter(EJ[task][b], arms[b]["success"], s=74, marker="*", c="white",
               edgecolor=HUE[task], linewidth=1.0, zorder=12 + zi)
ax.set_xscale("log")
ax.set_xlabel("calibrated energy per episode (J), log", fontsize=6.4, labelpad=1.5)
ax.set_ylabel("success rate (%)", fontsize=6.6, labelpad=2.0)
ax.tick_params(labelsize=5.8)
ax.set_title("success vs calibrated energy", fontsize=6.9, pad=2.6)
ax.margins(0.10); ax.set_ylim(0, ax.get_ylim()[1]); V.tidy(ax, grid="both")
h, lb = ax.get_legend_handles_labels()
o = [lb.index(SHORT[t] + (" (proxy)" if PROXY[t] else "")) for t, _, _ in L.TASKS]
ax.legend([h[i] for i in o], [lb[i] for i in o], fontsize=5.4, loc="lower left",
          handletextpad=0.25, borderpad=0.3, labelspacing=0.22, markerscale=0.95)

PERIODS = sorted({int(k.split("_")[0]) for k in warm})
WINDOWS = sorted({int(k.split("_")[1]) for k in warm})
fig.text(0.735, 0.578, "SUCCESS RATE on the schedule plane, per workload",
         ha="center", va="top", fontsize=6.8, fontweight="bold", color=V.INK)
for n, (task, _, _) in enumerate(L.TASKS):
    bx = fig.add_subplot(gsB[n // 2, 1 + n % 2])
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
                        ha="center", va="center", fontsize=3.2, color="#a8a7a1")
    b = BEST[task]
    bp_, bw_ = (int(z) for z in re.match(r"g(\d+)_(\d+)$", b).groups())
    bx.scatter(PERIODS.index(bp_), WINDOWS.index(bw_), s=110, marker="*", c="white",
               edgecolor=HUE[task], linewidth=1.0, zorder=8)
    bx.set_xticks(range(0, len(PERIODS), 2))
    bx.set_xticklabels([PERIODS[i] for i in range(0, len(PERIODS), 2)], fontsize=4.4, rotation=90)
    bx.set_yticks(range(0, len(WINDOWS), 2))
    bx.set_yticklabels([WINDOWS[i] for i in range(0, len(WINDOWS), 2)], fontsize=4.6)
    bx.tick_params(pad=1.0, length=1.4)
    bx.set_title(f"{SHORT[task]}   {lo:.0f}–{hi:.0f}%", fontsize=6.0, pad=1.8,
                 color=HUE[task], fontweight="bold")
    V.tidy(bx, grid="")
    if n // 2 == 1:
        bx.set_xlabel("period (ms)", fontsize=5.4, labelpad=0.8)
    if n % 2 == 0:
        bx.set_ylabel("window (ms)", fontsize=5.4, labelpad=1.2)
    cb = fig.colorbar(im, ax=bx, fraction=0.048, pad=0.03)
    cb.set_ticks([lo, hi]); cb.set_ticklabels([f"{lo:.0f}", f"{hi:.0f}"])
    cb.ax.tick_params(labelsize=4.4, length=1.2)

V.save(fig, "cand_c20_combined")

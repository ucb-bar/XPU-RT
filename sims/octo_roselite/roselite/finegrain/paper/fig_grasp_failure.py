#!/usr/bin/env python3
"""Latency does not stop the robot touching the object. It stops it CLOSING on it.

The companion to fig_trajectory.py: the same quantities, over every episode instead of
one, so the reading of the ladder is a measurement and not an anecdote.

  A  WHAT SURVIVES AND WHAT DOES NOT. Three MEASURED rates per schedule, over the full
     sweep (240 episodes per arm = 10 replicates x 24 configs). "Object moved > 3 cm"
     barely falls across the whole latency range; "grasped" collapses; task success
     follows "grasped", not "moved". The failure is specifically the GRASP.

  B  WHY. For every episode, the MEASURED 3D distance from the end-effector to the
     object at the instant the object first moves. On the ideal arm the gripper is on
     the object; by the CPU-only arm it is centimetres away and getting further -- the
     object is being SWIPED in passing, not picked up. This is also the number that
     makes drawing an object event on the end-effector wrong (see fig_trajectory.py).

  C  AND BY WHAT. The robot link whose centre of mass is closest to the object at that
     same instant, over all 14 widowx links. It is the gripper frame when the arm is
     controlled and increasingly a FINGER or something further up when it is not -- so
     an end-effector-only trajectory figure cannot account for all the object motion.

Panels A come from the 240-episode sweep (g5fine); panels B and C need per-tick object
and link positions and so come from the 24-episode trace set, whose per-arm rates are
consistent with the sweep. n is printed everywhere, because it varies: an arm that
never moves the object contributes no distance.

Success intervals in A are DESIGN-CORRECTED for clustering by episode config, the same
estimator as fig_metrics.py: episodes repeat the same 24 configs, so a naive binomial
overstates precision by 1 + (m-1) * ICC.

Medians, not means, in B: the distribution has a long right tail (one MEASURED case at
16.1 cm) and a mean would be steered by it.
"""
from __future__ import annotations
import json, glob, math, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).parent.parent
T = HERE / "traces_torque2"
OUT = Path(__file__).parent / "fig_grasp_failure.png"

FAM = {"ideal": "#7f8c8d", "pipelined": "#16a085",
       "serial": "#e67e22", "CPU only": "#c0392b"}
# arm -> family. Ages and rates are all read from the runs' own summary.json.
SWEEP_FAM = {"lat0": "ideal", "pipe110fix": "pipelined", "p105w300": "pipelined",
             "p130w275": "pipelined", "p150w300": "pipelined", "pipe200fix": "pipelined",
             "serial283": "serial", "fp32_555": "CPU only", "cpu685": "CPU only"}
# traces_torque2 covers all 9 arms for every task, so the trace panels now span the
# SAME arm set as the sweep panel. (The old traces_torque tree had only 6, under the
# mislabelled pipe110/pipe200 latencies.)
TRACE_FAM = dict(SWEEP_FAM)
SERIES = [("moved_correct_obj", "object moved > 3 cm", "#1c6dd0", "o"),
          ("is_src_obj_grasped", "GRASPED", "#8e44ad", "s"),
          ("src_on_target", "task success", "#111111", "^")]
GRIP = {"ee_gripper_link", "gripper_link", "gripper_bar_link", "gripper_prop_link",
        "ee_arm_link"}
FINGER = {"fingers_link", "left_finger_link", "right_finger_link"}


def design_ci(rows):
    """95% CI corrected for clustering by episode config (same as fig_metrics.py)."""
    by = collections.defaultdict(list)
    for e, ok in rows:
        by[e].append(ok)
    n, K = len(rows), len(by)
    p = sum(ok for _, ok in rows) / n
    if K < 2:
        return 100 * p, 100 * p, 100 * p
    ns = [len(v) for v in by.values()]
    m = sum(ns) / K
    msb = sum(len(v) * (sum(v) / len(v) - p) ** 2 for v in by.values()) / (K - 1)
    msw = sum(sum((z - sum(v) / len(v)) ** 2 for z in v) for v in by.values()) / max(n - K, 1)
    den = msb + (m - 1) * msw
    icc = max((msb - msw) / den, 0.0) if den > 0 else 0.0
    se = math.sqrt(max(p * (1 - p), 1e-12) / (n / (1 + (m - 1) * icc)))
    return 100 * p, 100 * max(p - 1.96 * se, 0), 100 * min(p + 1.96 * se, 1)


# ------------------------------------------------------- A: the 240-ep sweep
sweep = collections.defaultdict(list)     # arm -> [(episode_id, stats dict, success)]
sweep_age = collections.defaultdict(list)
for f in glob.glob(str(HERE / "g5fine" / "runs" / "*" / "*" / "summary.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    n = Path(f).parent.name
    if "_rng" not in n:
        continue
    head, _ = n.rsplit("_rng", 1)
    task, arm = head.split("_", 1)
    if task != "egg" or arm not in SWEEP_FAM:
        continue
    sweep_age[arm].append(d["age_mean_ms"])
    for e in d.get("episodes", []):
        sweep[arm].append((e["episode_id"], e.get("episode_stats", {}), bool(e["success"])))

A_ARMS = sorted(sweep, key=lambda a: np.median(sweep_age[a]))
A_AGE = [float(np.median(sweep_age[a])) for a in A_ARMS]

# ------------------------------------------- B, C: the per-tick trace set
def tdir(arm):
    return T / f"egg_{arm}"


dist, near, tr_age, tr_n = {}, {}, {}, {}
for arm in TRACE_FAM:
    d = tdir(arm)
    if not (d / "summary.json").exists():
        continue
    s = json.load(open(d / "summary.json"))
    names = json.load(open(d / "ep00_bodies.json"))["link_names"]
    tr_age[arm] = s["age_mean_ms"]
    tr_n[arm] = s["n_episodes"]
    dd, nn = [], []
    for e in s["episodes"]:
        ep = e["episode_id"]
        j = json.load(open(d / f"ep{ep:02d}_trace.json"))
        t = next((x["tick"] for x in j["events"] if x["stats"].get("moved_correct_obj")), None)
        if t is None:
            continue
        ee = np.load(d / f"ep{ep:02d}_ee_xyz.npy")
        ob = np.load(d / f"ep{ep:02d}_obj_xyz.npy")
        com = np.load(d / f"ep{ep:02d}_link_com.npy")
        t = min(t, len(ee) - 1)
        dd.append(100.0 * float(np.linalg.norm(ee[t] - ob[t])))
        dl = np.linalg.norm(com[t] - ob[t][None, :], axis=1)
        nn.append(names[int(np.argmin(dl))])
    dist[arm], near[arm] = np.array(dd), nn

B_ARMS = sorted(dist, key=lambda a: tr_age[a])

# --------------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.6), dpi=150,
                         gridspec_kw={"wspace": 0.215, "left": 0.048, "right": 0.995,
                                      "top": 0.855, "bottom": 0.335})

ax = axes[0]
# Categorical x, ORDERED by observation age -- deliberately not a continuous age axis.
# Success is not a function of age alone (that is the whole point of fig_operating_points),
# so a line over an age axis would assert a relationship the data contradicts: serial283
# at 405 ms scores below pipe110fix at 421 ms because their cadences differ.
xa = np.arange(len(A_ARMS))
LOFF = {"moved_correct_obj": (0, 11), "is_src_obj_grasped": (0, -15), "src_on_target": (0, 11)}
for key, lab, col, mk in SERIES:
    mu, lo, hi = [], [], []
    for a in A_ARMS:
        rows = [(e, bool(st.get(key, ok)) if key != "src_on_target" else ok)
                for e, st, ok in sweep[a]]
        m_, l_, h_ = design_ci(rows)
        mu.append(m_)
        lo.append(l_)
        hi.append(h_)
    ax.fill_between(xa, lo, hi, color=col, alpha=0.16, zorder=2)
    ax.plot(xa, mu, marker=mk, ms=6.5, lw=2.2, color=col, label=lab, zorder=4,
            mfc="white", mew=1.8)
    for i in (0, len(xa) - 1):
        dx, dy = LOFF[key]
        ax.annotate(f"{mu[i]:.1f}%", (xa[i], mu[i]), textcoords="offset points",
                    xytext=(dx + (7 if i == 0 else -7), dy),
                    ha="left" if i == 0 else "right", fontsize=7.6, color=col,
                    fontweight="bold")
ax.set_xticks(xa)
ax.set_xticklabels([f"{a}\n{g:.0f} ms" for a, g in zip(A_ARMS, A_AGE)], fontsize=7.0,
                   rotation=32, ha="right")
for lbl, a in zip(ax.get_xticklabels(), A_ARMS):
    lbl.set_color(FAM[SWEEP_FAM[a]])
ax.set_xlabel("schedules ORDERED by MEASURED observation age (not an age axis)", fontsize=9)
ax.set_ylabel("% of episodes  (design-corrected 95% CI)")
ax.set_title("A · the object still moves; nothing closes on it\n"
             "eggplant, 240 episodes per arm", fontsize=10.5)
ax.set_ylim(0, 108)
ax.legend(fontsize=8.4, loc="lower left", framealpha=0.95)

ax = axes[1]
bp = ax.boxplot([dist[a] for a in B_ARMS], patch_artist=True, widths=0.6,
                showfliers=False, medianprops=dict(color="black", lw=1.7))
for p_, a in zip(bp["boxes"], B_ARMS):
    p_.set_facecolor(FAM[TRACE_FAM[a]])
    p_.set_alpha(0.82)
    p_.set_edgecolor("black")
rng = np.random.default_rng(0)
for i, a in enumerate(B_ARMS):
    ax.scatter(rng.normal(i + 1, 0.055, len(dist[a])), dist[a], s=11, c="black",
               alpha=0.42, zorder=5, clip_on=True)
    ax.text(i + 1, np.percentile(dist[a], 75) + 0.30, f"med {np.median(dist[a]):.1f}",
            ha="center", va="bottom", fontsize=7.2, fontweight="bold")
worst = max(((v, a) for a in B_ARMS for v in dist[a]))
# Anchor the callout INWARD when the worst point sits on one of the last arms,
# otherwise a right-anchored label runs off the axis (it did).
_wi = B_ARMS.index(worst[1]) + 1
_right = _wi > len(B_ARMS) * 0.6
ax.annotate(f"worst single episode\n{worst[0]:.1f} cm — nothing was touching it",
            (_wi, worst[0]), textcoords="offset points",
            xytext=(-14, 10) if _right else (14, 2),
            ha="right" if _right else "left", va="center", fontsize=7.2, style="italic",
            color="#555", arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
ax.scatter([B_ARMS.index(worst[1]) + 1], [worst[0]], s=70, facecolor="none",
           edgecolor="#c0392b", linewidth=1.8, zorder=7)
ax.set_ylabel("MEASURED 3D end-effector → object distance (cm)")
ax.set_title("B · at the instant the object first moves,\nthe gripper is not there",
             fontsize=10.5)
ax.set_ylim(0, max(worst[0] * 1.16, 6))

ax = axes[2]
cats = [("gripper frame", "#16a085"), ("finger", "#e67e22"), ("further up the arm", "#c0392b")]
bot = np.zeros(len(B_ARMS))
for lab, col in cats:
    fr = []
    for a in B_ARMS:
        n = len(near[a])
        c = sum(1 for x in near[a] if (x in GRIP if lab == "gripper frame"
                                       else x in FINGER if lab == "finger"
                                       else x not in GRIP and x not in FINGER))
        fr.append(100.0 * c / n if n else 0.0)
    fr = np.array(fr)
    ax.bar(np.arange(len(B_ARMS)) + 1, fr, 0.62, bottom=bot, color=col,
           edgecolor="black", linewidth=0.5, zorder=3, label=lab)
    for i, v in enumerate(fr):
        if v >= 7:
            ax.text(i + 1, bot[i] + v / 2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=7.2, color="white", fontweight="bold")
    bot += fr
ax.set_ylabel("% of object-motion events")
ax.set_title("C · and it is not always the gripper that moved it\n"
             "nearest of all 14 links, in 3D", fontsize=10.5)
ax.set_ylim(0, 124)
ax.legend(fontsize=8.0, loc="upper center", framealpha=0.95, ncol=3, columnspacing=0.9,
          title="nearest link at that instant", title_fontsize=8.0)

for a_, arms, ages, famm in ((axes[1], B_ARMS, None, TRACE_FAM),
                             (axes[2], B_ARMS, None, TRACE_FAM)):
    a_.set_xticks(np.arange(len(arms)) + 1)
    a_.set_xticklabels([f"{a}\n{tr_age[a]:.0f} ms\nn={len(dist[a])}/{tr_n[a]}"
                        for a in arms], fontsize=7.0)
    for lbl, a in zip(a_.get_xticklabels(), arms):
        lbl.set_color(FAM[famm[a]])
for a_ in axes:
    a_.grid(True, axis="y", alpha=0.3, zorder=0)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)

mv = [design_ci([(e, bool(st.get("moved_correct_obj", False))) for e, st, _ in sweep[a]])[0]
      for a in A_ARMS]
gr = [design_ci([(e, bool(st.get("is_src_obj_grasped", False))) for e, st, _ in sweep[a]])[0]
      for a in A_ARMS]
fig.suptitle(f"Across the whole SCHEDULE range the eggplant still gets moved "
             f"({mv[0]:.1f}% → {mv[-1]:.1f}%) and almost never gets grasped "
             f"({gr[0]:.1f}% → {gr[-1]:.1f}%).",
             fontsize=13.0, y=0.974)

fig.text(0.5, 0.012, "\n".join([
    "Panel A is the 240-episode sweep (10 replicates x 24 configs per arm). Intervals are DESIGN-CORRECTED for clustering by episode config, the same estimator as fig_metrics.py;",
    "a naive binomial would be about half as wide. Arm labels are coloured by schedule family, and the x axis is an ORDERING by observation age, NOT an age axis — success is not a",
    "function of observation age alone (that is what fig_operating_points is for), so the connecting line is a reading aid, not a fitted relationship.",
    "Panels B and C need per-tick object and link positions, which only the 24-episode trace set carries, so n there is at most 24 and is printed per arm. n varies because an arm that",
    "never moves the object contributes no distance — which biases B DOWNWARD for the slow arms, against the effect shown. Every distance is a 3D Euclidean distance between",
    "MEASURED positions at the same tick, not a distance in the projected image. Event ticks are sampled every 5, so the instant is located to ±0.2 s. Medians, not means: the tail is long.",
    "In C the nearest-link search covers all 14 links; 'further up the arm' never wins, which is itself the finding — the contact is a finger or the gripper frame, never an elbow.",
]), ha="center", va="bottom", fontsize=7.7, style="italic", color="#555", linespacing=1.5)

fig.savefig(OUT)
print(f"[ok] {OUT}")
for a, g in zip(A_ARMS, A_AGE):
    r = {k: design_ci([(e, bool(st.get(k, ok)) if k != "src_on_target" else ok)
                       for e, st, ok in sweep[a]])[0] for k, _, _, _ in SERIES}
    print(f"  sweep {a:11s} age {g:7.1f}  n={len(sweep[a]):4d}  " +
          "  ".join(f"{k}={v:5.1f}%" for k, v in r.items()))
for a in B_ARMS:
    c = collections.Counter("gripper" if x in GRIP else "finger" if x in FINGER else "other"
                            for x in near[a])
    print(f"  trace {a:11s} age {tr_age[a]:7.1f}  n={len(dist[a]):2d}/{tr_n[a]}  "
          f"ee-obj median {np.median(dist[a]):5.2f} cm  max {dist[a].max():5.2f}  nearest {dict(c)}")
print(f"  worst single episode: {worst[0]:.1f} cm on {worst[1]}")

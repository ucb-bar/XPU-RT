#!/usr/bin/env python3
"""What the latency does to the robot, in the scene the policy saw.

Rows are TASKS, columns are the four schedule families at their MEASURED observation
age. Every panel is one rollout of the SAME episode configuration, projected onto that
rollout's own tick-0 frame with the camera's own intrinsic_cv / extrinsic_cv.

The grasp row falls apart down the ladder and the push row does not. That contrast is
the result; the panels are the mechanism.

FOUR THINGS THIS FIGURE DOES THAT THE DRAFT DID NOT
-------------------------------------------------
1. OBJECT EVENTS ARE ANCHORED TO THE OBJECT. `moved_correct_obj` is a statement about
   the OBJECT (it travelled > 3 cm), not about the gripper. Drawing it on the
   end-effector implies a contact that need not exist -- MEASURED here, the
   end-effector is a median 1.4 cm from the eggplant at that instant on the ideal arm
   and 4.8 cm on the CPU-only arm, and in the worst single case in this dataset it is
   16.1 cm away. Object events are therefore drawn on the OBJECT's own path, arm
   events on the arm's, and every object event carries a dashed leader to the nearest
   ROBOT LINK with the measured distance printed.

2. THE OBJECT HAS ITS OWN PATH. The background is the tick-0 frame, so object motion
   is otherwise invisible: a still frame plus an end-effector path silently asserts
   that the scene never moved.

3. CONTACT CAN COME FROM ANY LINK. The leader in (1) searches all 14 (widowx) link
   centres of mass, not just the gripper, and the arm's whole kinematic chain is drawn
   at the instant of the object event. On the CPU-only arm the nearest link at that
   instant is often a FINGER rather than the gripper frame -- the object is swiped,
   not grasped, and an end-effector-only figure cannot show that.

4. NO SINGLE EPISODE CARRIES A CLAIM. The harness is not run-to-run deterministic
   (NONDETERMINISM.md: three byte-identical invocations gave False / True / False), so
   each panel states BOTH the arm's aggregate success rate over the whole 24-config
   set AND, separately, the outcome of the one rollout drawn. The draft put only the
   single rollout's SUCCESS / FAILURE in the title, which is exactly the claim the
   harness cannot support.

Episode selection is disclosed on the figure and is deliberate, not random: the grasp
episode is one where every arm has an object-motion event to anchor (so the reader can
compare the anchoring across the ladder); the push episode is one where all four
rollouts succeed (which is the point of that row). The arm-level rates printed on
every panel are over all 24 configs and are NOT selected.

Caveat that cannot be removed: this is a 3D path projected to 2D, so an apparent
crossing is not necessarily a contact and depth is only implied. The measured 3D
distances printed next to each object event are the antidote.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import PIL.Image as PILImage

HERE = Path(__file__).parent.parent
T = HERE / "traces_torque2"
OUT = Path(__file__).parent / "fig_trajectory.png"

FAM = {"ideal": "#7f8c8d", "pipelined": "#16a085",
       "serial": "#e67e22", "CPU only": "#c0392b"}
OBJ_C = "#1c6dd0"          # object: outside the inferno ramp and the family palette
SKEL_C = "#00d1c1"
AGE_CMAP = "inferno"

# (arm dir suffix, family, short label). Latency / cadence / rates are read from the
# run's own summary.json -- nothing about the schedule is hardcoded here.
ARMS = [("lat0", "ideal", "IDEAL\n0 ms"),
        ("p150w300", "pipelined", "OURS\np150/w300"),
        ("pipe200fix", "pipelined", "PIPELINED\n219 ms cadence"),
        ("serial283", "serial", "3-WAY SERIAL"),
        ("cpu685", "CPU only", "CPU-ONLY")]

# Episode choice, and why (printed on the figure).
# SELECTION CRITERION, stated so it can be checked: a config where the ideal arm AND
# the recommended schedule both succeed, while the serial and CPU-only arms both fail.
# That is the degradation this work claims -- a faster SCHEDULE holds the task together
# where a slower one loses it -- and it cannot be shown by a config the method also fails.
# The previous selection maximised visible object-motion EVENTS instead, which happened
# to pick a config the method failed, and so illustrated the opposite of the result.
# 4 of 24 egg configs and 10 of 24 spoon configs qualify; these are the ones where the
# WHOLE fast group succeeds and the WHOLE slow group fails (pattern 11111 0000).
ROWS = [
    ("egg", 18, "eggplant into basket  ·  GRASP",
     "1 of 4 configs where ideal+fast succeed and serial+CPU both fail"),
    ("spoon", 6, "spoon on towel  ·  GRASP",
     "1 of 10 such configs on spoon; every fast arm succeeds, every slow arm fails"),
]

ARM_EV = {"gripper_closed":     ("gripper closes", "v"),
          "is_src_obj_grasped": ("grasped", "*"),
          "consecutive_grasp":  ("held 1 s", "P")}
OBJ_EV = {"moved_correct_obj": ("object moves > 3 cm", "o"),
          "moved_wrong_obj":   ("WRONG object moves", "X"),
          "src_on_target":     ("object on target", "D")}
# Kinematic chains, by link NAME -- the two robots have different link sets.
# Drawn from the upper arm outward, and further clipped to the links that actually
# project INSIDE the frame -- this camera does not see the robot base, so a
# proximal COM would draw a long line to nowhere. The NEAREST-LINK SEARCH that
# anchors each object event still covers EVERY link, in 3D, not just these.
CHAINS = [
    ["upper_arm_link", "upper_forearm_link", "lower_forearm_link", "wrist_link",
     "gripper_link", "ee_gripper_link"],
    ["link_bicep", "link_elbow", "link_forearm", "link_wrist", "link_gripper",
     "link_gripper_tcp"],
]


# traces_torque2 uses a uniform <task>_<arm> layout for all 4 tasks x 9 arms; the
# old traces_torque tree named the eggplant baselines without a prefix and used "drw".
TASKDIR = {"egg": "egg", "drw": "drawer", "spoon": "spoon", "coke": "coke"}


def dirof(task, arm):
    return T / f"{TASKDIR[task]}_{arm}"


def load(task, arm, ep):
    d = dirof(task, arm)
    j = json.load(open(d / f"ep{ep:02d}_trace.json"))
    s = json.load(open(d / "summary.json"))
    K = np.asarray(j["camera_param"]["intrinsic_cv"], float)
    E = np.asarray(j["camera_param"]["extrinsic_cv"], float)
    xyz = np.load(d / f"ep{ep:02d}_ee_xyz.npy")
    obj = np.load(d / f"ep{ep:02d}_obj_xyz.npy")
    com = np.load(d / f"ep{ep:02d}_link_com.npy")
    age = np.load(d / f"ep{ep:02d}_action_age_ms.npy")[:len(xyz)]
    names = json.load(open(d / f"ep{ep:02d}_bodies.json"))["link_names"]
    epi = {e["episode_id"]: e for e in s["episodes"]}
    return dict(j=j, s=s, K=K, E=E, xyz=xyz, obj=obj, com=com, names=names,
                age=np.nan_to_num(age, nan=0.0), tick=j["tick_ms"],
                bg=np.asarray(PILImage.open(d / f"ep{ep:02d}_bg.png"), float),
                roll_ok=bool(j["success"]), epi=epi.get(ep, {}),
                arm_sr=100.0 * s["success_rate"], n_ep=s["n_episodes"],
                age_mean=s["age_mean_ms"], period=s["issue_period_ms"],
                agree=sum(1 for e in s["episodes"]
                          if e["success"] == json.load(
                              open(d / f"ep{e['episode_id']:02d}_trace.json"))["success"]))


def project(P, K, E):
    """world -> camera (extrinsic_cv) -> pixels (intrinsic_cv). NaN behind the lens."""
    P = np.atleast_2d(np.asarray(P, float))
    cam = np.c_[P, np.ones(len(P))] @ E.T
    uvw = cam[:, :3] @ K.T
    w = uvw[:, 2:3].copy()
    w[np.abs(w) < 1e-9] = np.nan
    uv = uvw[:, :2] / w
    uv[cam[:, 2] <= 0] = np.nan
    return uv


def first_true(tr, key):
    for e in tr["j"]["events"]:
        if e["stats"].get(key):
            return e["tick"]
    return None


def gripper_close_tick(tr):
    g = [(e["tick"], e["grip"]) for e in tr["j"]["events"]]
    for a, b in zip(g, g[1:]):
        if a[1] >= 0.5 > b[1]:
            return b[0]
    return None


def cased(ax, uv, colour, lw, z, ls="-"):
    ax.plot(uv[:, 0], uv[:, 1], color="white", lw=lw + 1.8, ls=ls, zorder=z,
            solid_capstyle="round", alpha=0.85)
    ax.plot(uv[:, 0], uv[:, 1], color=colour, lw=lw, ls=ls, zorder=z + 1,
            solid_capstyle="round")


def age_path(ax, uv, age, vmax, lw=2.6, z=3):
    ok = np.isfinite(uv).all(1)
    ax.plot(uv[ok, 0], uv[ok, 1], color="white", lw=lw + 2.0, zorder=z,
            solid_capstyle="round", alpha=0.9)
    seg = np.stack([uv[:-1], uv[1:]], axis=1)
    lc = LineCollection(seg, cmap=AGE_CMAP, linewidths=lw, zorder=z + 1)
    lc.set_array(age[:len(seg)])
    lc.set_clim(0, vmax)
    ax.add_collection(lc)
    return lc


# ------------------------------------------------------------------ assemble
DATA = {(t, a): load(t, a, ep) for t, ep, _, _ in ROWS for a, _, _ in ARMS}
VMAX = 100 * np.ceil(max(d["age"].max() for d in DATA.values()) / 100.0)

# The four columns must be the SAME scene. Verify it geometrically -- the tick-0 PNGs
# differ by a few least-significant bits from render nondeterminism, so a pixel hash
# is the wrong test.
SCENE_MM = {}
for task, ep, _, _ in ROWS:
    p0 = [DATA[(task, a)]["xyz"][0] for a, _, _ in ARMS]
    o0 = [DATA[(task, a)]["obj"][0] for a, _, _ in ARMS]
    e = 1000.0 * float(np.nanmax(np.ptp(np.asarray(p0), axis=0)))
    o = 0.0 if np.isnan(o0[0]).all() else 1000.0 * float(np.nanmax(np.ptp(np.asarray(o0), axis=0)))
    SCENE_MM[task] = max(e, o)
    assert SCENE_MM[task] < 1.0, f"{task}: columns are not the same scene ({SCENE_MM[task]:.3f} mm)"

hs = [DATA[(t, ARMS[0][0])]["bg"].shape[0] / DATA[(t, ARMS[0][0])]["bg"].shape[1]
      for t, _, _, _ in ROWS]
fig = plt.figure(figsize=(18.2, 10.4), dpi=145)
gs = fig.add_gridspec(len(ROWS), len(ARMS), height_ratios=hs, hspace=0.30, wspace=0.035,
                      left=0.031, right=0.935, top=0.885, bottom=0.140)

for r, (task, ep, tlab, why) in enumerate(ROWS):
    for c, (arm, fam, alab) in enumerate(ARMS):
        tr = DATA[(task, arm)]
        ax = fig.add_subplot(gs[r, c])
        h, w = tr["bg"].shape[:2]
        ax.imshow(np.clip(0.62 * tr["bg"] + 0.38 * 255, 0, 255).astype(np.uint8))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(FAM[fam])
            sp.set_linewidth(2.2)

        uv_ee = project(tr["xyz"], tr["K"], tr["E"])
        lc = age_path(ax, uv_ee, tr["age"], VMAX)
        ax.scatter(*uv_ee[0], s=95, marker="s", c="white", edgecolor=FAM[fam],
                   linewidth=2.0, zorder=7)

        rows_box, has_obj = [], not np.isnan(tr["obj"]).all()
        if has_obj:
            uv_ob = project(tr["obj"], tr["K"], tr["E"])
            cased(ax, uv_ob, OBJ_C, 2.4, 4)
            ax.scatter(*uv_ob[0], s=80, marker="o", facecolor="white",
                       edgecolor=OBJ_C, linewidth=2.0, zorder=7)
            ax.scatter(*uv_ob[-1], s=90, marker="o", c=OBJ_C, edgecolor="white",
                       linewidth=1.4, zorder=7)
            trav = 100 * float(np.linalg.norm(tr["obj"][-1] - tr["obj"][0]))
            rows_box.append(("", OBJ_C, f"object path — travelled {trav:.1f} cm", None))

        # ---- object-anchored events, each with a leader to the nearest LINK
        for key, (lab, mk) in OBJ_EV.items():
            t = first_true(tr, key)
            if t is None or not has_obj:
                continue
            t = min(t, len(tr["xyz"]) - 1)
            o = tr["obj"][t]
            dl = np.linalg.norm(tr["com"][t] - o[None, :], axis=1)
            k = int(np.argmin(dl))
            uo, ul = project(o, tr["K"], tr["E"])[0], project(tr["com"][t][k], tr["K"], tr["E"])[0]
            ax.plot([uo[0], ul[0]], [uo[1], ul[1]], color=OBJ_C, lw=1.2, ls=(0, (3, 2)),
                    zorder=8, alpha=0.95)
            ax.scatter(*ul, s=26, marker="o", c="white", edgecolor=OBJ_C, linewidth=1.2,
                       zorder=9)
            ax.scatter(*uo, s=190 if mk in "DX" else 150, marker=mk, c=OBJ_C,
                       edgecolor="white", linewidth=1.6, zorder=10)
            rows_box.append((mk, OBJ_C,
                             f"{lab}  {t * tr['tick'] / 1000:.1f}s → {tr['names'][k].replace('_link', '')}"
                             f" {100 * dl[k]:.1f}cm",
                             None))
            if key == "moved_correct_obj":       # draw the arm that (maybe) did it
                chain = next((ch for ch in CHAINS
                              if sum(n in tr["names"] for n in ch) >= 4), None)
                if chain:
                    idx = [tr["names"].index(n) for n in chain if n in tr["names"]]
                    uvc = project(tr["com"][t][idx], tr["K"], tr["E"])
                    # The widowx base is mounted OUTSIDE this camera's view, so the
                    # proximal links project off-frame; drawing the leader to them
                    # would put a long line across the panel that means nothing.
                    off = ((uvc[:, 0] < 0) | (uvc[:, 0] > w) |
                           (uvc[:, 1] < 0) | (uvc[:, 1] > h))
                    uvc[off] = np.nan
                    # Dashed, thin and low-contrast on purpose: this is the arm's
                    # POSE at one instant, not a path, and it must not be mistaken
                    # for one.
                    ax.plot(uvc[:, 0], uvc[:, 1], color="white", lw=3.0, zorder=5,
                            alpha=0.45, solid_capstyle="round")
                    ax.plot(uvc[:, 0], uvc[:, 1], color=SKEL_C, lw=1.2, zorder=6,
                            ls=(0, (5, 2)), marker="o", ms=2.6, alpha=0.80)
                ax.scatter(*ul, s=125, marker="o", facecolor="none", edgecolor=SKEL_C,
                           linewidth=1.8, zorder=9)

        # ---- arm-anchored events
        for key, (lab, mk) in ARM_EV.items():
            # The gripper channel's sign convention is only established for the widowx
            # tasks; the google_robot drawer trace exposes no object events either, so
            # nothing arm-anchored is claimed there.
            if not has_obj:
                break
            t = gripper_close_tick(tr) if key == "gripper_closed" else first_true(tr, key)
            if t is None:
                continue
            t = min(t, len(tr["xyz"]) - 1)
            ax.scatter(*uv_ee[t], s=230 if mk == "*" else 130, marker=mk, c="#111",
                       edgecolor="white", linewidth=1.4, zorder=10)
            rows_box.append((mk, "#111", f"{lab}  {t * tr['tick'] / 1000:.1f}s", None))

        if not has_obj:      # the drawer evaluator exposes only the joint position
            q = tr["epi"].get("episode_stats", {}).get("qpos")
            if q is not None:
                rows_box.append(("", "#111",
                                 f"drawer joint {float(q):.3f} m  (closed ≤ 0.050)", None))

        handles = [Line2D([], [], ls="", marker=mk or "None", ms=8.5, color=col,
                          mec="white", mew=1.1, label=txt)
                   for mk, col, txt, _ in rows_box]
        if handles:
            pts = uv_ee[np.isfinite(uv_ee).all(1)]
            key = [uv_ee[0]]
            if has_obj:
                po = uv_ob[np.isfinite(uv_ob).all(1)]
                pts = np.vstack([pts, po])
                key += [uv_ob[0], uv_ob[-1]]
            # Put the box in whichever corner the drawn elements use least. Single
            # glyphs (path starts, the object's endpoints) count heavily: they are
            # one point each but hiding one loses the whole reading.
            key = np.asarray([k for k in key if np.isfinite(k).all()])
            pts = np.vstack([pts] + [key] * 40) if len(key) else pts
            # Score the four CORNERS by what the box would actually cover, not by
            # quadrant: the box is wider than half the panel, so a quadrant test
            # picks corners that still occlude.
            bw, bh = 0.62 * w, 0.055 * h * max(len(handles), 2) + 0.05 * h
            corners = {"lower left":  (0, bw, h - bh, h),
                       "lower right": (w - bw, w, h - bh, h),
                       "upper left":  (0, bw, 0, bh),
                       "upper right": (w - bw, w, 0, bh)}
            quad = {k: int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                            (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum())
                    for k, (x0, x1, y0, y1) in corners.items()}
            ax.legend(handles=handles, loc=min(quad, key=quad.get), fontsize=6.3,
                      framealpha=0.86, labelspacing=0.32, handletextpad=0.42,
                      borderpad=0.40, handlelength=1.0,
                      borderaxespad=0.35)

        ok = "SUCCESS" if tr["roll_ok"] else "FAILURE"
        ax.set_title(f"{alab}\ncadence {tr['period']:.0f} ms  ·  age {tr['age_mean']:.0f} ms\n"
                     f"arm {tr['arm_sr']:.1f}% over {tr['n_ep']} configs  ·  {ok}",
                     fontsize=7.8, color=FAM[fam], fontweight="bold", pad=4,
                     linespacing=1.35)
        if c == 0:
            ax.text(-0.055, 0.5, tlab, transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=11.5, fontweight="bold")

cax = fig.add_axes([0.945, 0.20, 0.011, 0.58])
cb = fig.colorbar(lc, cax=cax)
cb.set_label("MEASURED age of the observation being acted on (ms)", fontsize=9)

leg = [Line2D([], [], color=OBJ_C, lw=2.4, label="object path (MEASURED)"),
       Line2D([], [], color=OBJ_C, lw=1.2, ls=(0, (3, 2)),
              label="object event → nearest robot LINK, with the measured 3D distance"),
       Line2D([], [], color=SKEL_C, lw=1.4, ls=(0, (5, 2)), marker="o", ms=4,
              label="arm POSE at the object-motion tick (links in frame)"),
       Line2D([], [], ls="", marker="s", ms=8, mfc="white", mec="#555", mew=2,
              label="path start"),
       Line2D([], [], ls="", marker="o", ms=8, color="#111", mec="white",
              label="ARM-anchored event (black)"),
       Line2D([], [], ls="", marker="o", ms=8, color=OBJ_C, mec="white",
              label="OBJECT-anchored event (blue)")]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.48, 0.098), ncol=6,
           fontsize=8.4, framealpha=0.95, handlelength=2.0, columnspacing=1.6)

# The four arms differ in BOTH observation age and cadence, so this figure cannot
# attribute the collapse to either one; the 28,080-episode plane sweep does, and it
# says CADENCE (rho -0.52, p~0.0017 on both widowx tasks) while latency at fixed
# cadence is null on all three. Title states what the panels actually show.
fig.suptitle("The recommended schedule holds the grasp together; slower ones lose it — "
             "and the object still gets shoved around long after nothing can grasp it",
             fontsize=13.5, y=0.965)

agree = sum(d["agree"] for d in DATA.values())
fig.text(0.5, 0.010, "\n".join([
    "Paths are MEASURED, projected with the camera's own intrinsic_cv / extrinsic_cv onto that rollout's own tick-0 frame. A 3D path drawn in 2D: an apparent crossing is not "
    "necessarily a contact and depth is only implied — the printed 3D distances are.",
    f"Event ANCHORING: `moved_correct_obj` is a fact about the OBJECT, so it is drawn on the object with a leader to the nearest of the robot's link centres of mass — which is "
    f"often a FINGER, not the gripper frame. Arm events are drawn on the arm.",
    f"The four columns are the SAME episode config, verified geometrically (initial end-effector and object positions agree to "
    f"{max(SCENE_MM.values()):.3f} mm); the tick-0 PNGs differ by a few least-significant bits from render nondeterminism, so a pixel hash is the wrong test.",
    f"NO SINGLE EPISODE CARRIES A CLAIM: the harness is not run-to-run deterministic, so each panel prints the arm's rate over ALL {DATA[('egg', 'lat0')]['n_ep']} configs "
    f"separately from the one rollout drawn ({agree} of {len(DATA) * 24} stored rollouts match their own sweep's recorded outcome).",
    f"Source is traces_torque2 — all 36 cells pass a full integrity check: arrays present, lengths match ticks, all finite, per-episode flags reconcile with the headline.",
    f"Episode choice, disclosed: {ROWS[0][0]} — {ROWS[0][3]};  {ROWS[1][0]} — {ROWS[1][3]}.",
    f"The per-arm rates printed on every panel are over ALL configs and are NOT selected; only the single rollout DRAWN is chosen, by the criterion above. "
    "Event ticks are sampled every 5 ticks, so a time is accurate to ±0.2 s.",
]), ha="center", va="bottom", fontsize=7.6, style="italic", color="#555", linespacing=1.5)

fig.savefig(OUT)
print(f"[ok] {OUT}")
for task, ep, tlab, _ in ROWS:
    print(f"  {tlab}  ep{ep:02d}  scene agreement {SCENE_MM[task]:.4f} mm")
    for arm, fam, _ in ARMS:
        d = DATA[(task, arm)]
        print(f"    {arm:10s} age {d['age_mean']:7.1f} cadence {d['period']:6.1f} "
              f"arm SR {d['arm_sr']:5.1f}% (n={d['n_ep']})  rollout "
              f"{'SUCCESS' if d['roll_ok'] else 'FAILURE'}  "
              f"stored-vs-sweep agreement {d['agree']}/{d['n_ep']}")
print(f"  observation-age colour scale 0-{VMAX:.0f} ms")

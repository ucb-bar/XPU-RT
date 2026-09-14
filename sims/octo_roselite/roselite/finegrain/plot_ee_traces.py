#!/usr/bin/env python3
# SUPERSEDED: the pipe110 (117.7 ms) / pipe200 (231.8 ms) latencies below are
# THROUGHPUT (wall / n_instances), not per-inference LATENCY. True measured spans are
# pipe110 = 385.1 ms, pipe200 = 260.5 ms. Use paper/fig_*.py instead. Post-mortem:
# archive/2026-09-06_pipelined-latency-mislabelled/README
"""Static end-effector trajectory figures overlaid on the scene -- 4 candidates.

Everything drawn is MEASURED. The path is the env's own `tcp.pose` sampled every
tick, projected with the camera's OWN intrinsic_cv/extrinsic_cv onto the frame the
policy actually saw. The events come from the harness's evaluator
(`is_src_obj_grasped`, `consecutive_grasp`, `src_on_target`) and the commanded
gripper channel -- none of them are inferred from the shape of the path.

  A  path coloured by TIME              -- "what the robot did"
  B  path coloured by OBSERVATION AGE   -- where it was acting on stale data
  C  two arms OVERLAID on one scene     -- accelerated vs CPU-only, same scene
  D  small multiples across the ladder  -- progressive degradation

Caveat carried on every figure: this is a 3D path projected to 2D, so an apparent
crossing is not necessarily a contact and depth is only implied.
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
import mediapy as media

HERE = Path(__file__).parent
T = HERE / "traces"
OUT = HERE

ARMS = {"lat0": ("ideal (0 ms)", "#7f8c8d"),
        "pipe110": ("PIPELINED 110 ms (117.7 ms)", "#16a085"),
        "serial283": ("SERIAL 3-way (283.4 ms)", "#e67e22"),
        "cpu685": ("CPU-only (684.8 ms)", "#c0392b")}


def load(arm, ep=0):
    d = T / f"egg_{arm}"
    j = json.load(open(d / f"ep{ep:02d}_trace.json"))
    xyz = np.load(d / f"ep{ep:02d}_ee_xyz.npy")
    age = np.load(d / f"ep{ep:02d}_action_age_ms.npy")
    bg = media.read_image(d / f"ep{ep:02d}_bg.png")
    K = np.asarray(j["camera_param"]["intrinsic_cv"], dtype=float)
    E = np.asarray(j["camera_param"]["extrinsic_cv"], dtype=float)
    P = np.c_[xyz, np.ones(len(xyz))] @ E.T          # world -> camera (OpenCV)
    uvw = P[:, :3] @ K.T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return dict(uv=uv, xyz=xyz, age=age, bg=bg, j=j, tick=j["tick_ms"],
                success=j["success"], ev=j["events"])


def events(tr):
    """First tick at which each milestone is true, from the evaluator itself."""
    out, seen = {}, {}
    for e in tr["ev"]:
        for k, v in e["stats"].items():
            if v and k not in out:
                out[k] = e["tick"]
            seen[k] = v
    # gripper CLOSE command: the commanded channel crossing below 0.5
    grips = [(e["tick"], e["grip"]) for e in tr["ev"]]
    for i in range(1, len(grips)):
        if grips[i-1][1] >= 0.5 > grips[i][1]:
            out["gripper_closed"] = grips[i][0]; break
    return out


EVENT_STYLE = {
    "moved_correct_obj":  ("object moved",  "o", "#2980b9"),
    "gripper_closed":     ("gripper closes", "v", "#8e44ad"),
    "is_src_obj_grasped": ("GRASPED",       "*", "#27ae60"),
    "consecutive_grasp":  ("held 1 s",      "P", "#16a085"),
    "src_on_target":      ("ON TARGET",     "X", "#d35400"),
}


def draw_path(ax, tr, values, cmap, lw=2.6, alpha=1.0):
    uv = tr["uv"]
    seg = np.stack([uv[:-1], uv[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidths=lw, alpha=alpha, zorder=3)
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc


def mark_events(ax, tr, arm_col=None):
    """Draw the event markers on the path and RETURN the legend rows.

    Inline annotations collide badly (grasp, held and gripper-close land within a
    few pixels of each other), so the times go in a corner box instead.
    """
    ev = events(tr)
    uv, n = tr["uv"], len(tr["uv"])
    rows = []
    for key, (label, mk, col) in EVENT_STYLE.items():
        if key not in ev: continue
        t = min(ev[key], n - 1)
        ax.scatter(*uv[t], s=200 if mk in "*X" else 110, marker=mk,
                   c=arm_col or col, edgecolor="white", linewidth=1.5, zorder=6)
        rows.append((mk, arm_col or col, label, t * tr["tick"] / 1000.0))
    return rows


def event_box(ax, rows, loc="lower right", title=None, fs=8.0):
    """Compact legend of events and the time each first occurred."""
    handles = [Line2D([], [], ls="", marker=mk, ms=9, color=col, mec="white", mew=1.2,
                      label=f"{lab}  {t:.1f}s") for mk, col, lab, t in rows]
    if not handles: return
    lg = ax.legend(handles=handles, loc=loc, fontsize=fs, framealpha=0.86,
                   title=title, title_fontsize=fs, labelspacing=0.35,
                   handletextpad=0.5, borderpad=0.5)
    ax.add_artist(lg)
    return lg


def frame(ax, tr, title):
    ax.imshow(tr["bg"]); ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=9)
    h, w = tr["bg"].shape[:2]
    ax.set_xlim(0, w); ax.set_ylim(h, 0)


CAVEAT = ("3D end-effector path projected to 2D with the camera's own intrinsics — apparent crossings "
          "are not necessarily contacts, and depth is only implied.")

# ---------------- A: coloured by time ---------------------------------------
def candidate_A(arm="pipe110"):
    tr = load(arm)
    fig, ax = plt.subplots(figsize=(8.6, 7.0), dpi=140)
    frame(ax, tr, f"A · path coloured by TIME — {ARMS[arm][0]}"
                  f"  ({'SUCCESS' if tr['success'] else 'FAILURE'})")
    t_s = np.arange(len(tr["uv"])) * tr["tick"] / 1000.0
    lc = draw_path(ax, tr, t_s, "viridis")
    event_box(ax, mark_events(ax, tr), title="events (evaluator)")
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("elapsed time (s)", fontsize=9)
    fig.text(0.5, 0.015, CAVEAT, ha="center", fontsize=7.5, style="italic", color="#555")
    p = OUT / "eetrace_A_time.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {p}")

# ---------------- B: coloured by observation age ----------------------------
def candidate_B(arm="serial283"):
    tr = load(arm)
    fig, ax = plt.subplots(figsize=(8.6, 7.0), dpi=140)
    frame(ax, tr, f"B · path coloured by OBSERVATION AGE — {ARMS[arm][0]}"
                  f"  ({'SUCCESS' if tr['success'] else 'FAILURE'})")
    age = np.nan_to_num(tr["age"][:len(tr["uv"])], nan=0.0)
    lc = draw_path(ax, tr, age, "inferno")
    event_box(ax, mark_events(ax, tr), title="events (evaluator)")
    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("age of the observation being acted on (ms)", fontsize=9)
    fig.text(0.5, 0.015, "Bright = acting on stale data. " + CAVEAT,
             ha="center", fontsize=7.5, style="italic", color="#555")
    p = OUT / "eetrace_B_staleness.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {p}")

# ---------------- C: two arms overlaid --------------------------------------
def candidate_C(a="pipe110", b="cpu685"):
    ta, tb = load(a), load(b)
    fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=140)
    frame(ax, ta, "C · same scene, two hardware configurations")
    boxes = []
    for tr, arm, ls in ((ta, a, "-"), (tb, b, "--")):
        lbl, col = ARMS[arm]
        ax.plot(tr["uv"][:, 0], tr["uv"][:, 1], ls=ls, lw=2.8, color=col,
                zorder=3, solid_capstyle="round",
                label=f"{lbl} — {'SUCCESS' if tr['success'] else 'FAILURE'}")
        ax.scatter(*tr["uv"][0], s=90, marker="s", c="white", edgecolor=col,
                   linewidth=2, zorder=6)
        boxes.append((mark_events(ax, tr, arm_col=col), lbl, col))
    # NO "paths diverge" marker: the harness is not run-to-run deterministic
    # (see NONDETERMINISM.md), so two paths separating does not isolate latency
    # as the cause -- two runs of the SAME arm separate too.
    lg0 = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.add_artist(lg0)
    event_box(ax, boxes[0][0], loc="lower left", title=boxes[0][1], fs=7.5)
    event_box(ax, boxes[1][0], loc="lower right", title=boxes[1][1], fs=7.5)
    fig.text(0.5, 0.015, CAVEAT, ha="center", fontsize=7.5, style="italic", color="#555")
    p = OUT / "eetrace_C_overlay.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {p}")

# ---------------- D: small multiples ----------------------------------------
def candidate_D():
    arms = [a for a in ["lat0", "pipe110", "serial283", "cpu685"] if (T / f"egg_{a}").exists()]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.9*len(arms), 5.6), dpi=140)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        tr = load(arm)
        lbl, col = ARMS[arm]
        frame(ax, tr, f"{lbl}\n{'SUCCESS' if tr['success'] else 'FAILURE'}")
        age = np.nan_to_num(tr["age"][:len(tr["uv"])], nan=0.0)
        lc = draw_path(ax, tr, age, "inferno", lw=2.2)
        lc.set_clim(0, 1400)
        event_box(ax, mark_events(ax, tr), loc="lower left", fs=6.8)
    cb = fig.colorbar(lc, ax=axes.tolist(), fraction=0.02, pad=0.01)
    cb.set_label("observation age when acted on (ms)", fontsize=9)
    fig.suptitle("D · end-effector path across the latency ladder — eggplant, same scene, "
                 "colour = staleness", fontsize=13, y=1.02)
    fig.text(0.5, -0.02, CAVEAT, ha="center", fontsize=8, style="italic", color="#555")
    p = OUT / "eetrace_D_ladder.png"; fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {p}")


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401  (used via __import__ above)
    for f in (candidate_A, candidate_B, candidate_C, candidate_D):
        try:
            f()
        except Exception as e:
            print(f"[skip] {f.__name__}: {type(e).__name__}: {e}")

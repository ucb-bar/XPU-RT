#!/usr/bin/env python3
"""Splash figure: one scene, two object paths, no chrome.

The SAME episode config (eggplant into basket, ep06) under two schedules of the same
INT8 Octo policy on the same QRB5165 silicon:

  OURS        the pipelined 3-way schedule -- 150 ms cadence, 282 ms observation age.
              Object travels 30.7 cm and lands in the basket.
  QNN CPU     the CPU-only baseline -- 685 ms cadence, 1008 ms age.
              Object travels 183.8 cm, six times further, and never lands.
              (Both are EUCLIDEAN path length. An earlier draft of this docstring
              quoted 43.7 / 257.3, which is the per-axis absolute sum -- a different
              metric, not different data. The 6x ratio holds either way: 5.99 vs 5.89.)

Deliberately no title, caption or legend box: the two direct labels carry it.

Colours are checked, not chosen by eye: #00E5FF vs #FF2D55 is dE 42.9 in OKLab for
normal vision and 25.6 under the worst of protan/deutan/tritan simulation -- comfortably
above the >=15 normal and >=8 CVD floors. Both carry a dark halo so they survive over a
photographic background rather than depending on the backdrop staying light.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from scipy import ndimage
import PIL.Image as PILImage

HERE = Path(__file__).parent
T = HERE.parent / "traces_torque2"
OUT = HERE / "fig_splash.png"

TASK, EP = "egg", 6
OURS, BASE = "p150w300", "cpu685"
# SATURATED, on a light page. These sit on the PHOTO, not on white, and their
# contrast comes from a DARK halo (12.8:1 and 5.4:1 against it) rather than from the
# stroke's own luminance -- which is why the muted light-mode pair was the wrong fix:
# it solved a white-background problem these strokes never had.
# dE 42.9 normal / 25.6 under the worst of protan/deutan/tritan -- the widest of any
# pair tried in either mode.
C_OURS, C_BASE = "#00E5FF", "#FF2D55"


def project(P, K, E):
    P = np.atleast_2d(np.asarray(P, float))
    cam = np.c_[P, np.ones(len(P))] @ E.T
    uvw = cam[:, :3] @ K.T
    w = uvw[:, 2:3].copy(); w[np.abs(w) < 1e-9] = np.nan
    uv = uvw[:, :2] / w
    uv[cam[:, 2] <= 0] = np.nan
    return uv


def load(arm):
    d = T / f"{TASK}_{arm}"
    j = json.load(open(d / f"ep{EP:02d}_trace.json"))
    s = json.load(open(d / "summary.json"))
    K = np.asarray(j["camera_param"]["intrinsic_cv"], float)
    E = np.asarray(j["camera_param"]["extrinsic_cv"], float)
    obj = np.load(d / f"ep{EP:02d}_obj_xyz.npy")
    ok = next(e["success"] for e in s["episodes"] if e["episode_id"] == EP)
    trav = float(np.linalg.norm(np.diff(obj, axis=0), axis=1).sum())
    return dict(uv=project(obj, K, E), ok=bool(ok), trav=trav,
                bg=np.asarray(PILImage.open(d / f"ep{EP:02d}_bg.png").convert("RGB")))


a, b = load(OURS), load(BASE)
H, W = a["bg"].shape[:2]

# --- the eggplant, segmented from the tick-0 frame -------------------------------
# Purple is unambiguous in this scene: strong R and B, suppressed G. Used to (a) keep
# the object at FULL saturation under the scrim and (b) build the two end-state sprites.
hsv = np.asarray(PILImage.open(T / f"{TASK}_{OURS}" / f"ep{EP:02d}_bg.png")
                        .convert("HSV")).astype(float)
_H, _S, _V = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
_PURPLE = (_H > 250) & (_H < 330) & (_S > 0.30) & (_V > 0.10)
_STEM = (_H > 70) & (_H < 170) & (_S > 0.25) & (_V > 0.08)
# the calyx only counts where it touches the body, so unrelated green scene pixels
# cannot join the blob
_NEAR = ndimage.binary_dilation(_PURPLE, np.ones((9, 9)))
MASK = _PURPLE | (_STEM & _NEAR)
MASK = ndimage.binary_fill_holes(ndimage.binary_closing(MASK, np.ones((5, 5))))
_lab, _n = ndimage.label(MASK)                      # keep only the object itself
if _n:
    MASK = _lab == (1 + int(np.argmax(ndimage.sum(MASK, _lab, range(1, _n + 1)))))
_ys, _xs = np.nonzero(MASK)
Y0, Y1, X0, X1 = _ys.min(), _ys.max() + 1, _xs.min(), _xs.max() + 1
SPRITE = a["bg"][Y0:Y1, X0:X1].copy()
SPR_M = MASK[Y0:Y1, X0:X1]
SRC_C = np.array([(X0 + X1) / 2.0, (Y0 + Y1) / 2.0])


def stamp(ax, centre, colour, alpha=1.0, z=4):
    """Place the object's own pixels at `centre`, ringed in `colour`.

    A composited sprite, NOT a re-render: the harness stores only the tick-0 frame, so
    the end state is shown by translating the object's own pixels to its MEASURED final
    projected position. Pose change is therefore not depicted -- position is.
    """
    h, w = SPR_M.shape
    x0, y0 = centre[0] - w / 2.0, centre[1] - h / 2.0
    rgba = np.zeros((h, w, 4))
    rgba[..., :3] = SPRITE / 255.0
    rgba[..., 3] = SPR_M * alpha
    ax.imshow(rgba, extent=[x0, x0 + w, y0 + h, y0], zorder=z, interpolation="nearest")
    # The object is PURPLE and the "bad" colour is magenta, so ring and object blend.
    # A white under-stroke separates them whatever the hue.
    xs_, ys_ = np.linspace(x0, x0 + w, w), np.linspace(y0, y0 + h, h)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=["#0B0B0B"],
               linewidths=8.0, zorder=z + 1, alpha=0.92)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=[colour],
               linewidths=4.2, zorder=z + 2)

# The baseline drives the object 0.92 m DOWN -- off the counter, out of the camera
# (true final projected y = 656 in a 480 px frame). Leaving the frame is itself the
# failure, so the end state is TRUNCATED to the bottom edge rather than drawn at a
# coordinate outside the scene. The clamp is cosmetic; the measured fall is unchanged
# and is what `knocked off the counter` refers to.
YMAX = H
fig = plt.figure(figsize=(11.0, 11.0 * YMAX / W), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)          # below the photo the page shows through
ax.imshow(a["bg"], extent=[0, W, H, 0], zorder=0)
ax.set_xlim(0, W); ax.set_ylim(YMAX, 0); ax.axis("off")
# Slight scrim so saturated strokes read against a busy photo without hiding the scene.
# Light ground: the strokes are now DARK, so the scrim lifts the photo instead of
# dimming it. Same job -- separate marks from scene -- opposite direction.
ax.add_patch(Rectangle((0, 0), W, H, facecolor="white", alpha=0.30, zorder=1, lw=0))

# Keep the ROBOT at full saturation too -- it is the actor in the frame, and the flat
# scrim washes it out along with the scene. Mask is GEOMETRY-GATED (low-saturation pixels
# near a projected link COM), not colour alone: colour alone claims 14.5% of the eggplant
# frame and 0.6% of the coke frame, wrong in both directions. See splash_common.py.
from splash_common import robot_mask, unscrim_robot
_d0 = T / f"{TASK}_{OURS}"
_j0 = json.load(open(_d0 / f"ep{EP:02d}_trace.json"))
_ARM = robot_mask(_d0 / f"ep{EP:02d}_bg.png",
                  np.load(_d0 / f"ep{EP:02d}_link_com.npy")[0],
                  np.asarray(_j0["camera_param"]["intrinsic_cv"], float),
                  np.asarray(_j0["camera_param"]["extrinsic_cv"], float))
print(f"  arm mask: {unscrim_robot(ax, a['bg'], _ARM, W, H)} px kept at full saturation")
# Restore the object to FULL saturation: the scrim exists to make the strokes read,
# and dimming the thing the whole figure is about defeats it.
stamp(ax, SRC_C, "#F2F2F2", z=2)

# DARK halo: it is what makes a saturated stroke legible over a light scene.
halo = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.92), pe.Normal()]
for d, col, lab in ((b, C_BASE, "QNN CPU"), (a, C_OURS, "OURS")):
    uv = d["uv"]
    ax.plot(uv[:, 0], uv[:, 1], color=col, lw=5.4, solid_capstyle="round",
            zorder=3, path_effects=halo)
    # The final plunge out of frame crosses the path's OWN earlier passes near the
    # start. Drawn as one polyline it shares a single halo, so the crossings render
    # flat and the temporal order is unreadable. Re-draw the last descent on a higher
    # layer: its dark halo then cuts across the older track, showing which came last.
    fin = uv[~np.isnan(uv[:, 0])]
    if len(fin) > 4 and fin[-1, 1] > H - 4:
        above = np.nonzero(fin[:, 1] < fin[-1, 1] - 170)[0]
        if len(above):
            j = int(above[-1])
            ax.plot(fin[j:, 0], fin[j:, 1], color=col, lw=5.4, solid_capstyle="round",
                    zorder=6, path_effects=halo)
    # Arrowhead at MID-PATH by arc length, not at the endpoint. At the end it is
    # buried inside the end-state sprite; backing off by a clearance radius is
    # fragile because the two paths approach their ends at different angles.
    # Mid-path is unambiguous for direction and always on open trajectory.
    # Drawn from the immediately preceding sample so the connector is a couple of
    # pixels -- a longer one renders as a straight line across the scene. The dark
    # edge is a larger arrow layered underneath; a path_effects Stroke on the arrow
    # overrides lw=0 and redraws that connector.
    ins = uv[(~np.isnan(uv[:, 0])) & (uv[:, 1] <= H - 2)]
    if len(ins) > 3:
        step = np.linalg.norm(np.diff(ins, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        i = int(np.searchsorted(cum, cum[-1] * 0.55))
        i = min(max(i, 1), len(ins) - 2)
        tail, tip = ins[i], ins[i + 1]
        for c_, ms in (("#0B0B0B", 92), (col, 72)):
            ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                        arrowprops=dict(arrowstyle="-|>", color=c_, linewidth=0,
                                        mutation_scale=ms, shrinkA=0, shrinkB=0),
                        zorder=10)
    ax.scatter(*uv[0], s=340, marker="o", facecolor="white", edgecolor="#0B0B0B",
               linewidth=2.4, zorder=5)
    # The CPU path leaves the frame; put the outcome marker on the last point still
    # INSIDE it, or the failure marker is drawn where nobody can see it.
    # No end marker: the outlined end-state sprite below IS the outcome.

# End states, in the same good/bad colours as the paths.
SPR_H, SPR_W = SPR_M.shape
ENDS = {}
for d, col, key in ((a, C_OURS, "ours"), (b, C_BASE, "base")):
    good = d["uv"][~np.isnan(d["uv"][:, 0])]
    ex, ey = good[-1]
    clamped = ey > H - SPR_H / 2 - 8
    if clamped:
        seen = good[good[:, 1] <= H - 2]
        if len(seen):
            ex, ey = seen[-1] + np.array([-4.0, 18.0])   # ride the path, not beside it
    ey = min(ey, H - SPR_H / 2 - 8)
    ex = min(max(ex, SPR_W / 2 + 6), W - SPR_W / 2 - 6)
    ENDS[key] = (ex, ey, clamped)
    stamp(ax, (ex, ey), col, alpha=0.97, z=7)

# outcome glyphs: tick UNDER the success, cross BESIDE the failure
gl = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ox, oy, _ = ENDS["ours"]
ax.text(ox, oy + SPR_H / 2 + 52, "\u2714", ha="center", va="center", fontsize=132,
        color=C_OURS, zorder=9, path_effects=gl)
bx, by, _ = ENDS["base"]
ax.text(bx + SPR_W / 2 + 60, by, "\u2718", ha="center", va="center", fontsize=132,
        color=C_BASE, zorder=9, path_effects=gl)

# Dark outline on the label text too, so a saturated colour reads on a light page.
txt = [pe.Stroke(linewidth=5.2, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ox_, oy_, _ = ENDS["ours"]
ax.text(ox_ - SPR_W / 2 - 4, oy_ - SPR_H / 2 - 8, "XPURT", ha="left", va="bottom",
        fontsize=46, fontweight="bold", color=C_OURS, path_effects=txt, zorder=11)
ax.text(bx - SPR_W / 2 - 16, by, "Baseline", ha="right", va="center",
        fontsize=46, fontweight="bold", color=C_BASE, path_effects=txt, zorder=11)

fig.savefig(OUT, dpi=200, transparent=True)
print(f"[ok] {OUT}")
print(f"  ours  {OURS:10s} success={a['ok']}  object travel {100*a['trav']:.1f} cm")
print(f"  base  {BASE:10s} success={b['ok']}  object travel {100*b['trav']:.1f} cm")

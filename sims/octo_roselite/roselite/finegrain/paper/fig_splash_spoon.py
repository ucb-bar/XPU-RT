#!/usr/bin/env python3
"""Splash figure, spoon-on-towel: one scene, two object paths, no chrome.

Companion to fig_splash.py (eggplant). Same policy, same silicon, same episode
config -- only the SCHEDULE differs:

  XPURT       p150w300, the pipelined 3-way schedule -- 150 ms release cadence,
              282 ms latency, 337 ms mean action age (summary.json: issue_period_ms
              150.4, latency_ms 281.6, age_mean_ms 337.5). The spoon travels 38.5 cm
              and ends on the towel.
  Baseline    cpu685, the CPU-only baseline -- 685 ms cadence, 685 ms latency,
              1003 ms mean action age.
              The spoon travels 139.4 cm, is flung up and then dropped off the
              front of the table: its final projected y is 617 in a 480 px frame.

EPISODE SELECTION -- a criterion, not a browse.
Candidate set: episodes where cpu685 FAILS and at least one pipelined arm SUCCEEDS;
XPURT is then the highest-overall-success pipelined arm that succeeds on that episode
(spoon rates: p150w300 .458 > p105w300 = p130w275 = pipe110fix .333 > pipe200fix .208;
cpu685 0/24 = .000, so every episode where a pipelined arm succeeds is a candidate).
Rank by the outcome gap |object travel(XPURT) - travel(cpu685)|, subject to BOTH
projected paths being long enough to read (>= 100 px of on-screen arc length) -- a
path of zero pixel length carries no shape to compare, however large its travel gap.

  ep arm           trav_o  trav_b     gap   Lpx_o  Lpx_b   >=100px both
   4 p150w300        38.5   139.4   100.8     323    738   YES   <-- chosen
   1 pipe200fix      28.1   125.4    97.3     236    640   YES
  20 p130w275        29.8   107.5    77.6     359    566   YES
  23 p150w300        42.7     0.0    42.7     535      0   no  (baseline never moves it)
  15 p130w275        40.0     0.0    40.0     474      0   no
   0 p150w300        23.0     0.0    23.0     216      0   no
  19 p150w300        37.9    17.2    20.6     444    205   YES
   3 pipe200fix      28.3     7.9    20.4     233     76   no
  16 p150w300        10.2    28.9    18.7     114    295   YES
   6 p150w300        30.0    13.8    16.3     289    163   YES
  10 p150w300        29.2    15.6    13.6     256    180   YES
  21 pipe110fix      24.5    12.6    11.9     316    167   YES
  22 p150w300        26.9    15.6    11.4     344    181   YES
   8 p150w300        38.0    28.0    10.0     456    298   YES
  12 p150w300        34.3    27.0     7.3     334    241   YES
   2 p150w300        24.9    19.2     5.6     223    185   YES
  18 p105w300        34.2    29.5     4.7     365    347   YES
  (travel in cm, arc length in image px; all 17 candidates listed)

ep04 wins outright on the stated metric AND uses the top-rate pipelined arm.

Deliberately no title, caption or legend box: the two direct labels carry it.

Colours as in fig_splash.py: #00E5FF vs #FF2D55 is dE 42.9 in OKLab for normal vision
and 25.6 under the worst of protan/deutan/tritan -- comfortably above the >=15 normal
and >=8 CVD floors. Both carry a dark halo so they survive over a photographic
background rather than depending on the backdrop staying light.
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
OUT = HERE / "fig_splash_spoon.png"

TASK, EP = "spoon", 4
OURS, BASE = "p150w300", "cpu685"
# SATURATED, on a light page. These sit on the PHOTO, not on white, and their
# contrast comes from a DARK halo rather than from the stroke's own luminance.
C_OURS, C_BASE = "#00E5FF", "#FF2D55"
# Where along each path (fraction of IN-FRAME arc length) the direction arrowhead
# sits. Mid-path by default; nudged per path so neither head lands under the other
# track or inside a sprite.
ARROW_AT = {"XPURT": 0.55, "Baseline": 0.55}


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

# --- the spoon, segmented from the tick-0 frame ----------------------------------
# NOT the eggplant's purple rule. This object is two-toned: a saturated yellow-green
# HANDLE and a near-white BOWL, on a red-brown table in front of a grey wall.
#   handle  H 50..115 with S > 0.30 -- the only strongly green thing in the scene
#   bowl    near-white (S < 0.25, V > 0.68); white alone also matches the grey wall
#           and the specular table highlights, so only the white COMPONENTS that
#           touch the handle are taken. Growing by a fixed dilation radius (the
#           eggplant's calyx trick) is not enough here: the bowl is far bigger than
#           any sane radius, so it must be admitted whole, by connectivity.
# The bowl is partly occluded by the near gripper finger, which is why the mask
# centroid sits ~2.6 px from the projected obj_xyz[0] rather than exactly on it.
hsv = np.asarray(PILImage.open(T / f"{TASK}_{OURS}" / f"ep{EP:02d}_bg.png")
                        .convert("HSV")).astype(float)
_H, _S, _V = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255


def _largest(m):
    lab, n = ndimage.label(m)
    if not n:
        return m
    return lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))


def _touching(part, seed, grow=7):
    """whole components of `part` that touch a slightly grown `seed`"""
    lab, n = ndimage.label(part)
    if not n:
        return np.zeros_like(part)
    near = ndimage.binary_dilation(seed, np.ones((grow, grow)))
    keep = set(np.unique(lab[part & near])) - {0}
    return np.isin(lab, list(keep))


_HANDLE = _largest(ndimage.binary_closing(
    (_H > 50) & (_H < 115) & (_S > 0.30) & (_V > 0.35), np.ones((3, 3))))
_BOWL = _touching((_S < 0.25) & (_V > 0.68), _HANDLE)
MASK = ndimage.binary_fill_holes(
    ndimage.binary_closing(_HANDLE | _BOWL, np.ones((5, 5))))
MASK = _largest(MASK)                               # keep only the object itself
_ys, _xs = np.nonzero(MASK)
Y0, Y1, X0, X1 = _ys.min(), _ys.max() + 1, _xs.min(), _xs.max() + 1
SPRITE = a["bg"][Y0:Y1, X0:X1].copy()
SPR_M = MASK[Y0:Y1, X0:X1]
SRC_C = np.array([(X0 + X1) / 2.0, (Y0 + Y1) / 2.0])
# CHECK, printed at the end: the mask must be the object obj_xyz refers to.
_MASK_C = np.array([_xs.mean(), _ys.mean()])
_PROJ0 = a["uv"][0]

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
    # A dark under-stroke separates ring from object whatever the hue -- the pale
    # bowl would otherwise wash straight into a saturated ring.
    xs_, ys_ = np.linspace(x0, x0 + w, w), np.linspace(y0, y0 + h, h)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=["#0B0B0B"],
               linewidths=8.0, zorder=z + 1, alpha=0.92)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=[colour],
               linewidths=4.2, zorder=z + 2)

# The baseline lifts the spoon, carries it forward and drops it off the front edge of
# the table -- out of the camera (true final projected y = 617 in a 480 px frame).
# Leaving the frame IS the failure, so the end state is TRUNCATED to the bottom edge
# rather than drawn at a coordinate outside the scene. The clamp is cosmetic; the
# measured 139.4 cm of travel is unchanged.
YMAX = H
fig = plt.figure(figsize=(11.0, 11.0 * YMAX / W), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)          # below the photo the page shows through
ax.imshow(a["bg"], extent=[0, W, H, 0], zorder=0)
ax.set_xlim(0, W); ax.set_ylim(YMAX, 0); ax.axis("off")
# Slight scrim so saturated strokes read against a busy photo without hiding the scene.
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
for d, col, lab in ((b, C_BASE, "Baseline"), (a, C_OURS, "XPURT")):
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
    # Drawn from the immediately preceding sample so the connector is a couple of
    # pixels -- a longer one renders as a straight line across the scene.
    ins = uv[(~np.isnan(uv[:, 0])) & (uv[:, 1] <= H - 2)]
    if len(ins) > 3:
        step = np.linalg.norm(np.diff(ins, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        i = int(np.searchsorted(cum, cum[-1] * ARROW_AT[lab]))
        i = min(max(i, 1), len(ins) - 2)
        tail, tip = ins[i], ins[i + 1]
        for c_, ms in (("#0B0B0B", 92), (col, 72)):
            ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                        arrowprops=dict(arrowstyle="-|>", color=c_, linewidth=0,
                                        mutation_scale=ms, shrinkA=0, shrinkB=0),
                        zorder=10)
    ax.scatter(*uv[0], s=340, marker="o", facecolor="white", edgecolor="#0B0B0B",
               linewidth=2.4, zorder=5)
    # No end marker: the outlined end-state sprite below IS the outcome.

# End states, in the same good/bad colours as the paths.
SPR_H, SPR_W = SPR_M.shape
# Placed by the object's DISPLACEMENT rather than by centring the sprite's bounding box
# on the projected point; the two differ by SRC_C - projected obj_xyz[0] = (-0.1, -4.0)
# px here, since the gripper finger hides part of the bowl and biases the silhouette.
DELTA = SRC_C - _PROJ0
ENDS = {}
for d, col, key in ((a, C_OURS, "ours"), (b, C_BASE, "base")):
    good = d["uv"][~np.isnan(d["uv"][:, 0])]
    ex, ey = good[-1] + DELTA
    clamped = ey > H - SPR_H / 2 - 8
    if clamped:
        seen = good[good[:, 1] <= H - 2]
        if len(seen):
            ex, ey = seen[-1] + DELTA + np.array([0.0, 18.0])  # ride the path
    ey = min(ey, H - SPR_H / 2 - 8)
    ex = min(max(ex, SPR_W / 2 + 6), W - SPR_W / 2 - 6)
    ENDS[key] = (ex, ey, clamped)
    stamp(ax, (ex, ey), col, alpha=0.97, z=7)

# outcome glyphs: both end states sit LOW in this frame, so a glyph placed below
# either one would run off the page -- both go sideways instead, into the empty
# table on their own outer side, so each stays nearer its own sprite than the other's.
gl = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ox, oy, _ = ENDS["ours"]
bx, by, _ = ENDS["base"]
ax.text(ox + SPR_W / 2 + 56, oy + 18, "\u2714", ha="center", va="center", fontsize=132,
        color=C_OURS, zorder=9, path_effects=gl)
ax.text(bx - SPR_W / 2 - 52, by - 14, "\u2718", ha="center", va="center", fontsize=132,
        color=C_BASE, zorder=9, path_effects=gl)

# Dark outline on the label text too, so a saturated colour reads on a light page.
# XPURT rides above its tick, clear of the cyan descent at x~430; Baseline is
# right-aligned so it stops short of the red descent at x~241 rather than crossing it.
txt = [pe.Stroke(linewidth=5.2, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ax.text(ox + SPR_W / 2 + 56, oy - SPR_H / 2 - 10, "XPURT", ha="center", va="bottom",
        fontsize=46, fontweight="bold", color=C_OURS, path_effects=txt, zorder=11)
ax.text(bx - SPR_W / 2 - 14, by - SPR_H / 2 - 46, "Baseline", ha="right", va="bottom",
        fontsize=46, fontweight="bold", color=C_BASE, path_effects=txt, zorder=11)

fig.savefig(OUT, dpi=200, transparent=True)
print(f"[ok] {OUT}")
print(f"  ours  {OURS:10s} success={a['ok']}  object travel {100*a['trav']:.1f} cm")
print(f"  base  {BASE:10s} success={b['ok']}  object travel {100*b['trav']:.1f} cm")
print(f"  mask  {int(MASK.sum())} px  centroid {np.round(_MASK_C,1)}  "
      f"projected obj_xyz[0] {np.round(_PROJ0,1)}  "
      f"err {np.hypot(*(_MASK_C-_PROJ0)):.1f} px  "
      f"proj-inside-mask={bool(MASK[int(round(_PROJ0[1])), int(round(_PROJ0[0]))])}")

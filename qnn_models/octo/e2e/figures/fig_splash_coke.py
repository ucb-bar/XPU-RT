#!/usr/bin/env python3
"""Splash figure, pick-coke-can: one scene, two object paths, no chrome.

Companion to fig_splash.py (eggplant) and fig_splash_spoon.py. Same policy, same
silicon, same episode config -- only the SCHEDULE differs:

  XPURT       p105w300, the pipelined schedule -- 125 ms release cadence, 259 ms
              latency, 304 ms mean action age (summary.json: issue_period_ms 124.8,
              latency_ms 258.7, age_mean_ms 303.6).
              Grasps the can and takes it 24.5 cm (18.5 cm net) across the table.
              Almost all of that is in-plane: the can rises only 0.6 cm, so the shift
              up the frame is motion AWAY from the camera, not a lift.
  Baseline    cpu685, the CPU-only baseline -- 685 ms cadence, 685 ms latency,
              1008 ms mean action age.
              Never closes the gripper at all (gripper_frac_closed 0.000) and the can
              ends the episode at exactly its start pose: 0.0 cm of travel.

EPISODE SELECTION -- a criterion, not a browse.
Candidate set: episodes where cpu685 FAILS and at least one pipelined arm SUCCEEDS;
XPURT is then the highest-overall-success pipelined arm that succeeds on that episode
(coke rates: p105w300 .500 > p150w300 .458 > pipe110fix .417 > p130w275 = pipe200fix
.375; cpu685 3/24 = .125).  Rank by the outcome gap |object travel(XPURT) -
travel(cpu685)|, subject to two READABILITY tests: both projected paths long enough to
read (>= 100 px of on-screen arc length), and the two end-state sprites not overlapping
(they are stamps of the same object, so once they overlap the figure shows one smeared
can instead of two outcomes).

  ep arm           trav_o  trav_b     gap   Lpx_o  Lpx_b  endsep   sprite  overlap
  22 p105w300        50.8    17.1    33.7     236    116      50   66x103      39%
  14 p130w275         6.2    37.1    30.9      33    230     135   67x89        0%
  11 p150w300        19.2    44.1    24.9     111    295      52   115x54      36%
   7 p105w300        24.5     0.0    24.4     118      0      79   96x50        0%  <--
   8 p105w300        18.3    40.1    21.8      78    126      35   93x45       21%
  17 pipe200fix      16.9    37.0    20.0      83    160      65   63x98        1%
  18 p105w300         1.3    20.3    19.0       5     90      60   53x76       11%
  13 p105w300         8.0    21.5    13.5      34    116      28   90x47       39%
  12 p105w300         6.4    13.3     6.9      37     86      46   56x80       17%
   4 p105w300         7.6    12.7     5.1      48     95      74   53x91        0%
  20 p105w300         4.4     0.0     4.4      23      0       4   51x85       90%
  15 p105w300         7.7    11.4     3.7      51     49      40   57x88       38%
  21 p105w300         3.7     5.6     1.9      17     36      42   56x87       32%
  (travel in cm; arc length, separation and sprite size in image px; all 13 candidates)

NO candidate satisfies both tests, and that is a property of the task rather than of
the search: in `pick coke can` the can is grasped and shifted a few tens of cm on a
table it never leaves, so its projected motion is small in every episode and the two
end states usually land on top of each other. This is the least dramatic of the four
splash scenes and should not be sold as more than it is.

ep07 is chosen as the least-compromised: the highest-gap candidate whose two end states
are actually distinguishable (0% sprite overlap). Its compromise is the opposite one --
the baseline path has ZERO length, because under a 685 ms cadence the policy never
closes the gripper at all. That is the honest outcome, not a drawing artefact, and it
is why the baseline gets no direction arrowhead: there is no direction. The red ring
sits on the can in its ORIGINAL pose, which is exactly where the baseline left it --
so the white start marker is drawn ON TOP of the sprites here, as the one mark that
says the red outcome and the start are the same place.

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
OUT = HERE / "fig_splash_coke.png"

TASK, EP = "coke", 7
OURS, BASE = "p105w300", "cpu685"
# SATURATED, on a light page. These sit on the PHOTO, not on white, and their
# contrast comes from a DARK halo rather than from the stroke's own luminance.
C_OURS, C_BASE = "#00E5FF", "#FF2D55"
# Where along each path (fraction of IN-FRAME arc length) the direction arrowhead
# sits. Mid-path by default; a path with no length gets no arrowhead at all.
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

# --- the can, segmented from the tick-0 frame ------------------------------------
# NOT the eggplant's purple rule. This scene is beige-on-beige -- table, cabinet, wall
# and floor all sit around H 20-40 -- and the ONLY saturated red in it is the can:
#   body    H < 14 or H > 346 with S > 0.45; the beige never exceeds S ~ 0.35
#   lid     the brushed-aluminium top is desaturated, so a red-only mask decapitates
#           the sprite and drags the centroid ~10 px low. It is admitted the same way
#           the spoon's bowl is: whole low-saturation COMPONENTS that touch the body.
#           Stray table components picked up by that rule are then dropped by the
#           keep-largest step, since only the lid is actually joined to the body.
# Verified below: the mask centroid lands 5.0 px from the projected obj_xyz[0]
# (the can lies on its side here, so its silhouette centroid and its body-frame
# origin are not the same point; the projection falls well inside the mask).
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


_BODY = _largest(ndimage.binary_closing(
    ((_H < 14) | (_H > 346)) & (_S > 0.45) & (_V > 0.12), np.ones((3, 3))))
_LID = _touching((_S < 0.28) & (_V > 0.45), _BODY)
MASK = ndimage.binary_fill_holes(
    ndimage.binary_closing(_BODY | _LID, np.ones((5, 5))))
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
    # The can is RED and the "bad" colour is a red-magenta, so ring and object would
    # blend. A dark under-stroke separates them whatever the hue.
    xs_, ys_ = np.linspace(x0, x0 + w, w), np.linspace(y0, y0 + h, h)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=["#0B0B0B"],
               linewidths=8.0, zorder=z + 1, alpha=0.92)
    ax.contour(xs_, ys_, SPR_M.astype(float), levels=[0.5], colors=[colour],
               linewidths=4.2, zorder=z + 2)

# Nothing leaves the frame in this episode -- the can is lifted, not knocked off the
# table -- so no end state needs truncating here. The clamp below is still applied
# (it is what guarantees a sprite is never drawn outside the scene) and simply
# does nothing.
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

# End states are resolved FIRST: the arrowheads below need to know where the two
# sprites will land so they can dodge them. In this scene the object moves only ~77 px
# while the sprite is 50 px tall, so a blind mid-path arrowhead is swallowed whole.
SPR_H, SPR_W = SPR_M.shape
# The sprite is placed by the object's DISPLACEMENT, not by centring its bounding box
# on the projected point: those two differ by SRC_C - projected obj_xyz[0] = (5.3, -0.2)
# px here, because the can lies on its side and its silhouette centroid is not its
# body-frame origin. Centring on the projection would slide the baseline's stamp 5 px
# off the untouched can underneath it and leave a visible double-exposure fringe.
DELTA = SRC_C - _PROJ0
ENDS = {}
for d, key in ((a, "ours"), (b, "base")):
    good = d["uv"][~np.isnan(d["uv"][:, 0])]
    ex, ey = good[-1] + DELTA
    clamped = not (SPR_H / 2 + 8 <= ey <= H - SPR_H / 2 - 8)
    ey = min(max(ey, SPR_H / 2 + 8), H - SPR_H / 2 - 8)
    ex = min(max(ex, SPR_W / 2 + 6), W - SPR_W / 2 - 6)
    ENDS[key] = (ex, ey, clamped)


def _under_sprite(pt, pad=6.0):
    """True where a mark would be hidden by one of the two stamped end states."""
    for ex, ey, _ in ENDS.values():
        if abs(pt[0] - ex) < SPR_W / 2 + pad and abs(pt[1] - ey) < SPR_H / 2 + pad:
            return True
    return False


# DARK halo: it is what makes a saturated stroke legible over a light scene.
halo = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.92), pe.Normal()]
for d, col, lab in ((b, C_BASE, "Baseline"), (a, C_OURS, "XPURT")):
    uv = d["uv"]
    ax.plot(uv[:, 0], uv[:, 1], color=col, lw=5.4, solid_capstyle="round",
            zorder=3, path_effects=halo)
    # If a path ever did plunge out of frame it would cross its OWN earlier passes;
    # drawn as one polyline it would share a single halo and the temporal order would
    # be unreadable. Re-draw the last descent on a higher layer in that case. (No
    # path leaves the frame in this episode, so this is inert here -- kept so the
    # three splash figures stay one piece of code.)
    fin = uv[~np.isnan(uv[:, 0])]
    if len(fin) > 4 and fin[-1, 1] > H - 4:
        above = np.nonzero(fin[:, 1] < fin[-1, 1] - 170)[0]
        if len(above):
            j = int(above[-1])
            ax.plot(fin[j:, 0], fin[j:, 1], color=col, lw=5.4, solid_capstyle="round",
                    zorder=6, path_effects=halo)
    # Arrowhead at MID-PATH by arc length, not at the endpoint. At the end it is
    # buried inside the end-state sprite. Drawn from the immediately preceding sample
    # so the connector is a couple of pixels -- a longer one renders as a straight
    # line across the scene.
    # A path that never moves gets NO arrowhead: pointing one at a 4 px scribble would
    # invent a direction the run does not have.
    ins = uv[(~np.isnan(uv[:, 0])) & (uv[:, 1] <= H - 2)]
    span = float(np.linalg.norm(np.diff(ins, axis=0), axis=1).sum()) if len(ins) > 1 else 0.0
    if len(ins) > 3 and span >= 40.0:
        step = np.linalg.norm(np.diff(ins, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        i0 = int(np.searchsorted(cum, cum[-1] * ARROW_AT[lab]))
        i0 = min(max(i0, 1), len(ins) - 2)
        # walk outwards from mid-path to the nearest sample that is not buried under
        # an end-state sprite -- here the visible window is the ~27 px of open track
        # between the two cans, and mid-path is not in it.
        # ...and require the local tangent to agree with the run's NET displacement.
        # This path doubles back on itself while the gripper settles, so the sample
        # nearest mid-arc points the wrong way; an arrowhead there would claim the can
        # travelled back towards its start.
        net = ins[-1] - ins[0]
        net = net / (np.linalg.norm(net) + 1e-9)
        order = sorted(range(1, len(ins) - 1), key=lambda k: abs(k - i0))
        def _good(k):
            t = ins[k + 1] - ins[k]
            n = np.linalg.norm(t)
            return n > 0.5 and float(t @ net) / n > 0.6 and not _under_sprite(ins[k])
        i = next((k for k in order if _good(k)), i0)
        tail, tip = ins[i], ins[i + 1]
        # Smaller heads than the other splash figures: the open window between the two
        # cans is only ~27 px tall, and a 1-inch arrowhead would cover both of them.
        for c_, ms in (("#0B0B0B", 62), (col, 46)):
            ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                        arrowprops=dict(arrowstyle="-|>", color=c_, linewidth=0,
                                        mutation_scale=ms, shrinkA=0, shrinkB=0),
                        zorder=10)
    # The start marker rides ABOVE the sprites in this figure (zorder 12, not 5): the
    # baseline never moved the can, so its end-state stamp sits exactly on the start
    # and would otherwise bury the one mark that says so.
    ax.scatter(*uv[0], s=340, marker="o", facecolor="white", edgecolor="#0B0B0B",
               linewidth=2.4, zorder=12)
    # No end marker: the outlined end-state sprite below IS the outcome.

# Now stamp the end states, in the same good/bad colours as the paths.
for d, col, key in ((a, C_OURS, "ours"), (b, C_BASE, "base")):
    stamp(ax, ENDS[key][:2], col, alpha=0.97, z=7)

# outcome glyphs: tick LEFT of the lifted can, cross RIGHT of the untouched one, so
# each sits on empty table nearer its own sprite than the other's.
gl = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ox, oy, _ = ENDS["ours"]
bx, by, _ = ENDS["base"]
ax.text(ox - SPR_W / 2 - 62, oy - 4, "✔", ha="center", va="center", fontsize=132,
        color=C_OURS, zorder=9, path_effects=gl)
ax.text(bx + SPR_W / 2 + 66, by + 10, "✘", ha="center", va="center", fontsize=132,
        color=C_BASE, zorder=9, path_effects=gl)

# Dark outline on the label text too, so a saturated colour reads on a light page.
txt = [pe.Stroke(linewidth=5.2, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
# One line per run rather than a separate outcome word: on this task the difference is
# NOT visible from the paths (the can lifts under a centimetre and the baseline never
# closes the gripper), so the outcome has to be named -- but as part of the name, or the
# two texts compete for the same clear space and collide.
ax.text(ox - SPR_W / 2 - 62, oy - SPR_H / 2 - 10, "XPURT (Lifted)", ha="center",
        va="bottom", fontsize=44, fontweight="bold", color=C_OURS, path_effects=txt,
        zorder=11)
ax.text(bx + SPR_W / 2 + 66, by + 96, "Baseline (Missed)", ha="center", va="top",
        fontsize=44, fontweight="bold", color=C_BASE, path_effects=txt, zorder=11)

fig.savefig(OUT, dpi=200, transparent=True)
print(f"[ok] {OUT}")
print(f"  ours  {OURS:10s} success={a['ok']}  object travel {100*a['trav']:.1f} cm")
print(f"  base  {BASE:10s} success={b['ok']}  object travel {100*b['trav']:.1f} cm")
print(f"  mask  {int(MASK.sum())} px  centroid {np.round(_MASK_C,1)}  "
      f"projected obj_xyz[0] {np.round(_PROJ0,1)}  "
      f"err {np.hypot(*(_MASK_C-_PROJ0)):.1f} px  "
      f"proj-inside-mask={bool(MASK[int(round(_PROJ0[1])), int(round(_PROJ0[0]))])}")
print(f"  ends  ours {np.round(ENDS['ours'][:2],1)}  base {np.round(ENDS['base'][:2],1)}"
      f"  separation {np.hypot(ENDS['ours'][0]-ENDS['base'][0], ENDS['ours'][1]-ENDS['base'][1]):.0f} px"
      f"  (sprite {SPR_W}x{SPR_H})")

#!/usr/bin/env python3
"""Splash figure, close-middle-drawer: one scene, two gripper paths, no chrome.

Companion to fig_splash.py (eggplant), fig_splash_spoon.py and fig_splash_coke.py.
Same policy, same silicon, same episode config -- only the SCHEDULE differs.

THREE THINGS ARE DIFFERENT ABOUT THIS TASK, and none of them is cosmetic.

1.  THE PLOTTED PATH IS THE GRIPPER, NOT THE OBJECT.  There is no free object in this
    scene: the thing being manipulated is a prismatic joint of the cabinet. Every
    ep*_obj_xyz.npy under traces_torque2/drawer_* is all-NaN (checked: 216/216 files,
    0 finite samples), so no object trajectory exists to draw. What is drawn is
    ep*_ee_xyz.npy, the gripper TCP, projected through the same camera. The outcome
    itself is the scalar the harness does record: the drawer's final joint opening,
    episode_stats["qpos"]. The task starts the drawer fully open at 0.200 m and success
    is qpos <= 0.050 m -- checked across all 216 drawer episodes: the largest qpos among
    the 86 successes is 0.050, the smallest among the 130 failures is 0.054. Several
    failures end ABOVE 0.200, i.e. the arm pushed the drawer further open.

2.  NO OBJECT SPRITE.  The other three figures stamp the object's own pixels at its
    measured end position. That needs a colour mask, and this scene has no colour to
    key on -- drawer front, carcass, top and wall are all the same beige (H 20-40,
    S < 0.25). A mask cannot tell the middle drawer from the cabinet around it, and a
    wrong segmentation would be worse than none. So the end states here fall back to
    a SIMPLE MARKER: a filled disc in the run's colour with a dark under-stroke.

3.  THIS TASK IS THE ONE WHERE LATENCY DOES NOT PAY.  Over all 24 episodes cpu685
    scores 12/24 (.500) -- BETTER than every pipelined arm (p105w300 = pipe110fix =
    pipe200fix .417, p150w300 .375, p130w275 .333) and better than the zero-latency
    oracle lat0 (.458). Closing a drawer is a low-precision push with a large target;
    a stale action still lands. The episode below is a real episode, but it is NOT
    representative of the task, and this figure should never be shown as evidence
    that the schedule helps on `close middle drawer`. It does not.

EPISODE SELECTION -- a criterion, not a browse.
Candidate set: episodes where cpu685 FAILS and at least one pipelined arm SUCCEEDS;
XPURT is then the highest-overall-success pipelined arm that succeeds on that episode.
Rank by the outcome gap = final drawer opening(cpu685) - opening(XPURT), in mm, subject
to three readability tests:
  (i)   both projected paths >= 100 px of on-screen arc length;
  (ii)  the two end markers >= 3 marker radii (51 px) apart, so they do not merge;
  (iii) both end markers >= 90 px from EVERY frame edge. The other splash figures cope
        with an end state near an edge by swinging its glyph to the other side, but in
        this scene both runs finish their work in the same bottom-right corner, next to
        the robot; an end state there leaves no room for either glyph or label and the
        figure clips.

  ep arm            qpos_o  qpos_b     gap   Lpx_o  Lpx_b  endsep  edge_o  edge_b  pass
  13 pipe110fix       37.0   215.0   178.0     672    635      81      44      67   no (iii)
   0 p130w275         46.0   200.0   154.0     613    676     122     116      25   no (iii)
  22 p105w300         49.0   191.0   142.0     385    653      92     243     151   YES  <--
   9 p105w300         40.0   147.0   107.0     603    702      86      87      25   no (iii)
  14 pipe200fix       50.0   113.0    63.0     573    682      38     161     125   no (ii)
  (drawer opening in mm; lengths, separation and edge clearance in image px; all 5
  candidates listed. edge_* is the end marker's distance to the nearest frame edge,
  after the out-of-frame clamp.)

ep22 is the only candidate that survives, so it is the pick even though ep13 has the
larger closure gap. XPURT (p105w300 -- 125 ms release cadence, 259 ms latency, 304 ms
mean action age, against cpu685's 685 / 685 / 1008 ms) brings the gripper in along the
drawer face and shuts it from 200 mm to 49 mm. cpu685 overshoots to the right -- its
track runs off the right edge of the frame and back -- and gives up with the drawer
still 191 mm open, 9 mm off its fully-open start: essentially untouched.

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
import PIL.Image as PILImage

HERE = Path(__file__).parent
T = HERE.parent / "traces_torque2"
OUT = HERE / "fig_splash_drawer.png"

TASK, EP = "drawer", 22
OURS, BASE = "p105w300", "cpu685"
# SATURATED, on a light page. These sit on the PHOTO, not on white, and their
# contrast comes from a DARK halo rather than from the stroke's own luminance.
C_OURS, C_BASE = "#00E5FF", "#FF2D55"
# Where along each path (fraction of IN-FRAME arc length) the direction arrowhead sits.
ARROW_AT = {"XPURT": 0.55, "Baseline": 0.55}
# End-marker radius, in image px. The stand-in for the object sprite the other three
# splash figures stamp; see head of file for why there is no sprite here.
R_END = 17.0


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
    ee = np.load(d / f"ep{EP:02d}_ee_xyz.npy")           # NOT obj_xyz -- see head of file
    rec = next(e for e in s["episodes"] if e["episode_id"] == EP)
    return dict(uv=project(ee, K, E), ok=bool(rec["success"]),
                qpos=float(rec["episode_stats"]["qpos"]),
                bg=np.asarray(PILImage.open(d / f"ep{EP:02d}_bg.png").convert("RGB")))


a, b = load(OURS), load(BASE)
H, W = a["bg"].shape[:2]

# Sanity check that the no-object claim above still holds for this trace set.
_OBJ = np.load(T / f"{TASK}_{OURS}" / f"ep{EP:02d}_obj_xyz.npy")
assert not np.isfinite(_OBJ).any(), "obj_xyz is populated now -- draw the OBJECT, not the gripper"


def stamp(ax, centre, colour, z=7):
    """Simple end-state marker: a disc in the run's colour, dark-ringed.

    The fallback the head of this file explains. It says WHERE the gripper finished
    and nothing more -- unlike the other splash figures' sprites it does not claim to
    show the object, because in this scene there is no object to show.
    """
    ax.scatter(*centre, s=1500, marker="o", facecolor=colour, edgecolor="#0B0B0B",
               linewidth=5.0, zorder=z, alpha=0.98)
    ax.scatter(*centre, s=260, marker="o", facecolor="#0B0B0B", edgecolor="none",
               zorder=z + 1, alpha=0.85)


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

# End states first, so the arrowheads below can dodge them.
ENDS = {}
for d, key in ((a, "ours"), (b, "base")):
    good = d["uv"][~np.isnan(d["uv"][:, 0])]
    ex, ey = good[-1]
    clamped = not (R_END + 8 <= ey <= H - R_END - 8 and R_END + 8 <= ex <= W - R_END - 8)
    if clamped:
        # The baseline drives the arm down PAST the cabinet and out of the bottom of
        # the frame (true final projected y = 875 in a 512 px frame). Leaving the
        # frame is itself the failure, so the marker is TRUNCATED to the last point
        # still inside rather than drawn at a coordinate outside the scene. The clamp
        # is cosmetic; the measured drawer opening is unchanged.
        seen = good[(good[:, 1] <= H - 2) & (good[:, 0] >= 0) & (good[:, 0] < W)]
        if len(seen):
            ex, ey = seen[-1] + np.array([0.0, 10.0])
    ey = min(max(ey, R_END + 8), H - R_END - 8)
    ex = min(max(ex, R_END + 8), W - R_END - 8)
    ENDS[key] = (ex, ey, clamped)


# The arrowhead is drawn BACKWARDS from its tip, so a sample that merely sits outside
# a marker still paints a ~48 px triangle across it. Keep-out radii below are that
# length, not a cosmetic margin. The start marker counts too: both runs leave from it,
# so an arrowhead there would be stamped on top of the one mark they share.
_START = a["uv"][~np.isnan(a["uv"][:, 0])][0]
_KEEPOUT = [(ex, ey, R_END + 40.0) for ex, ey, _ in ENDS.values()]
_KEEPOUT.append((_START[0], _START[1], 46.0))


def _blocked(pt):
    return any(np.hypot(pt[0] - x, pt[1] - y) < r for x, y, r in _KEEPOUT)


# DARK halo: it is what makes a saturated stroke legible over a light scene.
halo = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.92), pe.Normal()]
for d, col, lab in ((b, C_BASE, "Baseline"), (a, C_OURS, "XPURT")):
    uv = d["uv"]
    ax.plot(uv[:, 0], uv[:, 1], color=col, lw=5.4, solid_capstyle="round",
            zorder=3, path_effects=halo)
    # The final plunge out of frame crosses the path's OWN earlier passes. Drawn as one
    # polyline it shares a single halo, so the crossings render flat and the temporal
    # order is unreadable. Re-draw the last descent on a higher layer: its dark halo
    # then cuts across the older track, showing which came last.
    fin = uv[~np.isnan(uv[:, 0])]
    if len(fin) > 4 and fin[-1, 1] > H - 4:
        above = np.nonzero(fin[:, 1] < fin[-1, 1] - 170)[0]
        if len(above):
            j = int(above[-1])
            ax.plot(fin[j:, 0], fin[j:, 1], color=col, lw=5.4, solid_capstyle="round",
                    zorder=6, path_effects=halo)
    # Arrowhead at MID-PATH by arc length, not at the endpoint -- at the end it is
    # buried under the end marker. Drawn from the immediately preceding sample so the
    # connector is a couple of pixels; a longer one renders as a straight line across
    # the scene. The sample is then walked outwards until its tangent agrees with the
    # run's net displacement and it is clear of both end markers: a 1000-sample
    # gripper track doubles back on itself constantly, and the raw mid-point sample
    # frequently points backwards.
    # x bounds as well as y: this baseline track leaves the RIGHT edge of the frame
    # and comes back, and an arrowhead placed on the off-frame excursion is invisible.
    ins = uv[(~np.isnan(uv[:, 0])) & (uv[:, 1] <= H - 2) & (uv[:, 0] >= 2) & (uv[:, 0] <= W - 2)]
    if len(ins) > 3:
        step = np.linalg.norm(np.diff(ins, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step)])
        i0 = int(np.searchsorted(cum, cum[-1] * ARROW_AT[lab]))
        i0 = min(max(i0, 1), len(ins) - 2)
        net = ins[-1] - ins[0]
        net = net / (np.linalg.norm(net) + 1e-9)
        order = sorted(range(1, len(ins) - 1), key=lambda k: abs(k - i0))

        def _good(k):
            t = ins[k + 1] - ins[k]
            n = np.linalg.norm(t)
            return n > 1.5 and float(t @ net) / n > 0.7 and not _blocked(ins[k])

        i = next((k for k in order if _good(k)), i0)
        tail, tip = ins[i], ins[i + 1]
        # Slightly smaller heads than the eggplant figure: these tracks are shorter
        # and the two end markers sit only 92 px apart.
        for c_, ms in (("#0B0B0B", 78), (col, 60)):
            ax.annotate("", xy=tuple(tip), xytext=tuple(tail),
                        arrowprops=dict(arrowstyle="-|>", color=c_, linewidth=0,
                                        mutation_scale=ms, shrinkA=0, shrinkB=0),
                        zorder=10)
    ax.scatter(*uv[0], s=340, marker="o", facecolor="white", edgecolor="#0B0B0B",
               linewidth=2.4, zorder=5)

for d, col, key in ((a, C_OURS, "ours"), (b, C_BASE, "base")):
    stamp(ax, ENDS[key][:2], col, z=7)

# outcome glyphs and labels. Everything that happens in this episode is inside one
# band, y 215..326 and x 397..640; outside it the frame is empty scene. So the two
# annotation stacks go OUT of that band -- XPURT left into the open drawer (no path
# reaches x < 397), Baseline up onto the wall -- which keeps each glyph nearer its own
# marker (84 and 94 px) than the other's (175 and 146 px) and off both tracks.
gl = [pe.Stroke(linewidth=9.0, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ox, oy, _ = ENDS["ours"]
bx, by, _ = ENDS["base"]
ax.text(ox - 84, oy, "\u2714", ha="center", va="center", fontsize=132,
        color=C_OURS, zorder=9, path_effects=gl)
ax.text(bx, by - 94, "\u2718", ha="center", va="center", fontsize=132,
        color=C_BASE, zorder=9, path_effects=gl)

# Dark outline on the label text too, so a saturated colour reads on a light page.
txt = [pe.Stroke(linewidth=5.2, foreground="#0B0B0B", alpha=0.95), pe.Normal()]
ax.text(ox - 84, oy - 66, "XPURT", ha="center", va="bottom",
        fontsize=46, fontweight="bold", color=C_OURS, path_effects=txt, zorder=11)
ax.text(bx, by - 158, "Baseline", ha="center", va="bottom",
        fontsize=46, fontweight="bold", color=C_BASE, path_effects=txt, zorder=11)

fig.savefig(OUT, dpi=200, transparent=True)
print(f"[ok] {OUT}")
print(f"  ours  {OURS:10s} success={a['ok']}  drawer left at {1000*a['qpos']:.0f} mm open")
print(f"  base  {BASE:10s} success={b['ok']}  drawer left at {1000*b['qpos']:.0f} mm open")
print(f"  paths gripper TCP (obj_xyz is all-NaN for this task)")
print(f"  ends  ours {np.round(ENDS['ours'][:2],1)} clamped={ENDS['ours'][2]}   "
      f"base {np.round(ENDS['base'][:2],1)} clamped={ENDS['base'][2]}")

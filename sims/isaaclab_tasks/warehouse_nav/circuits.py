# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Circuit LIBRARY for the warehouse drone-nav suite.

A curated set of hand-designed CIRCUITS (distinct challenge types), each with STRUCTURED
within-circuit randomization (obstacle positions sampled inside defined regions + rejection-
checked against the racks and the route, curriculum density, small start/goal/height jitter).
The circuit *structure* is fixed and meaningful; only the clutter/start/goal vary. Training
shuffles circuits across envs (CRL-style) so one policy handles the whole suite.

All coordinates are warehouse-local metres, grounded in the MEASURED geometry
(out/warehouse_geom.json): rack rows at x in {-25.3,-20.4,-15.4,-10.5,-5.5,-0.55,4.4}, each
spanning y in [8.9,24.9] up to z~6; six ~3.95 m-clear aisles between them (centrelines
x in {-22.8,-17.9,-12.9,-8.0,-3.0,1.9}); shelf decks at z ~ 1.15 / 2.65 / 3.95 / 5.8; ceiling 9.3.
"""

from __future__ import annotations

import math

# ---- real Isaac warehouse-prop USDs (relative to ISAAC_NUCLEUS_DIR; resolved by the spawner) ----
PROP_USD = {
    "pallet": "Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
    "crate":  "Environments/Simple_Warehouse/Props/SM_CratePlastic_A_01.usd",
    "box":    "Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd",
    "cone":   "Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    "klt":    "Props/KLT_Bin/small_KLT.usd",
    "forklift": "Props/Forklift/forklift.usd",
}
# "pole" is a primitive (thin vertical cylinder) authored by the spawner, not a USD.

RACK_ROWS_X = [-25.32, -20.37, -15.41, -10.46, -5.5, -0.55, 4.41]
RACK_HALF = 0.65        # keep obstacles this far from a rack-row centreline (in the rack y-band)
RACK_Y = (8.6, 25.1)
RACK_TOP_Z = 6.0
SHELF_DECKS_Z = [1.15, 2.65, 3.95, 5.8]


def point_clear_of_racks(x, y, margin=RACK_HALF):
    """True if (x,y) is not inside a rack footprint (safe to place an obstacle / fly)."""
    if not (RACK_Y[0] <= y <= RACK_Y[1]):
        return True
    return all(abs(x - rx) > margin for rx in RACK_ROWS_X)


# ---- circuit definitions ------------------------------------------------------------------
# Each circuit is a dict:
#   name, blurb
#   start, goal            : (x,y,z)
#   waypoints              : [(x,y,z), ...]  the intended route (preview flythrough + gate centres)
#   gates                  : [((x,y,z), yaw_facing), ...]  frames to thread ([] for none)
#   regions                : [ {"box": (xlo,xhi,ylo,yhi,zlo,zhi), "kinds": [...],
#                               "count": (min,max)} ]  structured obstacle-sample regions
#   vehicles               : [ {"kind": "forklift", "path": [(x,y,z,yawdeg), ...], "moving": bool} ]
#   cam_hint               : a good establishing-view target (x,y,z) for previews

# aisle centrelines (measured) + a rich clutter kind-mix, used to build the whole-warehouse circuit
_AISLE_CX = [-22.84, -17.89, -12.94, -7.98, -3.03, 1.93]
# realistic warehouse clutter (no abstract "pole" tubes — tall crate/box STACKS are the verticals)
_CLUTTER = ["pallet", "crate", "box", "cone", "klt", "crate", "box", "pallet"]


def _grand_regions():
    """Dense clutter across the ENTIRE warehouse footprint: the big open south hall (y[-18,7.5],
    ~50 m we never used) + all six rack aisles (y[9,24.5]). Sums to ~180-260 no-clip obstacles."""
    regs = [{"box": (-23.5, 3.5, -18.0, 7.5, 0.0, 0.0), "kinds": _CLUTTER,
             "count": (80, 120), "stack_prob": 0.7, "stack_h": (1.6, 3.2)}]
    for cx in _AISLE_CX:
        regs.append({"box": (cx - 1.4, cx + 1.4, 9.0, 24.5, 0.0, 0.0), "kinds": _CLUTTER,
                     "count": (16, 24), "stack_prob": 0.65, "stack_h": (1.5, 2.8)})
    return regs


CIRCUITS = [
    {
        "name": "warehouse_grand",
        "blurb": "The WHOLE warehouse packed — cross the dense south hall, then SERPENTINE through three rack aisles. Full 34x75 m footprint.",
        "start": (-8.0, -16.0, 1.4), "goal": (-3.03, 25.2, 1.6),
        # serpentine: south hall -> up aisle -12.94 -> N cross -> down aisle -7.98 -> S cross -> up aisle -3.03
        "waypoints": [(-8.0, -16.0, 1.4), (-11.0, -6.0, 1.4), (-12.94, 3.0, 1.3),
                      (-12.94, 12.0, 1.3), (-12.94, 24.0, 1.4), (-7.98, 25.4, 1.7),
                      (-7.98, 20.0, 1.3), (-7.98, 9.0, 1.3), (-3.03, 7.2, 1.5),
                      (-3.03, 13.0, 1.3), (-3.03, 24.0, 1.6), (-3.03, 25.2, 1.6)],
        "gates": [],
        "regions": _grand_regions(),
        # forklift drives ONE of these candidate lanes per episode (randomized). Obstacles are
        # carved clear of ALL lanes (in sample_obstacles) so it never clips. Lanes stay in the
        # OPEN south hall (racks are y>=8.6; the 3.5 m forklift needs the hall's room).
        # CLOSED-LOOP driving routes (first==last): move_forklift drives continuous forward LAPS
        # around the hall (no there-and-back ping-pong). Each loop stays in the open south hall
        # (racks y>=8.6; the 3.5 m forklift needs the room) and crosses the drone's path.
        "vehicles": [{"kind": "forklift", "moving": True, "lanes": [
            [(-18.0, -12.0, 0.0), (2.0, -12.0, 0.0), (2.0, 4.0, 0.0), (-18.0, 4.0, 0.0), (-18.0, -12.0, 0.0)],   # rectangular hall patrol
            [(-16.0, -10.0, 0.0), (0.0, 3.0, 0.0), (-16.0, 3.0, 0.0), (0.0, -10.0, 0.0), (-16.0, -10.0, 0.0)],   # bowtie (crosses centre)
            [(-18.0, -10.0, 0.0), (-4.0, -12.0, 0.0), (3.0, -7.0, 0.0), (2.0, 1.0, 0.0), (-6.0, 5.0, 0.0), (-17.0, 2.0, 0.0), (-18.0, -10.0, 0.0)],  # rounded oval
            [(-12.0, -11.0, 0.0), (3.0, -11.0, 0.0), (3.0, 3.0, 0.0), (-12.0, 3.0, 0.0), (-12.0, -11.0, 0.0)],   # tighter east-hall loop
        ]}],
        "cam_hint": (-10.0, 2.0, 1.5),
    },
    {
        "name": "aisle_slalom",
        "blurb": "Dense slalom straight down a rack aisle — tight weaving through a cluttered floor.",
        "start": (-8.0, 7.0, 1.2), "goal": (-8.0, 25.0, 1.3),
        "waypoints": [(-8.5, 9.0, 1.2), (-7.4, 12.0, 1.2), (-8.6, 15.0, 1.2),
                      (-7.4, 18.0, 1.2), (-8.5, 21.0, 1.2), (-8.0, 24.0, 1.3)],
        "gates": [],
        "regions": [{"box": (-9.6, -6.4, 8.0, 24.5, 0.0, 0.0),
                     "kinds": ["crate", "pallet", "cone", "box", "box", "klt", "cone", "crate"],
                     "count": (26, 36), "stack_prob": 0.65}],
        "vehicles": [],
        "cam_hint": (-8.0, 16.0, 1.2),
    },
    {
        "name": "cross_shelf",
        "blurb": "Climb OVER a loaded rack and drop into the next aisle — 3D traversal across the shelving.",
        "start": (-8.0, 11.0, 1.3), "goal": (-3.0, 22.0, 1.4),
        "waypoints": [(-8.0, 11.0, 1.3), (-8.0, 14.0, 3.6), (-6.6, 15.0, 6.6),
                      (-4.4, 15.2, 6.6), (-3.0, 16.5, 3.2), (-3.0, 19.0, 1.6), (-3.0, 22.0, 1.4)],
        "gates": [((-8.0, 14.0, 3.6), 0.0), ((-3.0, 19.0, 1.6), 0.0)],  # entry-climb + exit-descent gates
        "regions": [
            {"box": (-9.4, -6.6, 9.0, 13.0, 0.0, 0.0), "kinds": ["crate", "box", "pallet"], "count": (4, 7)},
            {"box": (-4.4, -1.7, 16.5, 22.0, 0.0, 0.0), "kinds": ["crate", "cone", "box", "box"], "count": (5, 9)},
        ],
        "vehicles": [],
        "cam_hint": (-5.5, 15.0, 3.5),
    },
    {
        "name": "vertical_gates",
        "blurb": "A racing line of gates at alternating heights — fly up over one, duck under the next.",
        "start": (-12.9, 8.0, 1.2), "goal": (-12.9, 21.5, 3.4),
        "waypoints": [(-13.0, 10.0, 1.0), (-12.4, 13.0, 2.9), (-13.4, 16.0, 1.3),
                      (-12.6, 19.0, 3.4), (-12.9, 21.5, 3.2)],
        "gates": [((-13.0, 10.0, 1.0), 0.0), ((-12.4, 13.0, 2.9), 0.0),
                  ((-13.4, 16.0, 1.3), 0.0), ((-12.6, 19.0, 3.4), 0.0)],
        "regions": [{"box": (-14.2, -11.6, 11.0, 20.0, 0.0, 0.0),
                     "kinds": ["crate", "cone", "box"], "count": (5, 9)}],
        "vehicles": [],
        "cam_hint": (-12.9, 15.0, 2.2),
    },
    {
        "name": "dynamic_dodge",
        "blurb": "Cross the open loading bay while a forklift drives across your path — time the gap.",
        # drone crosses the OPEN HALL south->north at x=-8; forklift drives E-W across that path
        # (both in the hall so the 3.5 m forklift has room and never clips the racks).
        "start": (-8.0, -13.0, 1.4), "goal": (-8.0, 6.5, 1.4),
        "waypoints": [(-8.0, -13.0, 1.4), (-8.0, -6.0, 1.4), (-8.0, 0.0, 1.4), (-8.0, 6.5, 1.4)],
        "gates": [],
        "regions": [
            {"box": (-17.0, 2.0, -12.0, 5.0, 0.0, 0.0),
             "kinds": ["cone", "pallet", "crate", "box", "klt"], "count": (26, 34), "stack_prob": 0.6, "stack_h": (1.5, 3.0)},
            {"box": (-20.0, -1.0, -16.0, -12.0, 0.0, 0.0),
             "kinds": ["crate", "box", "pallet", "cone", "klt"], "count": (10, 16), "stack_prob": 0.5},
        ],
        # forklift trajectory VARIETY: open routes drive forward once (accel/cruise/decel), closed
        # routes are driven as continuous patrol LAPS. Each seed picks a different lane => a
        # genuinely different way of moving. All repeatedly cross the drone's x=-8 corridor.
        "vehicles": [{"kind": "forklift", "moving": True, "lanes": [
            [(-16.0, -3.0, 0.0), (0.0, -3.0, 0.0), (2.0, -0.5, 0.0), (2.0, 3.5, 0.0), (-3.0, 4.0, 0.0)],   # (open) E across, turn N, back W
            [(2.0, 1.0, 0.0), (-6.0, 0.0, 0.0), (-14.0, -2.0, 0.0), (-15.0, -8.0, 0.0), (-6.0, -10.0, 0.0)],  # (open) W then S, sweeping the bay
            [(-15.0, -9.0, 0.0), (-2.0, -8.0, 0.0), (0.0, -3.0, 0.0), (-12.0, -1.0, 0.0), (-15.0, 4.0, 0.0)],  # (open) serpentine crossing x=-8 twice
            [(-15.0, -9.0, 0.0), (2.0, -9.0, 0.0), (2.0, 4.0, 0.0), (-15.0, 4.0, 0.0), (-15.0, -9.0, 0.0)],  # (loop) rectangular bay patrol - crosses x=-8 twice/lap
            [(-14.0, -8.0, 0.0), (-2.0, 4.0, 0.0), (-14.0, 4.0, 0.0), (-2.0, -8.0, 0.0), (-14.0, -8.0, 0.0)],  # (loop) bowtie/figure-8 - crosses centre every half-lap
        ]}],
        "cam_hint": (-8.0, -3.0, 1.5),
    },
    {
        "name": "mixed_clutter",
        "blurb": "A dense mixed-obstacle field — free navigation to a goal at a varied height.",
        "start": (-17.9, 8.0, 1.3), "goal": (-17.9, 24.0, 1.8),
        "waypoints": [(-17.9, 8.0, 1.3), (-18.4, 13.0, 1.6), (-17.4, 17.0, 1.4),
                      (-18.2, 21.0, 1.9), (-17.9, 24.0, 1.8)],
        "gates": [],
        "regions": [{"box": (-19.5, -16.3, 8.5, 24.0, 0.0, 0.0),
                     "kinds": ["crate", "pallet", "box", "cone", "box", "klt", "crate", "cone", "box"],
                     "count": (34, 48), "stack_prob": 0.65}],
        "vehicles": [],
        "cam_hint": (-17.9, 16.0, 1.5),
    },
]

CIRCUITS_BY_NAME = {c["name"]: c for c in CIRCUITS}


def sample_obstacles(circuit, rng, density=1.0, carve_lanes=None):
    """Structured obstacle sample for one episode via the collision-aware NO-CLIP placer
    (placement.sample_no_clip): true per-kind footprints, rejection vs the measured rack/wall
    AABBs + every placed obstacle + start/goal/gate exclusions, optional stacking. Count scaled
    by `density` (curriculum). Returns [{kind, pos:(x,y,z), yaw, r, stack}] with z already at the
    correct centre/stack height. Verified 0 clips across 200 seeds.

    carve_lanes: if given, keep ONLY these lanes clear (used by the preview which drives one
    chosen lane, so the other candidate lanes can be filled with obstacles); if None, every
    vehicle lane is carved clear (training default, where any lane may be picked per episode)."""
    from . import placement as P
    regions = []
    for reg in circuit["regions"]:
        b = reg["box"]
        regions.append({"box": (b[0], b[1], b[2], b[3]), "kinds": reg["kinds"],
                        "count": reg["count"], "density": density,
                        "stack_prob": reg.get("stack_prob", 0.35),
                        "stack_h": reg.get("stack_h", (1.5, 3.0))})
    excl = [(circuit["start"][0], circuit["start"][1], 1.0),
            (circuit["goal"][0], circuit["goal"][1], 1.0)]
    for gc, _ in circuit["gates"]:
        excl.append((gc[0], gc[1], 1.0))
    # carve forklift lane(s) clear of obstacles (dense exclusion points along each segment)
    if carve_lanes is not None:
        lanes_to_carve = carve_lanes
    else:
        lanes_to_carve = [lane for v in circuit.get("vehicles", []) for lane in v.get("lanes", [])]
    for lane in lanes_to_carve:
        for i in range(len(lane) - 1):
            a, b = lane[i], lane[i + 1]
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(2, int(seg / 1.0))
            for k in range(n + 1):
                t = k / n
                excl.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 1.6))
    # NOTE: the route corridor is intentionally NOT carved — obstacles fill the whole space
    # INCLUDING the drone's path (crowded + challenging, like the kinematic-preview look). The
    # showcase env flies through non-colliding props; the trainable env keeps colliders + curriculum.
    return P.sample_no_clip(regions, rng, exclusions=excl)

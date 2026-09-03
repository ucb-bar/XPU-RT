# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``dense_crossing`` scenario: cross the warehouse through an EXTREMELY packed clutter field.

Unlike the ``circuits`` library (one aisle at a time), this scenario floods the big OPEN
LOADING HALL to the SOUTH of the racks with obstacles across BOTH x and y: pallets, crates
and boxes STACKED 1-3 high, free-standing shelf units, poles, barrels and cones — a genuine
clutter maze. The task is a straight forward crossing from the south end (y ~ -35) to the
north end (y ~ +4), so the drone must WEAVE through the field, kept low by an invisible
"fake ceiling" (a reward/termination term, NOT a physical collider) so it can't cheat by
climbing over everything.

This module is PURE python (no isaaclab import), mirroring ``circuits.py`` — the same
generator feeds both the kinematic preview (``scripts/preview_dense.py``) and the training
env (which uses ``mdp_dense.py`` for the collidable pool + the fake-ceiling / speed reward
terms in ``mdp_nav.py``).

Coordinates are warehouse-local metres, grounded in the MEASURED geometry
(out/warehouse_geom.json): world bbox x[-26.5,7.2] y[-41.4,33.4] z[-0.01,9.3]; the 7 rack
rows occupy y[8.9,24.9], so the south hall (y < ~8) is genuinely empty and safe to pack.
"""

from __future__ import annotations

import math

from sims.isaaclab_tasks.warehouse_nav.circuits import PROP_USD, point_clear_of_racks  # noqa: F401

# ---- obstacle kinds --------------------------------------------------------------------------
# Real Isaac warehouse-prop USDs are reused for pallet/crate/box/cone/klt (see circuits.PROP_USD).
# "pole" and "barrel" are authored primitives (cylinders); "shelf" is a free-standing shelf unit
# approximated by the spawner as posts + decks (a warehouse rack USD did not reliably resolve
# from Environments/Simple_Warehouse/Props/, so we build a guaranteed-render primitive shelf;
# see NOTES). Heights below are the PHYSICAL height (m) used for stacking + collider sizing,
# taken from the measured typical_size z of each prop in out/warehouse_geom.json.
KIND_H = {
    "pallet": 0.21,   # SM_PaletteA (flat base — good stack pedestal)
    "crate":  0.24,   # SM_CratePlastic_A
    "box":    0.50,   # SM_CardBoxA
    "klt":    0.15,   # small KLT bin
    "cone":   0.46,   # S_TrafficCone
    "pole":   2.00,   # primitive cylinder
    "barrel": 0.90,   # primitive cylinder
    "shelf":  2.00,   # primitive free-standing shelf unit
}

# footprint "radius" (m) used for min-spacing rejection so NOTHING clips horizontally.
FOOTPRINT_R = {
    "pallet": 0.75, "crate": 0.30, "box": 0.42, "klt": 0.24,
    "cone": 0.25, "pole": 0.15, "barrel": 0.32, "shelf": 0.70,
}

# props sit on the floor with their pivot at the BASE (verified: circuits preview spawns them at
# translation z=0 and they rest on the ground). Primitives (pole/barrel/shelf) are authored by
# the spawner and handle their own pivot. This flag lets the spawner convert a floor-relative
# base-z into the right translation z per kind.
IS_USD_PROP = {"pallet": True, "crate": True, "box": True, "klt": True, "cone": True,
               "pole": False, "barrel": False, "shelf": False}

STACKABLE_BASE = ("pallet", "crate", "box")   # things you can stack ON
STACK_TOPPERS = ("crate", "box", "klt")        # things you stack on top

# weighted base-kind menu (repeats bias the mix toward a believable warehouse: mostly boxes /
# crates / pallets, punctuated by poles, barrels, the odd free-standing shelf and cones).
BASE_KIND_MENU = [
    "crate", "box", "pallet", "crate", "box", "pole", "pallet", "crate",
    "box", "barrel", "crate", "shelf", "box", "pole", "crate", "cone", "pallet",
]

MARGIN = 0.35   # extra clearance between two obstacle footprints (m)


# ---- scenario definition ---------------------------------------------------------------------
# start/goal: a straight south->north crossing (always-forward = +y). waypoints: a gentle
# x-weave used ONLY by the kinematic preview to carve a navigable weaving corridor through the
# packed field (obstacles within path_clearance of this polyline are rejected, so the preview
# drone threads a visible gap while the rest of the hall stays PACKED). Training does not use
# the corridor — the policy learns its own path.
DENSE_CROSSING = {
    "name": "dense_crossing",
    "blurb": "Cross the warehouse south->north through a wall-to-wall packed clutter maze, kept "
             "low by an invisible ceiling so you must weave, not climb.",
    "start": (-10.0, -35.0, 1.3),
    "goal":  (-10.0, 4.0, 1.4),
    "field": (-23.5, 4.0, -37.0, 6.0),   # (xlo, xhi, ylo, yhi) — packed across BOTH x and y
    "z_ceil": 2.5,                        # fake ceiling (m, env-local)
    "waypoints": [(-10.0, -35.0, 1.3), (-11.8, -27.0, 1.35), (-8.2, -19.0, 1.3),
                  (-11.8, -11.0, 1.35), (-8.2, -3.0, 1.3), (-10.0, 4.0, 1.4)],
    "path_clearance": 1.0,               # keep obstacles this far off the preview weave corridor
    "base_count": (150, 210),            # number of base clusters at density=1.0 (PACKED)
    "stack_prob": 0.55,                  # chance a stackable base gets 1-2 items on top
    "cam_hint": (-10.0, -16.0, 1.3),
}


def _dist_point_to_segment(px, py, ax, ay, bx, by):
    """Shortest distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def near_path(x, y, waypoints, clearance):
    """True if (x,y) is within `clearance` of the polyline through `waypoints` (xy only)."""
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        if _dist_point_to_segment(x, y, a[0], a[1], b[0], b[1]) < clearance:
            return True
    return False


def sample_dense_field(rng, density=1.0, carve_corridor=True):
    """Structured DENSE obstacle sample for the crossing (used by the preview).

    Fills the whole hall footprint with rejection-checked, non-clipping base clusters; a fraction
    of stackable bases get 1-2 items stacked centred on top. Returns a flat list of dicts::

        {"kind": str, "pos": (x, y, z_base), "yaw": float}

    where ``z_base`` is the floor-relative BOTTOM of that item (0.0 for a ground obstacle, or the
    top of the item beneath it in a stack). The spawner converts ``z_base`` to the correct
    translation per kind (USD props are base-pivoted; primitives are centred). Count scales with
    ``density`` in [0,1] (curriculum). If ``carve_corridor`` is True, obstacles near the preview
    weave path are rejected so a navigable weaving gap is left through the packed field.
    """
    spec = DENSE_CROSSING
    xlo, xhi, ylo, yhi = spec["field"]
    wp = spec["waypoints"]
    clear = spec["path_clearance"]
    cmin, cmax = spec["base_count"]
    n_target = max(0, int(round((cmin + (cmax - cmin) * rng.random()) * density)))

    sx, sy, _ = spec["start"]
    gx, gy, _ = spec["goal"]

    placed = []   # (x, y, footprint_r) for spacing rejection
    items = []
    tries = 0
    max_tries = n_target * 60 + 200
    while len(placed) < n_target and tries < max_tries:
        tries += 1
        x = xlo + (xhi - xlo) * rng.random()
        y = ylo + (yhi - ylo) * rng.random()

        base_kind = BASE_KIND_MENU[rng.randrange(len(BASE_KIND_MENU))]
        r = FOOTPRINT_R[base_kind]

        # keep out of the racks (harmless here — field is south of them — but faithful)
        if not point_clear_of_racks(x, y):
            continue
        # leave the preview weave corridor open
        if carve_corridor and near_path(x, y, wp, clear + r):
            continue
        # keep clear of the spawn + goal points
        if (x - sx) ** 2 + (y - sy) ** 2 < (1.2 + r) ** 2:
            continue
        if (x - gx) ** 2 + (y - gy) ** 2 < (1.2 + r) ** 2:
            continue
        # min-spacing vs everything already placed (no clipping)
        if any((x - px) ** 2 + (y - py) ** 2 < (r + pr + MARGIN) ** 2 for px, py, pr in placed):
            continue

        placed.append((x, y, r))
        yaw = rng.random() * 2 * math.pi
        items.append({"kind": base_kind, "pos": (x, y, 0.0), "yaw": yaw})

        # stacking: place 1-2 smaller items centred on top of a stackable base
        if base_kind in STACKABLE_BASE and rng.random() < spec["stack_prob"]:
            z = KIND_H[base_kind]
            n_top = 1 + (1 if rng.random() < 0.4 else 0)
            for _ in range(n_top):
                top_kind = STACK_TOPPERS[rng.randrange(len(STACK_TOPPERS))]
                # topper must not overhang the base -> only jitter within the spare footprint
                slack = max(0.0, r - FOOTPRINT_R[top_kind])
                jx = (rng.random() - 0.5) * 2 * min(0.08, slack)
                jy = (rng.random() - 0.5) * 2 * min(0.08, slack)
                items.append({"kind": top_kind, "pos": (x + jx, y + jy, z),
                              "yaw": rng.random() * 2 * math.pi})
                z += KIND_H[top_kind]
    return items

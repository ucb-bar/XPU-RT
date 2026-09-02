# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-aware NO-CLIP obstacle placement (W1).

The old sampler used a single fixed 0.55 m spacing, but props have very different real
footprints (measured via the REPL daemon): pallet 1.21x1.00, box 0.70x0.50, crate 0.62x0.50,
cone 0.34, klt 0.20x0.30, forklift 1.21x3.49. A fixed spacing therefore let pallets/boxes/
forklifts clip each other and the racks. This module places obstacles with TRUE per-kind
footprints, rejection-checked against (a) the measured rack/wall AABBs and (b) every already-
placed obstacle, plus route/start/goal exclusion — and provides a verifier that asserts zero
overlaps. Reference: VisFly `safe_generate` (100-retry collision-free), CRL `random_obstacle.py`
exclusion zones.

Coordinates are warehouse-local metres (out/warehouse_geom.json): rack rows along x at
RACK_ROWS_X, each occupying |x-row| < RACK_HALF for y in RACK_Y, up to z~6.
"""

from __future__ import annotations

import math

# per-kind conservative CIRCULAR footprint radius (metres) = 0.5*diagonal + 0.1 margin,
# from daemon-measured USD AABBs. Used for both prop-prop and prop-rack rejection.
FOOTPRINT_R = {
    "pole": 0.22,
    "cone": 0.30,
    "klt": 0.28,
    "crate": 0.50,
    "box": 0.55,
    "pallet": 0.88,
    "barrel": 0.40,
    "forklift": 1.95,
}
# ground-anchored centre-z per kind (props sit on the floor; pole is a 2 m cylinder)
KIND_Z = {"pole": 1.0, "cone": 0.0, "klt": 0.0, "crate": 0.0, "box": 0.0,
          "pallet": 0.0, "barrel": 0.0, "forklift": 0.0}
# height of the top face (for STACKING a smaller prop on top)
KIND_TOP = {"pallet": 0.21, "crate": 0.44, "box": 0.50, "klt": 0.15}

# BIGGER props: scale the clutter props up so fewer of them fill the same space (fewer physics
# bodies for the same visual clutter + tall-barrier avoidance). Applied to footprints AND stack
# heights here, and to the prop mesh scale at spawn (mdp_obstacles/_prop_spawn + preview). The
# forklift/pole keep their real size.
PROP_SCALE = 1.2
_SCALED = ("cone", "klt", "crate", "box", "pallet")
for _k in _SCALED:
    FOOTPRINT_R[_k] *= PROP_SCALE
    if _k in KIND_TOP:
        KIND_TOP[_k] *= PROP_SCALE

RACK_ROWS_X = [-25.32, -20.37, -15.41, -10.46, -5.5, -0.55, 4.41]
RACK_HALF = 0.55          # rack footprint half-width in x
RACK_Y = (8.6, 25.1)      # rack rows span this y-band
WALL_X = (-26.5, 7.2)     # outer walls (keep obstacles inside)
WALL_Y = (-41.4, 33.4)


def clear_of_racks(x, y, r):
    """True if a disc of radius r at (x,y) does not overlap any rack row or the outer walls."""
    if x - r < WALL_X[0] + 0.3 or x + r > WALL_X[1] - 0.3:
        return False
    if y - r < WALL_Y[0] + 0.3 or y + r > WALL_Y[1] - 0.3:
        return False
    if RACK_Y[0] - r <= y <= RACK_Y[1] + r:
        for rx in RACK_ROWS_X:
            if abs(x - rx) < RACK_HALF + r:
                return False
    return True


def sample_no_clip(regions, rng, exclusions=(), existing=None, max_tries_per=40):
    """Place obstacles with real footprints, zero clipping.

    regions: list of dicts {"box":(xlo,xhi,ylo,yhi), "kinds":[...], "count":(min,max), "density":f}
    exclusions: list of (x,y,radius) kept clear (start/goal/route waypoints/gates)
    existing: optional list of prior placements to avoid (each {"kind","pos","r"})
    returns: list of {"kind","pos":(x,y,z),"yaw","r","stack":bool}
    """
    placed = list(existing) if existing else []
    out = []

    def ok(x, y, r):
        if not clear_of_racks(x, y, r):
            return False
        for e in exclusions:
            if (x - e[0]) ** 2 + (y - e[1]) ** 2 < (e[2] + r) ** 2:
                return False
        for p in placed:
            if (x - p["pos"][0]) ** 2 + (y - p["pos"][1]) ** 2 < (p["r"] + r) ** 2:
                return False
        return True

    for reg in regions:
        xlo, xhi, ylo, yhi = reg["box"]
        cmin, cmax = reg["count"]
        dens = reg.get("density", 1.0)
        n = int(round((cmin + (cmax - cmin) * rng.random()) * dens))
        got, tries = 0, 0
        while got < n and tries < n * max_tries_per:
            tries += 1
            kind = reg["kinds"][got % len(reg["kinds"])]
            r = FOOTPRINT_R[kind]
            x = xlo + (xhi - xlo) * rng.random()
            y = ylo + (yhi - ylo) * rng.random()
            if not ok(x, y, r):
                continue
            item = {"kind": kind, "pos": (x, y, KIND_Z[kind]), "yaw": rng.random() * 2 * math.pi,
                    "r": r, "stack": False}
            placed.append(item); out.append(item); got += 1
            # STACK a column of smaller props on top of a stackable base (1-3 high) — real
            # warehouse pallet stacks. Each level's footprint must fit the one below (no overhang).
            # TALL vertical stacks: with prob stack_prob, build a column of crates/boxes on a
            # stackable base up to a target height (default 1.5-3.0 m) so obstacles are real
            # vertical barriers the drone must go AROUND, not over. Each level's footprint <= below.
            sp = reg.get("stack_prob", 0.35)
            hlo, hhi = reg.get("stack_h", (1.5, 3.0))
            if kind in KIND_TOP and rng.random() < sp:
                th = hlo + (hhi - hlo) * rng.random()
                ztop, cur_r, lvl = KIND_TOP[kind], r, 0
                while ztop < th and lvl < 8:
                    top_kinds = [k for k in ("crate", "box") if FOOTPRINT_R[k] <= cur_r + 1e-3]
                    if not top_kinds:
                        break
                    tk = top_kinds[rng.randrange(len(top_kinds))]
                    out.append({"kind": tk, "pos": (x, y, ztop), "yaw": rng.random() * 2 * math.pi,
                                "r": FOOTPRINT_R[tk], "stack": True})
                    cur_r = FOOTPRINT_R[tk]; ztop += KIND_TOP.get(tk, 0.45); lvl += 1
    return out


def verify_no_clip(items, exclusions=()):
    """Return a list of violation strings (empty == perfect). Checks prop-prop (same z-level),
    prop-rack, and exclusion overlaps using the circular footprints."""
    viol = []
    ground = [p for p in items if not p.get("stack")]
    for i, a in enumerate(ground):
        ax, ay, _ = a["pos"]
        if not clear_of_racks(ax, ay, a["r"]):
            viol.append(f"{a['kind']}@({ax:.1f},{ay:.1f}) overlaps rack/wall")
        for b in ground[i + 1:]:
            bx, by, _ = b["pos"]
            if (ax - bx) ** 2 + (ay - by) ** 2 < (a["r"] + b["r"]) ** 2:
                viol.append(f"{a['kind']}@({ax:.1f},{ay:.1f}) clips {b['kind']}@({bx:.1f},{by:.1f})")
        for e in exclusions:
            if (ax - e[0]) ** 2 + (ay - e[1]) ** 2 < (e[2] + a['r']) ** 2:
                viol.append(f"{a['kind']}@({ax:.1f},{ay:.1f}) in exclusion {e}")
    return viol

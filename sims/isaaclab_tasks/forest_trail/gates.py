"""Forest gate course — visual gate frames the drone navigates THROUGH in order (#56).

Adapts the warehouse gate machinery (``warehouse_nav.mdp_gates``) to the forest
frame: the forest trail runs along +x (warehouse ran +y), the drone cruises at
z≈1.0, so gates here FACE +x with their opening in the y-z plane, centred at
z=1.0, weaving within the ±1.5 m corridor. Each gate = 4 thin cuboid bars
(visual only — the expert flies through the centre; obstacles between gates give
the real avoidance challenge).

The gate CENTRES double as the goal-conditioned nav targets: goal = current gate
centre, advance when the drone passes within ``PASS_RADIUS``. This is the mapped
(privileged) goal for Stage 1; Stage 2 replaces it with a vision-derived goal.
"""

from __future__ import annotations

import os
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


def generate_forest_gates(seed: int = 0, n_gates: int = 4, z: float = 1.0):
    """Seeded gate course for DOMAIN RANDOMIZATION (#46). Gates march along +x with
    jittered spacing and alternating (jittered) lateral offset within the corridor,
    so each seed is a distinct but valid course. seed=0 → the canonical fixed course.
    """
    if seed == 0:
        return [(6.0, 0.6, z), (12.0, -0.6, z), (18.0, 0.6, z), (24.0, -0.6, z)]
    rng = random.Random(seed)
    gates, x = [], 6.0
    for i in range(n_gates):
        y = (0.6 if i % 2 == 0 else -0.6) + rng.uniform(-0.35, 0.35)   # jittered weave, stays |y|<1.0
        gates.append((round(x, 2), round(y, 2), z))
        x += rng.uniform(5.0, 7.0)                                     # jittered spacing
    return gates


# Course selected by the FOREST_GATE_SEED env var at IMPORT time (set before importing this
# module → the scene bakes that course). Default 0 = canonical fixed course. This is how
# collect_fused_data / eval_forest_nav_fused randomize the gate layout per run (--gate_seed).
_GATE_SEED = int(os.environ.get("FOREST_GATE_SEED", "0"))
FOREST_GATES = generate_forest_gates(_GATE_SEED)
FOREST_GATE_CENTERS_2D = tuple((x, y) for x, y, _ in FOREST_GATES)  # for the expert/collection goal
PASS_RADIUS = 1.0  # gate counts as passed when the drone comes within this of the centre

_OPEN = 1.4          # clear opening (m)
_BAR = 0.12          # bar cross-section (m)
_THICK = 0.12        # gate depth along the flight axis (x)
_OUTER = _OPEN + 2 * _BAR


def make_forest_gate_scene(gates=FOREST_GATES, collidable: bool = True) -> dict:
    """Return {name: AssetBaseCfg} of cuboid bars for every gate (4 bars each).

    A gate facing +x lies in the y-z plane: left/right vertical bars (offset in ±y),
    top/bottom horizontal bars (offset in ±z). Depth ``_THICK`` runs along x.

    ``collidable=True`` (default) makes the bars STATIC COLLIDERS — clipping a bar
    physically knocks the drone (→ crash), so a flight that threads the opening is
    genuine, not a pass-through of a visual frame. Set False for a soft visual course.
    """
    out: dict = {}
    half = _OPEN / 2 + _BAR / 2
    # (offset_xyz, size_xyz) in the gate-local frame (facing +x):
    bars = [
        ((0.0, -half, 0.0), (_THICK, _BAR, _OUTER)),   # left vertical  (-y)
        ((0.0, +half, 0.0), (_THICK, _BAR, _OUTER)),   # right vertical (+y)
        ((0.0, 0.0, +half), (_THICK, _OUTER, _BAR)),   # top    (+z)
        ((0.0, 0.0, -half), (_THICK, _OUTER, _BAR)),   # bottom (-z)
    ]
    coll = sim_utils.CollisionPropertiesCfg() if collidable else None
    for gi, (gx, gy, gz) in enumerate(gates):
        color = (0.9, 0.45, 0.1)
        for bi, (off, size) in enumerate(bars):
            pos = (gx + off[0], gy + off[1], gz + off[2])
            out[f"gate_{gi}_bar_{bi}"] = AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Gate_{gi}_bar_{bi}",
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, emissive_color=(0.35, 0.15, 0.0)),
                    collision_props=coll,          # static collider when collidable
                    semantic_tags=[("class", "gate")],
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            )
    return out

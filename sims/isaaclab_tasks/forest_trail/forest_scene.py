# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forest scene cfg: extends the steering scene with USD pine trees.

IsaacLab scene configs are dataclasses, so the set of fields has to be known
at class-definition time. Since we want a procedural tree count that's a
parameter (~60 trees by default), we build the configclass dynamically with
``dataclasses.make_dataclass``. This is a one-shot construction at import
time, not per-step or per-env.

Each tree is spawned from ``assets/pine_tree.usda`` (trunk + 3 layered foliage
cones) with per-tree scale jitter derived from a seeded RNG for visual variety.

Curved-trail scenes (``CurvedForestSceneCfg``) approximate the trail with a
polyline of short straight segments.  Trees and humans are placed relative to
the polyline normal rather than the global x-axis, and the trail strip is
rendered as one cuboid per segment.
"""

from __future__ import annotations

import math
from dataclasses import field, make_dataclass
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from sims.isaaclab_tasks.forest_trail.tree_layout import (
    CurvedTrailLayout,
    HumanLayout,
    StraightTrailLayout,
    _min_dist_to_polyline,
    _min_dist_to_trail_strips,
    generate_curved_human_positions,
    generate_curved_trail_segments,
    generate_curved_trail_trees,
    generate_curved_waypoints,
    generate_human_positions,
    generate_straight_trail,
)
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (
    SteeringSceneCfg_WithCamera,
)

# Poly Haven "Pine Sapling Small" (CC0) downloaded to assets/pine_sapling_small/.
# Real-world scale ≈ 1.5 m, so we scale up by ~3× to get ~4.5 m forest trees.
# Use the patched USDA that bypasses UsdTransform2d (unsupported in Isaac Sim RTX).
# Falls back to raw USDC, then to the procedural USDA if files are missing.
_ASSETS = Path(__file__).parent / "assets"
_POLYHAVEN_PATCHED = str(_ASSETS / "pine_sapling_small" / "pine_sapling_small_patched.usda")
_POLYHAVEN_USDC = str(_ASSETS / "pine_sapling_small" / "pine_sapling_small_1k.usdc")
_PINE_TREE_USD = (
    _POLYHAVEN_PATCHED if Path(_POLYHAVEN_PATCHED).is_file() else
    _POLYHAVEN_USDC if Path(_POLYHAVEN_USDC).is_file() else
    str(_ASSETS / "pine_tree.usda")
)
_POLYHAVEN_SCALE = 3.0   # base scale applied before per-tree jitter

# Local procedural fallback — always available offline.
_HUMAN_USD_LOCAL = str(_ASSETS / "human.usda")

# S3_ROOT for the Isaac Sim 5.1 public content bucket.  No Nucleus server
# required — Isaac Sim loads these HTTPS URLs directly.
_S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters"

# Candidate character USDs tried in order; first reachable one wins.
# These are photorealistic DH-Characters from NVIDIA's public Isaac Sim bucket.
# Set to [] to always use the local procedural fallback.
HUMAN_USD_CANDIDATES: list[str] = [
    f"{_S3}/F_Business_02/F_Business_02.usd",
    f"{_S3}/M_Medical_01/M_Medical_01.usd",
    f"{_S3}/F_Medical_01/F_Medical_01.usd",
    f"{_S3}/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    f"{_S3}/female_adult_police_01_new/female_adult_police_01_new.usd",
]

_resolved_human_usd: str | None = None


def _https_reachable(url: str, timeout: float = 3.0) -> bool:
    """HEAD request to confirm the S3 asset exists. No omni.client needed."""
    import urllib.request, urllib.error
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _resolve_human_usd() -> str:
    """Return the best available human USD path (cached after first call).

    Tries each HTTPS candidate with a HEAD request; uses the first reachable
    one.  Falls back to the local procedural USDA if none are reachable (e.g.
    no internet access).  Can be called at any point — does not require the
    Omniverse runtime.
    """
    global _resolved_human_usd
    if _resolved_human_usd is not None:
        return _resolved_human_usd

    for url in HUMAN_USD_CANDIDATES:
        if _https_reachable(url):
            print(f"[forest_scene] Using S3 human asset: {url}")
            _resolved_human_usd = url
            return _resolved_human_usd

    print(f"[forest_scene] S3 not reachable — using local procedural fallback: {_HUMAN_USD_LOCAL}")
    _resolved_human_usd = _HUMAN_USD_LOCAL
    return _resolved_human_usd


# Approximate bounding height of the pine tree USDA (tip of the top cone).
# Used by terminations and layout helpers that need a rough tree size.
TRUNK_HEIGHT = 4.5


# Ground + trail-strip configuration. Using PreviewSurfaceCfg (solid PBR
# colors) avoids Nucleus MDL fetches that may silently fall back to a flat
# default. To upgrade to real textured ground later, swap to MdlFileCfg with
# a verified Nucleus path on your install (e.g. via the Omniverse browser).
GROUND_SIZE = (80.0, 30.0, 0.05)
GROUND_CENTER = (15.0, 0.0, -GROUND_SIZE[2] / 2.0)
GROUND_COLOR = (0.18, 0.28, 0.10)  # forest-floor green
GROUND_ROUGHNESS = 0.95

TRAIL_LENGTH = 30.0
TRAIL_WIDTH = 1.6  # m; matches roughly the corridor we'd want the drone to fly along
TRAIL_THICKNESS = 0.008
TRAIL_CENTER_Z = TRAIL_THICKNESS / 2.0 + 0.002  # tiny lift to avoid z-fighting with ground
TRAIL_COLOR = (0.45, 0.33, 0.20)  # dirt path brown
TRAIL_ROUGHNESS = 0.9


def _make_textured_ground_cfg() -> AssetBaseCfg:
    """Replace the default grid ground plane with a colored PBR cuboid."""
    return AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=GROUND_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=GROUND_COLOR,
                roughness=GROUND_ROUGHNESS,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=GROUND_CENTER),
    )


def _make_curved_ground_cfg(waypoints: list) -> AssetBaseCfg:
    """Ground plane sized to the bounding box of the curved trail waypoints."""
    xs = [w[0] for w in waypoints]
    ys = [w[1] for w in waypoints]
    pad = 15.0  # m of clearance beyond the trail bounding box
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    sx = max(max(xs) - min(xs) + 2 * pad, GROUND_SIZE[0])
    sy = max(max(ys) - min(ys) + 2 * pad, GROUND_SIZE[1])
    return AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, GROUND_SIZE[2]),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=GROUND_COLOR,
                roughness=GROUND_ROUGHNESS,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(cx, cy, -GROUND_SIZE[2] / 2.0)),
    )


def _make_trail_strip_cfg() -> AssetBaseCfg:
    """Brown dirt strip running along the trail centerline.

    Per-env (sits under ``{ENV_REGEX_NS}``) so each env has its own visible
    trail starting at the env origin and going +x.
    """
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TrailStrip",
        spawn=sim_utils.CuboidCfg(
            size=(TRAIL_LENGTH, TRAIL_WIDTH, TRAIL_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TRAIL_COLOR,
                roughness=TRAIL_ROUGHNESS,
                metallic=0.0,
            ),
        ),
        # Centered along the trail length; lifted slightly above ground.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TRAIL_LENGTH / 2.0, 0.0, TRAIL_CENTER_Z)),
    )


def _make_trail_segment_cfg(
    idx: int,
    cx: float,
    cy: float,
    heading: float,
    length: float,
) -> AssetBaseCfg:
    """One cuboid segment of a curved trail strip, rotated to match the heading."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/TrailSeg_{idx:03d}",
        spawn=sim_utils.CuboidCfg(
            size=(length, TRAIL_WIDTH, TRAIL_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=TRAIL_COLOR,
                roughness=TRAIL_ROUGHNESS,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(cx, cy, TRAIL_CENTER_Z),
            rot=_yaw_to_quat(heading),
        ),
    )


def _make_tree_cfg(idx: int, x: float, y: float, scale_xy: float = 1.0, scale_z: float = 1.0) -> AssetBaseCfg:
    """Pine-tree USD asset positioned at ``(x, y)`` in env-local frame.

    Uses the Poly Haven "pine_sapling_small" USDC (real-world scale ~1.5 m)
    with a ``_POLYHAVEN_SCALE`` multiplier to reach full forest-tree height.
    Falls back to the procedural USDA if the USDC file hasn't been downloaded.
    Per-tree scale jitter comes from the caller.
    """
    sx = _POLYHAVEN_SCALE * scale_xy
    sz = _POLYHAVEN_SCALE * scale_z
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Tree_{idx:03d}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PINE_TREE_USD,
            scale=(sx, sx, sz),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 0.0)),
    )


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` radians about +Z."""
    half = yaw / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _make_human_cfg(idx: int, x: float, y: float, yaw: float) -> AssetBaseCfg:
    """Human figure at ``(x, y)`` with given yaw.

    Uses a Nucleus-hosted character when available (see NUCLEUS_HUMAN_CANDIDATES),
    falling back to the procedural ``assets/human.usda``.  Resolution is cached
    after the first call, so all humans in one env use the same asset.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Human_{idx:02d}",
        spawn=sim_utils.UsdFileCfg(usd_path=_resolve_human_usd()),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(x, y, 0.0),
            rot=_yaw_to_quat(yaw),
        ),
    )


def _build_forest_scene_cfg_class(
    layout: StraightTrailLayout,
    human_layout: HumanLayout | None = None,
    class_name: str = "ForestSceneCfg",
) -> type:
    """Dynamically build a configclass with one field per tree (and optional humans).

    The ``ground`` field of the inherited scene cfg is overridden here too,
    so the grid-textured default ground is replaced with a textured cuboid.

    Note: do NOT stash layout/positions as attributes on the resulting class.
    InteractiveScene's _add_entities_from_cfg iterates the cfg attributes and
    rejects anything that isn't a recognized asset cfg type. We expose layout
    information via module-level constants instead.
    """
    positions = generate_straight_trail(layout)
    fields: list[tuple[str, type, "field"]] = []

    # Override the inherited grid ground plane with a textured cuboid.
    fields.append((
        "ground",
        AssetBaseCfg,
        field(default_factory=_make_textured_ground_cfg),
    ))

    # Visible dirt-colored strip along the trail centerline so DroNet's
    # "stay on the trail" cue has something to latch onto visually.
    fields.append((
        "trail_strip",
        AssetBaseCfg,
        field(default_factory=_make_trail_strip_cfg),
    ))

    # Per-tree scale jitter for visual variety. Seeded so the forest layout
    # is reproducible across runs with the same StraightTrailLayout.seed.
    rng = np.random.default_rng(layout.seed + 1000)
    for i, (x, y) in enumerate(positions):
        sxy = float(rng.uniform(0.80, 1.20))  # width/depth ±20 %
        sz = float(rng.uniform(0.75, 1.25))   # height ±25 %
        # Closure capture: bind i, x, y, scales at field-creation time.
        def _factory(idx=i, x=x, y=y, scale_xy=sxy, scale_z=sz):
            return _make_tree_cfg(idx, x, y, scale_xy, scale_z)
        fields.append((f"tree_{i:03d}", AssetBaseCfg, field(default_factory=_factory)))

    # Optional human figures along the trail.
    if human_layout is not None:
        for i, (hx, hy, hyaw) in enumerate(generate_human_positions(human_layout)):
            def _human_factory(idx=i, x=hx, y=hy, yaw=hyaw):
                return _make_human_cfg(idx, x, y, yaw)
            fields.append((f"human_{i:02d}", AssetBaseCfg, field(default_factory=_human_factory)))

    cls = make_dataclass(
        class_name,
        fields=fields,
        bases=(SteeringSceneCfg_WithCamera,),
    )
    return configclass(cls)


def _save_layout_debug_plot(
    waypoints: list,
    tree_positions: list,
    trail_half_w: float,
    class_name: str,
    out_path: str = "/tmp/curved_trail_layout.png",
) -> None:
    """Save a top-down PNG of the curved-trail layout to ``out_path``.

    Shows the trail-strip rectangles, tree trunk positions (black dots),
    foliage canopy circles (translucent green), and the polyline waypoints.

    Uses the matplotlib OO interface (``Figure`` + ``FigureCanvasAgg``)
    deliberately — calling ``matplotlib.use(...)`` or ``pyplot`` here would
    swap the global backend and break the pilot script's interactive FPV
    window.  ``Figure().savefig(...)`` writes a PNG without any backend
    side-effects.
    """
    import math as _math
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle as _Rect, Circle as _Circle
    from matplotlib.transforms import Affine2D as _Aff

    foliage_r = 0.4 * _POLYHAVEN_SCALE * 1.2
    fig = Figure(figsize=(12, 8))
    FigureCanvasAgg(fig)  # attach a non-interactive canvas
    ax = fig.add_subplot(111)
    ax.plot([w[0] for w in waypoints], [w[1] for w in waypoints],
            "k--", lw=0.5, alpha=0.5)
    for i in range(len(waypoints) - 1):
        p0, p1 = waypoints[i], waypoints[i + 1]
        cx = (p0[0] + p1[0]) / 2.0
        cy = (p0[1] + p1[1]) / 2.0
        h = _math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        seg_len = _math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        rect = _Rect((-seg_len / 2, -trail_half_w),
                     seg_len, 2 * trail_half_w,
                     facecolor="saddlebrown", alpha=0.6,
                     edgecolor="black", linewidth=0.5)
        rect.set_transform(_Aff().rotate(h).translate(cx, cy) + ax.transData)
        ax.add_patch(rect)
    for tx, ty in tree_positions:
        ax.add_patch(_Circle((tx, ty), foliage_r,
                             facecolor="green", alpha=0.30,
                             edgecolor="darkgreen", linewidth=0.5))
        ax.plot(tx, ty, "ko", markersize=2)
    for i, w in enumerate(waypoints):
        ax.plot(w[0], w[1], "r^", markersize=8)
        ax.annotate(f"{i}", (w[0], w[1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{class_name}: {len(tree_positions)} trees, "
                 f"trail_half_w={trail_half_w:.2f} m, foliage_r={foliage_r:.2f} m")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"[forest_scene] debug plot saved to {out_path}")


def _build_curved_forest_scene_cfg_class(
    layout: CurvedTrailLayout,
    waypoints: list,
    human_layout: HumanLayout | None = None,
    class_name: str = "CurvedForestSceneCfg",
    save_debug_plot: bool = False,
) -> type:
    """Dynamically build a configclass for a curved-trail forest scene.

    The trail strip is rendered as ``len(waypoints)-1`` per-env cuboid
    segments, each rotated to align with its polyline segment.  Trees are
    placed relative to the polyline normal so they border the trail
    regardless of its curvature.

    Args:
        layout: Curved trail parameters (density, turn limits, seed).
        waypoints: Pre-computed polyline from ``generate_curved_waypoints``.
            Pass explicitly so the same waypoints feed both this function and
            the termination term.
        human_layout: Optional human placement parameters.
        class_name: Name of the generated configclass.
    """
    tree_positions = generate_curved_trail_trees(layout, waypoints)
    trail_segments = generate_curved_trail_segments(waypoints)

    # Diagnostic — surface tree placement stats so a trunk-on-trail bug
    # shows up in the build log before the visual appears.  Both metrics
    # are reported: polyline-perpendicular distance (how the band floor is
    # measured) and strip-rectangle distance (how the rejection check
    # guarantees trunks stay off the visible trail).
    trail_half_w = TRAIL_WIDTH / 2.0
    if tree_positions:
        poly_dists = [_min_dist_to_polyline(x, y, waypoints) for x, y in tree_positions]
        strip_dists = [_min_dist_to_trail_strips(x, y, waypoints, trail_half_w)
                       for x, y in tree_positions]
        on_strip = sum(1 for d in strip_dists if d < 1e-6)
        print(
            f"[forest_scene] curved trail: {len(tree_positions)}/"
            f"{2 * layout.trees_per_side} trees placed; "
            f"min trunk→polyline = {min(poly_dists):.2f} m, "
            f"min trunk→strip = {min(strip_dists):.2f} m; "
            f"on-trail trunks = {on_strip}"
        )
        # Top-down debug PNG (opt-in) so a discrepancy between the layout
        # data and the rendered scene can be diagnosed without staring at
        # the FPV.  Only enabled for runtime-built scenes (i.e., callers
        # who pass save_debug_plot=True), so the import-time default
        # builds don't import matplotlib.
        if save_debug_plot:
            try:
                _save_layout_debug_plot(
                    waypoints, tree_positions, trail_half_w, class_name,
                )
            except Exception as exc:
                print(f"[forest_scene] debug plot skipped: {exc}")

    fields: list[tuple[str, type, "field"]] = []

    # Ground sized to the bounding box of all waypoints.
    def _gnd_factory(wp=waypoints):
        return _make_curved_ground_cfg(wp)
    fields.append(("ground", AssetBaseCfg, field(default_factory=_gnd_factory)))

    # Trail strip: one cuboid per segment, each rotated to match heading.
    for i, (cx, cy, heading, length) in enumerate(trail_segments):
        def _seg_factory(idx=i, cx=cx, cy=cy, h=heading, ln=length):
            return _make_trail_segment_cfg(idx, cx, cy, h, ln)
        fields.append((f"trail_seg_{i:03d}", AssetBaseCfg, field(default_factory=_seg_factory)))

    # Per-tree scale jitter, seeded reproducibly from the layout seed.
    rng = np.random.default_rng(layout.seed + 1000)
    for i, (x, y) in enumerate(tree_positions):
        sxy = float(rng.uniform(0.80, 1.20))
        sz  = float(rng.uniform(0.75, 1.25))
        def _tree_factory(idx=i, x=x, y=y, scale_xy=sxy, scale_z=sz):
            return _make_tree_cfg(idx, x, y, scale_xy, scale_z)
        fields.append((f"tree_{i:03d}", AssetBaseCfg, field(default_factory=_tree_factory)))

    # Optional human figures beside the curved trail.
    if human_layout is not None:
        human_positions = generate_curved_human_positions(human_layout, waypoints)
        for i, (hx, hy, hyaw) in enumerate(human_positions):
            def _human_factory(idx=i, x=hx, y=hy, yaw=hyaw):
                return _make_human_cfg(idx, x, y, yaw)
            fields.append((f"human_{i:02d}", AssetBaseCfg, field(default_factory=_human_factory)))

    cls = make_dataclass(
        class_name,
        fields=fields,
        bases=(SteeringSceneCfg_WithCamera,),
    )
    return configclass(cls)


# ── Straight-trail module-level instances ─────────────────────────────────────
# Default layouts. Variant scene classes are built once at import time so they
# can be referenced as fields on env configclasses (which need a stable type).
DEFAULT_STRAIGHT_LAYOUT = StraightTrailLayout()
DEFAULT_STRAIGHT_POSITIONS = generate_straight_trail(DEFAULT_STRAIGHT_LAYOUT)
DEFAULT_HUMAN_LAYOUT = HumanLayout(trail_length=DEFAULT_STRAIGHT_LAYOUT.trail_length)

ForestSceneCfg = _build_forest_scene_cfg_class(DEFAULT_STRAIGHT_LAYOUT)
ForestSceneCfgWithHumans = _build_forest_scene_cfg_class(
    DEFAULT_STRAIGHT_LAYOUT,
    human_layout=DEFAULT_HUMAN_LAYOUT,
    class_name="ForestSceneCfgWithHumans",
)

# ── Curved-trail module-level instances ───────────────────────────────────────
DEFAULT_CURVED_LAYOUT = CurvedTrailLayout()
DEFAULT_CURVED_WAYPOINTS: list[tuple[float, float, float]] = generate_curved_waypoints(
    DEFAULT_CURVED_LAYOUT
)
# 2-D (x, y) only — used by the off_trail_curved termination term.
DEFAULT_CURVED_WAYPOINTS_2D: tuple[tuple[float, float], ...] = tuple(
    (x, y) for x, y, _ in DEFAULT_CURVED_WAYPOINTS
)
DEFAULT_CURVED_HUMAN_LAYOUT = HumanLayout(
    trail_length=DEFAULT_CURVED_LAYOUT.trail_length
)

CurvedForestSceneCfg = _build_curved_forest_scene_cfg_class(
    DEFAULT_CURVED_LAYOUT,
    DEFAULT_CURVED_WAYPOINTS,
)
CurvedForestSceneCfgWithHumans = _build_curved_forest_scene_cfg_class(
    DEFAULT_CURVED_LAYOUT,
    DEFAULT_CURVED_WAYPOINTS,
    human_layout=DEFAULT_CURVED_HUMAN_LAYOUT,
    class_name="CurvedForestSceneCfgWithHumans",
)

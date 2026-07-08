# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural tree-position generators for the forest-trail testbed.

Layouts are deterministic for a given seed so that all envs in a run see the
same world and so that re-runs are visually consistent (useful for sanity-
checking DroNet's behaviour at fixed locations).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class StraightTrailLayout:
    """Parameters for the straight-trail tree scatter."""

    trail_length: float = 30.0
    """Trail length along the +x axis (m)."""

    corridor_half_width: float = 1.5
    """Drone-clear half-width of the trail (m)."""

    tree_band: tuple[float, float] = (2.5, 6.0)
    """Min/max distance (m) of trees from the trail centreline, on each side."""

    trees_per_side: int = 30
    """Number of trees per side of the trail."""

    margin: float = 1.5
    """Empty buffer (m) at the start of the trail, so the spawn point is open."""

    seed: int = 42


def generate_straight_trail(layout: StraightTrailLayout) -> list[tuple[float, float]]:
    """Return ``[(x, y), ...]`` tree positions in env-local frame for a straight trail.

    ``x`` runs along the trail (from 0 to ``trail_length``); ``y`` is lateral,
    with negative-y being one side and positive-y the other. Half the trees
    are on each side, in the band ``tree_band``.
    """
    rng = random.Random(layout.seed)
    positions: list[tuple[float, float]] = []
    for side in (-1.0, +1.0):
        for _ in range(layout.trees_per_side):
            x = rng.uniform(layout.margin, layout.trail_length)
            y = side * rng.uniform(*layout.tree_band)
            positions.append((x, y))
    return positions


@dataclass
class HumanLayout:
    """Parameters for procedural human placement on a straight trail."""

    trail_length: float = 30.0
    """Trail length along the +x axis (m). Should match the tree layout."""

    num_humans: int = 4
    """Total number of humans to place along the trail."""

    x_min: float = 5.0
    """Earliest x along the trail. Leaves clearance from the spawn point."""

    x_end_margin: float = 2.0
    """Empty buffer at the trail end."""

    lateral_offset: float = 1.3
    """Base distance (m) from the trail centreline. Humans are placed at
    ``±lateral_offset ± lateral_jitter``, alternating sides, so they are
    clearly off the 0.8 m half-width trail."""

    lateral_jitter: float = 0.3
    """±variation (m) added to ``lateral_offset`` for natural spread."""

    seed: int = 7


def generate_human_positions(
    layout: HumanLayout,
) -> list[tuple[float, float, float]]:
    """Return ``[(x, y, yaw_rad), ...]`` for procedural humans on a straight trail.

    Spaced roughly evenly along the trail with small per-human jitter; yaw is
    randomised so the figures face different directions. Deterministic per seed.
    Humans alternate left/right of the trail, offset by ``lateral_offset`` so
    they stand clearly beside the trail rather than on it.
    """
    rng = random.Random(layout.seed)
    if layout.num_humans <= 0:
        return []

    x_max = max(layout.x_min + 0.1, layout.trail_length - layout.x_end_margin)
    span = x_max - layout.x_min
    step = span / max(1, layout.num_humans)

    positions: list[tuple[float, float, float]] = []
    for i in range(layout.num_humans):
        # Even spacing along x, with ±25% jitter so figures don't form a perfect grid.
        x_centre = layout.x_min + (i + 0.5) * step
        x = x_centre + rng.uniform(-0.25 * step, 0.25 * step)
        side = 1.0 if i % 2 == 0 else -1.0   # alternate left / right
        y = side * (layout.lateral_offset + rng.uniform(-layout.lateral_jitter, layout.lateral_jitter))
        yaw = rng.uniform(-math.pi, math.pi)
        positions.append((x, y, yaw))
    return positions


# ── Curved-trail layout ───────────────────────────────────────────────────────

@dataclass
class CurvedTrailLayout:
    """Parameters for a procedurally curved trail.

    The trail is built as a polyline of ``num_segments`` straight segments,
    each ``segment_length`` m long, with a random heading change of up to
    ``max_turn_deg`` between consecutive segments.  The first segment always
    runs along +x so the drone spawn point (near the origin, facing +x) is
    always valid regardless of seed.
    """

    num_segments: int = 8
    """Number of straight segments that approximate the curve."""

    segment_length: float = 4.0
    """Length of each segment (m). Total arc ≈ num_segments × segment_length."""

    max_turn_deg: float = 35.0
    """Maximum heading change (degrees) between consecutive segments.
    Default of 35° gives a clearly curvy trail without doubling back; bump
    to 45–60° for more aggressive turns, or down to 10–15° for gentle bends."""

    corridor_half_width: float = 1.5
    """Drone-clear half-width around the trail centreline (m)."""

    tree_band: tuple[float, float] = (2.5, 5.0)
    """Min/max lateral distance (m) from the centreline for tree placement.

    Floor of 2.5 m gives ~1.7 m clearance from the 0.8 m trail half-width —
    enough that trunks (at 3× Poly Haven scale ≈ 0.3 m radius) stay well
    clear of the trail even with scale jitter.  Some upper-canopy foliage
    will still overhang the trail which is the desired forest look.

    Per-tree rejection sampling against ``_min_dist_to_trail_strips`` (not the
    polyline) ensures no trunk lands on a trail-strip cuboid, even at the
    apex of sharp turns where one strip's rectangle can extend past another's
    perpendicular distance from the polyline."""

    trees_per_side: int = 40
    """Number of trees per side of the trail.  Slightly denser than the
    straight default (30) to compensate for the rejection-sampling losses at
    high-curvature segments."""

    seed: int = 1337

    @property
    def trail_length(self) -> float:
        """Approximate total arc length (m)."""
        return self.num_segments * self.segment_length


def generate_curved_waypoints(
    layout: CurvedTrailLayout,
) -> list[tuple[float, float, float]]:
    """Return ``[(x, y, heading_rad), ...]`` — one vertex per waypoint.

    The first waypoint is always ``(0, 0, 0)`` and the first segment always
    runs straight along +x, so the drone spawn near the origin is always
    cleanly inside the corridor regardless of seed.
    """
    rng = random.Random(layout.seed)
    max_turn = math.radians(layout.max_turn_deg)
    waypoints: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    x, y, heading = 0.0, 0.0, 0.0
    for i in range(layout.num_segments):
        if i > 0:  # keep first segment straight for clean drone spawn
            heading += rng.uniform(-max_turn, max_turn)
        x += layout.segment_length * math.cos(heading)
        y += layout.segment_length * math.sin(heading)
        waypoints.append((x, y, heading))
    return waypoints


def _min_dist_to_polyline(
    px: float,
    py: float,
    waypoints: list[tuple[float, float, float]],
) -> float:
    """Minimum perpendicular distance from ``(px, py)`` to any segment of the polyline."""
    min_d = float("inf")
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            continue
        t = ((px - p0[0]) * dx + (py - p0[1]) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        cx = p0[0] + t * dx
        cy = p0[1] + t * dy
        d = math.hypot(px - cx, py - cy)
        if d < min_d:
            min_d = d
    return min_d


def _min_dist_to_trail_strips(
    px: float,
    py: float,
    waypoints: list[tuple[float, float, float]],
    half_width: float,
) -> float:
    """Minimum distance from ``(px, py)`` to any trail-strip rectangle.

    A trail strip is a ``segment_length × (2 * half_width)`` rectangle
    centred at the segment midpoint, rotated by the segment heading.  This
    is a tighter bound than ``_min_dist_to_polyline`` at sharp turns because
    the strip rectangle extends along its segment beyond the perpendicular
    line you'd test against.  Returns 0 if the point is inside any strip.
    """
    min_d = float("inf")
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-9:
            continue
        # Rotate (px, py) into strip-local frame (origin = strip centre,
        # +x = along-segment, +y = perpendicular).
        cx = (p0[0] + p1[0]) * 0.5
        cy = (p0[1] + p1[1]) * 0.5
        cos_h = dx / seg_len
        sin_h = dy / seg_len
        rx = (px - cx) * cos_h + (py - cy) * sin_h
        ry = -(px - cx) * sin_h + (py - cy) * cos_h
        # Distance to axis-aligned rectangle in local frame.
        ex = max(0.0, abs(rx) - seg_len * 0.5)
        ey = max(0.0, abs(ry) - half_width)
        d = math.hypot(ex, ey)
        if d < min_d:
            min_d = d
    return min_d


def generate_curved_trail_trees(
    layout: CurvedTrailLayout,
    waypoints: list[tuple[float, float, float]],
    trail_half_width: float = 0.8,
) -> list[tuple[float, float]]:
    """Return ``[(x, y), ...]`` tree positions beside a curved trail.

    Each tree is placed by:
    1. Picking a random segment and position ``t ∈ [0, 1]`` along it.
    2. Stepping laterally ``d ∈ tree_band`` metres along that segment's normal.
    3. **Rejecting** the candidate if it intrudes on any trail-strip rectangle
       — i.e. the trunk-to-strip distance is below ``tree_band[0] - trail_half_width``.
       This is the actual visible trail (not the polyline) so it correctly
       handles foldback-style sharp turns where one strip rectangle extends
       far past the polyline-perpendicular distance from another.

    Rejected candidates are retried up to ``max_attempts`` times; if none of
    them fit, that tree slot is silently dropped (better fewer trees than
    trees on the trail).
    """
    rng = random.Random(layout.seed + 100)
    n_segs = len(waypoints) - 1
    positions: list[tuple[float, float]] = []
    band_lo, band_hi = layout.tree_band
    band_span = band_hi - band_lo
    # Required clearance from the strip rectangle.  band_lo measures from the
    # centreline; the strip itself spans ±trail_half_width, so the trunk-to-
    # strip clearance must be at least band_lo - trail_half_width.  Add a
    # 5 cm safety margin to absorb floating-point edge cases and any small
    # overshoot in the rendered strip cuboid extents.
    min_strip_clear = max(0.05, band_lo - trail_half_width + 0.05)
    max_attempts = 30
    for side in (-1.0, +1.0):
        for _ in range(layout.trees_per_side):
            for _attempt in range(max_attempts):
                seg_idx = rng.randint(0, n_segs - 1)
                t = rng.uniform(0.0, 1.0)
                p0, p1 = waypoints[seg_idx], waypoints[seg_idx + 1]
                cx = p0[0] + t * (p1[0] - p0[0])
                cy = p0[1] + t * (p1[1] - p0[1])
                seg_heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                nx = -math.sin(seg_heading)
                ny =  math.cos(seg_heading)
                # Bias placement toward the inner edge of the band so the
                # forest looks densest right next to the trail and thins out
                # with distance.  rand ** 2 squashes values toward 0.
                d = band_lo + band_span * (rng.random() ** 2.0)
                px = cx + side * d * nx
                py = cy + side * d * ny
                strip_d = _min_dist_to_trail_strips(px, py, waypoints, trail_half_width)
                if strip_d >= min_strip_clear:
                    positions.append((px, py))
                    break
            # else: gave up after max_attempts; skip this tree slot
    return positions


def generate_curved_trail_segments(
    waypoints: list[tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    """Return ``[(cx, cy, heading_rad, length), ...]`` — one entry per segment.

    Used to build the trail-strip cuboids in ``forest_scene.py``.
    """
    segments: list[tuple[float, float, float, float]] = []
    for i in range(len(waypoints) - 1):
        p0, p1 = waypoints[i], waypoints[i + 1]
        cx = (p0[0] + p1[0]) / 2.0
        cy = (p0[1] + p1[1]) / 2.0
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        segments.append((cx, cy, heading, length))
    return segments


def generate_curved_human_positions(
    layout: HumanLayout,
    waypoints: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Return ``[(x, y, yaw_rad), ...]`` for humans placed beside a curved trail.

    Arc-length equivalent of ``generate_human_positions``: humans are evenly
    spaced along the total arc with ±25 % jitter, alternating left/right,
    offset by ``layout.lateral_offset`` from the centreline.
    """
    rng = random.Random(layout.seed)
    if layout.num_humans <= 0:
        return []

    # Pre-compute cumulative arc lengths for each segment.
    seg_lens: list[float] = []
    for i in range(len(waypoints) - 1):
        p0, p1 = waypoints[i], waypoints[i + 1]
        seg_lens.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    total_arc = sum(seg_lens)

    x_min_frac = layout.x_min / total_arc
    x_max_frac = max(x_min_frac + 0.01, 1.0 - layout.x_end_margin / total_arc)
    span_frac = x_max_frac - x_min_frac
    step_frac = span_frac / max(1, layout.num_humans)

    def _point_at_arc(target_arc: float):
        """Return (x, y, seg_heading) at a given arc-length distance."""
        arc = 0.0
        for j, slen in enumerate(seg_lens):
            if arc + slen >= target_arc or j == len(seg_lens) - 1:
                t = (target_arc - arc) / max(slen, 1e-9)
                t = max(0.0, min(1.0, t))
                p0, p1 = waypoints[j], waypoints[j + 1]
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                h = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                return x, y, h
            arc += slen
        # fallback: last waypoint
        p = waypoints[-1]
        return p[0], p[1], p[2]

    positions: list[tuple[float, float, float]] = []
    for i in range(layout.num_humans):
        frac = x_min_frac + (i + 0.5) * step_frac
        frac += rng.uniform(-0.25 * step_frac, 0.25 * step_frac)
        frac = max(0.0, min(1.0, frac))

        cx, cy, seg_heading = _point_at_arc(frac * total_arc)

        nx = -math.sin(seg_heading)
        ny =  math.cos(seg_heading)
        d = layout.lateral_offset + rng.uniform(-layout.lateral_jitter, layout.lateral_jitter)
        yaw = rng.uniform(-math.pi, math.pi)

        # Try both sides; pick whichever has more clearance from the polyline.
        # On the inside of a sharp turn the default "alternating side" choice
        # can place the human on the trail itself.
        default_side = 1.0 if i % 2 == 0 else -1.0
        best = None
        best_clear = -1.0
        for side in (default_side, -default_side):
            hx = cx + side * d * nx
            hy = cy + side * d * ny
            clear = _min_dist_to_polyline(hx, hy, waypoints)
            if clear > best_clear + 1e-6:   # default side wins ties
                best_clear = clear
                best = (hx, hy)
        positions.append((best[0], best[1], yaw))

    return positions

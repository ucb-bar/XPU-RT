"""Privileged analytic expert for forest-trail nav (BC teacher for FusedSensorNet).

Uses GROUND-TRUTH geometry (trail centreline + tree/human positions) that the
learned student never sees, and emits the ideal control in the SAME interface
the steering inner-loop consumes: ``(yaw_rate, forward_speed)``. So:

  * BC label   = expert (yaw_rate, forward_speed)   -> FusedSensorNet out_dim=2
  * flight seam = student (yaw_rate, forward_speed)  -> steering_term (reused verbatim
                  from eval_forest_nav / the DroNet pilot)

Control law (pure numpy, batched over N envs, testable without Isaac):
  1. Pure-pursuit centreline follow — project the drone onto the centreline
     (straight: y=0 line; curved: nearest polyline segment), steer the heading
     toward a lookahead point on the line (tangent heading corrected by the
     signed cross-track error).
  2. Potential-field avoidance — obstacles inside ``avoid_radius`` in the forward
     cone push the yaw away and, if close/ahead, reduce forward speed. On the
     default trails obstacles sit outside the 1.5 m corridor so this is ~inert;
     it activates on a harder in-corridor course.

All positions are ENV-LOCAL (subtract ``env.scene.env_origins`` before calling).
"""

from __future__ import annotations

import math

import numpy as np


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class ForestExpert:
    def __init__(
        self,
        mode: str,                       # "straight" | "curved"
        trail_length: float = 30.0,
        waypoints_2d=None,               # (M,2) for curved
        obstacles_xy=None,               # (K,2) env-local tree+human positions
        obstacle_radius: float = 0.4,
        base_speed: float = 1.0,
        lookahead: float = 2.0,
        k_head: float = 2.5,             # heading P-gain -> yaw_rate
        k_cross: float = 1.2,            # cross-track -> heading correction
        goal_slow_radius: float = 1.5,   # start slowing within this range of the goal
        max_yaw_rate: float = 1.5,
        avoid_radius: float = 1.6,       # start avoiding within this range
        k_avoid: float = 1.8,            # repulsive yaw strength
        min_speed_frac: float = 0.4,     # slow to this fraction when blocked ahead
        turn_slow: bool = False,         # slow down when the required heading change is large
        turn_slow_ref: float = 0.9,      # heading error (rad) that maps to full slowdown
        turn_min_frac: float = 0.35,     # floor of the turn-slowdown factor
        yaw_ema: float = 0.0,            # EMA smoothing on the yaw-rate output (0=off) — makes the
                                         # command stream smooth/low-jerk so a BC student can imitate
                                         # it (jerky potential-field yaw is hard to learn from sensors)
    ):
        assert mode in ("straight", "curved")
        self.mode = mode
        self.trail_length = float(trail_length)
        self.base_speed = float(base_speed)
        self.lookahead = float(lookahead)
        self.k_head = float(k_head)
        self.k_cross = float(k_cross)
        self.goal_slow_radius = float(goal_slow_radius)
        self.max_yaw_rate = float(max_yaw_rate)
        self.obstacle_radius = float(obstacle_radius)
        self.avoid_radius = float(avoid_radius)
        self.k_avoid = float(k_avoid)
        self.min_speed_frac = float(min_speed_frac)
        self.turn_slow = bool(turn_slow)
        self.turn_slow_ref = float(turn_slow_ref)
        self.turn_min_frac = float(turn_min_frac)
        self.yaw_ema = float(yaw_ema)
        self._yr_prev = None  # per-env EMA state; reset via reset_smoothing()
        self.obstacles = None if obstacles_xy is None or len(obstacles_xy) == 0 \
            else np.asarray(obstacles_xy, dtype=np.float64).reshape(-1, 2)
        if mode == "curved":
            assert waypoints_2d is not None and len(waypoints_2d) >= 2
            self.pts = np.asarray(waypoints_2d, dtype=np.float64).reshape(-1, 2)
            self.seg = self.pts[1:] - self.pts[:-1]
            self.seg_len_sq = np.maximum((self.seg ** 2).sum(1), 1e-9)

    # -- centreline: tangent heading + signed cross-track error (left +) --
    def _centreline(self, xy: np.ndarray):
        """xy: (N,2) -> (tangent_heading[N], cross_err[N]) env-local."""
        if self.mode == "straight":
            tangent = np.zeros(len(xy))          # +x
            cross = xy[:, 1]                      # y offset from y=0 centreline
            return tangent, cross
        # curved: nearest segment
        p0 = self.pts[:-1]                        # (M-1,2)
        diff = xy[:, None, :] - p0[None, :, :]    # (N,M-1,2)
        t = np.clip((diff * self.seg[None]).sum(2) / self.seg_len_sq[None], 0.0, 1.0)
        closest = p0[None] + t[:, :, None] * self.seg[None]   # (N,M-1,2)
        d = np.linalg.norm(xy[:, None, :] - closest, axis=2)  # (N,M-1)
        j = d.argmin(1)                                       # (N,)
        segs = self.seg[j]                                    # (N,2)
        tangent = np.arctan2(segs[:, 1], segs[:, 0])
        # signed perpendicular offset (left of tangent positive)
        rel = xy - closest[np.arange(len(xy)), j]
        cross = np.cos(tangent) * rel[:, 1] - np.sin(tangent) * rel[:, 0]
        return tangent, cross

    def command(self, xy, yaw, z=None, goal_xy=None):
        """xy:(N,2) env-local, yaw:(N,) world heading -> (yaw_rate[N], speed[N]).

        If ``goal_xy`` (N,2) is given, PURSUE the goal point (goal-conditioned nav)
        instead of the trail centreline: steer the heading toward the goal and slow
        as we close in. Obstacle avoidance is applied either way. This is the
        privileged teacher for the mapped-goal / gate task (#56 Stage 1).
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        yaw = np.asarray(yaw, dtype=np.float64).reshape(-1)
        N = len(xy)

        if goal_xy is not None:
            goal = np.asarray(goal_xy, dtype=np.float64).reshape(-1, 2)
            to_goal = goal - xy                               # (N,2)
            dist_goal = np.linalg.norm(to_goal, axis=1)
            desired_heading = np.arctan2(to_goal[:, 1], to_goal[:, 0])
            speed = self.base_speed * np.clip(dist_goal / self.goal_slow_radius, 0.3, 1.0)
        else:
            tangent, cross = self._centreline(xy)
            # steer back toward the line: subtract a term proportional to cross error
            desired_heading = tangent - np.arctan2(self.k_cross * cross, self.lookahead)
            speed = np.full(N, self.base_speed)
        heading_err = _wrap(desired_heading - yaw)
        yaw_rate = self.k_head * heading_err

        # turn-aware slowdown: cruise fast when aligned, slow into sharp heading changes so a
        # forward-only (yaw-then-translate) controller doesn't overshoot laterally into a wall.
        if self.turn_slow:
            speed = speed * np.clip(1.0 - np.abs(heading_err) / self.turn_slow_ref,
                                    self.turn_min_frac, 1.0)

        # potential-field avoidance
        if self.obstacles is not None:
            for i in range(N):
                d = self.obstacles - xy[i]                    # (K,2)
                dist = np.linalg.norm(d, axis=1) - self.obstacle_radius
                near = dist < self.avoid_radius
                if not near.any():
                    continue
                # bearing of each obstacle relative to the drone heading
                bearing = _wrap(np.arctan2(d[near, 1], d[near, 0]) - yaw[i])
                dn = np.maximum(dist[near], 1e-2)
                infront = np.abs(bearing) < (np.pi / 2)
                if infront.any():
                    # repulsive yaw: steer AWAY from obstacles ahead (opposite their bearing side)
                    w = (1.0 / dn[infront]) * (self.avoid_radius)
                    push = -np.sign(bearing[infront] + 1e-6) * self.k_avoid * w
                    yaw_rate[i] += push.sum()
                    # slow down for the closest obstacle dead ahead
                    closest_ahead = dn[infront].min()
                    frac = np.clip(closest_ahead / self.avoid_radius, self.min_speed_frac, 1.0)
                    speed[i] *= frac

        yaw_rate = np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        speed = np.clip(speed, 0.0, self.base_speed)

        # EMA-smooth the yaw-rate stream so the BC target is low-jerk / imitable.
        if self.yaw_ema > 0.0:
            if self._yr_prev is None or len(self._yr_prev) != len(yaw_rate):
                self._yr_prev = yaw_rate.copy()
            yaw_rate = self.yaw_ema * self._yr_prev + (1.0 - self.yaw_ema) * yaw_rate
            self._yr_prev = yaw_rate.copy()
        return yaw_rate, speed

    def reset_smoothing(self):
        """Clear the yaw-rate EMA state (call at the start of each episode)."""
        self._yr_prev = None


if __name__ == "__main__":
    # smoke: straight trail, drone off-centre to the left, no obstacles -> should
    # command a RIGHT (negative) yaw to return to y=0.
    ex = ForestExpert("straight", trail_length=30.0)
    yr, sp = ex.command(xy=[[5.0, 0.8]], yaw=[0.0])
    print(f"[straight] off-left y=0.8 -> yaw_rate={yr[0]:+.3f} (expect <0), speed={sp[0]:.2f}")
    yr, sp = ex.command(xy=[[5.0, -0.8]], yaw=[0.0])
    print(f"[straight] off-right y=-0.8 -> yaw_rate={yr[0]:+.3f} (expect >0), speed={sp[0]:.2f}")
    yr, sp = ex.command(xy=[[5.0, 0.0]], yaw=[0.0])
    print(f"[straight] centred -> yaw_rate={yr[0]:+.3f} (expect ~0), speed={sp[0]:.2f}")
    # obstacle dead ahead at (7,0), drone at (5,0) heading +x -> slow + steer aside
    ex2 = ForestExpert("straight", obstacles_xy=[[7.0, 0.0]])
    yr, sp = ex2.command(xy=[[5.0, 0.05]], yaw=[0.0])
    print(f"[avoid] obstacle ahead -> yaw_rate={yr[0]:+.3f} (nonzero), speed={sp[0]:.2f} (<1)")
    # curved
    wps = [(0, 0), (5, 0), (10, 2), (15, 2)]
    exc = ForestExpert("curved", waypoints_2d=wps)
    yr, sp = exc.command(xy=[[7.0, 0.0]], yaw=[0.0])
    print(f"[curved] near bend -> yaw_rate={yr[0]:+.3f}, speed={sp[0]:.2f}")
    print("[smoke] expert OK")

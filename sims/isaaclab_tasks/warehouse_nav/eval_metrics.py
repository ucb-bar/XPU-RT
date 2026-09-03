# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantitative navigation-eval metrics (W4). PURE numpy — NO Isaac import, so it is unit-
testable offline and reused by both the live rollout harness (sims/scripts/eval_suite.py) and
any post-hoc trajectory analysis.

Metric definitions follow the reference frameworks so our numbers are comparable:
  * success  = reached goal AND zero collisions  (CRL/VisFly `test()` convention).
  * collision-rate, timeout-rate                  (mutually exclusive with success).
  * time-to-goal, path length, SPL                (SPL = success-weighted path length, the
                                                    standard nav-generalization scalar).
  * min clearance (closest-obstacle distance)     (CRL closest-obstacle plot).
  * mean/max speed, mean |accel|, mean |jerk|     (vitfly gen_metrics smoothness sweep).

A per-episode "trajectory" is a dict of arrays sampled at the CONTROL rate (dt seconds):
  pos        (T,3)  world position
  vel        (T,3)  world linear velocity            (optional; derived from pos if absent)
  min_obs_d  (T,)   nearest-active-obstacle distance (optional; NaN-safe)
  action     (T,A)  raw policy action                (optional; for command smoothness)
  goal       (3,)   goal position                    (optional; for SPL straight-line ref)
plus scalar outcome flags captured from the env at episode end:
  success (bool), collision (bool), timeout (bool).
"""

from __future__ import annotations

import numpy as np


def _finite_diff(x, dt):
    """d/dt of an (T,·) array via forward difference -> (T-1,·)."""
    return np.diff(x, axis=0) / dt


def episode_metrics(traj: dict, dt: float) -> dict:
    """Reduce one episode's sampled trajectory to a flat dict of scalar metrics."""
    pos = np.asarray(traj["pos"], dtype=np.float64)
    T = pos.shape[0]
    m = {
        "steps": int(T),
        "duration_s": float(T * dt),
        "success": bool(traj.get("success", False)),
        "collision": bool(traj.get("collision", False)),
        "timeout": bool(traj.get("timeout", False)),
    }
    if T < 2:
        # degenerate episode (reset-and-die): report zeros, keep the outcome flags.
        m.update(path_len_m=0.0, straight_len_m=0.0, spl=0.0, min_clearance_m=float("nan"),
                 mean_speed_ms=0.0, max_speed_ms=0.0, mean_accel_ms2=0.0, mean_jerk_ms3=0.0,
                 mean_action_rate=0.0, time_to_goal_s=float("nan"))
        return m

    # --- path geometry ---
    step_d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_len = float(step_d.sum())
    m["path_len_m"] = path_len
    # straight-line reference = start -> goal (falls back to start -> last pos if no goal given)
    goal = np.asarray(traj["goal"], dtype=np.float64) if "goal" in traj else pos[-1]
    straight = float(np.linalg.norm(goal - pos[0]))
    m["straight_len_m"] = straight
    # SPL: success-weighted path efficiency in [0,1]; 0 on failure, ~1 on a straight successful run
    m["spl"] = float(m["success"]) * (straight / max(path_len, straight, 1e-6))

    # --- speed / smoothness (prefer recorded vel; else differentiate pos) ---
    if "vel" in traj and np.asarray(traj["vel"]).shape[0] == T:
        vel = np.asarray(traj["vel"], dtype=np.float64)
    else:
        vel = _finite_diff(pos, dt)
    speed = np.linalg.norm(vel, axis=1)
    m["mean_speed_ms"] = float(speed.mean())
    m["max_speed_ms"] = float(speed.max())
    accel = _finite_diff(vel, dt)
    m["mean_accel_ms2"] = float(np.linalg.norm(accel, axis=1).mean()) if accel.shape[0] else 0.0
    jerk = _finite_diff(accel, dt)
    m["mean_jerk_ms3"] = float(np.linalg.norm(jerk, axis=1).mean()) if jerk.shape[0] else 0.0

    # --- command smoothness (mean per-step action change) ---
    if "action" in traj and np.asarray(traj["action"]).shape[0] == T:
        act = np.asarray(traj["action"], dtype=np.float64)
        da = _finite_diff(act, dt)
        m["mean_action_rate"] = float(np.linalg.norm(da, axis=1).mean()) if da.shape[0] else 0.0
    else:
        m["mean_action_rate"] = float("nan")

    # --- clearance (closest obstacle over the whole episode) ---
    if "min_obs_d" in traj:
        d = np.asarray(traj["min_obs_d"], dtype=np.float64)
        d = d[np.isfinite(d)]
        m["min_clearance_m"] = float(d.min()) if d.size else float("nan")
    else:
        m["min_clearance_m"] = float("nan")

    # time-to-goal only meaningful on success (else NaN so it's excluded from means)
    m["time_to_goal_s"] = float(T * dt) if m["success"] else float("nan")
    return m


def aggregate(episodes: list[dict]) -> dict:
    """Aggregate a list of per-episode metric dicts into a summary (rates + NaN-safe means)."""
    n = len(episodes)
    if n == 0:
        return {"n_episodes": 0}
    out = {"n_episodes": n}
    out["success_rate"] = float(np.mean([e["success"] for e in episodes]))
    out["collision_rate"] = float(np.mean([e["collision"] for e in episodes]))
    out["timeout_rate"] = float(np.mean([e["timeout"] for e in episodes]))
    for k in ("spl", "path_len_m", "min_clearance_m", "mean_speed_ms", "max_speed_ms",
              "mean_accel_ms2", "mean_jerk_ms3", "mean_action_rate", "time_to_goal_s"):
        vals = np.array([e.get(k, np.nan) for e in episodes], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
    return out


def speed_sweep(episodes_by_setpoint: dict) -> list[dict]:
    """vitfly-style sweep: {speed_setpoint: [episode dicts]} -> one summary row per setpoint,
    so success/jerk/etc. can be plotted against the commanded forward speed."""
    rows = []
    for sp in sorted(episodes_by_setpoint):
        row = {"speed_setpoint_ms": float(sp)}
        row.update(aggregate(episodes_by_setpoint[sp]))
        rows.append(row)
    return rows

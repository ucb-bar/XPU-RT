# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forest-trail env cfg.

Extends ``TrackSteeringEnvCfg`` (from ``track_steering_vision``) with:

- A scene that adds ~60 cylinder "trees" along a straight or curved trail.
- A reset that spawns the drone at the start of the trail facing +x with a
  small randomization, instead of anywhere in a 2 m square.
- An additional ``off_trail`` termination so the drone resets cleanly when it
  leaves the corridor.

Everything else (action space, observations, inner-loop reward terms) is
inherited unchanged so the trained velocity-tracker checkpoint and DroNet's
output convention work without modification.

Curved-trail variants (``ForestTrailEnvCfg_Curved_*``) use the same spawn
and play-mode reset — the curved trail always starts at the origin going +x
so the straight-trail spawn logic works unchanged.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp import reset_root_state_uniform

from sims.isaaclab_tasks.forest_trail import mdp_terminations
from sims.isaaclab_tasks.forest_trail.forest_scene import (
    DEFAULT_STRAIGHT_LAYOUT,
    DEFAULT_CURVED_WAYPOINTS_2D,
    ForestSceneCfg,
    ForestSceneCfgWithHumans,
    CurvedForestSceneCfg,
    CurvedForestSceneCfgWithHumans,
    _build_curved_forest_scene_cfg_class,
)
from sims.isaaclab_tasks.forest_trail.tree_layout import (
    CurvedTrailLayout,
    HumanLayout,
    generate_curved_waypoints,
)
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (
    TrackSteeringEnvCfg,
    TerminationsCfg as BaseTerminationsCfg,
    EventCfg as BaseEventCfg,
)


@configclass
class ForestEventCfg(BaseEventCfg):
    """Reset spawns drone at trail start, facing +x. Used for training-style runs."""

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            # Spawn near (x ≈ 0.5, y ≈ 0) — i.e. inside the trail corridor at
            # the trail start. Yaw is small so drone faces roughly +x.
            "pose_range": {
                "x": (0.0, 1.0),
                "y": (-0.3, 0.3),
                "z": (0.8, 1.2),
                "yaw": (-math.pi / 12.0, math.pi / 12.0),
                "roll": (-math.pi / 24.0, math.pi / 24.0),
                "pitch": (-math.pi / 24.0, math.pi / 24.0),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class ForestEventCfg_PLAY(BaseEventCfg):
    """Near-deterministic reset for inference / qualitative runs.

    Cuts the broad training-style randomization down to near-zero so the drone
    spawns reliably upright, on the trail, at hover height, with no initial
    velocity. Removes the "erratic startup" you'd otherwise observe while the
    inner-loop policy stabilizes from a perturbed initial state.

    The curved-trail variants reuse this same cfg: the curved trail always
    begins at the origin going +x, so the spawn point is always valid.
    """

    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.4, 0.6),               # 20 cm uncertainty along trail start
                "y": (-0.05, 0.05),            # 10 cm lateral
                "z": (0.95, 1.05),             # ±5 cm of hover height
                "yaw": (-math.pi / 180.0, math.pi / 180.0),    # ±1° yaw
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
            },
            "velocity_range": {                # essentially zero — drone hovers in place
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


# ── Straight-trail terminations ───────────────────────────────────────────────

@configclass
class ForestTerminationsCfg(BaseTerminationsCfg):
    """Adds an off-trail termination on top of time-out + crash."""

    off_trail = DoneTerm(
        func=mdp_terminations.off_trail,
        params={
            # corridor_half_width=1.5 + 1.5 m slack = drone resets when |y|>3
            "lateral_margin": 3.0,
            "end_overrun": 2.0,
            "trail_length": DEFAULT_STRAIGHT_LAYOUT.trail_length,
        },
    )


# ── Curved-trail terminations ─────────────────────────────────────────────────

@configclass
class CurvedForestTerminationsCfg(BaseTerminationsCfg):
    """Off-trail termination for curved trails.

    Uses perpendicular distance to the nearest polyline segment rather than
    the straight-trail axis-aligned check.
    """

    off_trail = DoneTerm(
        func=mdp_terminations.off_trail_curved,
        params={
            "waypoints": DEFAULT_CURVED_WAYPOINTS_2D,
            "lateral_margin": 3.0,
            "end_margin": 2.0,
        },
    )


# ── Straight-trail env cfgs ───────────────────────────────────────────────────

@configclass
class ForestTrailEnvCfg(TrackSteeringEnvCfg):
    """Forest-trail env cfg (full / training-style)."""

    scene: ForestSceneCfg = ForestSceneCfg(num_envs=4096, env_spacing=10.0)
    terminations: ForestTerminationsCfg = ForestTerminationsCfg()
    events: ForestEventCfg = ForestEventCfg()


@configclass
class ForestTrailEnvCfg_PLAY(ForestTrailEnvCfg):
    """Play / inference variant: single env, deterministic obs, tight reset."""

    # Single env for hand-pilot / DroNet inspection.
    scene: ForestSceneCfg = ForestSceneCfg(num_envs=1, env_spacing=10.0)
    # Near-deterministic spawn so we don't waste the first ~second of the run
    # on the inner-loop policy unwinding randomized initial conditions.
    events: ForestEventCfg_PLAY = ForestEventCfg_PLAY()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class ForestTrailEnvCfg_PLAY_WithHumans(ForestTrailEnvCfg_PLAY):
    """Play env with procedural humans placed on the trail.

    Same env as :class:`ForestTrailEnvCfg_PLAY` but the scene cfg is swapped
    for one that adds ~4 humanoid figures along the trail centreline. Useful
    for stressing DroNet against non-tree obstacles in the forward field of
    view.
    """

    scene: ForestSceneCfgWithHumans = ForestSceneCfgWithHumans(num_envs=1, env_spacing=10.0)


# ── Curved-trail env cfgs ─────────────────────────────────────────────────────

@configclass
class ForestTrailEnvCfg_Curved(TrackSteeringEnvCfg):
    """Curved forest-trail env cfg (full / training-style)."""

    scene: CurvedForestSceneCfg = CurvedForestSceneCfg(num_envs=4096, env_spacing=15.0)
    terminations: CurvedForestTerminationsCfg = CurvedForestTerminationsCfg()
    events: ForestEventCfg = ForestEventCfg()


@configclass
class ForestTrailEnvCfg_Curved_PLAY(ForestTrailEnvCfg_Curved):
    """Curved trail, play / inference variant."""

    scene: CurvedForestSceneCfg = CurvedForestSceneCfg(num_envs=1, env_spacing=15.0)
    events: ForestEventCfg_PLAY = ForestEventCfg_PLAY()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class ForestTrailEnvCfg_Curved_PLAY_WithHumans(ForestTrailEnvCfg_Curved_PLAY):
    """Curved trail, play variant with procedural humans."""

    scene: CurvedForestSceneCfgWithHumans = CurvedForestSceneCfgWithHumans(
        num_envs=1, env_spacing=15.0
    )


# ── Custom curved-trail builder ───────────────────────────────────────────────

def make_curved_env_cfg(
    layout: CurvedTrailLayout,
    with_humans: bool = False,
    play: bool = True,
    num_envs: int = 1,
    env_spacing: float = 15.0,
    human_layout: HumanLayout | None = None,
) -> ForestTrailEnvCfg_Curved:
    """Build a curved-trail env cfg with custom layout parameters.

    The default :class:`CurvedForestSceneCfg` is baked at import time from
    :data:`DEFAULT_CURVED_LAYOUT`.  This helper rebuilds the scene class for
    a user-supplied :class:`CurvedTrailLayout` and rewires the off-trail
    termination to the matching waypoints.

    Args:
        layout: Custom curved-trail layout (curvature, length, seed).
        with_humans: If True, populate the trail with procedural humans.
        play: If True, return a PLAY-mode env (1 env, deterministic spawn).
            If False, return the training-mode env.
        num_envs: Number of parallel envs in the scene.
        env_spacing: Spacing between env origins (m).
        human_layout: Optional override for the HumanLayout.  Defaults to
            the standard layout sized to ``layout.trail_length``.

    Returns:
        A fully-wired env cfg ready to pass to ``gym.make(..., cfg=...)``.
    """
    waypoints = generate_curved_waypoints(layout)
    waypoints_2d = tuple((x, y) for x, y, _ in waypoints)

    if with_humans and human_layout is None:
        human_layout = HumanLayout(trail_length=layout.trail_length)

    scene_cls = _build_curved_forest_scene_cfg_class(
        layout,
        waypoints,
        human_layout=human_layout if with_humans else None,
        class_name=f"CurvedForestSceneCfg_Custom_seed{layout.seed}",
        save_debug_plot=True,
    )

    if play:
        cfg_cls = (
            ForestTrailEnvCfg_Curved_PLAY_WithHumans if with_humans
            else ForestTrailEnvCfg_Curved_PLAY
        )
    else:
        cfg_cls = ForestTrailEnvCfg_Curved

    cfg = cfg_cls()
    # Replace the default-layout scene with one built from the user's layout.
    cfg.scene = scene_cls(num_envs=num_envs, env_spacing=env_spacing)
    # Rewire the curved off-trail termination to the new waypoints.
    cfg.terminations.off_trail.params["waypoints"] = waypoints_2d

    # Sanity check: verify the runtime override actually took effect.  If a
    # configclass quirk silently reverted to the import-time default, the
    # rendered scene would not match the overridden layout.
    actual_cls_name = type(cfg.scene).__name__
    if actual_cls_name != scene_cls.__name__:
        raise RuntimeError(
            f"make_curved_env_cfg: scene override did not take effect. "
            f"Expected {scene_cls.__name__}, got {actual_cls_name}. "
            f"This means the rendered scene is using the default "
            f"DEFAULT_CURVED_LAYOUT instead of the user's custom layout."
        )
    print(
        f"[forest_env_cfg] make_curved_env_cfg: scene={scene_cls.__name__}, "
        f"waypoints={len(waypoints_2d)}, max_turn={layout.max_turn_deg}°, "
        f"seed={layout.seed}, with_humans={with_humans}"
    )
    return cfg

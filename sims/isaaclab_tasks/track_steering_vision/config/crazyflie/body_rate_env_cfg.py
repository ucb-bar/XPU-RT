# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment config for full 3-axis CTBR body-rate tracking with Crazyflie.

This is a sibling of ``track_steering_env_cfg.py``: SAME hover-derived structure
(``DirectThrustMomentAction``, the ``SteeringSceneCfg``, the
``SteeringTrackingPPORunnerCfg``), but the command is the full CTBR interface
``[wx, wy, wz, collective_thrust]`` and the reward tracks all three body rates
instead of yaw alone. Nothing in the steering task is modified, and
``model_6998.pt`` stays loadable against its original gym-id.

The policy learns to be a body-rate CONTROLLER: given a commanded rate + IMU
state it outputs thrust + moments so the achieved ``root_ang_vel_b`` tracks the
command. That is exactly the interface a real flight stack (Betaflight/PX4/
Crazyflie rate mode) exposes, so the policy is a drop-in learned rate loop.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from sims.isaaclab_tasks.track_steering_vision import mdp_commands, mdp_rewards
from sims.isaaclab_tasks.track_steering_vision.mdp_actions import DirectThrustMomentActionCfg
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (
    SteeringSceneCfg,
    SteeringSceneCfg_WithCamera,
)

# Standard MDP terms (base_lin_vel, base_ang_vel, projected_gravity, generated_commands,
# last_action, action_l2, is_terminated, time_out, root_height_below_minimum,
# reset_root_state_uniform, ...)
from isaaclab.envs.mdp import *


##
# MDP settings
##


@configclass
class BodyRateCommandsCfg:
    """CTBR body-rate command: [wx, wy, wz, collective_thrust_norm]."""

    body_rate_command = mdp_commands.BodyRateCommandCfg(
        resampling_time_range=(0.5, 1.5),
        ranges=mdp_commands.BodyRateCommandCfg.Ranges(
            wx=(-3.0, 3.0),   # roll rate (rad/s)
            wy=(-3.0, 3.0),   # pitch rate (rad/s)
            wz=(-2.0, 2.0),   # yaw rate (rad/s)
            thrust=(0.4, 0.7),  # normalized collective thrust; hover ~= 1/1.9 = 0.53
        ),
    )


@configclass
class ActionsCfg:
    """Same actuation as the hover/steering task: direct thrust + 3 moments."""

    thrust_moment = DirectThrustMomentActionCfg(
        asset_name="robot",
        body_name="body",
        thrust_to_weight=1.9,
        moment_scale=0.01,
    )


@configclass
class ObservationsCfg:
    """Observations for the MDP (18-D once the command widens to 4)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])
        # The 4-D CTBR command (was the 2-D steering command).
        target_command = ObsTerm(func=generated_commands, params={"command_name": "body_rate_command"})
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset randomization. Wider attitude/rate than steering so the policy
    learns to track from a range of rotational states (agility)."""

    # Near-level starts (matches the proven steering-env reset). Wide-attitude /
    # inverted-recovery starts caused ~97% instant crashes before the policy could
    # learn thrust; agility here comes from the wide COMMAND ranges, not the reset.
    # Wider-attitude recovery is a follow-up curriculum stage.
    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.8, 2.0),
                "yaw": (-math.pi, math.pi),
                "roll": (-math.pi / 12.0, math.pi / 12.0),
                "pitch": (-math.pi / 12.0, math.pi / 12.0),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Body-rate tracking rewards.

    Deliberately NO upright / height-setpoint / lateral-drift terms: a full-agility
    rate tracker must be free to roll/pitch, and an attitude/height *setpoint* term
    directly fights the wx/wy command (the documented "height=50 swamped steering"
    trap). Altitude is handled by the light collective-thrust term + crash penalty.
    """

    # Primary: track the full 3-axis body rate.
    body_rate_tracking = RewTerm(
        func=mdp_rewards.body_rate_tracking,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.6,
                "command_name": "body_rate_command"},
    )

    # Light: produce the commanded collective thrust (keeps it aloft).
    thrust_tracking = RewTerm(
        func=mdp_rewards.collective_thrust_tracking,
        weight=2.0,
        params={"std": 0.3, "command_name": "body_rate_command"},
    )

    # Smoothness. NOTE: use the standard positive-valued action_rate_l2 (sum of
    # squared action deltas) with a NEGATIVE weight so it is a genuine penalty.
    # (mdp_rewards.action_smoothness returns a NEGATIVE value, which combined with
    # a negative weight would *reward* jitter and swamp the tracking objective.)
    action_rate = RewTerm(func=action_rate_l2, weight=-0.01)
    action_magnitude = RewTerm(func=action_l2, weight=-0.001)

    # Termination penalty (crash / timeout).
    termination_penalty = RewTerm(func=is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    crash = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.1})


##
# Environment configuration
##


@configclass
class TrackBodyRateEnvCfg(ManagerBasedRLEnvCfg):
    """CTBR body-rate tracking RL environment."""

    scene: SteeringSceneCfg = SteeringSceneCfg(num_envs=4096, env_spacing=5.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: BodyRateCommandsCfg = BodyRateCommandsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2       # 50 Hz control (100 Hz physics / 2)
        self.episode_length_s = 10.0
        self.sim.dt = 0.01        # 100 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )


@configclass
class TrackBodyRateEnvCfg_EASY(TrackBodyRateEnvCfg):
    """Narrow-range curriculum stage: gentle rates near hover. Resume the EASY
    checkpoint into the full task (``--resume``) for a two-stage curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.body_rate_command.ranges = mdp_commands.BodyRateCommandCfg.Ranges(
            wx=(-1.0, 1.0), wy=(-1.0, 1.0), wz=(-1.0, 1.0), thrust=(0.45, 0.6),
        )
        # gentler resets for the easy stage
        self.events.reset_base.params["pose_range"].update({
            "roll": (-math.pi / 12.0, math.pi / 12.0),
            "pitch": (-math.pi / 12.0, math.pi / 12.0),
        })


@configclass
class TrackBodyRateEnvCfg_PLAY(TrackBodyRateEnvCfg):
    """Play/eval config: camera scene, few envs."""

    scene: SteeringSceneCfg_WithCamera = SteeringSceneCfg_WithCamera(num_envs=50, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class TrackBodyRateEnvCfg_DR(TrackBodyRateEnvCfg):
    """Domain-randomized body-rate env (anti-overfit / sim2real).

    Randomizes the plant per episode so the policy learns a ROBUST rate-control law
    instead of exploiting one fixed Crazyflie's dynamics: actuation authority
    (thrust-to-weight, moment scale), first-order motor lag, sensor noise (already
    via Unoise), and random in-flight velocity pushes (disturbance rejection).
    Compare against the non-DR policy with the ROB (robustness) eval.
    """

    def __post_init__(self):
        super().__post_init__()
        # actuation-authority + motor-lag DR on the shared thrust/moment term
        self.actions.thrust_moment.randomize_thrust_to_weight = (1.6, 2.2)   # ~1.9 +/- 15%
        self.actions.thrust_moment.randomize_moment_scale = (0.008, 0.013)   # ~0.01 +/- 25%
        self.actions.thrust_moment.motor_tau = 0.03                          # 30 ms motor lag
        # observation corruption ON (sensor-noise DR)
        self.observations.policy.enable_corruption = True
        # random in-flight velocity pushes (gust / disturbance rejection)
        self.events.push = EventTerm(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": {
                "x": (-0.6, 0.6), "y": (-0.6, 0.6), "z": (-0.4, 0.4),
                "roll": (-0.8, 0.8), "pitch": (-0.8, 0.8), "yaw": (-0.8, 0.8),
            }},
        )

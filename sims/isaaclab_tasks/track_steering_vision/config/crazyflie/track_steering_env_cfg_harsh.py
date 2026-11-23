# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harsh reward configuration for steering tracking task.

This configuration uses much tighter constraints and harsher penalties
to train a more stable policy.
"""

from __future__ import annotations

import math
import torch

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.sim as sim_utils

# Import MDP functions
import isaaclab.envs.mdp as mdp
from isaaclab.managers import CommandTermCfg

# Import custom MDP functions
from sims.isaaclab_tasks.track_steering_vision import mdp_commands, mdp_rewards
from isaaclab_contrib.mdp import ThrustActionCfg

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene
##

@configclass
class SteeringSceneCfg(InteractiveSceneCfg):
    """Configuration for the steering tracking scene."""

    # Ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Lights
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.9, 0.9, 0.9),
        ),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/distantLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    # Crazyflie drone with thrust actuators
    robot: MultirotorCfg = MultirotorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=MultirotorCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            rps={
                "m1_prop": 0.0,
                "m2_prop": 0.0,
                "m3_prop": 0.0,
                "m4_prop": 0.0,
            },
        ),
        actuators={
            "thrusters": ThrusterCfg(
                thrust_range=(0.0, 6.0),
                thrust_const_range=(1.0e-5, 2.0e-5),
                tau_inc_range=(0.02, 0.05),
                tau_dec_range=(0.005, 0.01),
                torque_to_thrust_ratio=0.06,
                thruster_names_expr=["m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            ),
        },
        rotor_directions=[1, -1, 1, -1],  # CW, CCW, CW, CCW
        # Allocation matrix: [fx, fy, fz, tx, ty, tz] from 4 rotors
        allocation_matrix=[
            [0.0, 0.0, 0.0, 0.0],  # fx (forward)
            [0.0, 0.0, 0.0, 0.0],  # fy (sideways)
            [1.0, 1.0, 1.0, 1.0],  # fz (thrust up)
            [-0.04, -0.04, 0.04, 0.04],  # tx (roll)
            [-0.04, 0.04, 0.04, -0.04],  # ty (pitch)
            [-0.06, 0.06, -0.06, 0.06],  # tz (yaw)
        ],
    )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # Steering command (yaw rate + forward velocity)
    steering_command = mdp_commands.SteeringCommandCfg(
        resampling_time_range=(5.0, 5.0),
        ranges=mdp_commands.SteeringCommandCfg.Ranges(
            yaw_rate=(-1.0, 1.0),  # rad/s
            velocity=(0.5, 2.0),  # m/s
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Direct thrust control for each propeller (Crazyflie has 4 props)
    thrust = ThrustActionCfg(
        asset_name="robot",
        scale=3.0,
        offset=3.0,
        preserve_order=False,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # Robot state
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))

        # Height above ground
        base_height = ObservationTermCfg(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])

        # Command (target yaw rate and velocity)
        target_command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "steering_command"})

        # Last action
        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Reset with random initial state
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.4, 0.6),  # Start near target height
                "yaw": (-math.pi / 4, math.pi / 4),  # Reduced yaw range
                "roll": (-math.pi / 18.0, math.pi / 18.0),  # ±10 degrees
                "pitch": (-math.pi / 18.0, math.pi / 18.0),
            },
            "velocity_range": {
                "x": (0.0, 0.5),  # Small forward velocity
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    """Harsh reward configuration with tight constraints."""

    # PRIMARY: Track steering angle (yaw rate) - tighter tolerance
    steering_tracking = RewardTermCfg(
        func=mdp_rewards.steering_angle_tracking,
        weight=15.0,  # Increased from 10.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.15,  # Tighter from 0.3
            "command_name": "steering_command",
        },
    )

    # Track forward velocity - tighter tolerance
    velocity_tracking = RewardTermCfg(
        func=mdp_rewards.forward_velocity_tracking,
        weight=8.0,  # Increased from 5.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.25,  # Tighter from 0.5
            "command_name": "steering_command",
        },
    )

    # Stay upright - tighter tolerance
    upright = RewardTermCfg(
        func=mdp_rewards.upright_orientation,
        weight=5.0,  # Increased from 3.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.15,  # Tighter from 0.3
        },
    )

    # Maintain height - corrected target and tighter tolerance
    height_tracking = RewardTermCfg(
        func=mdp_rewards.height_tracking,
        weight=4.0,  # Increased from 2.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 0.5,  # Fixed from 1.0
            "std": 0.25,  # Tighter from 0.5
        },
    )

    # Minimize lateral drift
    no_lateral_drift = RewardTermCfg(
        func=mdp_rewards.lateral_drift_penalty,
        weight=2.0,  # Increased from 1.0
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,  # Tighter from 0.5
        },
    )

    # NEW: Harsh penalty for excessive yaw rate (beyond command range)
    yaw_rate_limit_penalty = RewardTermCfg(
        func=lambda env: -torch.clamp(torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2]) - 2.0, min=0.0),
        weight=5.0,  # Strong penalty
    )

    # NEW: Harsh penalty for excessive velocity (beyond reasonable limits)
    velocity_limit_penalty = RewardTermCfg(
        func=lambda env: -torch.clamp(torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0]) - 3.0, min=0.0),
        weight=3.0,
    )

    # NEW: Penalty for excessive height deviation
    height_limit_penalty = RewardTermCfg(
        func=lambda env: -torch.clamp(torch.abs((env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2] - 0.5) - 0.5, min=0.0),
        weight=3.0,
    )

    # Penalize action rate (smoothness) - stronger
    action_rate = RewardTermCfg(
        func=mdp_rewards.action_smoothness,
        weight=-0.02,  # Doubled from -0.01
    )

    # Penalize large actions - stronger
    action_magnitude = RewardTermCfg(func=mdp.action_l2, weight=-0.01)  # Doubled from -0.005

    # Much stronger termination penalty
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-50.0)  # Increased from -10.0


@configclass
class TerminationsCfg:
    """Harsh termination conditions."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    # Terminate if drone crashes (too low)
    crash = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15},  # Raised from 0.1
    )

    # Terminate if drone flies too high
    too_high = TerminationTermCfg(
        func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2] > 2.0,  # Lowered from 3.0
    )

    # NEW: Terminate if spinning too fast (uncontrolled)
    excessive_spin = TerminationTermCfg(
        func=lambda env: torch.abs(env.scene["robot"].data.root_ang_vel_b[:, 2]) > 5.0,  # 5 rad/s limit
    )

    # NEW: Terminate if moving too fast (out of control)
    excessive_velocity = TerminationTermCfg(
        func=lambda env: torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0]) > 5.0,  # 5 m/s limit
    )

    # NEW: Terminate if tilted too much (falling over)
    excessive_tilt = TerminationTermCfg(
        func=lambda env: torch.abs(mdp.root_quat_w(env)[..., 1:3]).max(dim=-1)[0] > 0.7,  # ~89 degrees tilt
    )


##
# Environment configuration
##


@configclass
class TrackSteeringEnvCfg_Harsh(ManagerBasedRLEnvCfg):
    """Harsh configuration for the steering tracking RL environment."""

    # Scene settings
    scene: SteeringSceneCfg = SteeringSceneCfg(num_envs=4096, env_spacing=5.0)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz simulation
        self.decimation = 2  # 50 Hz control
        self.episode_length_s = 10.0  # 10 seconds per episode

        # Set actuator dt to match sim dt
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        # Viewer settings
        self.viewer.eye = (5.0, 5.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class TrackSteeringEnvCfg_Harsh_PLAY(TrackSteeringEnvCfg_Harsh):
    """Configuration for playing with trained policy (fewer envs)."""

    def __post_init__(self):
        super().__post_init__()
        # Make smaller for play
        self.scene.num_envs = 50
        self.episode_length_s = 20.0  # Longer episodes for observation

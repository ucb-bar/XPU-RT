# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for vision-based steering angle tracking with Crazyflie."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import our custom MDP functions (absolute import from FreshScheduler root)
from sims.isaaclab_tasks.track_steering_vision import mdp_rewards
from sims.isaaclab_tasks.track_steering_vision import mdp_commands
from sims.isaaclab_tasks.track_steering_vision.mdp_actions import DirectThrustMomentActionCfg

# Import standard MDP functions from IsaacLab
from isaaclab.envs.mdp import *

# Import Crazyflie configuration (plain articulation without thrusters)
from isaaclab_assets import CRAZYFLIE_CFG


# NOTE: Using plain ArticulationCfg (CRAZYFLIE_CFG from isaaclab_assets)
# instead of MultirotorCfg to avoid conflict between thruster actuators
# and direct wrench control in DirectThrustMomentAction


##
# Scene definition
##


@configclass
class SteeringSceneCfg(InteractiveSceneCfg):
    """Configuration for the steering tracking scene with Crazyflie."""

    # Robot (using plain Articulation for direct wrench control)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # FPV Camera (disabled for state-based training, enable with --enable_cameras for vision-based)
    # fpv_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam",
    #     update_period=0.1,  # 10 Hz for training
    #     height=200,  # Smaller for faster training
    #     width=200,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=18.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.05, 50.0),
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.06, 0.0, 0.01),
    #         rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention
    #         convention="ros",
    #     ),
    # )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.4, 0.6, 0.4)),
    )

    # Lighting
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    steering_command = mdp_commands.SteeringCommandCfg(
        resampling_time_range=(5.0, 5.0),
        ranges=mdp_commands.SteeringCommandCfg.Ranges(
            yaw_rate=(-1.0, 1.0),  # rad/s
            velocity=(0.0, 1.0),  # m/s (easier range: 0-1 instead of 0.5-2.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Direct thrust and moment control (high-level control like working quadcopter env)
    # Action[0]: Total vertical thrust
    # Actions[1:4]: Roll, pitch, yaw moments
    thrust_moment = DirectThrustMomentActionCfg(
        asset_name="robot",
        body_name="body",
        thrust_to_weight=1.9,  # Max thrust = 1.9x robot weight (same as working env)
        moment_scale=0.01,  # Moment scaling (same as working env)
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        base_lin_vel = ObsTerm(func=base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))

        # Height above ground
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])

        # Command (target yaw rate and velocity)
        target_command = ObsTerm(func=generated_commands, params={"command_name": "steering_command"})

        # Last action
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Reset with random initial state
    reset_base = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),  # Start at reasonable flying height
                "yaw": (-math.pi, math.pi),
                "roll": (-math.pi / 12.0, math.pi / 12.0),
                "pitch": (-math.pi / 12.0, math.pi / 12.0),
            },
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.2, 0.2),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.3, 0.3),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary: track steering angle (yaw rate)
    steering_tracking = RewTerm(
        func=mdp_rewards.steering_angle_tracking,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,
            "command_name": "steering_command",
        },
    )

    # Track forward velocity
    velocity_tracking = RewTerm(
        func=mdp_rewards.forward_velocity_tracking,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
            "command_name": "steering_command",
        },
    )

    # Stay upright
    upright = RewTerm(
        func=mdp_rewards.upright_orientation,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.3,
        },
    )

    # Maintain height (CRITICAL: prevents shooting into sky)
    height_tracking = RewTerm(
        func=mdp_rewards.height_tracking,
        weight=50.0,  # Very high weight to compete with action penalty
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": 1.0,
            "std": 0.5,
        },
    )

    # Minimize lateral drift
    no_lateral_drift = RewTerm(
        func=mdp_rewards.lateral_drift_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.5,
        },
    )

    # Penalize action rate (smoothness)
    action_rate = RewTerm(
        func=mdp_rewards.action_smoothness,
        weight=-0.01,
    )

    # Penalize large actions (reduced weight so it doesn't dominate)
    action_magnitude = RewTerm(func=action_l2, weight=-0.001)

    # Termination penalty
    termination_penalty = RewTerm(func=is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=time_out, time_out=True)

    # Terminate if drone crashes (too low)
    crash = DoneTerm(
        func=root_height_below_minimum,
        params={"minimum_height": 0.1},
    )

    # NOTE: Removed "too_high" termination - height_tracking reward provides incentive to stay at target height


##
# Environment configuration
##


@configclass
class TrackSteeringEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the steering tracking RL environment."""

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
        # General settings
        self.decimation = 2  # 50 Hz control (100 Hz physics / 2)
        self.episode_length_s = 10.0

        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )


@configclass
class SteeringSceneCfg_WithCamera(SteeringSceneCfg):
    """Scene configuration with FPV camera enabled for visualization."""

    # FPV Camera
    fpv_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam",
        update_period=0.1,  # 10 Hz
        height=480,  # Higher res for visualization
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.06, 0.0, 0.01),
            rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention
            convention="ros",
        ),
    )


@configclass
class TrackSteeringEnvCfg_PLAY(TrackSteeringEnvCfg):
    """Configuration for playing/testing the trained policy."""

    # Use scene with camera
    scene: SteeringSceneCfg_WithCamera = SteeringSceneCfg_WithCamera(num_envs=50, env_spacing=5.0)

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Disable observation corruption
        self.observations.policy.enable_corruption = False

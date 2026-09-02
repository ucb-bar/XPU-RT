# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``dense_crossing``: cross the warehouse south->north through a wall-to-wall packed clutter field.

A sibling of ``warehouse_nav_env_cfg`` that swaps the single-aisle gate course for a straight
crossing of the OPEN LOADING HALL, flooded with a dense ``RigidObjectCollection`` pool (crates,
boxes, pallets STACKED 1-3 high, poles, barrels, shelf blocks — see ``mdp_dense.py``). An
invisible "fake ceiling" reward/termination (``mdp_nav.fake_ceiling_penalty`` /
``above_fake_ceiling``) keeps the drone low so it must WEAVE, not climb. A forward-speed bonus
rewards fast completion.

Reuses the proven stack: ``CRAZYFLIE_CFG`` + ``VelocityCommandActionCfg`` action, the
``GoalPositionCommand`` (a fixed far-north goal), ``navigation_reward`` (progress + heading +
arrival + obstacle-proximity + -100 collision), and ``obstacle_count_curriculum`` (ramps the
active obstacle count as success rises).
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.mdp import *  # noqa: F401,F403  (base MDP terms)

from sims.isaaclab_tasks.warehouse_nav import mdp_nav
from sims.isaaclab_tasks.warehouse_nav import mdp_obstacles
from sims.isaaclab_tasks.warehouse_nav import mdp_dense
from sims.isaaclab_tasks.warehouse_nav import scenario_dense as SD
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg
from isaaclab_assets import CRAZYFLIE_CFG

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"

_XLO, _XHI, _YLO, _YHI = SD.DENSE_CROSSING["field"]
_Z_CEIL = SD.DENSE_CROSSING["z_ceil"]        # penalty ceiling (m, env-local)
_Z_CEIL_TERM = _Z_CEIL + 1.0                 # hard termination ceiling (grace band above)
_START = SD.DENSE_CROSSING["start"]
_GOAL = SD.DENSE_CROSSING["goal"]


##
# Scene
##
@configclass
class DenseCrossingSceneCfg(InteractiveSceneCfg):
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
    )
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    # dense packed clutter pool (crates/boxes/pallets/poles/barrels/shelves), real colliders
    obstacles = mdp_dense.make_dense_obstacle_collection_cfg(mdp_dense.N_DENSE)
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/body", update_period=0.0, history_length=3)
    dome = AssetBaseCfg(prim_path="/World/DomeLight",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.9, 0.9, 0.95)))


##
# MDP
##
@configclass
class CommandsCfg:
    # A single fixed goal near the far (north) end of the hall — a straight crossing objective.
    goal = mdp_nav.GoalPositionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=mdp_nav.GoalPositionCommandCfg.Ranges(
            pos_x=(_GOAL[0] - 1.0, _GOAL[0] + 1.0),
            pos_y=(_GOAL[1] - 1.5, _GOAL[1] + 1.0),
            pos_z=(1.1, 1.7),
        ),
    )


@configclass
class ActionsCfg:
    velocity = VelocityCommandActionCfg(asset_name="robot", body_name="body")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel)            # noqa: F405
        base_ang_vel = ObsTerm(func=base_ang_vel)            # noqa: F405
        projected_gravity = ObsTerm(func=projected_gravity)  # noqa: F405
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])
        goal = ObsTerm(func=mdp_nav.goal_vector_b, params={"command_name": "goal"})
        obstacles = ObsTerm(func=mdp_nav.nearest_obstacles_b, params={"k": 5})
        last_action = ObsTerm(func=last_action)              # noqa: F405

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # spawn at the SOUTH end of the hall, FACING NORTH (+y) so the forward-only controller heads
    # straight into the crossing; the north goal requires traversing the whole packed field.
    reset_base = EventTerm(
        func=reset_root_state_uniform,  # noqa: F405
        mode="reset",
        params={
            "pose_range": {"x": (_START[0] - 1.0, _START[0] + 1.0),
                           "y": (_START[1] - 1.0, _START[1] + 1.0),
                           "z": (1.0, 1.5),
                           "yaw": (math.pi / 2 - math.pi / 12, math.pi / 2 + math.pi / 12)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)},
        },
    )
    # flood the hall with a packed, stacked, non-clipping obstacle field (curriculum count)
    reset_obstacles = EventTerm(
        func=mdp_dense.reset_dense_field,
        mode="reset",
        params={"cell": 1.7, "stack_prob": 0.4, "margin": 0.15},
    )
    reset_reached = EventTerm(func=mdp_dense.reset_reached_flag, mode="reset")
    randomize_light = EventTerm(
        func=mdp_obstacles.randomize_dome_light,
        mode="reset",
        params={"intensity_range": (150.0, 650.0)},
    )


@configclass
class RewardsCfg:
    # progress + heading + arrival + obstacle-proximity (VisFly/aerial_gym port)
    navigation = RewTerm(func=mdp_nav.navigation_reward, weight=1.0, params={"command_name": "goal"})
    # SPEED bonus: reward closing speed toward the goal (faster completion)
    speed = RewTerm(func=mdp_nav.forward_speed_bonus, weight=0.5,
                    params={"command_name": "goal", "cap": 2.5})
    # FAKE CEILING: penalize (per metre) crossing the invisible z limit so it can't fly over
    ceiling = RewTerm(func=mdp_nav.fake_ceiling_penalty, weight=-5.0, params={"z_ceil": _Z_CEIL})
    # hard -100 on any failure termination (collision / ground / blatant ceiling fly-over)
    collision = RewTerm(func=is_terminated, weight=-100.0)  # noqa: F405
    # success bookkeeping (0 reward; stashes _ep_success/_ep_crash for the curriculum)
    success_metric = RewTerm(func=mdp_dense.update_goal_success, weight=1.0,
                             params={"command_name": "goal", "radius": 1.2})


@configclass
class CurriculumCfg:
    # ramp the active obstacle count from a modest field to the full packed pool as success rises
    obstacle_count = CurrTerm(
        func=mdp_obstacles.obstacle_count_curriculum,
        params={"min_level": 12, "max_level": mdp_dense.N_DENSE, "check_after": 512,
                "up": 0.7, "down": 0.6, "inc": 8, "dec": 4},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)          # noqa: F405
    crash_ground = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.1})  # noqa: F405
    collision = DoneTerm(
        func=illegal_contact,                                  # noqa: F405
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact", body_names="body")},
    )
    # hard backstop: end the episode on a blatant fly-over above the ceiling (grace band)
    ceiling = DoneTerm(func=mdp_nav.above_fake_ceiling, params={"z_ceil": _Z_CEIL_TERM})


@configclass
class DenseCrossingEnvCfg(ManagerBasedRLEnvCfg):
    # env_spacing must exceed the ~43 m field footprint so per-env warehouses don't overlap
    scene: DenseCrossingSceneCfg = DenseCrossingSceneCfg(num_envs=64, env_spacing=80.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2           # 50 Hz control
        self.episode_length_s = 30.0  # a full hall crossing needs travel time
        self.sim.dt = 0.01            # 100 Hz physics
        self.sim.render_interval = self.decimation


@configclass
class DenseCrossingEnvCfg_PLAY(DenseCrossingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crowd-navigation drone task: cross an open loading hall full of walking people at a FIXED
altitude, from a start on the south side to a goal on the far (north) side, without colliding
with anyone.

Shares the proven warehouse-nav stack (Crazyflie articulation, geometric velocity controller,
proprioceptive obs, aerial_gym/VisFly-style reward, contact-based collision termination) and
swaps in the crowd scenario (``scenario_crowd``):

  * PLANAR control — ``PlanarVelocityCommandAction`` maps [-1,1]^4 -> [vx, vy, vz=0, yawrate];
    the drone moves forward/back/sideways + yaws at a held altitude, never up/down. An
    ``altitude_hold_penalty`` reward pins z to Z_TARGET against residual drift.
  * A CROWD of ``N_CROWD`` person capsules that mutually avoid each other (social-force /
    VO-lite ``move_crowd``) and walk in many directions across the crowd zone.
  * Reward = progress-to-goal + people-proximity penalty + people-approach penalty
    + altitude-hold penalty + −100 collision.
"""

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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.mdp import *  # noqa: F401,F403  (base MDP terms: last_action, base_lin_vel, ...)

from sims.isaaclab_tasks.warehouse_nav import mdp_nav
from sims.isaaclab_tasks.warehouse_nav import mdp_obstacles
from sims.isaaclab_tasks.warehouse_nav import scenario_crowd
from isaaclab_assets import CRAZYFLIE_CFG

# open loading-hall crowd zone (env-local metres) + the held cruise altitude.
_ZONE = scenario_crowd.CROWD_ZONE
_Z = scenario_crowd.Z_TARGET

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"


##
# Scene
##
@configclass
class CrowdSceneCfg(InteractiveSceneCfg):
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
    )
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    # the walking crowd: N_CROWD person capsules, real colliders + person tag
    crowd = scenario_crowd.make_crowd_collection_cfg()
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/body", update_period=0.0, history_length=3)
    dome = AssetBaseCfg(prim_path="/World/DomeLight",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.9, 0.9, 0.95)))


##
# MDP
##
@configclass
class CommandsCfg:
    # A single far-side goal, sampled once per episode near the NORTH edge of the crowd zone
    # (the drone spawns on the SOUTH edge), so the objective is to cross the crowd. Held at the
    # fixed cruise altitude (matches the planar constraint).
    goal = mdp_nav.GoalPositionCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        ranges=mdp_nav.GoalPositionCommandCfg.Ranges(
            pos_x=(-6.0, 1.0), pos_y=(_ZONE["y"][1] - 2.0, _ZONE["y"][1]), pos_z=(_Z, _Z),
        ),
    )


@configclass
class ActionsCfg:
    # PLANAR velocity command: [-1,1]^4 -> [vx (fwd/back), vy (lateral), vz=0, yawrate].
    velocity = scenario_crowd.PlanarVelocityCommandActionCfg(asset_name="robot", body_name="body")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel)              # noqa: F405
        base_ang_vel = ObsTerm(func=base_ang_vel)              # noqa: F405
        projected_gravity = ObsTerm(func=projected_gravity)   # noqa: F405
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])
        goal = ObsTerm(func=mdp_nav.goal_vector_b, params={"command_name": "goal"})
        # nearest person (privileged): distance the vision policy would infer from depth
        nearest_person = ObsTerm(func=lambda env: scenario_crowd.nearest_person_dist(env).unsqueeze(-1))
        last_action = ObsTerm(func=last_action)                # noqa: F405

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # spawn on the SOUTH edge of the crowd zone at the cruise altitude, facing NORTH (+y) toward
    # the goal on the far side.
    reset_base = EventTerm(
        func=reset_root_state_uniform,  # noqa: F405
        mode="reset",
        params={
            "pose_range": {"x": (-5.0, -1.0), "y": (_ZONE["y"][0], _ZONE["y"][0] + 1.5), "z": (_Z, _Z),
                           "yaw": (math.pi / 2 - math.pi / 12, math.pi / 2 + math.pi / 12)},
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
        },
    )
    # (re)spawn the crowd with random start/goal/speed, de-overlapped
    reset_crowd = EventTerm(
        func=scenario_crowd.reset_crowd,
        mode="reset",
        params={"zone": _ZONE, "speed_range": scenario_crowd.WALK_SPEED_RANGE},
    )
    # advance the crowd every control step (social-force mutual avoidance)
    move_crowd = EventTerm(
        func=scenario_crowd.move_crowd,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every step
    )
    # lighting domain randomization (reuse the warehouse-nav helper)
    randomize_light = EventTerm(
        func=mdp_obstacles.randomize_dome_light,
        mode="reset",
        params={"intensity_range": (150.0, 650.0)},
    )


@configclass
class RewardsCfg:
    navigation = RewTerm(func=scenario_crowd.crowd_navigation_reward, weight=1.0,
                         params={"command_name": "goal"})
    # explicit people-proximity term (in addition to the one folded into navigation) — bump its
    # weight to make crowd-avoidance more/less conservative.
    people_prox = RewTerm(func=scenario_crowd.people_proximity_penalty, weight=1.0,
                          params={"mag": 6.0, "exp": 0.5})
    # PLANAR / fixed-altitude enforcement in the reward
    altitude_hold = RewTerm(func=scenario_crowd.altitude_hold_penalty, weight=20.0,
                            params={"target_z": _Z})
    # hard −100 collision override (contact with a person / structure)
    collision = RewTerm(func=is_terminated, weight=-100.0)     # noqa: F405
    # success bookkeeping (0 reward): reached the far-side goal without crashing
    success_metric = RewTerm(func=scenario_crowd.update_success_metric, weight=1.0,
                             params={"command_name": "goal", "radius": 0.9})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)          # noqa: F405
    crash_ground = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.1})  # noqa: F405
    collision = DoneTerm(
        func=illegal_contact,                                  # noqa: F405
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact", body_names="body")},
    )
    reached_goal = DoneTerm(func=scenario_crowd.reached_goal,
                            params={"command_name": "goal", "radius": 0.9})


@configclass
class CrowdNavEnvCfg(ManagerBasedRLEnvCfg):
    scene: CrowdSceneCfg = CrowdSceneCfg(num_envs=16, env_spacing=60.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2           # 50 Hz control
        self.episode_length_s = 25.0  # crossing a ~28 m crowded hall needs travel time
        self.sim.dt = 0.01            # 100 Hz physics
        self.sim.render_interval = self.decimation


@configclass
class CrowdNavEnvCfg_PLAY(CrowdNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False

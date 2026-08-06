# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-conditioned drone navigation in Isaac's photoreal warehouse.

This replaces the old steering task (which had NO real objective: it tracked a randomly
resampled yaw command uncorrelated with the scene, nothing collided, and the policy was
blind).  Here the drone must fly to a goal point at the far end of the warehouse without
hitting the racking/forklift/walls, in a genuinely photoreal, YOLO-detectable environment.

Reuses the proven stack from ``track_steering_vision``:
  * ``CRAZYFLIE_CFG`` articulation + ``DirectThrustMomentAction`` (thrust + 3 moments),
  * the same proprioceptive observations,
so only the SCENE and the MDP change.  The reward/command are ported from aerial_gym's
``navigation_task`` (see ``warehouse_nav/mdp_nav.py``).
"""

import math
from dataclasses import field, make_dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.mdp import *  # noqa: F401,F403  (base MDP terms: last_action, base_lin_vel, ...)

from sims.isaaclab_tasks.warehouse_nav import mdp_nav
from sims.isaaclab_tasks.warehouse_nav import mdp_obstacles
from sims.isaaclab_tasks.warehouse_nav import mdp_gates
from sims.isaaclab_tasks.warehouse_nav import mdp_waypoints
from sims.isaaclab_tasks.warehouse_nav.mdp_velocity_action import VelocityCommandActionCfg
from sims.isaaclab_tasks.forest_trail import sensors as _sensors
from isaaclab_assets import CRAZYFLIE_CFG

# Navigable zone (warehouse-local metres). The measured full_warehouse (out/warehouse_geom.json)
# has 7 rack rows along x at x~=[-25.3,-20.4,-15.4,-10.5,-5.5,-0.55,4.4], each spanning
# y in [8.9,24.9] and rising to z~6, with six ~3.95 m-clear AISLES between them. We fly the
# aisle centred on x=-8.0 (between rack rows at x=-10.46 and x=-5.5): heading north in +y from
# the aisle mouth (y~6) up to the far end (y~24), threading gates and dodging an obstacle field
# scattered on the aisle floor. The loaded pallet racks are photoreal scenery + a hard collision
# boundary. Obstacle-placement volume = the aisle clear corridor (x in [-9.96,-6.0]; we inset to
# keep obstacles ~0.4 m off the rack faces), y between the spawn and the last gate.
_ZONE = {"x": (-9.5, -6.5), "y": (7.5, 22.5), "z": (0.5, 1.5)}

# The full_warehouse variant (racking + boxes + forklift + cones) — richest for detection/nav.
WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"


##
# Scene — base fields + injected gate frames (make_dataclass, one-shot at import)
##
@configclass
class _WarehouseSceneBase(InteractiveSceneCfg):
    warehouse = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
    )
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    # obstacle field: poles + crates (curriculum count) + patrolling people, real colliders
    obstacles = mdp_obstacles.make_obstacle_collection_cfg()
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/body", update_period=0.0, history_length=3)
    dome = AssetBaseCfg(prim_path="/World/DomeLight",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.9, 0.9, 0.95)))


def _build_warehouse_scene_cfg():
    """Add one field per gate-frame bar (collidable cuboids) to the base scene."""
    fields = []
    for name, cfg in mdp_gates.make_gate_scene().items():
        fields.append((name, AssetBaseCfg, field(default_factory=(lambda c=cfg: c))))
    cls = make_dataclass("WarehouseSceneCfg", fields=fields, bases=(_WarehouseSceneBase,))
    return configclass(cls)


WarehouseSceneCfg = _build_warehouse_scene_cfg()


@configclass
class WarehouseSceneCfg_WithCamera(WarehouseSceneCfg):
    """Scene + the FPV Crazyflie's onboard nose camera and a free chase camera. Reused verbatim
    from ``track_steering_vision`` (the FPV variant Dima asked us to re-use): the fpv_cam is
    parented to the drone body so it flies with real dynamics; the chase cam is a free prim the
    renderer repositions behind/above the drone each step. RENDER-only (needs --enable_cameras)."""

    fpv_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam",
        update_period=0.0,            # every render (record each control step)
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.01), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    chase_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam",
        update_period=0.0,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 200.0)),
        offset=CameraCfg.OffsetCfg(pos=(-1.3, 0.0, 0.8), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


##
# Sensor-equipped scene (fused-BC nav): the real onboard suite (HM01B0 front greyscale cam +
# 4x VL53L5CX ToF cross) from ``forest_trail/sensors.py`` — env-agnostic, parents to
# ``{ENV_REGEX_NS}/Robot/body`` exactly like the FPV cam — plus a world-anchored chase camera
# for demos, on top of a COLLIDABLE gate course (the fused expert threads them cleanly, so
# collidable frames make "flew THROUGH the opening" an honest test; racks/obstacles/people are
# already real colliders). This is the photoreal replacement for the forest ``*_WithSensors``
# scenes — same sensor contract, far richer scene.
##
def _build_ws_scene(collidable: bool):
    """Factory: gate-course + full onboard sensor rig + chase cam on the warehouse base scene.

    ``collidable`` toggles the gate frames only. COLLECTION uses VISUAL gates (the privileged
    expert + DART recovery noise then fly clean full-course demonstrations without dying every
    time the noise nudges the drone into a thin bar); EVAL/DEMO uses COLLIDABLE gates so 100%
    gate success is an honest "flew THROUGH the opening" result. Racks, the prop field and the
    patrolling people are REAL colliders in BOTH — the warehouse is a genuine collision test
    regardless of the gate setting.
    """
    fields = []
    for name, cfg in mdp_gates.make_gate_scene(collidable=collidable, gates=mdp_gates.FUSED_GATES).items():
        fields.append((name, AssetBaseCfg, field(default_factory=(lambda c=cfg: c))))
    cams = [("front_camera", _sensors.front_greyscale_camera_cfg()),
            ("tof_n", _sensors.tof_camera_cfg("N")), ("tof_e", _sensors.tof_camera_cfg("E")),
            ("tof_s", _sensors.tof_camera_cfg("S")), ("tof_w", _sensors.tof_camera_cfg("W")),
            ("chase_camera", _sensors.chase_camera_cfg(width=960, height=540))]
    for name, camcfg in cams:
        fields.append((name, CameraCfg, field(default_factory=(lambda c=camcfg: c))))
    suffix = "_Coll" if collidable else ""
    cls = make_dataclass(f"WarehouseSceneCfg_WithSensors{suffix}", fields=fields, bases=(_WarehouseSceneBase,))
    return configclass(cls)


WarehouseSceneCfg_WithSensors = _build_ws_scene(collidable=False)        # collection: clean expert
WarehouseSceneCfg_WithSensors_Coll = _build_ws_scene(collidable=True)    # eval/demo: honest gates


##
# MDP
##
@configclass
class CommandsCfg:
    # A course of 4 gates the drone must pass in sequence (ordered along +y through the open
    # band). The command tracks the CURRENT gate of the FIXED course; navigation_reward drives
    # toward it, and the drone must fly THROUGH the real (collidable) gate frames in mdp_gates.
    goal = mdp_gates.FixedGateCourseCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        success_radius=0.9,
    )


@configclass
class ActionsCfg:
    # Velocity-command interface (faithful to aerial_gym): policy emits [-1,1]^4 -> polar
    # velocity setpoint [vx>=0, 0, vz, yawrate] -> fixed geometric controller -> wrench.
    # Verified in isolation: tracks vx/vz/yawrate accurately, stays upright, holds altitude.
    velocity = VelocityCommandActionCfg(asset_name="robot", body_name="body")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel)          # noqa: F405
        base_ang_vel = ObsTerm(func=base_ang_vel)          # noqa: F405
        projected_gravity = ObsTerm(func=projected_gravity)  # noqa: F405
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])
        goal = ObsTerm(func=mdp_nav.goal_vector_b, params={"command_name": "goal"})
        # privileged nearest-obstacle info (state-based variant; moves to critic-only w/ vision)
        obstacles = ObsTerm(func=mdp_nav.nearest_obstacles_b, params={"k": 5})
        last_action = ObsTerm(func=last_action)            # noqa: F405

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # spawn at the aisle mouth (south, y~6), centred in the x=-8 aisle, FACING NORTH (+y) down
    # the corridor so the forward-only controller flies straight into the course; the +y goal
    # (last gate at y~21) requires traversing the full aisle.
    reset_base = EventTerm(
        func=reset_root_state_uniform,  # noqa: F405
        mode="reset",
        params={
            "pose_range": {"x": (-8.6, -7.4), "y": (5.0, 6.5), "z": (1.0, 1.5),
                           "yaw": (math.pi / 2 - math.pi / 12, math.pi / 2 + math.pi / 12)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1)},
        },
    )
    # place patrolling people + curriculum-count ground obstacles (poles/crates), rest dumped.
    reset_obstacles = EventTerm(
        func=mdp_obstacles.reset_obstacle_field,
        mode="reset",
        params={"volume": _ZONE, "half_density_prob": 0.15, "walk_speed": 0.8},
    )
    # patrol the people every control step (natural constant-speed straight paths)
    move_people = EventTerm(
        func=mdp_obstacles.move_people,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # every step
    )
    # drive the forklift along its circuit lane every control step (no-op if none active)
    move_forklift = EventTerm(
        func=mdp_obstacles.move_forklift,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )
    # lighting domain randomization: jitter the supplementary dome intensity each reset
    randomize_light = EventTerm(
        func=mdp_obstacles.randomize_dome_light,
        mode="reset",
        params={"intensity_range": (150.0, 650.0)},
    )


@configclass
class RewardsCfg:
    # closing-speed + heading + arrival + obstacle-proximity toward the CURRENT gate (VisFly-style)
    navigation = RewTerm(func=mdp_nav.navigation_reward, weight=1.0, params={"command_name": "goal"})
    # +bonus each time a gate is passed, and while the course is complete (loiter-after-finish)
    gate_pass = RewTerm(func=mdp_gates.gate_pass_bonus, weight=20.0, params={"command_name": "goal"})
    course_done = RewTerm(func=mdp_gates.course_complete_bonus, weight=5.0, params={"command_name": "goal"})
    # collision: hard -100 override (aerial_gym) via the terminating contact
    collision = RewTerm(func=is_terminated, weight=-100.0)       # noqa: F405
    # success metric bookkeeping (0 reward; stashes _ep_success/_ep_crash for the curriculum)
    success_metric = RewTerm(func=mdp_nav.update_success_metric, weight=1.0,
                             params={"command_name": "goal"})


@configclass
class CurriculumCfg:
    # Start easy (few obstacles) and ramp to a dense field as success rate rises. min_level=2
    # lets the drone first learn goal-reaching (aerial_gym's 15 traps from-scratch RL in a
    # safe-hover optimum). check_after=512 (vs aerial_gym 2048) so the curriculum visibly
    # progresses within a feasible run. max_level capped at the pool size (30).
    obstacle_count = CurrTerm(
        func=mdp_obstacles.obstacle_count_curriculum,
        params={"min_level": 2, "max_level": mdp_obstacles.N_STATIC, "check_after": 512,
                "up": 0.7, "down": 0.6, "inc": 2, "dec": 1},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)          # noqa: F405
    crash_ground = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.1})  # noqa: F405
    collision = DoneTerm(
        func=illegal_contact,                                  # noqa: F405
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact", body_names="body")},
    )


@configclass
class WarehouseNavEnvCfg(ManagerBasedRLEnvCfg):
    scene: WarehouseSceneCfg = WarehouseSceneCfg(num_envs=64, env_spacing=40.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2          # 50 Hz control
        self.episode_length_s = 20.0  # longer than steering task: goal-reaching needs travel time
        self.sim.dt = 0.01           # 100 Hz physics
        self.sim.render_interval = self.decimation


@configclass
class WarehouseNavEnvCfg_PLAY(WarehouseNavEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False


@configclass
class WarehouseNavEnvCfg_FPV(WarehouseNavEnvCfg):
    """Single-env, camera-equipped variant for validating the ONBOARD FPV footage of the physics
    drone (Dima's request). Same physics/controller/colliders as training; adds the fpv+chase
    cameras and thins the obstacle field a little so a scripted route gives clean footage."""

    scene: WarehouseSceneCfg_WithCamera = WarehouseSceneCfg_WithCamera(num_envs=1, env_spacing=40.0)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
        # start the curriculum low so the aisle is flyable for a clean validation pass
        self.curriculum.obstacle_count.params["min_level"] = 2


@configclass
class WarehouseNavEnvCfg_PLAY_WithSensors(WarehouseNavEnvCfg):
    """Single-env, full-sensor-suite variant for the fused-BC gate-nav pipeline (collect / train /
    fly / demo). Same physics, controller, gate course and REAL colliders (racks + gate frames +
    obstacle field + patrolling people) as training; adds the onboard sensor rig + chase cam.
    RENDER-only (needs --enable_cameras). Obstacle count starts moderate so the expert can weave a
    clean demonstration (BC teacher needs feasible avoidance trajectories)."""

    scene: WarehouseSceneCfg_WithSensors = WarehouseSceneCfg_WithSensors(num_envs=1, env_spacing=40.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        # a moderate, weavable obstacle field (start = end so density is fixed for the demo)
        self.curriculum.obstacle_count.params["min_level"] = 6
        # give the slower BC flight time to traverse the full aisle + weave
        self.episode_length_s = 30.0
        # Spawn AT the cruise altitude (the altitude-hold autopilot targets 2.0 m; FUSED_GATES are
        # centred at z=2.0). The RL base spawns at z∈(1.0,1.5) and climbs, but a drone still climbing
        # when it reaches the low aisle-mouth clutter (rack-shelf edge at y≈10.8, z≲1.9) clips it.
        # Starting near 2.0 with small jitter gives clearance from step 0. Fused-nav variant ONLY —
        # the RL env/base spawn is untouched.
        self.events.reset_base.params["pose_range"]["z"] = (1.9, 2.1)
        # Stage-1 = pure gate-following: clear the aisle of the tall random prop field (1.5-3 m
        # stacks that reach through the 2.0 m cruise altitude — the Stage-2 ToF-avoidance task).
        # Patrolling people stay (they top out at 1.7 m, harmless below the cruise, and keep the
        # photoreal warehouse scene alive). Overridable for the Stage-2 avoidance collect/eval.
        self.events.reset_obstacles.params["prop_density"] = 0.0


@configclass
class WarehouseNavEnvCfg_PLAY_WithSensors_Coll(WarehouseNavEnvCfg_PLAY_WithSensors):
    """Eval/demo variant: identical to the collection env but with COLLIDABLE gate frames, so a
    reported 100% gate-pass rate is an honest 'flew THROUGH the opening' result (clipping a bar
    crashes)."""

    scene: WarehouseSceneCfg_WithSensors_Coll = WarehouseSceneCfg_WithSensors_Coll(
        num_envs=1, env_spacing=40.0)


@configclass
class WarehouseNavEnvCfg_PLAY_WithSensors_Coll_Crowded(WarehouseNavEnvCfg_PLAY_WithSensors_Coll):
    """THE complex collidable course (for Dima / RoSE): identical to the Coll eval variant but with
    the tall-thin stacked-box PROP FIELD turned ON (density 0.3) — so gates + rack rows + tall stacked
    props + patrolling people are ALL real colliders (contact → termination). This is the crowded
    ToF-avoidance nav env; the base Coll variant carries only gates+racks+people (props off). One
    gym-id, no flags — use this to be sure you get the crowded, colliding scene."""

    def __post_init__(self):
        super().__post_init__()
        # turn the aisle prop field back on (the WithSensors base zeroes it for pure gate-following);
        # tall-thin stacked boxes are the default kind mix in mdp_obstacles.reset_obstacle_field.
        self.events.reset_obstacles.params["prop_density"] = 0.3


# =============================================================================================
# CIRCUIT-parametrized physics env (proper-B): the whole circuit library (circuits.py) becomes
# physics/FPV/trainable envs. No aisle gates — the route is the circuit's fixed waypoint course,
# and obstacles fill the circuit's FULL regions via the shared no-clip sampler.
# =============================================================================================
@configclass
class WarehouseSceneCfg_CircuitCam(_WarehouseSceneBase):
    """Base scene (NO aisle gate frames) + the FPV + chase cameras, for circuit routes.
    Uses the BIG prop pool so a single-env showcase holds a full-density circuit layout (matches
    the kinematic-preview clutter 1:1)."""

    obstacles = mdp_obstacles.make_obstacle_collection_cfg(mdp_obstacles.PROP_POOL_FULL, collide=False)

    fpv_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam", update_period=0.0,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 60.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.01), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"))
    chase_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 200.0)),
        offset=CameraCfg.OffsetCfg(pos=(-1.3, 0.0, 0.8), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"))


# ---- RICH showcase scene: MINIMAL physics collection (people capsules + forklift only); the ~800
# dense obstacles are injected as cheap STATIC VISUAL prims by fpv_rich_flight.py (bypasses the
# RigidObjectCollection pool cap). Adds a fixed top-down OVERVIEW camera for the full-warehouse shot.
_RICH_MINIMAL_POOL = {"pallet": 0, "crate": 0, "box": 0, "cone": 0, "klt": 0}


@configclass
class WarehouseSceneCfg_RichCam(_WarehouseSceneBase):
    obstacles = mdp_obstacles.make_obstacle_collection_cfg(_RICH_MINIMAL_POOL, collide=False)
    fpv_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/fpv_cam", update_period=0.0,
        height=480, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 80.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.06, 0.0, 0.01), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"))
    chase_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0,
        height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 300.0)))
    overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCam", update_period=0.0,
        height=540, width=960, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 400.0)))


def make_circuit_rich_cfg(circuit_name: str, start, yaw_center: float):
    Base = make_circuit_fpv_cfg(circuit_name, start, yaw_center)

    @configclass
    class _Rich(Base):
        scene: WarehouseSceneCfg_RichCam = WarehouseSceneCfg_RichCam(num_envs=1, env_spacing=60.0)

        def __post_init__(self):
            super().__post_init__()
            # showcase flythrough: don't end the episode on contact/low-altitude so the drone flies
            # the full route through the (non-colliding) crowd. time_out still bounds the clip.
            self.terminations.crash_ground = None
            self.terminations.collision = None

    return _Rich


def make_circuit_fpv_cfg(circuit_name: str, start, yaw_center: float):
    """Build a single-env, camera-equipped physics env cfg that flies a named circuit's route
    through its full-warehouse obstacle field (proper-B). Used to validate FPV per circuit."""

    @configclass
    class _Cfg(WarehouseNavEnvCfg):
        scene: WarehouseSceneCfg_CircuitCam = WarehouseSceneCfg_CircuitCam(num_envs=1, env_spacing=60.0)

        def __post_init__(self):
            super().__post_init__()
            self.observations.policy.enable_corruption = False
            # route = the circuit's fixed waypoint course (not the aisle gate course)
            self.commands.goal = mdp_waypoints.FixedCircuitCourseCommandCfg(
                circuit=circuit_name, success_radius=1.3,
                resampling_time_range=(1.0e9, 1.0e9))
            # spawn at the circuit start, facing the first leg
            self.events.reset_base.params["pose_range"] = {
                "x": (start[0] - 0.5, start[0] + 0.5), "y": (start[1] - 0.5, start[1] + 0.5),
                "z": (start[2] - 0.2, start[2] + 0.2),
                "yaw": (yaw_center - math.pi / 12, yaw_center + math.pi / 12)}
            # obstacles fill the circuit's FULL regions at FULL density (showcase = preview 1:1)
            self.events.reset_obstacles.params["circuit"] = circuit_name
            self.events.reset_obstacles.params["full_density"] = True
            # pass-bonus / course-complete read the same interface on the waypoint command
            self.rewards.gate_pass.func = mdp_waypoints.waypoint_pass_bonus
            self.rewards.course_done.func = mdp_waypoints.course_complete_bonus

    return _Cfg


# =============================================================================================
# LOCOMOTION-POLICY variant: same circuit scene, but the low-level flight is Dima's TRAINED
# steering policy (DirectThrustMoment action + steering obs) driven by a route-steering command,
# instead of the geometric velocity controller. Reuses the steering task's action/obs verbatim so
# the checkpoint loads; the command steers toward the circuit waypoints.
# =============================================================================================
from sims.isaaclab_tasks.track_steering_vision.config.crazyflie.track_steering_env_cfg import (  # noqa: E402
    ActionsCfg as _SteerActionsCfg, ObservationsCfg as _SteerObsCfg)


def make_circuit_locomotion_cfg(circuit_name: str, start, yaw_center: float):
    @configclass
    class _LocoCmds:
        steering_command = mdp_waypoints.RouteSteeringCommandCfg(
            circuit=circuit_name, success_radius=1.3, resampling_time_range=(1.0e9, 1.0e9))

    @configclass
    class _LocoRewards:
        alive = RewTerm(func=is_terminated, weight=0.0)  # noqa: F405  (running a policy, not training)

    @configclass
    class _Cfg(ManagerBasedRLEnvCfg):
        scene: WarehouseSceneCfg_CircuitCam = WarehouseSceneCfg_CircuitCam(num_envs=1, env_spacing=60.0)
        observations: _SteerObsCfg = _SteerObsCfg()
        actions: _SteerActionsCfg = _SteerActionsCfg()
        commands: _LocoCmds = _LocoCmds()
        rewards: _LocoRewards = _LocoRewards()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()

        def __post_init__(self):
            self.decimation = 2
            self.episode_length_s = 40.0
            self.sim.dt = 0.01
            self.sim.render_interval = self.decimation
            self.observations.policy.enable_corruption = False
            self.events.reset_base.params["pose_range"] = {
                "x": (start[0] - 0.5, start[0] + 0.5), "y": (start[1] - 0.5, start[1] + 0.5),
                "z": (start[2] - 0.2, start[2] + 0.2),
                "yaw": (yaw_center - math.pi / 12, yaw_center + math.pi / 12)}
            self.events.reset_obstacles.params["circuit"] = circuit_name
            self.events.reset_obstacles.params["full_density"] = True

    return _Cfg


WarehouseGrandLocoEnvCfg_FPV = make_circuit_locomotion_cfg("warehouse_grand", (-8.0, -16.0, 1.4), 1.86)

# the whole circuit library as physics/FPV envs (start + initial yaw from circuits.py routes)
WarehouseGrandEnvCfg_FPV = make_circuit_fpv_cfg("warehouse_grand", (-8.0, -16.0, 1.4), 1.86)
AisleSlalomEnvCfg_FPV = make_circuit_fpv_cfg("aisle_slalom", (-8.0, 7.0, 1.2), 1.22)
CrossShelfEnvCfg_FPV = make_circuit_fpv_cfg("cross_shelf", (-8.0, 11.0, 1.3), 1.57)
VerticalGatesEnvCfg_FPV = make_circuit_fpv_cfg("vertical_gates", (-12.9, 8.0, 1.2), 1.37)
DynamicDodgeEnvCfg_FPV = make_circuit_fpv_cfg("dynamic_dodge", (-8.0, -13.0, 1.4), 1.57)
MixedClutterEnvCfg_FPV = make_circuit_fpv_cfg("mixed_clutter", (-17.9, 8.0, 1.3), 1.67)

# RICH showcase variants (minimal collection + injected ~800 visual props + overview cam)
_CIRCUIT_STARTS = {
    "warehouse_grand": ((-8.0, -16.0, 1.4), 1.86), "aisle_slalom": ((-8.0, 7.0, 1.2), 1.22),
    "cross_shelf": ((-8.0, 11.0, 1.3), 1.57), "vertical_gates": ((-12.9, 8.0, 1.2), 1.37),
    "dynamic_dodge": ((-8.0, -13.0, 1.4), 1.57), "mixed_clutter": ((-17.9, 8.0, 1.3), 1.67),
}
WarehouseGrandRichEnvCfg = make_circuit_rich_cfg("warehouse_grand", *_CIRCUIT_STARTS["warehouse_grand"])
AisleSlalomRichEnvCfg = make_circuit_rich_cfg("aisle_slalom", *_CIRCUIT_STARTS["aisle_slalom"])
CrossShelfRichEnvCfg = make_circuit_rich_cfg("cross_shelf", *_CIRCUIT_STARTS["cross_shelf"])
VerticalGatesRichEnvCfg = make_circuit_rich_cfg("vertical_gates", *_CIRCUIT_STARTS["vertical_gates"])
DynamicDodgeRichEnvCfg = make_circuit_rich_cfg("dynamic_dodge", *_CIRCUIT_STARTS["dynamic_dodge"])
MixedClutterRichEnvCfg = make_circuit_rich_cfg("mixed_clutter", *_CIRCUIT_STARTS["mixed_clutter"])

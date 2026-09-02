"""RL velocity-controller env (Agent B, Track 1c-b — the RL upgrade to the distilled MLP).

A learned low-level controller that tracks the SAME 2-D command the nav model emits
(yaw_rate, forward_speed) AND holds altitude, outputting thrust + moments. This is the direct
RL analog of the distilled warehouse_mlp_control (which imitates the geometric Lee law): here the
controller is learned end-to-end by PPO instead of distilled, and both share the DirectThrustMoment
interface so they are drop-in comparable (eval matrix E1 distilled vs E1-RL).

Sibling of body_rate_env_cfg.py: same scene / action / PPO runner, but the command is the nav
interface (steering yaw-rate + forward velocity) and the reward tracks forward velocity + yaw rate
+ a fixed cruise altitude, keeping the drone roughly level (a cruise controller, not a full-agility
rate tracker). Reuses the existing steering/velocity/height rewards verbatim.

Train (after a GPU window):
    <env_isaaclab py> sims/scripts/train_steering_tracking.py --task Isaac-Track-VelocityCtrl-Crazyflie-v0 \
        --num_envs 4096 --max_iterations 1500 --headless
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
    SteeringSceneCfg, SteeringSceneCfg_WithCamera)
from isaaclab.envs.mdp import *  # noqa: F401,F403

TARGET_H = 2.0  # cruise altitude (matches the warehouse nav altitude-hold setpoint)
MAX_YAWRATE = 1.047198


@configclass
class VelocityCommandsCfg:
    """The nav interface: [target_yaw_rate, target_forward_velocity]."""
    steering_command = mdp_commands.SteeringCommandCfg(
        resampling_time_range=(0.5, 2.0),
        ranges=mdp_commands.SteeringCommandCfg.Ranges(
            yaw_rate=(-MAX_YAWRATE, MAX_YAWRATE),
            velocity=(0.1, 2.0),
        ),
    )


@configclass
class ActionsCfg:
    thrust_moment = DirectThrustMomentActionCfg(
        asset_name="robot", body_name="body", thrust_to_weight=1.9, moment_scale=0.01)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        base_height = ObsTerm(func=lambda env: (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, 2:3])
        target_command = ObsTerm(func=generated_commands, params={"command_name": "steering_command"})
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (1.5, 2.5),
                           "yaw": (-math.pi, math.pi),
                           "roll": (-math.pi / 12, math.pi / 12), "pitch": (-math.pi / 12, math.pi / 12)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                               "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3)},
        },
    )


@configclass
class RewardsCfg:
    """Track forward velocity + yaw rate + cruise altitude. NO upright term: a quadrotor moves
    forward by PITCHING, so rewarding 'stay level' directly fights the forward-velocity command
    (the documented reward-conflict trap — first RL run hovered 0/8 with upright weight 2).
    Forward tracking is the PRIMARY objective here (weighted highest)."""
    # yaw tracking weighted EQUAL to forward (was 4 vs 8 -> the policy ignored yaw and saturated the
    # yaw moment, spinning the drone; the warehouse gate-nav then ran away). Tight yaw std for sharp
    # tracking + a meaningful action penalty to stop moment saturation.
    # yaw tracking now PRIMARY + tight std (the DR policy over-yawed ~1.4x, command-insensitive ->
    # crashed on gate weave). Stronger altitude (sagged to 1.6). Bigger action penalty to curb the
    # yaw-moment over-actuation.
    forward_velocity = RewTerm(func=mdp_rewards.forward_velocity_tracking, weight=6.0,
                               params={"std": 0.5, "command_name": "steering_command"})
    yaw_rate = RewTerm(func=mdp_rewards.steering_angle_tracking, weight=10.0,
                       params={"std": 0.25, "command_name": "steering_command"})
    height = RewTerm(func=mdp_rewards.height_tracking, weight=6.0,
                     params={"target_height": TARGET_H, "std": 0.4})
    # COORDINATED TURN (the key fix): lateral_drift_penalty is MISNAMED -- it returns exp(-vy^2), a
    # POSITIVE reward maxed when body-y velocity=0. It needs a POSITIVE weight. It was NEGATIVE (-3),
    # which REWARDED sideslip (the documented sign trap) -> the drone crabbed sideways at ~2.4 m/s,
    # yawing its heading without turning its flight path, and drifted out of the aisle. Positive +6
    # (on par with forward/yaw/height) forces vy~0 -> a coordinated turn that redirects velocity with
    # heading -> straight, steerable flight that threads gates.
    lateral_drift = RewTerm(func=mdp_rewards.lateral_drift_penalty, weight=6.0, params={"std": 0.3})
    action_rate = RewTerm(func=action_rate_l2, weight=-0.03)
    action_magnitude = RewTerm(func=action_l2, weight=-0.03)
    termination_penalty = RewTerm(func=is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=time_out, time_out=True)
    crash = DoneTerm(func=root_height_below_minimum, params={"minimum_height": 0.2})


@configclass
class TrackVelocityCtrlEnvCfg(ManagerBasedRLEnvCfg):
    scene: SteeringSceneCfg = SteeringSceneCfg(num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: VelocityCommandsCfg = VelocityCommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0)


@configclass
class TrackVelocityCtrlEnvCfg_PLAY(TrackVelocityCtrlEnvCfg):
    scene: SteeringSceneCfg_WithCamera = SteeringSceneCfg_WithCamera(num_envs=50, env_spacing=5.0)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class TrackVelocityCtrlEnvCfg_DR(TrackVelocityCtrlEnvCfg):
    """Domain-randomized velocity controller. The non-DR policy overfit the open SteeringScene and
    went OUT-OF-DISTRIBUTION in the warehouse closed-loop (saturated yaw moment -> spin). Wider reset
    states + in-flight pushes + actuation DR + obs corruption broaden the training distribution to
    cover the warehouse's nav-driven states, matching WHY the distilled MLP (broad analytic teacher)
    generalizes. Longer resample so the policy tracks sustained commands (as nav produces)."""

    def __post_init__(self):
        super().__post_init__()
        # sustained-command tracking (nav holds commands) + full command range
        self.commands.steering_command.resampling_time_range = (1.0, 4.0)
        # wider reset states so warehouse velocities/attitudes are in-distribution
        self.events.reset_base.params["velocity_range"] = {
            "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
            "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.5, 1.5)}
        # actuation-authority + motor-lag DR (robust control law, not one plant's exploit)
        self.actions.thrust_moment.randomize_thrust_to_weight = (1.6, 2.2)
        self.actions.thrust_moment.randomize_moment_scale = (0.008, 0.013)
        self.actions.thrust_moment.motor_tau = 0.03
        self.observations.policy.enable_corruption = True
        # random in-flight pushes (disturbance rejection)
        self.events.push = EventTerm(
            func=push_by_setting_velocity, mode="interval", interval_range_s=(1.5, 3.5),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (-0.4, 0.4),
                                       "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0)}})

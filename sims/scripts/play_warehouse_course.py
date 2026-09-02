"""Render a trained warehouse-course policy: the drone flies a 4-gate course through the
warehouse, avoiding walking people (blue capsules) and static clutter (boxes/posts).

Loads a checkpoint, runs the policy in a few PLAY envs, follows env 0 with a chase camera,
and writes an MP4.

    python sims/scripts/play_warehouse_course.py --headless --checkpoint <model.pt> \
        --video out/warehouse_course.mp4 --steps 900
"""
import argparse, os, sys
freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab","isaaclab_assets","isaaclab_rl","isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--video", type=str, default="out/warehouse_course.mp4")
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--task", type=str, default="Isaac-Drone-Warehouse-Nav-Crazyflie-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli); simulation_app = app.app

import importlib, numpy as np, torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from sims.isaaclab_tasks.warehouse_nav.config import crazyflie  # noqa: F401


def _sphere(color, r=0.45):
    return sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=color, emissive_color=tuple(0.5 * c for c in color)))


def main():
    spec = gym.spec(args_cli.task)
    mod, cls = spec.kwargs["env_cfg_entry_point"].split(":")
    cfg = getattr(importlib.import_module(mod), cls)()
    cfg.scene.num_envs = 4
    cfg.sim.device = args_cli.device
    amod, acls = spec.kwargs["rsl_rl_cfg_entry_point"].split(":")
    agent_cfg = getattr(importlib.import_module(amod), acls)()

    # add a chase camera to the SCENE cfg so the InteractiveScene initializes it (a standalone
    # Camera created after the env is not initialized -> '_ALL_INDICES' error).
    cfg.scene.chase_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ChaseCam", update_period=0.0, height=720, width=1280,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 200)))

    env = gym.make(args_cli.task, cfg=cfg)
    unwrapped = env.unwrapped
    cam = unwrapped.scene["chase_camera"]

    wrapped = RslRlVecEnvWrapper(env)
    runner_cfg = {
        "algorithm": agent_cfg.algorithm.to_dict(),
        "actor": {"class_name": agent_cfg.actor.class_name, "hidden_dims": agent_cfg.actor.hidden_dims,
                  "activation": agent_cfg.actor.activation, "obs_normalization": agent_cfg.actor.obs_normalization,
                  "distribution_cfg": agent_cfg.actor.distribution_cfg.to_dict()},
        "critic": {"class_name": agent_cfg.critic.class_name, "hidden_dims": agent_cfg.critic.hidden_dims,
                   "activation": agent_cfg.critic.activation, "obs_normalization": agent_cfg.critic.obs_normalization},
        "obs_groups": agent_cfg.obs_groups, "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": 1, "save_interval": 1, "experiment_name": agent_cfg.experiment_name,
        "empirical_normalization": False,
    }
    runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)
    print(f"[play] loaded {args_cli.checkpoint}", flush=True)

    # The gates are now REAL collidable frames in the scene (mdp_gates) — no ball markers.
    # A small green orb marks only the CURRENT target gate (so the sequence reads on screen).
    gate_markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/World/CurrentGateMarker", markers={"current": _sphere((0.1, 0.9, 0.2), r=0.2)}))

    # Walking human MESHES: VisualizationMarkers can't host skinned character USDs (point-
    # instancer rejects them), so spawn them as regular prims and drive their transforms via
    # XFormPrim (writes to Fabric, so the render updates). We hide env 0's capsule people and
    # overlay a human mesh on each; the capsule stays the collidable/detectable physics body.
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from sims.isaaclab_tasks.warehouse_nav.mdp_obstacles import N_HUMANS
    from isaacsim.core.prims import XFormPrim
    from pxr import UsdGeom
    chars = ["F_Business_02", "male_adult_construction_01_new",
             "female_adult_police_01_new", "M_Medical_01"]
    for i in range(N_HUMANS):
        cfg_h = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/People/Characters/{chars[i]}/{chars[i]}.usd")
        cfg_h.func(f"/World/Human_{i}", cfg_h)
    # a scaled REAL Crazyflie mesh overlaid on the (tiny) drone so it's visible as a quadrotor
    drone_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd", scale=(6.0, 6.0, 6.0))
    drone_cfg.func("/World/DroneVis", drone_cfg)
    # hide env 0's capsule people so only the human meshes show
    stage = unwrapped.sim.stage
    for i in range(N_HUMANS):
        pr = stage.GetPrimAtPath(f"/World/envs/env_0/Person_{i}")
        if pr.IsValid():
            UsdGeom.Imageable(pr).MakeInvisible()

    obs, _ = wrapped.reset()
    robot = unwrapped.scene["robot"]
    course = unwrapped.command_manager.get_term("goal")
    obstacles = unwrapped.scene["obstacles"]
    humans_view = XFormPrim("/World/Human_.*", name="humans")   # after reset -> sim playing
    drone_view = XFormPrim("/World/DroneVis", name="dronevis")
    frames = []
    dev = args_cli.device
    import imageio.v2 as imageio
    for i in range(args_cli.steps):
        with torch.no_grad():
            act = policy(obs)
        obs, _, _, _ = wrapped.step(act)
        # small green orb on the CURRENT target gate (the real frames are in the scene)
        gate_markers.visualize(translations=course.goal_pos_w[0:1])
        p = robot.data.root_pos_w                          # (num_envs,3)
        # overlay the scaled Crazyflie mesh on env 0's drone, yawed to its heading
        from isaaclab.utils.math import yaw_quat as _yq
        drone_view.set_world_poses(positions=p[0:1], orientations=_yq(robot.data.root_quat_w[0:1]))
        # move the human meshes to env 0's patrolling people (feet on floor; face travel dir)
        ppos = obstacles.data.object_pos_w[0, :N_HUMANS].clone()   # (H,3) capsule centers
        ppos[:, 2] = 0.0
        wd = unwrapped.person_dir[0]                               # (H,2) patrol direction
        yaw = torch.atan2(wd[:, 1], wd[:, 0])
        pq = torch.zeros(N_HUMANS, 4, device=dev)
        pq[:, 0] = torch.cos(yaw / 2); pq[:, 3] = torch.sin(yaw / 2)
        humans_view.set_world_poses(positions=ppos, orientations=pq)
        # aisle chase cam: follow from BEHIND (south, -y) and above, looking north up the aisle
        # (slight -x so the drone doesn't fully occlude the corridor; stays within the ~4 m aisle).
        eye = torch.stack([p[:, 0] - 0.6, p[:, 1] - 4.0, p[:, 2] + 1.9], dim=1)
        tgt = torch.stack([p[:, 0], p[:, 1] + 4.0, p[:, 2] + 0.2], dim=1)
        cam.set_world_poses_from_view(eye, tgt)
        cam.update(dt=unwrapped.physics_dt)
        if i % 2 == 0:  # ~25 fps from 50 Hz
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            frames.append(rgb)
    outp = os.path.join(freshscheduler_root, args_cli.video) if not os.path.isabs(args_cli.video) else args_cli.video
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    imageio.mimwrite(outp, frames, fps=25, quality=7)
    print(f"[play] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main(); simulation_app.close()

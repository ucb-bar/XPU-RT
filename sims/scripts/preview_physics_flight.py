"""PHYSICS-driven flight preview: fly the drone through waypoints with its REAL geometric
velocity controller, so banking / pitch / overshoot come from the dynamics instead of a
kinematic spline (which looked fake). No trained policy needed — a simple waypoint-follower
feeds velocity setpoints into the same controller the RL policy uses. Reusable to preview a
circuit's flyability + look before committing to training. Multi-camera + propeller spin.

    <xpurt python> sims/scripts/preview_physics_flight.py --headless --views chase --frames 120
"""
import argparse, math, os, sys

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--views", type=str, default="chase")
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--dome", type=float, default=250.0)
parser.add_argument("--sub", type=int, default=4, help="physics substeps per captured frame")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (matrix_from_quat, normalize, yaw_quat, quat_apply,
                                 euler_xyz_from_quat)
from isaaclab_assets import CRAZYFLIE_CFG
from sims.isaaclab_tasks.warehouse_nav import mdp_gates

WH = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd"
MAX_SPEED, MAX_YAWRATE = 2.0, 1.047


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Flight:
    """Extracted geometric velocity controller (from mdp_velocity_action) — vel setpoint -> wrench."""
    def __init__(self, robot, sim, dev):
        self.robot, self.dev = robot, dev
        self.body_id = robot.find_bodies("body")[0]
        self.mass = robot.root_physx_view.get_masses()[0].sum().item()
        self.g = torch.tensor(sim.cfg.gravity, device=dev)
        inertia = robot.root_physx_view.get_inertias()[0, self.body_id[0]].to(dev)
        self.J = torch.tensor([inertia[0], inertia[4], inertia[8]], device=dev)
        self.Kv, self.KR, self.Kw = 2.0, 200.0, 20.0
        self.thrust = torch.zeros(1, 1, 3, device=dev)
        self.moment = torch.zeros(1, 1, 3, device=dev)

    def apply(self, vel_sp):  # vel_sp (1,4): [vx_fwd, 0, vz_up, yawrate]
        d = self.robot.data
        quat = d.root_quat_w
        R = matrix_from_quat(quat)
        v_w = d.root_lin_vel_w
        w_b = d.root_ang_vel_b
        v_sp_yaw = torch.zeros_like(v_w)
        v_sp_yaw[:, 0] = vel_sp[:, 0]; v_sp_yaw[:, 2] = vel_sp[:, 2]
        v_des_w = quat_apply(yaw_quat(quat), v_sp_yaw)
        a_des = self.Kv * (v_des_w - v_w) - self.g
        forces_w = self.mass * a_des
        body_z = R[:, :, 2]
        self.thrust[:, 0, 2] = (forces_w * body_z).sum(dim=1).clamp(min=0.0)
        z_des = normalize(forces_w)
        _, _, cy = euler_xyz_from_quat(quat)
        x_c = torch.stack([torch.cos(cy), torch.sin(cy), torch.zeros_like(cy)], dim=1)
        y_des = normalize(torch.cross(z_des, x_c, dim=1))
        x_des = torch.cross(y_des, z_des, dim=1)
        R_des = torch.stack([x_des, y_des, z_des], dim=2)
        errM = 0.5 * (torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des))
        e_R = torch.stack([errM[:, 2, 1], errM[:, 0, 2], errM[:, 1, 0]], dim=1)
        w_des = torch.zeros_like(w_b); w_des[:, 2] = vel_sp[:, 3]
        moment = self.J.unsqueeze(0) * (-self.KR * e_R - self.Kw * (w_b - w_des))
        self.moment[:, 0, :] = moment
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self.body_id, forces=self.thrust, torques=self.moment)


def waypoint_vel(robot, target, dev):
    """Proportional waypoint-follower -> velocity setpoint [vx_fwd,0,vz,yawrate]."""
    d = robot.data
    pos = d.root_pos_w
    _, _, cy = euler_xyz_from_quat(d.root_quat_w)
    to_t = target - pos
    dist_h = to_t[:, :2].norm(dim=1)
    des_yaw = torch.atan2(to_t[:, 1], to_t[:, 0])
    yaw_err = _wrap(des_yaw - cy)
    yawrate = (2.0 * yaw_err).clamp(-MAX_YAWRATE, MAX_YAWRATE)
    fwd = (1.2 * dist_h).clamp(0.0, MAX_SPEED) * torch.cos(yaw_err).clamp(min=0.0)
    vz = (1.5 * to_t[:, 2]).clamp(-1.5, 1.5)
    sp = torch.zeros(1, 4, device=dev)
    sp[:, 0] = fwd; sp[:, 2] = vz; sp[:, 3] = yawrate
    return sp, dist_h.item()


def cam_for(view, p, dev):
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    if view == "fpv":
        e, t = [px, py + 0.25, pz + 0.1], [px, py + 4, pz - 0.2]
    elif view == "topdown":
        e, t = [px + 0.4, py - 0.6, pz + 7.5], [px, py, pz]
    elif view == "side":
        e, t = [px + 1.3, py - 1.8, pz + 1.2], [px - 0.1, py + 0.4, pz]
    else:
        e, t = [px - 0.4, py - 2.8, pz + 1.1], [px, py + 0.6, pz - 0.2]
    return torch.tensor([e], device=dev), torch.tensor([t], device=dev)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim_utils.UsdFileCfg(usd_path=WH).func("/World/WH", sim_utils.UsdFileCfg(usd_path=WH))
    d = sim_utils.DomeLightCfg(intensity=args_cli.dome, color=(0.9, 0.9, 0.95)); d.func("/World/Dome", d)
    for name, cfg in mdp_gates.make_gate_scene().items():
        cfg.spawn.func(cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/c"), cfg.spawn,
                       translation=tuple(cfg.init_state.pos), orientation=tuple(cfg.init_state.rot))
    robot = Articulation(CRAZYFLIE_CFG.replace(prim_path="/World/Robot"))
    cam = Camera(CameraCfg(prim_path="/World/Cam", update_period=0.0, height=args_cli.height,
                           width=args_cli.width, data_types=["rgb"],
                           spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 400))))
    sim.reset()
    dev = sim.device
    # place the drone at the aisle mouth, FACING +y (yaw=pi/2) so it flies straight into the course
    start = torch.tensor([[-8.0, 6.0, 1.2]], device=dev)
    root = robot.data.default_root_state.clone()
    root[:, :3] = start
    root[:, 3] = 0.70711; root[:, 4] = 0.0; root[:, 5] = 0.0; root[:, 6] = 0.70711  # quat wxyz, yaw +90deg
    robot.write_root_state_to_sim(root)
    robot.reset()
    ctl = Flight(robot, sim, dev)
    # a scaled visual quad overlaid on the (tiny, ~9 cm) physics body: respawned each frame at the
    # body's REAL pose+orientation, so it shows the true banking/pitch at a visible size.
    CF2X = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
    dcfg = sim_utils.UsdFileCfg(usd_path=CF2X, scale=(12.0, 12.0, 12.0))
    # prop spin: set a constant velocity on any rotor joints (visual)
    spin_ids = [i for i, n in enumerate(robot.joint_names) if "prop" in n.lower() or n.lower().startswith("m")]
    waypts = torch.tensor([list(g[0]) for g in mdp_gates.GATES] + [[-8.0, 24.0, 1.3]], device=dev)

    # settle
    for _ in range(60):
        ctl.apply(torch.zeros(1, 4, device=dev)); robot.write_data_to_sim(); sim.step(); robot.update(0.005)

    views = [v.strip() for v in args_cli.views.split(",") if v.strip()]
    import imageio.v2 as imageio
    # we fly ONCE and capture all requested views' frames in lockstep (re-fly per view is costly);
    # simplest: fly once per view (deterministic controller -> identical path).
    for view in views:
        # reset to start for each view
        robot.write_root_state_to_sim(root); robot.reset()
        for _ in range(40):
            ctl.apply(torch.zeros(1, 4, device=dev)); robot.write_data_to_sim(); sim.step(); robot.update(0.005)
        wp_idx = 0
        frames = []
        for i in range(args_cli.frames):
            for _ in range(args_cli.sub):
                tgt = waypts[wp_idx:wp_idx + 1]
                sp, dist = waypoint_vel(robot, tgt, dev)
                if dist < 0.9 and wp_idx < len(waypts) - 1:
                    wp_idx += 1
                ctl.apply(sp)
                if spin_ids:
                    jv = robot.data.joint_vel.clone(); jv[:, spin_ids] = 60.0
                    robot.set_joint_velocity_target(jv[:, spin_ids], joint_ids=spin_ids)
                robot.write_data_to_sim(); sim.step(); robot.update(0.005)
            p = robot.data.root_pos_w[0]
            q = robot.data.root_quat_w[0]
            # overlay a scaled visual quad at the body's real pose+orientation (respawn = only
            # reliable per-frame move in a bare ctx; shows the true banking at a visible size)
            sim.stage.RemovePrim("/World/DroneVis")
            dcfg.func("/World/DroneVis", dcfg, translation=tuple(p.tolist()), orientation=tuple(q.tolist()))
            e, t = cam_for(view, p, dev)
            cam.set_world_poses_from_view(e, t)
            simulation_app.update(); simulation_app.update()
            cam.update(dt=0.005)
            frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            if i % 40 == 0:
                print(f"[flight:{view}] frame {i}/{args_cli.frames} pos={[round(x,1) for x in p.tolist()]} wp={wp_idx}", flush=True)
        outp = os.path.join(freshscheduler_root, "out", f"flight_preview__{view}.mp4")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        imageio.mimwrite(outp, frames, fps=25, quality=7)
        print(f"[flight] wrote {outp} ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main(); simulation_app.close()

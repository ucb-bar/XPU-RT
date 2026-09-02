# Warehouse HIL nav — flight → schedule → paper figure (reproduce)

A runbook from a fresh clone of this repo to the **warehouse mega figure**: the paper-friendly
replacement for the flight video. It ties together three things, all in this repo:

1. an **Isaac Lab** warehouse flight (a Crazyflie flying a 4-gate aisle past static props and 4
   patrolling people), captured with the full onboard **sensor bank** (FPV grey + YOLO gate/person,
   4× cross-ToF, optical flow, IMU, altitude, goal command);
2. the **onboard K1 schedule** for the deployed nets (CTRL = mlp_control, NAV = fused_full,
   YOLO = yolov8_nano), solved by the XPU-RT greedy solver and, for contrast, by a naive ROS
   per-net-pinning baseline;
3. a single tall composite (`sims/scripts/compose_mega_figure.py`) — top-down aisle, four in-flight
   moments, telemetry, and the annotated schedule as one figure.

Every number in the figure is measured — the flight is a real Isaac run, the schedule uses the
measured K1 profile shipped in `gen_mb/`. Caveats are in [§7](#7-honest-caveats).

> **Paths.** Everything below is **relative to the repo root** — run every command from the top of
> your clone. There is no dependency on any specific machine.

## Environments

Two Python environments; set each once to the interpreter from your install, then the commands use
`$ISAAC_PY` / `$XPURT_PY`:

```bash
# from the repo root of your clone:
export ISAAC_PY=python   # the Python from your Isaac Lab install (see below)
export XPURT_PY=python   # the Python from the XPU-RT venv (see docs/xpurt_env_setup.md; needs ortools)
```

- **Isaac Lab** (render + training): install **NVIDIA Isaac Sim** and **Isaac Lab** from their
  public releases (isaac-sim.github.io / the Isaac Lab docs) plus **rsl_rl**. The warehouse task
  under `sims/` imports as a package, so run render/train commands **from the repo root** (or put the
  repo root on `PYTHONPATH`). `docs/xpurt_env_setup.md` records the exact versions used here.
- **XPU-RT venv** (scheduler + Gantt): a plain venv with this repo's requirements; it needs
  `ortools`. See `docs/xpurt_env_setup.md`.

## What ships in the repo vs what you build

**In the repo (you get these on clone):**
- `sims/isaaclab_tasks/warehouse_nav/` — the warehouse env (scene, MDP, gates, obstacles, sensors).
- `sims/scripts/record_sensor_demo.py`, `sims/scripts/compose_mega_figure.py` — render + compose.
- `sims/scripts/train_steering_tracking.py`, `train_warehouse_nav.py`, `train_yolo.py` — trainers.
- **All three trained models the render loads**, so it runs from a clone out-of-the-box:
  `sims/models/warehouse/nav_fused_v12_cnn.pt` (nav), `rl_controller_velctrl_dr4.pt` (MLP
  controller), `yolov8n_gate_person_128x192.pt` (YOLO).
- `data/toplevel/networks_k1_flight_deployed.json`, `scripts/ros_pinning_periodic.py`,
  `scripts/plot_solver_gantt_annotated.py` — schedule spec, ROS baseline, annotated Gantt.
- `gen_mb/` — the K1 dispatch graphs (`vmfb/`) **and the measured spacemit_x60 profiles** for all
  three deployed nets, so `--profiled` is genuinely measured (no board needed to reproduce; only to
  re-measure the profile).

**You build (nothing external needed):** the flight capture (`figdata_mega/`, §1) and the figures
(§5). The repo ships trained models, so **you do not need to train anything** to reproduce the
figure. Training from scratch is optional — see the next section. (The published figure used one
specific controller checkpoint, `model_3250.pt`, which is a training output and is not committed;
the shipped default controller reproduces an equivalent flight, and training to iteration 3250
reproduces the exact one.)

### (Optional) Train the models from scratch

Run from the repo root with the Isaac Lab env. The controller and nav net train in Isaac; YOLO is a
standard ultralytics fit on rendered gate/person crops.

```bash
# 1. MLP flight controller (the --rl_ckpt). The published figure used --actor_hidden_dims 512,512,256,128
$ISAAC_PY sims/scripts/train_steering_tracking.py --headless \
  --task Isaac-Track-Steering-Vision-Crazyflie-v0 --actor_hidden_dims 512,512,256,128 \
  --run_note larger_ctrl --max_iterations 4000 --seed 42
#   -> logs/rsl_rl/crazyflie_steering_tracking/<run>/model_3250.pt  (point --rl_ckpt at it)
# 2. Collision-avoidance nav net (the --weights)
$ISAAC_PY sims/scripts/train_warehouse_nav.py --headless
# 3. YOLOv8n gate/person detector (the --yolo)
$ISAAC_PY sims/scripts/train_yolo.py
```

The warehouse env: aisle along +Y at x≈−8, cruise z≈2; the four gates at world y≈9, 13, 17, 21
(also stored as `gates_world` in the dump). The drone spawns at the low-y (south) end.

---

## 1. Render the flight + sensor bank

`--controller rl` runs the trained MLP as the real thrust/moment controller (not the analytic Lee
law); `--dump_figure_data <dir>` writes poses, obstacle tracks, per-moment dense frames, and the
full sensor bank. `--rl_ckpt` / `--weights` / `--yolo` all default into `sims/models/warehouse/`, so
this runs straight from a clone.

```bash
$ISAAC_PY sims/scripts/record_sensor_demo.py --headless \
  --controller rl --moment_scale 0.006 \
  --dump_figure_data sims/out/figdata_mega \
  --save_video sims/out/_mega.mp4 \
  --episodes 6 --max_steps 1100 --fps 50 --seed 1000
```

To reproduce the *exact* published figure, add `--rl_ckpt <your model_3250.pt>` (§ above).

Notes that cost time if you skip them:
- `--moment_scale` is the controller's moment gain. The MLP was learned at 50 Hz; at 100 Hz
  (decimation 1) **halve it** (≈0.003) or the flight drifts.
- Dump the episode that passes **4/4 gates** — change `--seed` until one does, and confirm (§2).
- The 4 people are blue capsules that patrol ~5 m back-and-forth (YOLO class "person"); the rest of
  the obstacle field is static props. The dump records both so the figure can tell them apart.

### 1b. Clean (drone-hidden) background — once per env layout

The top-down panel draws the path over a **clean** overhead plate. `--clean_overview` resets the
env, hides the roof and the drone prim, and captures the overhead background with calibration:

```bash
$ISAAC_PY sims/scripts/record_sensor_demo.py --headless --clean_overview \
  --dump_figure_data sims/out/figdata_mega
```

Writes `sims/out/figdata_mega/clean_bg.npz` (`ov_bg/ovK/ovpos/ovquat`). Calibration gotcha:
`set_world_poses_from_view` does **not** refresh `data.pos_w`/`quat` unless the camera cfg sets
`update_latest_camera_pose=True` **and** a render tick runs after — else the saved pose is
`[0,0,0]`/identity. The mode already handles this.

---

## 2. What got dumped (data schema)

`sims/out/figdata_mega/figure_data.npz` (one successful episode, `T` control steps):

| key | shape | meaning |
|---|---|---|
| `poses` | (T, 7) | drone pose; `[:, :3]` = xyz world |
| `t_s` | (T,) | flight time (s) |
| `obst_pos` | (T, M, 3) | every obstacle's world pos each step (M≈113) |
| `person_mask` | (M,) bool | which obstacles are the 4 patrolling people |
| `gates_world` | (4, 3) | gate centers |
| `imu_w` | (T, 3) | body gyro ω (roll/pitch/yaw) |
| `goal_cmd` | (T, 2) | desired-velocity command to the next gate |
| `alt_dtof`, `alt_baro` | (T,) | down-ToF / barometer altitude |
| `frame_steps` | (58,) | step indices of the dense per-frame captures |
| `chase`,`fpv`,`tof`,`det` | in `frames/frame_NNN.npz` | chase RGB (540,960,3), FPV grey (60,90), cross-ToF (4,8,8), YOLO boxes (≤4, 6) |

`clean_bg.npz` adds `ov_bg`/`ovK`/`ovpos`/`ovquat`. Verify after rendering (from the repo root):

```bash
$ISAAC_PY - <<'PY'
import numpy as np
d=np.load("sims/out/figdata_mega/figure_data.npz",allow_pickle=True)
print({k:getattr(d[k],'shape',None) for k in d.files})
op,pm=d["obst_pos"],np.asarray(d["person_mask"]).astype(bool)
print("people net displacement:",np.linalg.norm(op[-1,pm,:2]-op[0,pm,:2],axis=1).round(2),
      "(patrols return near start; travel-range is larger)")
PY
```

---

## 3. Solve the onboard K1 schedules

The deployed flight stack is three nets — **CTRL** `mlp_control` (100 Hz), **NAV** `fused_full`
(50 Hz), **YOLO** `yolov8_nano_64x96` (pipelined). The spec `networks_k1_flight_deployed.json`
encodes those rates over an ~110 ms window (12/6/5 instances).

```bash
# XPU-RT dynamic schedule (greedy, shards YOLO across all 8 harts)
$XPURT_PY scripts/run_xpurt_schedule.py \
  --networks-json data/toplevel/networks_k1_flight_deployed.json \
  --solver greedy --profiled --max-periodic-iters 1
# -> schedules/scheduled_networks_k1_flight_deployed_greedy_profiled.json

# ROS baseline: each net pinned to ONE hart, periodic releases, YOLO serial (no sharding)
$XPURT_PY scripts/ros_pinning_periodic.py \
  --from schedules/scheduled_networks_k1_flight_deployed_greedy_profiled.json \
  --out schedules/scheduled_ros_periodic_deployed.json
```

The contrast is the point: greedy shards YOLO across 8 harts (~20 ms/frame, meets the 22 ms
budget); ROS pins YOLO to one hart (24 ms/frame serial) → misses every deadline, 5 harts idle.
YOLO's real per-core scaling (list-scheduled from its dispatch graph): 1 core 24.0 ms, 2 cores
12.0, 4 cores 6.1, 8 cores 3.1 — see [§7](#7-honest-caveats).

---

## 4. Render the annotated Gantts

`plot_solver_gantt_annotated.py` draws the real per-hart dispatches, colored by net, with the
period-window shading (NAV 20 ms light purple, CTRL 10 ms light green nested), red sensor-in /
colored output arrows, and the YOLO per-frame deadline status (`--yolo-deadline`).

```bash
# XPU-RT: meets every YOLO deadline (green "✓ slack")
$XPURT_PY scripts/plot_solver_gantt_annotated.py --window-ms 112 --yolo-deadline 22 \
  --out results/hil_figures/solver_gantt_annotated

# ROS: misses every deadline (red "✗ +N ms"), diverging backlog + crash banner + idle harts
$XPURT_PY scripts/plot_solver_gantt_annotated.py \
  --sched schedules/scheduled_ros_periodic_deployed.json --window-ms 124 --yolo-deadline 22 --crash-note \
  --title "Onboard K1 schedule — ROS per-net pinning + greedy (baseline)" \
  --desc "naive ROS + greedy pinning: each net on its own hart, YOLO serial (no sharding) — 5 harts idle, YOLO misses every deadline" \
  --out results/hil_figures/solver_gantt_annotated_ros
```

---

## 5. Compose the mega figure(s)

`compose_mega_figure.py` is matplotlib-only (no Isaac). Run it with either env; from the repo root
it reads `sims/out/figdata_mega/` and the Gantt PNGs under `results/hil_figures/`.

```bash
# XPU-RT variant (drone completes the flight)
$ISAAC_PY sims/scripts/compose_mega_figure.py --data-dir sims/out/figdata_mega \
  --gantt results/hil_figures/solver_gantt_annotated.png \
  --out sims/out/paper_figure_mega
# ROS variant (same flight, ROS crash schedule + crash-truncated trajectory)
$ISAAC_PY sims/scripts/compose_mega_figure.py --data-dir sims/out/figdata_mega \
  --gantt results/hil_figures/solver_gantt_annotated_ros.png --crash-step 417 \
  --out sims/out/paper_figure_mega_ros
```

Layout (top→bottom): tall top-down aisle (time-colored path, gates, dotted-purple patrol paths,
4 numbered moment markers with descriptors) beside a 2×2 of the four moments (closer chase crop +
FPV/YOLO + detailed 4×8×8 cross-ToF); a telemetry row (velocity vectors, 3-axis IMU gyro, goal
heading); and the annotated K1 schedule. Useful flags: `--td-rot {1,3}` (aisle orientation;
1 = G1 at bottom), `--path-start N` (trim the pre-gate-1 approach), `--crash-step N` (ROS variant).

---

## 6. Output

`sims/out/paper_figure_mega.png` (XPU-RT solver — drone completes) and
`sims/out/paper_figure_mega_ros.png` (ROS baseline — crash-truncated), plus their `.pdf` twins, are
the deliverable.

---

## 7. Honest caveats

- **Sim speed cap.** The velocity action clamps to `max_speed = 2.0 m/s`
  (`sims/isaaclab_tasks/warehouse_nav/mdp_velocity_action.py`). The captured flight peaks ~1.36 m/s.
  A genuinely faster physical render needs that cap raised, and the controller was trained around the
  nominal speed — push it too far and the drone fails on **dynamics**, not scheduling.
- **YOLO parallelizes almost linearly** (measured, list-scheduled from the real dispatch graph):
  1 core 24.0 ms → 2 cores 12.0 → 4 cores 6.1 → 8 cores 3.1. So a "reserve the 4 P-cores for YOLO"
  partition **meets** the 22 ms budget (6.1 ms) and does **not** crash at cruise; only the naive
  **1-hart-per-net** pinning (YOLO stuck on one core = 24 ms > 22 ms) crashes. The crash in the ROS
  figure is that naive baseline. A multi-core partition only breaks under a much shorter
  (fly-faster) budget — around a 3.7× tighter deadline the 6.1 ms partition misses while XPU-RT's
  8-hart 3.1 ms still holds.
- **The ROS crash is schedule-level.** The flight footage is the real successful run. The ROS figure
  shows that the *same workload is infeasible under naive pinning* — not that the drone in those
  frames physically crashed.
- **Deployed-rate spec.** `networks_k1_flight_deployed.json` is a clone of `networks_k1_allneural`
  with the deployed rates. `fused_full` is the nav-lane proxy for the larger FusedSensorViT until
  that compiles (see the spec's `_comment`).

---

## Files this runbook uses (all repo-relative)

- `sims/scripts/record_sensor_demo.py` — render (`--dump_figure_data`, `--clean_overview`)
- `sims/scripts/compose_mega_figure.py` — the composite
- `sims/scripts/train_{steering_tracking,warehouse_nav,yolo}.py` — the trainers
- `sims/isaaclab_tasks/warehouse_nav/` — the env · `sims/models/warehouse/*.pt` — the models
- `scripts/run_xpurt_schedule.py` — the solver
- `scripts/ros_pinning_periodic.py` — the ROS per-net-pinning baseline
- `scripts/plot_solver_gantt_annotated.py` — the annotated Gantt
- `data/toplevel/networks_k1_flight_deployed.json` — the spec · `gen_mb/` — dispatch graphs + K1 profiles

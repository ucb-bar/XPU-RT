# Reproducing the warehouse sensor-fusion nav + HIL + K1 schedule experiments

How to get the full three-model warehouse pilot — the **FusedSensorNet** nav
model (grey FPV + 4× cross-ToF + state → yaw_rate/forward_speed), a low-level
velocity controller (RL PPO or the distilled MLP), and a **YOLOv8n**
gate/person/obstacle detector — flying, evaluated, recorded, and compiled +
scheduled onto a **SpaceMiT K1**, starting from a clone of this repo.

This is the warehouse counterpart of
[`docs/replicate_forest_trail_demo.md`](../Demo/replicate_forest_trail_demo.md). The
scene is the photoreal `full_warehouse` rack aisle with a 4-gate course; the
sensor rig, the models, and the K1 co-design are all documented below.

Reference command (what you're working toward — the composite demo video):

```bash
$PY XPU-RT/sims/scripts/record_sensor_demo.py --headless \
    --controller rl --safety \
    --weights   train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --rl_ckpt   train_out/logs/rsl_rl/crazyflie_steering_tracking/2026-08-28_00-04-21_velctrl_dr4/model_400.pt \
    --yolo      train_out/warehouse_yolov8n/train/weights/best.pt \
    --gantt_schedule /scratch2/agustin/XPU-RT/schedules/scheduled_networks_k1_mb_4model_4hz_fused_edf_profiled.json \
    --save_video out/v12_crowded_sensor_demo.mp4
```

## Dependency graph

```
env_isaaclab conda python  +  IsaacLab  (sim side)     merlin-dev env (cvxpy+mosek)  (schedule side)
        │                                                       │
        ▼                                                       │
warehouse_nav task + forest_trail/sensors.py                    │
(Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-                  │
 Play-WithSensors-Coll-v0)                                      │
        │                                                       │
        ├──► FusedSensorNet nav  (vitfly/models/fused_model.py) │
        │    collect_fused_warehouse.py → train_fused.py        │
        │    → train_out/fused_bc_warehouse_v12_mixed_cnn/…/best.pt
        │                                                       │
        ├──► RL velocity controller (rsl_rl PPO)                │
        │    train_steering_tracking.py --task                  │
        │    Isaac-Track-VelocityCtrl-DR-Crazyflie-v0           │
        │    → …/crazyflie_steering_tracking/*velctrl_dr4/model_400.pt
        │                                                       │
        └──► YOLOv8n {gate,person,obstacle}                     │
             gen_yolo_dataset.py → train_yolo.py                │
             → train_out/warehouse_yolov8n/train/weights/best.pt│
        │                                                       │
        ▼                                                       ▼
eval_fused_warehouse.py / eval_rl_controller_warehouse.py    ModelBlaster: 3 models → rvv_x60 int8
record_sensor_demo.py (chase+FPV+sensors+overhead+YOLO)      → gen/mb/ profiles
                                                             → run_xpurt_schedule.py --solver milp
                                                               --scheduler mosek → schedules/*.json
```

---

## 0. Prerequisites

- NVIDIA GPU with ≥ 6 GB **free** VRAM for the photoreal render (the warehouse
  spatial-hash grid OOMs below that on a shared card).
- Two conda environments (see §1): `env_isaaclab` for everything on the sim
  side, and `merlin-dev` for the MOSEK MILP scheduler.
- No `$DISPLAY` required — everything runs headless.
- Paths in this doc assume the DIMA project root as the working directory:

  ```bash
  cd /scratch/agustin/projects/DIMA          # train_out/, out/, logs/ resolve from here
  export PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
  ```

---

## 1. Environments

### 1a. `env_isaaclab` (sim, training, eval, video)

Everything in §2–§5 runs under this interpreter:

```bash
/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python   # = $PY
```

It carries Isaac Sim + IsaacLab + PyTorch + rsl_rl + ultralytics. The
env-building recipe (Isaac Sim install, `isaaclab.sh --install`, the version
pins, and the EULA/cuDNN gotchas) is documented once in
[`docs/Artifact/xpurt_env_setup.md`](../Artifact/xpurt_env_setup.md) — that doc builds an env named
`xpurt`; `env_isaaclab` is the same recipe under a different name and is the
interpreter actually used for the warehouse runs.

Set these for **every** sim invocation:

```bash
export TMPDIR=/tmp/agustin_isaac      # else PermissionError on /tmp/isaaclab (owned by another user)
export OMNI_KIT_ACCEPT_EULA=Y         # per-process EULA bypass (see env-setup doc)
```

- Always pass `--headless`. The eval/record scripts additionally hard-set
  `args_cli.headless = True` and `args_cli.enable_cameras = True`, but pass
  `--headless` anyway — omitting the offscreen kit is the root cause of frozen /
  black camera videos.
- Keep Isaac at **≤ 2048 envs**. Above that it competes for RAM with any
  concurrent Vivado/synthesis job and gets OOM-killed. The eval/record scripts
  use `num_envs=1`; only training sweeps hit this.

### 1b. `merlin-dev` (MOSEK scheduler only)

The MILP scheduler (§6) needs `cvxpy` + `mosek`, which live in `merlin-dev`,
**not** in `env_isaaclab`:

```bash
conda activate merlin-dev
export MOSEKLM_LICENSE_FILE=~/mosek/mosek.lic
```

> **Gotcha.** Running `run_xpurt_schedule.py --solver milp` in `env_isaaclab`
> does **not** fall back gracefully — it crashes on the missing `cvxpy` import.
> A "30-second success" there is an import crash, not a solve. Always run the
> MILP path from `merlin-dev` with the license exported.

---

## 2. The warehouse sensor-fusion nav experiment (the core)

### The task / env

Registered in
`sims/isaaclab_tasks/warehouse_nav/config/crazyflie/__init__.py`. The honest
eval env is:

```
Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0
    → WarehouseNavEnvCfg_PLAY_WithSensors_Coll
```

The `-Coll` variant makes the gate frames, rack rows, tall-thin stacked props,
and patrolling people **all real colliders**, so a mis-fly ends the episode via
the collision termination — a 4/4 gate pass is an honest "flew through the
opening", not a visual pass-through. (`-WithSensors` without `-Coll` uses visual
gates, used for data collection; `-Coll-Crowded-v0` is the densest showcase.)

- **Course**: 4 gates weaving the x≈−8 aisle in +y, at
  `FUSED_GATES = [(-8.05,9,2), (-8.30,13,2), (-7.75,17,2), (-8.05,21,2)]`
  (`warehouse_nav/mdp_gates.py`), 1.5 m openings, z = 2.0 m cruise. These are a
  gentler ±0.25 m weave than the RL `GATES` course, kept separate so the trained
  warehouse policy stays byte-identical.
- **Difficulty knobs**: `--obstacle_level` (active prop count) and
  `--prop_density` (the tall-thin stacked-prop field, `0..1`). The crowded eval
  is `prop_density 0.30`, all colliders.
- **Flight seam**: the warehouse's own cascade-stable geometric
  `VelocityCommandAction` (`mdp_velocity_action.py`), mapping nav's
  `(yaw_rate, forward_speed)` to the 4-D action
  (`a0 = fwd/(max_speed/2)−1`, `a2 = yaw/max_yawrate`; max_speed 2.0,
  max_yawrate 1.047 rad/s), with an altitude-hold autopilot decoding vz from the
  inclination channel toward TARGET_H = 2.0 m. Nav does horizontal guidance only.

### The onboard sensor suite

All sensors are defined once in `sims/isaaclab_tasks/forest_trail/sensors.py`
(env-agnostic — parents to `Robot/body`, reused verbatim by the warehouse):

| sensor | real part | sim cfg | model input |
|---|---|---|---|
| front camera | Himax **HM01B0** mono | RGB `CameraCfg` 320×240 (QVGA), ~87° HFOV, faces +x; luma via `front_greyscale()` (Rec.601) | `front_grey` (B,1,H,W), interpolated to **60×90** in the net |
| 4× cross ToF | ST **VL53L5CX** | four 8×8 depth cameras N/E/S/W, 63° diagonal FoV, 0.02–4 m; `tof_stack()` → (B,4,8,8), `tof_cross_composite()` → 24×24 + layout | `tof_cross` (B,4,8,8) normalized [0,1] |
| optical flow | PMW3901 | ground-relative (dx,dy) | `optical_flow` (B,2) |
| down ToF | VL53L1X | downward range → height AGL | `down_tof` (B,1) |
| baro / IMU | — | barometer + Madgwick attitude (`StateEstimator`) | `baro` (B,2), `quat` (B,4), `body_rates` (B,3) |
| goal | — | body-frame direction to the next gate | `desired_vel` (B,3) |

A 3rd-person `chase_camera_cfg()` (640×400, world-anchored) is also in that file
and is what the `--save_video` flag prefers.

### The nav model — FusedSensorNet

Defined in `vitfly/models/fused_model.py` (imported by the scripts via a
`sys.path` insert of `../vitfly/models`). It generalizes vitfly's `LSTMNetVIT`:

```
front_grey → vision encoder (vit | cnn) → 512
tof_cross  → conv → depth feat
[optical_flow, down_tof, baro, quat, body_rates, desired_vel] + group-present flags → state
concat → 3-layer LSTM (hidden 128) → FC → command
```

- The vision encoder always interpolates its input to **60×90** first.
- `vision_encoder="cnn"` (60×90 → 30×45 → 15×23 → 8×12 → 4×6, standard convs) is
  the deployable option (~1.55 M params / 6.45 M MACs, Gemmini/RVV-friendly). The
  `vit` variant is ~28.6 M MACs and is **10 Hz-infeasible** at the K1's clock —
  the shipped model is CNN. The eval scripts auto-detect the encoder from the
  checkpoint's state_dict keys.
- `out_dim=2` for the warehouse (`yaw_rate`, `forward_speed`); the LSTM hidden is
  carried across steps and reset per episode.

---

## 3. The models + how to get them

All three checkpoints already exist in `train_out/`. To retrain from scratch:

### 3a. Nav — FusedSensorNet (ship model = v12 CNN, camera-primary)

Pipeline: privileged goal-pursuit expert → DART-noise collection → BC
(truncated BPTT). The ship model was trained on **mixed** data (crowded demos +
the prop-free gate-follow set `fused_warehouse_gate_v9`, which re-injects
"go to the gate" so the net doesn't freeze in clutter):

```bash
# collect (crowded) — see sims/training/collect_fused_warehouse.py
$PY XPU-RT/sims/training/collect_fused_warehouse.py --headless \
    --prop_density 0.3 --obstacle_level 8 --base_speed 1.4 --noise_std 0.15 ...
# train the CNN on mixed data
$PY XPU-RT/sims/training/train_fused.py \
    --data <crowded>.pt <gate_v9>.pt --vision_encoder cnn --epochs 60
```

Ship checkpoint (already present):

```
train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt
```

> **Expected results:** yaw_sign_agree ≈ 0.96 offline. Closed-loop: **~100 %
> prop-free**, **~25–42 % crowded** (`prop_density 0.30`) — the honest crowded
> number is a genuine collision rate in the dense gate1→2 aisle, not freezing.
> Finding: the model is **camera-primary** — masking `tof_cross` does not hurt;
> the 4×8×8 ToF is a weaker avoidance sensor than the greyscale camera here.

### 3b. RL velocity controller (rsl_rl PPO)

A learned low-level velocity tracker consuming
`[base_lin_vel(3), base_ang_vel(3), projected_gravity(3), base_height(1),
steering_cmd(2), last_action(4)]` (steering_cmd = nav's (yaw_rate,
forward_speed)) → DirectThrustMoment action; actor is an ELU MLP
(16→256/128/64→4, no obs-norm).

```bash
$PY XPU-RT/sims/scripts/train_steering_tracking.py --headless \
    --task Isaac-Track-VelocityCtrl-DR-Crazyflie-v0 \
    --num_envs 2048 --max_iterations 400 --run_note velctrl_dr4
```

(`train_steering_tracking.py` flags: `--task`, `--num_envs`, `--max_iterations`,
`--seed`, `--resume`, `--entropy_coef`, `--init_noise_std`, `--actor_hidden_dims`,
`--run_note`, `--video`.) Checkpoints land under
`train_out/logs/rsl_rl/crazyflie_steering_tracking/<ts>_velctrl_dr4/`. Ship
checkpoint:

```
train_out/logs/rsl_rl/crazyflie_steering_tracking/2026-08-28_00-04-21_velctrl_dr4/model_400.pt
```

> **Expected results:** DR4 reward ≈ 258; **50 % gate-nav** at
> `moment_scale 0.006`, cruise ≈ 1.2 (beats the distilled MLP at 37.5 % and the
> classical law at 25 %). Note the sign-trap that produced it: the
> `lateral_drift_penalty` reward is `exp(−vy²)` (a *positive* reward maxed at
> zero sideslip) and needs a **positive** weight; a negative weight rewards
> sideslip and every controller circles.

### 3c. YOLOv8n {gate, person, obstacle}

Synthetic dataset flown by the v12 nav model (real onboard viewpoint
distribution), boxes derived from `instance_segmentation_fast` masks (IsaacLab's
Camera has no bbox annotator); the 8 scene classes collapse to
{0 gate, 1 person, 2 obstacle}, split by seed to avoid frame leakage.

```bash
# dataset (flags: --weights, --episodes, --seed, --stride, --base_speed,
#          --out_root, --min_box_px, --max_frames)
$PY XPU-RT/sims/scripts/gen_yolo_dataset.py --headless \
    --weights train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --episodes 24 --out_root out/yolo_warehouse

# train (flags: --data, --weights, --epochs, --imgsz, --rect/--no-rect,
#        --batch, --out, --device). --rect is ON by default; on 90×60 (W×H)
#        frames ultralytics letterboxes to 64×96 (H×W, 3:2) with ~0 padding.
$PY XPU-RT/sims/scripts/train_yolo.py \
    --data out/yolo_warehouse/dataset.yaml \
    --epochs 60 --imgsz 96 --out train_out/warehouse_yolov8n           # 64×96 rect build
# larger, more accurate build for the same 3:2 aspect:
$PY XPU-RT/sims/scripts/train_yolo.py \
    --data out/yolo_warehouse/dataset.yaml \
    --epochs 60 --imgsz 192 --out train_out/warehouse_yolov8n_128x192  # 128×192 rect build
```

Ship checkpoint: `train_out/warehouse_yolov8n/train/weights/best.pt`.

> **Expected results:** at **64×96** ≈ mAP gate 0.85 / person 0.58; at
> **128×192** ≈ mAP gate 0.975 / person 0.79 (overall mAP50 0.974 / mAP50-95
> 0.819). The 128×192 build is 3.5× the compute — see the co-design finding in
> §6. The YOLOv8n operator topology is preserved so ModelBlaster's existing
> `yolov8_nano` int8/fusion path applies unchanged.

---

## 4. Evaluation — warehouse gate-nav success rate

The eval is the crowded collidable course (`prop_density 0.30`): success = flew
through all 4 gates without a collision.

### Nav model (classical or distilled-MLP controller)

`sims/scripts/eval_fused_warehouse.py` — key flags: `--weights` (required),
`--episodes`, `--prop_density`, `--obstacle_level`, `--fixed_speed`,
`--controller {geom,mlp}`, `--mask_off <mods>` (ablation / Stage-2 vision via
`desired_vel`), `--yolo <best.pt>`, `--safety`, `--visual_gates`, `--save_video`,
`--dump_calib`/`--calib_max` (int8 calibration capture).

```bash
$PY XPU-RT/sims/scripts/eval_fused_warehouse.py --headless \
    --weights train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --episodes 12 --prop_density 0.3 --obstacle_level 8 --fixed_speed 1.3
```

> **Expected results:** ~**100 %** prop-free (`--prop_density 0` or omitted on
> the plain `-WithSensors` env), ~**25–42 %** crowded (`--prop_density 0.3`).
> Add `--mask_off desired_vel` for the camera-only Stage-2 vision-goal test.

### RL velocity controller

`sims/scripts/eval_rl_controller_warehouse.py` — flags: `--weights` (nav,
required), `--rl_checkpoint` (required), `--episodes`, `--prop_density`
(default 0.3), `--obstacle_level` (default 8), `--cruise_speed`, `--yaw_scale`,
`--moment_scale`.

```bash
$PY XPU-RT/sims/scripts/eval_rl_controller_warehouse.py --headless \
    --weights train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --rl_checkpoint train_out/logs/rsl_rl/crazyflie_steering_tracking/2026-08-28_00-04-21_velctrl_dr4/model_400.pt \
    --episodes 8 --prop_density 0.3 --moment_scale 0.006 --cruise_speed 1.2
```

> **Expected results:** RL controller ≈ **50 %** (4/8) at `moment_scale 0.006`.
> The default `--moment_scale 0.01` over-yaws; use 0.006 for the ship result.

---

## 5. The demo video

`sims/scripts/record_sensor_demo.py` renders one composite `.mp4`: a chase cam
following a bright-red drone, the full onboard sensor bank exactly as the model
receives each input (FPV grey 60×90, cross-ToF heatmap, flow arrow, altitude
traces, IMU/attitude, goal cmd), an elongated fixed overhead camera with the
red flight-path trace, optional YOLO boxes, and an embedded K1-schedule Gantt
strip with a playhead synced to sim time.

Flags: `--weights`, `--controller {classical,rl}`, `--rl_ckpt`, `--safety`,
`--yolo`, `--gantt_schedule`, `--save_video`, `--fixed_speed`, `--prop_density`,
`--obstacle_level`, `--moment_scale`, `--cruise_speed`, `--fps`.

```bash
$PY XPU-RT/sims/scripts/record_sensor_demo.py --headless \
    --controller rl --safety \
    --weights   train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --rl_ckpt   train_out/logs/rsl_rl/crazyflie_steering_tracking/2026-08-28_00-04-21_velctrl_dr4/model_400.pt \
    --yolo      train_out/warehouse_yolov8n/train/weights/best.pt \
    --gantt_schedule /scratch2/agustin/XPU-RT/schedules/scheduled_networks_k1_mb_4model_4hz_fused_edf_profiled.json \
    --save_video out/v12_crowded_sensor_demo.mp4
```

`--controller rl --safety --yolo …` exercises all three learned nets at once
(nav LSTM-conv + MLP/RL control + YOLO, with detections routed through
`hil/safety_layer.py` so a person/obstacle-ahead slows+steers; gates never
braked). Output is 1800×1100 @ 50 fps; it records the first 4/4 success on the
crowded collidable course (deepest run as fallback).

> **Gotchas:** needs `TMPDIR=/tmp/agustin_isaac` and ≥ 6 GB free VRAM; the
> overhead panel calls `hide_roof(stage)` so it doesn't image the ceiling. Empty
> `--gantt_schedule ""` disables the strip.

---

## 6. The K1 co-design (compile + schedule)

This is the ModelBlaster + XPU-RT half. It has its own full runbook — see
[`docs/K1/k1_modelblaster_xpurt_closed_loop.md`](../K1/k1_modelblaster_xpurt_closed_loop.md)
(profile → schedule → build → run → advise → rewrite on the physical board).
This section only names where the three warehouse models plug in.

### The three models on the board

All three are compiled to `rvv_x60` int8 (a K1-specific build of the `rvv`
core-kind), bit-exact against their goldens:

| model | role | ModelBlaster tag |
|---|---|---|
| nav (FusedSensorNet) | stateful gate nav | `fused_full` |
| MLP control | low-level velocity → wrench | `mlp_control` |
| YOLOv8n | gate/person/obstacle detect | `yolov8_nano` (+ `_64x96`, `_128x192` rect builds) |

Single-core int8 profiles live under
`/scratch2/agustin/XPU-RT/gen/mb/profile/rvv_x60/spacemit_x60/{fused_full,mlp_control,yolov8_nano,yolov8_nano_64x96,yolov8_nano_128x192,…}/`.
Note nav is **stateful** (LSTM): `nav[k] → nav[k+1]` must stay serial and must
**not** be parallelized like the stateless MLP/YOLO replicas — a scheduling
constraint the workload JSON encodes.

### Generate a schedule (MOSEK, in `merlin-dev`)

```bash
conda activate merlin-dev && export MOSEKLM_LICENSE_FILE=~/mosek/mosek.lic
cd /scratch2/agustin/XPU-RT
python3 scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_k1_mb_4model_4hz_fused.json \
    --solver milp --scheduler mosek --profiled
```

(`--solver {milp,greedy,greedy_periodic,decomposed}` — `milp` is the global
cvxpy/MOSEK solve; `--scheduler` default `mosek`; `--time-limit` caps the MILP
seconds; there are many `networks_k1_mb_*.json` workloads, from `1model` up to
`4model_4hz_fused` and per-Hz YOLO sweeps.) Outputs, keyed by config basename:

```
schedules/scheduled_networks_k1_mb_4model_4hz_fused_milp_profiled.json          the schedule
schedules/scheduled_networks_k1_mb_4model_4hz_fused_milp_profiled_metrics.json  makespan, misses
schedules/scheduled_networks_k1_mb_4model_4hz_fused_milp_profiled_report.json   SchedulerReport
```

> **Gotchas:** run this in `merlin-dev`, not `env_isaaclab` (missing cvxpy → a
> fast crash that looks like a solve). Read `makespan` from the `*_ms` fields /
> `units_note`, not the mislabeled `*_us` keys (see the K1 doc §5).

### Render the Gantt

```bash
python3 scripts/plot_scheduled_json.py \
    schedules/scheduled_networks_k1_mb_4model_4hz_fused_milp_profiled.json \
    --save plots/networks_k1_mb_4model_4hz_fused_milp_profiled.png
```

This is the same `scheduled_*.json` that `record_sensor_demo.py --gantt_schedule`
embeds as its playhead strip.

### The co-design finding (high level)

The dominant lever is YOLO input geometry, not the kernels: a **160²** square
letterbox is a predicted **226.9 ms** dispatch, of which ~26 ms convolves grey
padding bars; the aspect-matched **64×96** rect keeps every real pixel of the
90×60 FPV frame and is **≈ 46 ms — a 4.9× speedup at no accuracy cost**, because
the content box *is* 64×96. The 128×192 rect is the next legal 3:2 step (every
dim must be a multiple of 32) at 3.5× the cost, buying the higher mAP in §3c.
Full detail (rectangular-input rationale, PTQ calibration traps, the
core-kind-vs-backend-tag pitfall) is in the K1 doc.

---

## 7. Gotchas, in one place

| symptom | cause | fix |
|---|---|---|
| `PermissionError` on `/tmp/isaaclab/logs` | logger writes a dir owned by another user | `export TMPDIR=/tmp/agustin_isaac` |
| frozen / black camera video | offscreen kit not loaded | always pass `--headless` |
| Isaac OOM-killed mid-run | too many envs vs a concurrent synthesis job | keep **≤ 2048** envs; watch `free` |
| checkpoint glob resolves to nothing | `ls` into a var captured ANSI color codes | `ls --color=never …` |
| MILP "succeeds" in 30 s with no schedule | `cvxpy` missing in `env_isaaclab` | run `--solver milp` in **`merlin-dev`** + `MOSEKLM_LICENSE_FILE` |
| EULA prompt hangs a fresh Isaac process | per-process bypass not set | `export OMNI_KIT_ACCEPT_EULA=Y` every invocation |
| RL controller circles / crabs | `moment_scale` too high, or the `exp(-vy²)` sign trap | `--moment_scale 0.006`; positive lateral reward weight |
| nav does 0/4 in HIL | nav run at 10 Hz camera rate, not 50 Hz control rate | run the nav model every control step (LSTM timescale = 50 Hz) |

## Related docs

- [`docs/replicate_forest_trail_demo.md`](../Demo/replicate_forest_trail_demo.md) — the forest-trail sibling pilot.
- [`docs/Artifact/xpurt_env_setup.md`](../Artifact/xpurt_env_setup.md) — building the Isaac/IsaacLab conda env.
- [`docs/K1/k1_modelblaster_xpurt_closed_loop.md`](../K1/k1_modelblaster_xpurt_closed_loop.md) — the full K1 profile→schedule→build→run runbook.
</content>
</invoke>

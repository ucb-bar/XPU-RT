# Warehouse gate-nav — handoff for RoSE integration (for Dima)

The Isaac side of "a network navigating the warehouse (obstacles + people + checkpoints)" is
done. On the **clean** gate course it flies **100 %**; on the **complex collidable** course
(collidable gates + dense tall-thin stacked props + patrolling people, all real colliders) the
crowded-trained net is at **~42 %** — an honest number bounded by the analytic teacher, not the
HW. This is everything you need to plug our nav net into your RoSE-in-Isaac low-level loop.
**No RoSE bridge is included** — you own the Isaac↔RoSE side (RoSE already flies in Isaac on
your end); we own the net + env + sensors + the command seam documented below.

## Where everything lives (absolute paths on this host — you have read access)

Everything is under `/scratch/agustin/projects/DIMA` (world-readable on this machine, so just
`grep`/read in place; copy into your own `/scratch2/dima/...` if you want to modify/retrain).

| thing | absolute path |
|---|---|
| this doc + the env | `/scratch/agustin/projects/DIMA/XPU-RT/sims/isaaclab_tasks/warehouse_nav/` |
| sensor rig + estimator | `/scratch/agustin/projects/DIMA/XPU-RT/sims/isaaclab_tasks/forest_trail/{sensors,state_estimator}.py` |
| eval / reference driver | `/scratch/agustin/projects/DIMA/XPU-RT/sims/scripts/eval_fused_warehouse.py` |
| model class `FusedSensorNet` | `/scratch/agustin/projects/DIMA/vitfly/models/fused_model.py` *(separate `vitfly` repo)* |
| **crowded ship checkpoint (v12, CNN)** | `/scratch/agustin/projects/DIMA/train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt` |
| clean-course checkpoints (v8/v9) | `/scratch/agustin/projects/DIMA/train_out/fused_bc_warehouse_v9_stage1/2026-07-28_08-33-38/best.pt` |
| demo videos (crowded collidable) | `/scratch/agustin/projects/DIMA/out/v12_crowded_collidable_{chase,fullrun}.mp4` |
| conda python | `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python` |

Read it straight off the filesystem — **nothing is pushed to git** (the env/eval/docs are
uncommitted working-tree changes in the local **XPU-RT** checkout; the model class is in the
local **vitfly** checkout). Copy what you need into your own space; ping me if you ever want a
branch.

---

## What's included

| piece | path |
|---|---|
| Env (clean, non-collidable gates — expert/collect) | gym id `Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-v0` |
| Env (collidable gates + racks + prop field + people — honest eval/demo) | gym id `Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-v0` |
| Scene + sensor factory | `warehouse_nav/config/crazyflie/warehouse_nav_env_cfg.py` (`WarehouseNavEnvCfg_PLAY_WithSensors[_Coll]`) |
| Sensor rig (shared w/ forest) | `forest_trail/sensors.py` + `forest_trail/state_estimator.py` |
| **Trained net — crowded ship model (v12, CNN+LSTM)** | `train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt` |
| Trained net (v9, clean course; stage-1 flies 12/12) | `train_out/fused_bc_warehouse_v9_stage1/2026-07-28_08-33-38/best.pt` |
| Trained net (v9 vision-goal, camera-only, no goal vector) | `train_out/fused_bc_warehouse_v9_stage2vis/2026-07-28_10-39-48/best.pt` |
| Eval / reference driver | `sims/scripts/eval_fused_warehouse.py` |
| Demo video (crowded collidable, single clean 4/4 weave) | `out/v12_crowded_collidable_chase.mp4` |
| Demo video (crowded collidable, full 12-ep run, crashes incl.) | `out/v12_crowded_collidable_fullrun.mp4` |

Model class: `FusedSensorNet(out_dim=2)` in `vitfly/models/fused_model.py`. One class, two
interchangeable vision encoders (both emit a 512-D feature → identical downstream):
- **`vision_encoder="cnn"`** — the **crowded ship model** (v12). All int8/Gemmini-friendly
  conv2d + ReLU, no attention/softmax/LayerNorm. This is the HW-deployable one (see Loren's
  `XPU-RT/docs/hw_codesign_model_spec.md`). ~1.55 M params.
- **`vision_encoder="vit"`** — the Segformer ViT-LSTM (3.67 M params) that flies the *clean*
  course 100 %; too heavy for the 60 MHz SoC at 10 Hz, so not the crowded/deploy target.

Both carry an LSTM hidden state across steps. **Load the encoder that matches the checkpoint**
(v12 = `cnn`); the eval driver already infers it from the state-dict.

## Sensors simulated (the onboard suite)

Built by the `_WithSensors` scene; read via `forest_trail/sensors.py`. This is the ToF you were
building dummy geometry for — it's already faithful here (4× VL53L5CX, 8×8, N/E/S/W cross):

- **HM01B0 front greyscale cam** — `S.front_greyscale(env)` → `(N,1,H,W)`, 87° HFOV.
- **4× VL53L5CX cross ToF** — `S.tof_stack(env)` → `(N,4,8,8)` (N,E,S,W), 63° FoV, 0.02–4 m.
- **down-ToF** (VL53L1X) `S.down_tof`, **optical flow** (PMW3901) `S.optical_flow`, **barometer**
  `S.barometer`, and **Madgwick attitude** via `state_estimator.StateEstimator`.

The model input dict (see `sense()` in the eval, lines ~169–181):
`{front_grey, tof_cross, optical_flow, down_tof, baro, quat, body_rates, desired_vel, flags}`.
`desired_vel` is the goal direction × cruise speed (mapped goal); mask it (`--mask_off
desired_vel`) to run **camera-only** (vision-goal, no YOLO).

---

## The command seam (the integration interface)

This is where RoSE plugs in. **Model runs at 10 Hz; the low-level tracker runs at 100 Hz.**

```
FusedSensorNet(sensors) @10Hz ──► (yaw_rate, forward_speed)
                                       │
                     cmd_to_action()   │  (+ altitude-hold autopilot, see below)
                                       ▼
                 VelocityCommandAction  ──►  low-level tracker  ──► motors
                 (4-ch polar velocity)       (ours: geometric ctrl;
                                              YOURS: RoSE/TinyMPC replaces this)
```

**`VelocityCommandAction`** (`warehouse_nav/mdp_velocity_action.py`) — 4-ch action in `[-1,1]`:

| ch | meaning | mapping |
|---|---|---|
| a0 | forward speed | `speed = a0+1 ∈ [0,2]`; `vx = speed·cos(max_incl·a1)·(max_speed/2)` |
| a1 | inclination → climb | `vz = speed·sin(max_incl·a1)·(max_speed/2)` |
| a2 | yaw rate | `yawrate = a2·max_yawrate` |
| a3 | unused | — |

Limits: `max_speed=2.0 m/s`, `max_yawrate=π/3 rad/s`, `max_inclination=π/4`. Our controller is a
cascade-stable geometric velocity controller (Isaac port of aerial_gym `LeeVelocityController`),
gains `vel=2.0, att=200.0, rate=20.0`.

**Where RoSE goes:** your RoSE/TinyMPC low-level controller **replaces `VelocityCommandAction`
as the setpoint tracker**. The nav net emits a body-frame velocity/yaw setpoint at 10 Hz; RoSE
tracks it at 100 Hz. The contract is exactly the 4-ch polar velocity command above (or the raw
`(yaw_rate, forward_speed)` + a climb rate if you'd rather own altitude — see next).

**Altitude-hold:** in our eval, `cmd_to_action()` (`eval_fused_warehouse.py:98`) adds a fixed
altitude-hold autopilot (`TARGET_H=2.0 m`, back-solves a1 from a P-loop on height). The nav net
does **horizontal guidance only**. If RoSE owns altitude, drop this and feed only `(yaw_rate,
forward_speed)` + your own vz.

---

## Run it

```bash
# crowded collidable course with the v12 CNN ship model (records first clean weave-through):
/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python \
    XPU-RT/sims/scripts/eval_fused_warehouse.py --headless \
    --weights train_out/fused_bc_warehouse_v12_mixed_cnn/2026-08-03_19-51-49/best.pt \
    --prop_density 0.30 --obstacle_level 8 --fixed_speed 0.9 --episodes 12 \
    [--record_all] [--save_video out/demo.mp4]
```
- default = collidable gates (honest: a mis-fly ends the episode on contact). `--prop_density 0.30
  --obstacle_level 8` = the crowded prop field; `--visual_gates` = non-collidable; `--mask_off
  desired_vel` = camera-only vision-goal. `--fixed_speed` sets cruise (0.9 is v12's best on the
  crowded course; the speed head underfits, so the net's real job is steering).
- `--save_video` records the first successful weave-through (deepest run as fallback); add
  `--record_all` to record every episode into one continuous video.
- `success = flew through all 4 gates`. **v12 CNN: ~42 % on the crowded collidable course**
  (teacher-bounded), **12/12 on the clean course**.

Python env: `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python`. IsaacLab source at
`/scratch2/dima/IsaacLab/source/*` (added to path by the eval).

---

## The COMPLEX COLLIDABLE course (obstacles + people + gates, all colliders)

This is the crowded scene — read this, because two defaults will otherwise bite you:

- **Use gym-id `Isaac-Drone-Warehouse-Gates-Vision-Crazyflie-Play-WithSensors-Coll-Crowded-v0`.**
  It is the collidable gate course + rack rows + **tall-thin stacked-box prop field** + **patrolling
  people**, and every one of those is a **real collider** (contact → episode termination). One id, no
  flags — the safe way to get the crowded, colliding scene.
- **GOTCHA 1 — props default OFF.** The plain `...WithSensors-Coll-v0` (and the collect/eval scripts)
  set `reset_obstacles.params["prop_density"] = 0.0` for pure gate-following, so the tall prop field is
  absent unless you (a) use the `-Coll-Crowded-v0` id above, or (b) pass `--prop_density 0.3` to
  `eval_fused_warehouse.py`. Without one of those you get an empty aisle and think that's "the env".
- **GOTCHA 2 — the ~800-obstacle showcase is NOT a physics course.** `WarehouseSceneCfg_RichCam`
  builds its props with `collide=False` (visual prims only, drone flies a scripted route). Do **not**
  use it for collision testing — it looks crowded but nothing collides. The `-Coll-Crowded` env is the
  physics one.
- **Difficulty knobs:** `--prop_density` (0→1, default 0.3 in the Crowded env), `--obstacle_level`
  (curriculum count), and the tall-thin stack mix / height in `mdp_obstacles.py::reset_obstacle_field`
  (`kinds=box/crate/klt`, `stack_prob 0.85`, `stack_h 2.0–3.5 m`). People are always present in `Coll*`.

**One-command reproducer (launch + record + metrics):**
```bash
bash XPU-RT/sims/scripts/run_crowded_demo.sh [WEIGHTS.pt] [EPISODES]
```
Defaults to the crowded-trained CNN+LSTM checkpoint, records a chase-cam mp4, and writes a metrics
JSON (success = flew all 4 gates without hitting a prop/rack/person/gate). Run it once to **confirm
the crowded collidable scene loads and collides** before wiring RoSE.

**For RoSE:** the collision test is inherent — you keep this exact env + sensors and swap only the
low-level tracker (your TinyMPC/RoSE controller in place of `VelocityCommandAction`). Nothing extra to
enable; just point at the `-Coll-Crowded-v0` id.

## Notes for parallelizing

- Use `eval_fused_warehouse.py` as the reference driver — its `sense()` + `cmd_to_action()` are
  the exact glue you'd re-implement on the RoSE side (sensor dict assembly, LSTM hidden carry,
  command mapping).
- Gate course geometry (the "checkpoints"): `warehouse_nav/mdp_gates.py::FUSED_GATES` (4 gates,
  aisle centerline x≈−8, y = 9/13/17/21, z = 2.0).
- The prop field + patrolling people are real colliders in the `-Coll` variant; density is the
  `obstacle_level` arg (0 = gates only, up to a full curriculum field).

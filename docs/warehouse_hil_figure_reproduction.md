# Warehouse HIL nav — flight → schedule → paper figure (reproduce)

A runbook from a fresh checkout to the **warehouse mega figure**: the paper-friendly
replacement for the flight video. It ties together three things that live in this repo:

1. an **Isaac Lab** warehouse flight (a Crazyflie flying a 4-gate aisle past static props
   and 4 patrolling people), captured with the full onboard **sensor bank** (FPV grey + YOLO
   gate/person, 4× cross-ToF, optical flow, IMU, altitude, goal command);
2. the **onboard K1 schedule** that runs the deployed nets (CTRL = mlp_control, NAV = fused_full,
   YOLO = yolov8_nano), solved by the XPU-RT greedy solver and, for contrast, by a naive ROS
   per-net pinning baseline;
3. a single tall composite (`compose_mega_figure.py`) that renders the top-down aisle, four
   in-flight moments, telemetry, and the annotated schedule as one figure.

Every number in the figure is measured — the flight is a real Isaac run, the schedule uses the
measured K1 profile. Where a step is a modelling choice or has a caveat, it is called out in
[§7](#7-honest-caveats).

Two environments are used and they are **not** interchangeable:

| env | python | used for |
|---|---|---|
| Isaac Lab | `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python` | the flight render (steps 1–2) |
| XPU-RT | `/scratch2/agustin/XPU-RT/.venv/bin/python` | the scheduler + Gantt (steps 3–4) |

Paths below are absolute so the runbook is copy-pasteable. The sim scripts live in
`XPU-RT/sims/scripts/`; the scheduler scripts in `XPU-RT/scripts/`. Nothing here pushes to git.

---

## 0. One-time: what you need on disk

- The trained MLP flight controller checkpoint:
  `/scratch/agustin/projects/DIMA/logs/rsl_rl/crazyflie_steering_tracking/2026-08-29_19-27-23_larger_ctrl_512_512_256_128/model_3250.pt`
- The YOLOv8n gate/person model: `XPU-RT/sims/models/warehouse/yolov8n_gate_person_128x192.pt`
- A measured K1 profile checked into the scheduler (`--profiled` reads it); no board needed to
  reproduce the figure, only to re-measure the profile.

The warehouse env is `warehouse_nav` (the `WithSensors` variant). The aisle runs along +Y at
x≈−8, cruise z≈2; the four gates are at world y≈9, 13, 17, 21 (also stored as `gates_world` in
the dumped data). The drone spawns at the low-y (south) end.

---

## 1. Render the flight + sensor bank

Run headless with the Isaac python. `--controller rl` runs the trained MLP as the real
thrust/moment controller (not the analytic Lee law); `--dump_figure_data <dir>` writes the poses,
obstacle tracks, per-moment dense frames, and the full sensor bank.

```bash
cd /scratch/agustin/projects/DIMA
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
$PY XPU-RT/sims/scripts/record_sensor_demo.py --headless \
  --controller rl \
  --rl_ckpt logs/rsl_rl/crazyflie_steering_tracking/2026-08-29_19-27-23_larger_ctrl_512_512_256_128/model_3250.pt \
  --moment_scale 0.006 \
  --dump_figure_data XPU-RT/sims/out/figdata_mega \
  --save_video XPU-RT/sims/out/_mega.mp4 \
  --episodes 6 --max_steps 1100 --fps 50 --seed 1000
```

Notes that cost time if you skip them:
- `--moment_scale` is the controller's moment gain. The MLP was learned at 50 Hz; at 100 Hz
  (decimation 1) **halve it** (≈0.003) or the flight drifts.
- Dump the episode that actually passes **4/4 gates** — re-run / change `--seed` until one does,
  and confirm before composing (step 5's verify).
- The people are 4 blue capsules that patrol ~5 m back-and-forth (YOLO class "person"); the rest
  of the obstacle field is static props. The dump records both so the figure can tell them apart.

### 1b. Clean (drone-hidden) backgrounds — once per env layout

The top-down panel draws the flight path over a **clean** overhead plate (no drone baked in). A
`--clean_overview` mode resets the env, hides the roof, hides the drone prim, and captures the
overhead (and an isometric) background with correct calibration:

```bash
$PY XPU-RT/sims/scripts/record_sensor_demo.py --headless --clean_overview \
  --dump_figure_data XPU-RT/sims/out/figdata_mega
```

This writes `figdata_mega/clean_bg.npz` (`ov_bg/ovK/ovpos/ovquat` = top-down plate + calibration).
Calibration gotcha: `set_world_poses_from_view` does **not** refresh `data.pos_w`/`quat` unless the
camera cfg sets `update_latest_camera_pose=True` **and** a render tick runs after — otherwise the
saved pose is `[0,0,0]`/identity and the projection is garbage. The mode already does this.

---

## 2. What got dumped (data schema)

`figdata_mega/figure_data.npz` (one successful episode, `T` control steps):

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

`clean_bg.npz` adds `ov_bg`/`ovK`/`ovpos`/`ovquat`. Verify after rendering:

```bash
$PY - <<'PY'
import numpy as np
d=np.load("XPU-RT/sims/out/figdata_mega/figure_data.npz",allow_pickle=True)
print({k:getattr(d[k],'shape',None) for k in d.files})
op,pm=d["obst_pos"],np.asarray(d["person_mask"]).astype(bool)
disp=np.linalg.norm(op[-1,pm,:2]-op[0,pm,:2],axis=1)
print("people net displacement:",disp.round(2),"(patrols return near start; travel-range is larger)")
PY
```

---

## 3. Solve the onboard K1 schedules

Switch to the XPU-RT python. The deployed flight stack is three nets — **CTRL** `mlp_control`
(100 Hz), **NAV** `fused_full` (50 Hz), **YOLO** `yolov8_nano_64x96` (pipelined). The spec
`networks_k1_flight_deployed.json` encodes those rates over an ~110 ms window (12/6/5 instances).

```bash
cd /scratch2/agustin/XPU-RT
V=.venv/bin/python
# XPU-RT dynamic schedule (greedy, shards YOLO across all 8 harts)
$V scripts/run_xpurt_schedule.py --networks-json data/toplevel/networks_k1_flight_deployed.json \
  --solver greedy --profiled --max-periodic-iters 1
# -> schedules/scheduled_networks_k1_flight_deployed_greedy_profiled.json

# ROS baseline: each net pinned to ONE hart, periodic releases, YOLO serial (no sharding)
$V scripts/ros_pinning_periodic.py \
  --from schedules/scheduled_networks_k1_flight_deployed_greedy_profiled.json \
  --out schedules/scheduled_ros_periodic_deployed.json
```

The contrast is the point: greedy shards YOLO across 8 harts (~20 ms/frame, meets the 22 ms
budget); ROS pins YOLO to one hart (24 ms/frame serial), so it misses every deadline and 5 harts
sit idle. YOLO's real per-core scaling (list-scheduled from its dispatch graph): 1 core 24.0 ms,
2 cores 12.0, 4 cores 6.1, 8 cores 3.1 — see [§7](#7-honest-caveats) on what this means for a
"reserve the P-cores for YOLO" partition.

---

## 4. Render the annotated Gantts

`plot_solver_gantt_annotated.py` draws the real per-hart dispatches, colored by net, with the
period-window shading (NAV 20 ms light purple, CTRL 10 ms light green nested), red sensor-in /
colored output arrows, and the YOLO per-frame deadline status (`--yolo-deadline`).

```bash
# XPU-RT: meets every YOLO deadline (green "✓ slack")
$V scripts/plot_solver_gantt_annotated.py --window-ms 112 --yolo-deadline 22 \
  --out results/hil_figures/solver_gantt_annotated

# ROS: misses every deadline (red "✗ +N ms"), diverging backlog + crash banner + idle harts
$V scripts/plot_solver_gantt_annotated.py \
  --sched schedules/scheduled_ros_periodic_deployed.json --window-ms 124 --yolo-deadline 22 --crash-note \
  --title "Onboard K1 schedule — ROS per-net pinning + greedy (baseline)" \
  --desc "naive ROS + greedy pinning: each net on its own hart, YOLO serial (no sharding) — 5 harts idle, YOLO misses every deadline" \
  --out results/hil_figures/solver_gantt_annotated_ros
```

---

## 5. Compose the mega figure(s)

Back in the Isaac python (matplotlib only — no Isaac needed here, but its python has the deps).
`compose_mega_figure.py` reads `figdata_mega/` and embeds one of the Gantt PNGs.

```bash
cd /scratch/agustin/projects/DIMA
PY=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
G=/scratch2/agustin/XPU-RT/results/hil_figures
# XPU-RT variant (drone completes the flight)
$PY XPU-RT/sims/scripts/compose_mega_figure.py --data-dir XPU-RT/sims/out/figdata_mega \
  --gantt $G/solver_gantt_annotated.png --out out/paper_figure_mega
# ROS variant (same flight, ROS crash schedule at the bottom)
$PY XPU-RT/sims/scripts/compose_mega_figure.py --data-dir XPU-RT/sims/out/figdata_mega \
  --gantt $G/solver_gantt_annotated_ros.png --out out/paper_figure_mega_ros
```

Output: `out/paper_figure_mega{,_ros}.{png,pdf}`. Layout (top→bottom): tall top-down aisle
(time-colored path, gates, dotted-purple patrol paths for the people, 4 numbered moment markers
with descriptors) beside a 2×2 of the four moments (closer chase crop + FPV/YOLO + detailed
4×8×8 cross-ToF); a telemetry row (velocity vectors, 3-axis IMU gyro, goal heading); and the
annotated K1 schedule.

Useful `compose_mega_figure.py` flags: `--td-rot {1,3}` (aisle orientation; 1 = G1 at bottom),
`--path-start N` (trim the pre-gate-1 approach line), `--td-flipx`.

---

## 6. Optional: the gallery

`scratchpad/build_gallery.py` embeds all the co-design figures (data-URI) into one self-contained
`figure_gallery.html`. Rebuild it and open locally after regenerating any figure.

---

## 7. Honest caveats

- **Sim speed cap.** The velocity action clamps to `max_speed = 2.0 m/s`
  (`isaaclab_tasks/warehouse_nav/mdp_velocity_action.py`). The captured flight peaks ~1.36 m/s.
  A genuinely faster physical render needs that cap raised, and the controller was trained around
  the nominal speed — push it too far and the drone fails on **dynamics**, not scheduling.
- **YOLO parallelizes almost linearly** (measured, list-scheduled from the real dispatch graph):
  1 core 24.0 ms → 2 cores 12.0 → 4 cores 6.1 → 8 cores 3.1. So a "reserve the 4 P-cores for
  YOLO" partition **meets** the 22 ms budget (6.1 ms) and does **not** crash at cruise; only the
  naive **1-hart-per-net** pinning (YOLO stuck on one core = 24 ms > 22 ms) crashes. The crash in
  the ROS figure is that naive baseline. A multi-core partition only breaks under a much shorter
  (fly-faster) budget — around a 3.7× tighter deadline the 6.1 ms partition misses while XPU-RT's
  8-hart 3.1 ms still holds.
- **The ROS crash is schedule-level.** The flight footage is the real successful run (flown under
  a feasible schedule). The ROS figure shows that the *same workload is infeasible under naive
  pinning* — not that the drone in those frames physically crashed. Frame captions accordingly.
- **Deployed-rate spec.** `networks_k1_flight_deployed.json` is a clone of `networks_k1_allneural`
  with the deployed rates (CTRL 100 Hz, NAV 50 Hz, YOLO pipelined). `fused_full` is the nav-lane
  proxy for the larger FusedSensorViT until that compiles (see the spec's `_comment`).

---

## Scripts touched by this runbook

- `XPU-RT/sims/scripts/record_sensor_demo.py` — flight render, `--dump_figure_data`, `--clean_overview`
- `XPU-RT/sims/scripts/compose_mega_figure.py` — the composite
- `XPU-RT/scripts/run_xpurt_schedule.py` — the solver
- `XPU-RT/scripts/ros_pinning_periodic.py` — the ROS per-net-pinning baseline schedule
- `XPU-RT/scripts/plot_solver_gantt_annotated.py` — the annotated Gantt
- specs: `XPU-RT/data/toplevel/networks_k1_flight_deployed.json`

# Figure runbook — how each headline PNG is made (inputs + exact command)

Every figure below is reproducible from on-disk artifacts. Repo root = `/scratch2/agustin/XPU-RT`.
Scheduler figures use the XPU-RT venv; the two warehouse mega plots + the HIL scatter need the Isaac env
(`env_isaaclab`) to (re)generate their flight data. `export XPURT_CPSAT_WORKERS=0` for any CP-SAT solve.

Composition scripts marked `scripts/…` live in the repo. Scripts marked `(scratchpad)` are the newer
figure builders that should be copied into `scripts/` for a clean checkout (they only read on-disk artifacts).

---

## 1. `schedule_evolution_mega` — the co-design "Gantt after Gantt" (the 3rd mega plot)
**Story**: og (fused, RVV) → +sharding → +unfuse (ModelBlaster graph rewrite) → runtime feedback
(board-calibrated re-solve) → +other. Each panel a REAL CP-SAT schedule; per-round makespan Δ + verdict.
- **Generate the sequence** (re-solves 5 lever configs fresh with CP-SAT):
  `.venv/bin/python scratchpad/evolution_seq.py`  → writes `schedules/scheduled__evo_*_cpsat_profiled.json`
  and `scratchpad/evo_panels.json`.
- **Render**:
  `.venv/bin/python scratchpad/evolution_mega.py --spec data/toplevel/_evo_og.json \
     $(python -c "import json;print(' '.join('--panel \"'+p+'\"' for p in json.load(open('scratchpad/evo_panels.json'))))")`
  → `results/codesign_feedback/schedule_evolution_mega.{png,pdf}`.
- **Inputs**: the 5 `_evo_*` schedules + their `_metrics.json`; the YOLO fused/unfused dispatch graphs under
  `gen_mb/vmfb/yolov8_nano/…`; `k1_board_calibration.json` (for the runtime-feedback round).

## 2. `hil_ablation_scatter` — HIL speed × frequency × crash/success (GPU)
- **Generate** the per-flight grid (real Isaac flights; ~120 flights, GPU-hours):
  `bash scratchpad/hil_ablation_grid.sh`  → appends rows to `scratchpad/hil_grid/hil_ablation.csv`
  (via `sims/scripts/sweep_rate_demo.py --sweep-csv`; one row/episode:
  `seed,cruise_speed,control_dt_ms,sched_latency_ms,hold_steps,eff_cmd_hz,…,outcome`).
- **Render**:
  `.venv/bin/python scratchpad/hil_ablation_scatter.py --csv scratchpad/hil_grid/hil_ablation.csv`
  → `results/codesign_feedback/hil_ablation_scatter.{png,pdf}`. Overlays the analytic crash-frontier from
  `results/microros_baseline_k1/flyfaster_crash_band.json`.

## 3–4. `mega_warehouse_xpurt` / `mega_warehouse_ros` — warehouse HIL mega plots (GPU to regen flight)
- **Flight data** (one clean successful weave; dump poses + a drone-free overhead background):
  `<env_isaaclab>/python sims/scripts/record_sensor_demo.py --headless --controller rl \
     --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sched_latency_ms 12.40 --decimation 1 \
     --prop_density 0.35 --obstacle_level 8 --fixed_speed 1.2 --episodes 4 --seed 2000 \
     --dump_figure_data <dir>` then once more with `--clean_overview --clean_out <dir>` for `clean_bg.npz`.
- **Schedule Gantt** (bottom panel): `.venv/bin/python scripts/plot_solver_gantt_annotated.py \
     --sched schedules/scheduled__flight_deployed_2frame_greedy_profiled.json --window-ms 44 \
     --out results/codesign_feedback/gantt_annotated_cpsat` (CP-SAT variant when a feasible full-frame CP-SAT
  schedule is available; the deployed 2-frame CP-SAT does not converge feasibly, so use a feasible schedule).
- **Compose**: `<env_isaaclab>/python sims/scripts/compose_mega_figure.py --data-dir <dir> \
     --gantt <gantt.png> [--crash-step 779] --out results/codesign_feedback/mega_warehouse_{xpurt,ros}`.
  `--crash-step` (ROS variant only) truncates the weave mid-course at a crate tower with a crash marker.
- **Notes**: the top-down uses the realistic overhead render (`clean_bg.npz`; without it the drone is baked in)
  with crate-tower box markers overlaid; IMU is smoothed; moment times come from `t_s[step]`.

## 5. `solver_win_sensor` — CP-SAT beats greedy (contended sensor workload)
`.venv/bin/python scripts/compose_solver_win.py \
   --greedy schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_greedy_profiled.json \
   --cpsat  schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_cpsat_profiled.json \
   --spec   data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0.json --window 32 \
   --out results/codesign_feedback/solver_win_sensor`
(greedy 4 misses / INFEASIBLE vs CP-SAT 0 misses / PROVEN OPTIMAL, 27.85 vs 30.64 ms).

## 6. Supporting Gantts / comparison
- `gantt_annotated_{cpsat,ros}` — `scripts/plot_solver_gantt_annotated.py --sched <schedule> --window-ms 44
  [--crash-note --yolo-deadline 22]` (ROS variant shows serial-YOLO backlog → crash).
- `warehouse_crash_speed` / `warehouse_solver_flight` / `solver_comparison_*` — scratchpad builders reading
  `schedules/*_metrics.json` + the Isaac sweep CSV.

---
**Regenerate all** (from cached artifacts where possible): `bash scripts/make_all_codesign_figures.sh`.

# Figure runbook — how each headline PNG is made (inputs + exact command)

Every figure below is reproducible from on-disk artifacts. Repo root = `/scratch2/agustin/XPU-RT`.
Scheduler figures use the XPU-RT venv; the two warehouse mega plots + the HIL scatter need the Isaac env
(`env_isaaclab`) to (re)generate their flight data. `export XPURT_CPSAT_WORKERS=0` for any CP-SAT solve.

Composition scripts marked `scripts/…` live in the repo. Scripts marked `(scratchpad)` are the newer
figure builders that should be copied into `scripts/` for a clean checkout (they only read on-disk artifacts).

---

## 1. `schedule_evolution_mega` — the co-design "Gantt after Gantt" (the 3rd mega plot)
**Story** (contended sensor-fusion workload, deadline-miss trajectory **4 → 0 → 4 → 0**, each panel a REAL
solve or a measured re-cost — nothing faked):
1. **og** — RVV, singletons (1 net per hart): misses control deadlines.
2. **+ shard + IME** — AOT levers (multi-hart widths + matrix-engine routing): meets every deadline ON THE GANTT
   (4 → 0). `Δ makespan` and `deadlines X→Y miss` annotated between panels.
3. **runtime feedback** — panel-2 schedule RE-COST on the measured board (+31% per-op inflation): deadlines the
   optimistic Gantt promised are now MISSED (0 → 4). Panel tinted (board-calibrated round).
4. **re-schedule on board-calibrated costs** — CP-SAT re-solves knowing the true costs: deadlines RECOVERED
   (4 → 0). This is the loop closing: adjust after runtime feedback, before any further optimization.

Idle stretches are compressed into grey break columns (real-ms x-axis kept); misses are ringed red.
- **Generate the sequence** (`XPURT_CPSAT_WORKERS=0`; CP-SAT solves for og/AOT/fix + a board re-cost):
  `XPURT_PY=$(command -v python) python scripts/gen_schedule_evolution.py`
  → writes the 4 panel schedules + `panels.json` under `results/codesign_feedback/sensor_evo/`.
  (The board re-cost is `scripts/recost_schedule_on_board.py` — re-times a fixed schedule under the board's
  per-op multipliers; the software twin of "how the run differs from the Gantt".)
- **Render**:
  `.venv/bin/python scripts/compose_schedule_evolution.py \
     --spec data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
     --panels-json results/codesign_feedback/sensor_evo/panels.json`
  → `results/codesign_feedback/schedule_evolution_mega.{png,pdf}`.
- **Inputs**: the sensor workload spec `_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json` + its profiles
  under `gen_mb/…`; `k1_board_calibration.json` (drives both the re-cost and the board-calibrated re-solve).

## 2. `hil_ablation_phase` — HIL command-rate phase diagram (GPU for the grid)
- **Generate** the per-flight grid (real Isaac flights; 5 speeds × 4 rates × 6 seeds = 120 flights, GPU-hours).
  Needs the conda `env_isaaclab` python — pass it via `ISAAC_PY` (see `docs/REPRODUCE.md` §1):
  `ISAAC_PY=<env_isaaclab>/bin/python bash scripts/hil_ablation_grid.sh`
  → appends rows to `results/codesign_feedback/hil_grid/hil_ablation.csv` (via
  `sims/scripts/sweep_rate_demo.py --sweep-csv`; one row/episode:
  `seed,cruise_speed,sim_dt,decimation,control_dt_ms,sched_latency_ms,hold_steps,eff_cmd_hz,…,outcome`).
  Overridable env: `XPURT_REPO`, `HIL_OUTDIR`, `HIL_WEIGHTS`. The 120-flight CSV is committed at
  `results/codesign_feedback/hil_ablation.csv` so the diagram re-renders without the GPU sweep.
- **Render**:
  `.venv/bin/python scripts/hil_ablation_phase.py --csv results/codesign_feedback/hil_ablation.csv`
  → `results/codesign_feedback/hil_ablation_phase.{png,pdf}`. A smooth safe→crash surface over (speed, command
  rate) with the crash frontier drawn, the raw 6-seed cells overlaid, and the schedulers' sustainable command
  rates marked (XPU-RT shard 204 Hz deep-safe, greedy 125 Hz, ROS 81 Hz on the frontier). Replaces the old
  `hil_ablation_scatter` (kept for reference).

## 3–4. Warehouse figures — combined showdown + the two mega plots (GPU to regen flights)
- **Flight data** — TWO dumps: the XPU-RT successful weave and the ROS crash:
  `<env_isaaclab>/python sims/scripts/record_sensor_demo.py --headless --controller rl \
     --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sched_latency_ms 12.40 --decimation 1 \
     --prop_density 0.35 --obstacle_level 8 --fixed_speed 1.2 --episodes 4 --seed 2000 \
     --dump_figure_data <xpu-dir>` (+ once with `--clean_overview --clean_out <xpu-dir>` for `clean_bg.npz`);
  the ROS crash dump is the same command with the ROS-rate latency (crashes ~y=10, past gate 1) → `<ros-dir>`.
- **`warehouse_showdown`** (the combined, horizontal figure — the headline): both paths on one top-down aisle,
  2 ROS + 2 XPU snapshots, IMU/goal/speed/velocity comparing both, and the combined XPU-over-ROS Gantt:
  `<env_isaaclab>/python sims/scripts/compose_warehouse_showdown.py --xpu-dir <xpu-dir> --ros-dir <ros-dir> \
     --rot 0` → `results/codesign_feedback/warehouse_showdown.{png,pdf}`. Schedules default to the committed
  `scheduled__flight_deployed_2frame_cpsat_profiled.json` (XPU) and `scheduled_ros_partition_deployed.json` (ROS).
- **`mega_warehouse_xpurt` / `mega_warehouse_ros`** (the per-scheduler mega plots) — Gantt strip via
  `scripts/plot_solver_gantt_annotated.py --sched schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json`
  (the CP-SAT deployed schedule: 40.4 ms 0-miss, balanced), then
  `<env_isaaclab>/python sims/scripts/compose_mega_figure.py --data-dir <xpu-dir> --gantt <gantt.png>
   [--crash-step 779] --out results/codesign_feedback/mega_warehouse_{xpurt,ros}`.
- **Notes**: people are projected at their real height (z≈0.85, not 2.0); moment markers are placed at the
  actual gate crossings; IMU is smoothed. `make_all_codesign_figures.sh §5` drives all three (set
  `WAREHOUSE_FIGDATA`/`WAREHOUSE_ROS_FIGDATA`).

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

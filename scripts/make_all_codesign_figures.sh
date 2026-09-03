#!/usr/bin/env bash
# Regenerate the headline co-design figures from on-disk artifacts.
# CPU-only steps run here; GPU steps (Isaac flight data for the warehouse mega plots + HIL scatter grid) are
# NOTED and skipped unless their inputs already exist. See docs/figure_runbook.md for full commands.
set -u
REPO=/scratch2/agustin/XPU-RT
PY=$REPO/.venv/bin/python
SP=/scratch/agustin/tmp/claude-2621/-scratch-agustin-projects-DIMA/057226a3-598b-40aa-8396-ef0c5c742cd9/scratchpad
R=$REPO/results/codesign_feedback
export XPURT_CPSAT_WORKERS=0
cd "$REPO"
say(){ echo "=== $* ==="; }

# 1. Solver-win Gantt (CP-SAT beats greedy) — CPU, cached schedules
S=schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0
if [ -f ${S}_cpsat_profiled.json ]; then
  say "solver_win_sensor"; $PY scripts/compose_solver_win.py --greedy ${S}_greedy_profiled.json \
    --cpsat ${S}_cpsat_profiled.json --spec data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0.json \
    --window 32 --out $R/solver_win_sensor \
    --title "Exact scheduling recovers what greedy drops — s5.0 sensor-fusion + IME" \
    --annotate "ffn_block0 overruns its 20 ms deadline\n(late dispatches ringed red)@24,4" || true
fi

# 2. Annotated onboard K1 Gantts (XPU-RT interleaved / ROS backlog) — CPU
if [ -f schedules/scheduled__flight_deployed_2frame_greedy_profiled.json ]; then
  say "gantt_annotated_cpsat"; $PY scripts/plot_solver_gantt_annotated.py \
    --sched schedules/scheduled__flight_deployed_2frame_greedy_profiled.json --window-ms 44 \
    --out $R/gantt_annotated_cpsat \
    --title "Onboard K1 warehouse schedule — CTRL 100Hz + NAV 50Hz interleaved per YOLO frame, YOLO sharded over 8 cores" \
    --desc "real XPU-RT schedule, K1 measured profile" || true
  say "gantt_annotated_ros"; $PY scripts/plot_solver_gantt_annotated.py \
    --sched schedules/scheduled_ros_partition_deployed.json --window-ms 44 --crash-note --yolo-deadline 22 \
    --out $R/gantt_annotated_ros --title "ROS static partition — serial YOLO overruns 22 ms every frame, E-cores idle → crash" \
    --desc "ROS per-net pinning, K1 measured profile" || true
fi

# 3. Schedule-evolution mega plot — CPU, needs the fresh _evo_* schedules (run scratchpad/evolution_seq.py first)
if [ -f $SP/evo_panels.json ]; then
  say "schedule_evolution_mega"
  PANELS=$($PY -c "import json;print(' '.join('--panel '+repr(p) for p in json.load(open('$SP/evo_panels.json'))))")
  eval $PY $SP/evolution_mega.py --spec data/toplevel/_evo_og.json $PANELS || true
fi

# 4. HIL ablation scatter — needs the grid CSV (run scratchpad/hil_ablation_grid.sh on a GPU box first)
if [ -f $SP/hil_grid/hil_ablation.csv ]; then
  say "hil_ablation_scatter"; $PY $SP/hil_ablation_scatter.py --csv $SP/hil_grid/hil_ablation.csv || true
else
  echo "(skip hil_ablation_scatter: run scratchpad/hil_ablation_grid.sh on a GPU box first)"
fi

# 5. Warehouse mega plots — need Isaac flight dumps (GPU). Regenerate only if the figdata exists.
CFD=$SP/crash_demo/complete_figdata
if [ -f $CFD/figure_data.npz ] && [ -f $CFD/clean_bg.npz ]; then
  ISAAC=/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python
  say "mega_warehouse_xpurt"; $ISAAC sims/scripts/compose_mega_figure.py --data-dir $CFD \
    --gantt $R/gantt_annotated_cpsat.png --out $R/mega_warehouse_xpurt || true
  say "mega_warehouse_ros"; $ISAAC sims/scripts/compose_mega_figure.py --data-dir $CFD \
    --gantt $R/gantt_annotated_ros.png --crash-step 779 --out $R/mega_warehouse_ros || true
else
  echo "(skip warehouse mega plots: need Isaac flight dump; see docs/figure_runbook.md §3-4)"
fi

say "done — figures in $R"

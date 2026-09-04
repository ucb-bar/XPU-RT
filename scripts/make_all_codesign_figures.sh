#!/usr/bin/env bash
# Regenerate the headline co-design figures from on-disk artifacts.
# CPU-only steps run here; GPU steps (Isaac flight data for the warehouse mega plots + HIL scatter grid) are
# NOTED and skipped unless their inputs already exist. See docs/figure_runbook.md for full commands.
set -u
REPO="${XPURT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${XPURT_PY:-$REPO/.venv/bin/python}"
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
    --sched schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json --window-ms 44 \
    --out $R/gantt_annotated_cpsat \
    --title "Onboard K1 warehouse schedule — CP-SAT balances CTRL 100Hz + NAV 50Hz + YOLO sharded across 8 cores (max hart 24ms vs greedy 38ms)" \
    --desc "real CP-SAT schedule, K1 measured profile, 40.4ms 0-miss" || true
  say "gantt_annotated_ros"; $PY scripts/plot_solver_gantt_annotated.py \
    --sched schedules/scheduled_ros_partition_deployed.json --window-ms 44 --crash-note --yolo-deadline 22 \
    --out $R/gantt_annotated_ros --title "ROS static partition — serial YOLO overruns 22 ms every frame, E-cores idle → crash" \
    --desc "ROS per-net pinning, K1 measured profile" || true
fi

# 3. Schedule-evolution mega plot — CPU. og -> +shard+IME -> board re-cost (feedback) -> board-cal re-solve (fix).
EVO=data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json
if [ ! -f $R/sensor_evo/panels.json ]; then
  say "gen_schedule_evolution (re-solving og/AOT/fix with CP-SAT + board re-cost)"; $PY scripts/gen_schedule_evolution.py || true
fi
if [ -f $R/sensor_evo/panels.json ]; then
  say "schedule_evolution_mega"
  $PY scripts/compose_schedule_evolution.py --spec $EVO --panels-json $R/sensor_evo/panels.json \
    --title "Co-design schedule evolution — contended sensor-fusion workload: AOT opts meet the Gantt, runtime feedback exposes board deadline misses, re-scheduling recovers them" || true
fi

# 4. HIL command-rate PHASE DIAGRAM — needs the grid CSV (committed copy at $R/hil_ablation.csv; regen via
#    scripts/hil_ablation_grid.sh on a GPU box with ISAAC_PY set — see docs/figure_runbook.md §2).
CSV=$R/hil_ablation.csv; [ -f $R/hil_grid/hil_ablation.csv ] && CSV=$R/hil_grid/hil_ablation.csv
if [ -f "$CSV" ]; then
  say "hil_ablation_phase"; $PY scripts/hil_ablation_phase.py --csv "$CSV" || true
else
  echo "(skip hil_ablation_phase: run scripts/hil_ablation_grid.sh on a GPU box first)"
fi

# 5. Warehouse figures — need Isaac flight dumps (GPU). XPU-RT success dump + ROS crash dump.
CFD="${WAREHOUSE_FIGDATA:-$R/crash_demo/complete_figdata}"       # XPU-RT (successful) flight
RFD="${WAREHOUSE_ROS_FIGDATA:-$R/crash_demo/crash_figdata}"      # ROS (crash) flight
ISAAC="${ISAAC_PY:-python}"
if [ -f $CFD/figure_data.npz ] && [ -f $RFD/figure_data.npz ]; then
  say "warehouse_showdown (combined XPU-RT vs ROS)"; $ISAAC sims/scripts/compose_warehouse_showdown.py \
    --xpu-dir $CFD --ros-dir $RFD --rot 0 --out $R/warehouse_showdown || true
fi
if [ -f $CFD/figure_data.npz ] && [ -f $CFD/clean_bg.npz ]; then
  say "mega_warehouse_xpurt"; $ISAAC sims/scripts/compose_mega_figure.py --data-dir $CFD \
    --gantt $R/gantt_annotated_cpsat.png --out $R/mega_warehouse_xpurt || true
  say "mega_warehouse_ros"; $ISAAC sims/scripts/compose_mega_figure.py --data-dir $CFD \
    --gantt $R/gantt_annotated_ros.png --crash-step 779 --out $R/mega_warehouse_ros || true
else
  echo "(skip warehouse figures: need Isaac flight dumps; see docs/figure_runbook.md §2-4)"
fi

say "done — figures in $R"

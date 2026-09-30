#!/usr/bin/env bash
# =====================================================================
# DENSE HIL flight-envelope grid: drone SPEED x command RATE x crash/success.
#
# This is the dense-grid successor to hil_ablation_grid.sh. That script sets
# the command rate with a ZOH *latency* hold at a 100 Hz base, so it tops out
# at 100 Hz and cannot exercise the higher rates XPU-RT's schedule unlocks
# (greedy 125 Hz, feedback-sharded 204 Hz). This one reaches them.
#
# Rate model (exact-rate via decimation, identical physics in every cell):
#   * sim.dt is FIXED across the whole grid (SIM_DT, default 0.001 s = 1000 Hz
#     physics), so the disturbance a given moment_scale injects per physics
#     step is identical in every cell -> outcome depends only on
#     (cruise_speed, command_rate, seed), which is what the 3-panel figure needs.
#   * the command is recomputed once per `decimation` physics steps and HELD in
#     between (zero-order hold) -> effective command rate = 1 / (sim.dt * dec).
#     decimation = round( 1 / (rate * sim.dt) ), so eff_cmd_hz ~= the target rate
#     (recorded exactly per row as `eff_cmd_hz`; the plot bins on that column).
#   * sched_latency_ms is left at 0: the decimation hold already IS the ZOH.
#   * flight duration is held ~constant (FLIGHT_S, default 14 s) by scaling
#     max_steps = round(rate * FLIGHT_S), so every cell flies the same seconds.
#
# Controller-gain calibration across the rate axis:
#   The RL/MLP controller was trained at 50 Hz; its action->moment gain
#   (moment_scale) is calibrated for a 50 Hz closed loop. Running it at another
#   command rate WITHOUT rescaling the gain changes the effective closed-loop
#   gain, so a fixed moment_scale=0.0055 is under-authority at
#   25-50 Hz and over-authority at 200 Hz -- a gain artifact masquerading as a
#   rate effect. Here moment_scale is set PER CELL to hold the closed-loop gain
#   constant:  moment_scale = MOMENT_C / eff_cmd_hz   (MOMENT_C=0.5 -> 0.01 @50Hz,
#   0.005 @100Hz, matching the sim's own note; the proven demos used 0.0055 @~90Hz
#   = 0.5/90). This isolates the REAL command-rate effect from the gain artifact.
#   Force a single fixed value instead with  MOMENT=<val>  (e.g. for an A/B check).
#
# Everything else matches the proven grid: --controller rl, the v12 CNN nav
# weights, seeds via --episodes.
#
# Grid (defaults): 6 speeds x 8 rates x 8 seeds = 384 real Isaac flights.
#   Est. ~9-24 h depending on the box; crashes early-terminate, so failing
#   cells are cheaper. All axes are env-overridable (see below).
#
# SMOKE TEST FIRST (one easy cell, ~2 min) before the full run:
#   SPEEDS=1.2 RATES=100 EPISODES=2 ISAAC_PY=<env_isaaclab py> bash scripts/hil_dense_grid.sh
#
# FULL RUN:
#   ISAAC_PY=<the Isaac Sim / IsaacLab interpreter> \
#     bash scripts/hil_dense_grid.sh
#
# Output: results/codesign_feedback/hil_dense.csv  (additive; append-only).
# Then render the figure:
#   <env_isaaclab py> scripts/hil_ablation_3panel.py \
#     --csv results/codesign_feedback/hil_dense.csv \
#     --out results/codesign_feedback/hil_ablation_3panel
# =====================================================================
set -u

ROOT="${XPURT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
# sweep_rate_demo.py imports hil/safety_layer.py (unconditional, line ~148). Some worktrees
# lack the hil/ dir; $XPURT_SIM_TREE (scripts/env.sh, scripts/env.local.sh) names one that has it.
if [ ! -f "$ROOT/hil/safety_layer.py" ] && [ -f "${XPURT_SIM_TREE:-}/hil/safety_layer.py" ]; then
  echo "[driver] $ROOT lacks hil/safety_layer.py -> using \$XPURT_SIM_TREE ($XPURT_SIM_TREE)" >&2
  ROOT="$XPURT_SIM_TREE"
fi
OUT="${HIL_OUTDIR:-$ROOT/results/codesign_feedback/hil_dense_grid}"
mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
PY="${ISAAC_PY:-python}"                          # conda env_isaaclab python (docs/Artifact/REPRODUCE.md)
W="${HIL_WEIGHTS:-$ROOT/sims/models/warehouse/nav_fused_v12_cnn.pt}"
CSV="${HIL_CSV:-$ROOT/results/codesign_feedback/hil_dense.csv}"

SIM_DT="${SIM_DT:-0.001}"                          # fixed physics dt (s); 0.001 -> 1000 Hz base
FLIGHT_S="${FLIGHT_S:-14}"                         # target flight seconds per cell (scales max_steps)
MOMENT="${MOMENT:-}"                               # empty = per-cell calibrated (MOMENT_C/eff);
                                                   # set a number to FORCE a fixed gain (A/B check)
MOMENT_C="${MOMENT_C:-0.5}"                         # closed-loop-gain constant: moment_scale=MOMENT_C/eff_hz
EPISODES="${EPISODES:-8}"                          # seeds per cell
SEED0="${SEED0:-1000}"
PER_CELL_TIMEOUT="${PER_CELL_TIMEOUT:-1800}"       # seconds; generous, high-rate cells step more

# Axes (space-separated, env-overridable). Rates chosen to bracket the scheme
# caps: ROS 81, XPU-RT greedy 125, XPU-RT feedback-sharded 204 Hz.
SPEEDS="${SPEEDS:-1.0 1.2 1.4 1.6 1.8 2.0}"
RATES="${RATES:-25 33 50 81 100 125 165 204}"

SUMMARY="$OUT/DENSE_SUMMARY.txt"; : > "$SUMMARY"
cd "$ROOT"
mkdir -p "$(dirname "$CSV")"

echo "grid: speeds=[$SPEEDS] rates=[$RATES] episodes=$EPISODES sim_dt=$SIM_DT flight_s=$FLIGHT_S" | tee -a "$SUMMARY"
echo "gain: ${MOMENT:+FIXED moment_scale=$MOMENT}${MOMENT:-calibrated moment_scale=$MOMENT_C/eff_hz (0.01@50Hz .. 0.0025@200Hz)}" | tee -a "$SUMMARY"
echo "csv:  $CSV" | tee -a "$SUMMARY"

for cruise in $SPEEDS; do
  for rate in $RATES; do
    # decimation so the ZOH command rate lands on `rate`; max_steps for ~FLIGHT_S seconds
    dec=$(awk -v r="$rate" -v dt="$SIM_DT" 'BEGIN{d=int(1.0/(r*dt)+0.5); if(d<1)d=1; print d}')
    steps=$(awk -v r="$rate" -v t="$FLIGHT_S" 'BEGIN{s=int(r*t+0.5); if(s<1)s=1; print s}')
    eff=$(awk -v dt="$SIM_DT" -v d="$dec" 'BEGIN{printf "%.1f", 1.0/(dt*d)}')
    # per-cell controller gain: calibrated MOMENT_C/eff unless MOMENT is forced fixed
    if [ -n "$MOMENT" ]; then ms="$MOMENT"; else
      ms=$(awk -v c="$MOMENT_C" -v e="$eff" 'BEGIN{printf "%.5f", c/e}'); fi
    tag="c${cruise}_r${rate}"
    echo "=== $(date +%H:%M:%S) START $tag  (dec=$dec eff~=${eff}Hz max_steps=$steps moment=$ms) ===" | tee -a "$SUMMARY"
    timeout "$PER_CELL_TIMEOUT" $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
        --weights "$W" --sim_dt "$SIM_DT" --decimation "$dec" --max_steps "$steps" \
        --moment_scale "$ms" --cruise_speed "$cruise" --sched_latency_ms 0 \
        --episodes "$EPISODES" --seed "$SEED0" --sweep-csv "$CSV" \
        > "$OUT/${tag}.log" 2>&1
    rc=$?
    res=$(grep -aE "\[SWEEP\]" "$OUT/${tag}.log" | tail -1 | sed -E 's/.*SUCCESS ([0-9]+\/[0-9]+).*/\1/')
    echo "  exit=$rc success=${res:-NA}" | tee -a "$SUMMARY"
  done
done
echo "=== $(date +%H:%M:%S) DENSE GRID DONE ($([ -f "$CSV" ] && wc -l < "$CSV" || echo 0) rows incl header) ===" | tee -a "$SUMMARY"

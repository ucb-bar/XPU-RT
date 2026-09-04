#!/usr/bin/env bash
# HIL-in-loop ablation grid: drone SPEED x command FREQUENCY x crash/success.
# Frequency is set by the schedule latency via the ZOH hold: at control_dt=10ms (100Hz base),
#   hold = ceil(latency/10ms) -> eff command rate 100/hold Hz.
#   lat 8 -> 1-step -> 100 Hz | 18 -> 2 -> 50 Hz | 28 -> 3 -> 33 Hz | 38 -> 4 -> 25 Hz
# 5 speeds x 4 frequencies x 6 seeds = 120 real Isaac flights. Rows -> hil_ablation.csv.
set -u
# All paths overridable by env. ROOT = the XPU-RT repo root (this script lives in $ROOT/scripts).
ROOT="${XPURT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${HIL_OUTDIR:-$ROOT/results/codesign_feedback/hil_grid}"; mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
PY="${ISAAC_PY:-python}"                         # conda env_isaaclab python (see docs/REPRODUCE.md)
W="${HIL_WEIGHTS:-$ROOT/sims/models/warehouse/nav_fused_v12_cnn.pt}"
CSV=$OUT/hil_ablation.csv
SUMMARY=$OUT/GRID_SUMMARY.txt; : > "$SUMMARY"
rm -f "$CSV"
cd "$ROOT"

for cruise in 1.0 1.2 1.4 1.6 1.8; do
  for lat in 8 18 28 38; do
    tag="c${cruise}_lat${lat}"
    echo "=== $(date +%H:%M:%S) START $tag ===" | tee -a "$SUMMARY"
    timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
        --weights "$W" --sim_dt 0.01 --decimation 1 --moment_scale 0.0055 \
        --cruise_speed "$cruise" --sched_latency_ms "$lat" \
        --episodes 6 --seed 1000 --sweep-csv "$CSV" > "$OUT/${tag}.log" 2>&1
    res=$(grep -aE "\[SWEEP\]" "$OUT/${tag}.log" | tail -1 | sed -E 's/.*SUCCESS ([0-9]+\/[0-9]+).*/\1/')
    echo "  exit=$? success=$res" | tee -a "$SUMMARY"
  done
done
echo "=== $(date +%H:%M:%S) GRID DONE ($(wc -l < "$CSV") rows incl header) ===" | tee -a "$SUMMARY"

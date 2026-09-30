#!/usr/bin/env bash
# APPEND more seeds to the committed envelope CSV (results/codesign_feedback/hil_ablation.csv).
# Identical conditions to scripts/hil_ablation_grid.sh (same gain 0.0055, same ZOH latencies,
# same speeds) so the new rows pool validly with the existing 120 — the ONLY differences are:
#   * NO `rm` of the CSV (append-only; sweep_rate_demo.py --sweep-csv appends, header if new)
#   * SEED0 and EPISODES are env-overridable (default SEED0=1006 EPISODES=6 -> seeds 1006..1011,
#     i.e. +6 seeds/cell across 5x4 cells = +120 flights, taking each cell 6 -> 12 seeds).
set -u
ROOT="${XPURT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${HIL_OUTDIR:-$ROOT/results/codesign_feedback/hil_grid_more}"; mkdir -p "$OUT/tmp"; export TMPDIR="$OUT/tmp"
PY="${ISAAC_PY:-python}"
W="${HIL_WEIGHTS:-$ROOT/sims/models/warehouse/nav_fused_v12_cnn.pt}"
CSV="${HIL_CSV:-$ROOT/results/codesign_feedback/hil_ablation.csv}"     # the committed figure source
SEED0="${SEED0:-1006}"; EPISODES="${EPISODES:-6}"
SUMMARY=$OUT/MORESEEDS_SUMMARY.txt; : > "$SUMMARY"
cd "$ROOT"
echo "append: seeds ${SEED0}..$((SEED0+EPISODES-1)) into $CSV" | tee -a "$SUMMARY"
for cruise in 1.0 1.2 1.4 1.6 1.8; do
  for lat in 8 18 28 38; do
    tag="c${cruise}_lat${lat}_s${SEED0}"
    echo "=== $(date +%H:%M:%S) START $tag ===" | tee -a "$SUMMARY"
    timeout 900 $PY sims/scripts/sweep_rate_demo.py --headless --controller rl \
        --weights "$W" --sim_dt 0.01 --decimation 1 --moment_scale 0.0055 \
        --cruise_speed "$cruise" --sched_latency_ms "$lat" \
        --episodes "$EPISODES" --seed "$SEED0" --sweep-csv "$CSV" > "$OUT/${tag}.log" 2>&1
    res=$(grep -aE "\[SWEEP\]" "$OUT/${tag}.log" | tail -1 | sed -E 's/.*SUCCESS ([0-9]+\/[0-9]+).*/\1/')
    echo "  exit=$? success=${res:-NA}" | tee -a "$SUMMARY"
  done
done
echo "=== $(date +%H:%M:%S) MORESEEDS DONE ($(wc -l < "$CSV") rows incl header) ===" | tee -a "$SUMMARY"

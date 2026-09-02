#!/usr/bin/env bash
# Launch / verify THE complex collidable warehouse course (gates + rack rows + tall-thin stacked
# props + patrolling people, ALL real colliders). One command — no flags to remember.
#
#   bash run_crowded_demo.sh [WEIGHTS.pt] [EPISODES]
#
# Defaults to the crowded-trained CNN+LSTM checkpoint. Records a chase-cam video and writes a
# metrics JSON (success = flew through all 4 gates without hitting a prop / rack / person / gate).
set -euo pipefail

REPO="/scratch/agustin/projects/DIMA"
PY="/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python"
cd "$REPO"

# crowded-course checkpoint (CNN+LSTM). Override by passing a path as $1 (e.g. the ViT+LSTM one).
WEIGHTS="${1:-$(ls -td train_out/fused_bc_warehouse_v11_crowded_cnn/*/ 2>/dev/null | head -1)best.pt}"
EPISODES="${2:-12}"
OUT="$REPO/train_out/fused_bc_warehouse/warehouse_crowded_demo.mp4"

echo "[run_crowded_demo] weights = $WEIGHTS"
echo "[run_crowded_demo] course  = collidable gates + dense tall-thin stacked props + people"

# --prop_density turns the tall-thin stacked-prop field ON (the eval env defaults it OFF for pure
# gate-following). Collidable gates are the default (no --visual_gates). This IS the crowded course.
"$PY" XPU-RT/sims/scripts/eval_fused_warehouse.py --headless \
  --weights "$WEIGHTS" \
  --prop_density 0.3 --obstacle_level 8 \
  --fixed_speed 1.3 --episodes "$EPISODES" \
  --save_video "$OUT" \
  --out "$REPO/train_out/fused_bc_warehouse/warehouse_crowded_demo.json"

echo "[run_crowded_demo] video  -> $OUT"
echo "[run_crowded_demo] metrics-> $REPO/train_out/fused_bc_warehouse/warehouse_crowded_demo.json"

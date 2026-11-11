#!/bin/bash
# usage: run_arm.sh <name> <latency_ms> <period_ms|auto> <seed> [extra args...]
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
NAME=$1; LAT=$2; PER=$3; SEED=$4; shift 4
EXTRA=""
if [ "$PER" != "auto" ]; then EXTRA="--issue-period-ms $PER"; fi
mkdir -p logs
python finegrain_eval.py --n 24 --init-rng $SEED --latency-ms $LAT $EXTRA \
    --out /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/runs/${NAME}_rng${SEED} \
    "$@" > logs/${NAME}_rng${SEED}.log 2>&1
echo "DONE ${NAME} rng${SEED}: $(grep 'SUCCESS RATE' logs/${NAME}_rng${SEED}.log)"

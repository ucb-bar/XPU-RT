#!/usr/bin/env bash
# Regenerate the raw analysis blocks that RESULTS_GOOGLE.txt is written from.
set -u
cd "$(dirname "$0")"
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh 2>/dev/null && conda activate octo_sim
OUT=RESULTS_raw_google.txt
{
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) from $(find runs -name summary.json | wc -l) summaries"
  echo "harness md5 $(md5sum ../finegrain_eval.py | cut -c1-12)"
  echo; echo "################ analyze_fine.py ################"; python analyze_fine.py
  echo; echo "################ compare_tasks.py ################"; python compare_tasks.py
  echo; echo "################ arm_timing.py ################"; python arm_timing.py
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < $OUT) lines)"

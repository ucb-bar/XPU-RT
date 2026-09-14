#!/bin/bash
# Phase 1: find, LOCALLY, episodes where the accelerated arms succeed and the
# CPU-only arm fails. The sweep's per-episode outcomes cannot be replayed here --
# it ran on AWS A10G, this box is a TITAN RTX, and float differences in the policy
# compound in closed loop (verified: identical timing, different trajectory).
# So the videos depict locally-rendered episodes of the same configuration.
# No video in this pass; rendering every tick is ~5x slower and we only need outcomes.
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
OUT=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/scan_local
mkdir -p "$OUT"
run () {  # key task seed arm lat cadence
  python finegrain_eval_videosnap.py --task "$2" --latency-ms "$5" --issue-period-ms "$6" \
     --init-rng "$3" --n 24 --save-video-every 0 --out "$OUT/$1_$4" > "$OUT/$1_$4.log" 2>&1
  echo "done $1 $4 : $(grep -h 'SUCCESS RATE' "$OUT/$1_$4.log" | tail -1)"
}
export -f run; export OUT
# 3 concurrent on the one local GPU
( run egg widowx_put_eggplant_in_basket 100 pipe110   117.7 117.6
  run egg widowx_put_eggplant_in_basket 100 serial283 283.4 283.4 ) &
( run egg widowx_put_eggplant_in_basket 100 pipe200   231.8 203.0
  run egg widowx_put_eggplant_in_basket 100 cpu685    684.8 684.8 ) &
( run spoon widowx_spoon_on_towel 91 pipe110   117.7 117.6
  run spoon widowx_spoon_on_towel 91 serial283 283.4 283.4
  run spoon widowx_spoon_on_towel 91 pipe200   231.8 203.0
  run spoon widowx_spoon_on_towel 91 cpu685    684.8 684.8 ) &
wait
echo SCAN_DONE

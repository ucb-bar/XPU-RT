#!/bin/bash
# Re-run a few episodes per arm with rendering on EVERY 40 ms tick so the
# zero-order-hold on stale data is visible per tick, not per 200 ms granule.
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
mkdir -p runs_video logs
rec () {  # name lat cadence nep
  python finegrain_eval.py --n "$4" --init-rng 0 --latency-ms "$2" --issue-period-ms "$3" \
    --save-video-every 1 \
    --out /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/runs_video/"$1" \
    > logs/video_"$1".log 2>&1
  echo "DONE video $1: $(grep 'SUCCESS RATE' logs/video_$1.log)"
}
rec "$@"

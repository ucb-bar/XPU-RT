#!/bin/bash
# Render a same-scene 4-arm ladder per environment: pipelined 110, pipelined 200,
# accelerated serial, CPU-only. Target outcome is SUCCESS for the three accelerated
# arms and FAILURE for the CPU-only arm.
#
# The harness is NOT run-to-run deterministic (three identical invocations of the
# same config gave False/True/False), so an episode's outcome cannot be selected in
# advance -- we retry until the run produces the target outcome and keep THAT run.
# Each kept video is internally consistent: frames and the age trace come from the
# same run. No video replays a specific AWS-swept episode; the arm success rates
# quoted in the titles are the AWS sweep's (n=240/arm).
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
OUT=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/videos_ladder
mkdir -p "$OUT"

try () {  # key task seed ep arm lat cadence target(True|False)
  local K=$1 T=$2 S=$3 EP=$4 ARM=$5 LAT=$6 PER=$7 WANT=$8
  for att in 1 2 3 4 5 6 7 8; do
    local d="$OUT/${K}_${ARM}"
    rm -rf "$d"
    python finegrain_eval_videosnap.py --task "$T" --latency-ms "$LAT" \
       --issue-period-ms "$PER" --init-rng "$S" --n 1 --ep-start "$EP" \
       --save-video-every 1 --out "$d" > "$OUT/${K}_${ARM}.log" 2>&1
    if ls "$d"/ep*_success_${WANT}.mp4 >/dev/null 2>&1; then
      echo "OK ${K} ${ARM}: got success=${WANT} on attempt ${att} -> $(ls $d/ep*.mp4)"
      return 0
    fi
  done
  echo "GAVEUP ${K} ${ARM}: wanted success=${WANT}, 8 attempts -> $(ls $d/ep*.mp4 2>/dev/null)"
}

( try egg widowx_put_eggplant_in_basket 100 0 pipe110   117.7 117.6 True
  try egg widowx_put_eggplant_in_basket 100 0 pipe200   231.8 203.0 True
  try egg widowx_put_eggplant_in_basket 100 0 serial283 283.4 283.4 True
  try egg widowx_put_eggplant_in_basket 100 0 cpu685    684.8 684.8 False ) &
( try spoon widowx_spoon_on_towel 91 21 pipe110   117.7 117.6 True
  try spoon widowx_spoon_on_towel 91 21 pipe200   231.8 203.0 True
  try spoon widowx_spoon_on_towel 91 21 serial283 283.4 283.4 True
  try spoon widowx_spoon_on_towel 91 21 cpu685    684.8 684.8 False ) &
wait
echo LADDER_DONE

#!/bin/bash
# Render the HW-accelerated SUCCESS vs CPU-only FAILURE contrast, per environment.
#
# Both arms are SERIAL schedules, so the only difference is whether the DSP and HTA
# are used: serial283 = 3-way CPU+DSP+HTA at 283.4 ms, cpu685 = CPU-only int8
# monolith at 684.8 ms. Same scene, same policy seed, same episode -- the only
# variable is the measured board latency.
#
# Episode 0 ONLY. The policy RNG advances across episodes (jax.random.split per
# inference), so a 1-episode run reproduces the sweep exactly only at episode 0.
#
# Uses finegrain_eval_videosnap.py, md5 cb302b43f647 -- byte-identical to the
# harness that produced the sweep numbers, and isolated from in-flight edits.
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
OUT=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/videos_contrast
mkdir -p "$OUT"

render () {  # task_key env_task seed arm lat cadence
  local k=$1 T=$2 S=$3 ARM=$4 LAT=$5 PER=$6
  local d="$OUT/${k}_${ARM}_rng${S}"
  echo "=== rendering $k $ARM (lat ${LAT} ms) seed $S ep 0 ==="
  python finegrain_eval_videosnap.py --task "$T" --latency-ms "$LAT" \
      --issue-period-ms "$PER" --init-rng "$S" --n 1 --ep-start 0 \
      --save-video-every 1 --out "$d" > "$OUT/${k}_${ARM}_rng${S}.log" 2>&1
  echo "   rc=$? : $(grep -h 'SUCCESS RATE' "$OUT/${k}_${ARM}_rng${S}.log" | tail -1)"
  ls "$d"/*.mp4 2>/dev/null
}

render egg   widowx_put_eggplant_in_basket 100 serial283 283.4 283.4
render egg   widowx_put_eggplant_in_basket 100 cpu685    684.8 684.8
render spoon widowx_spoon_on_towel          91 serial283 283.4 283.4
render spoon widowx_spoon_on_towel          91 cpu685    684.8 684.8
echo "VIDEOS_DONE"

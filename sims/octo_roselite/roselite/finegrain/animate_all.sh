#!/bin/bash
# Compose the forest-trail layout for every rendered ladder video:
# rollout on top, MEASURED QRB5165 lane gantt below with a sweeping playhead.
source /scratch2/dima/miniforge3/etc/profile.d/conda.sh
conda activate octo_sim
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain
V=videos_ladder
TR=/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs
OUT=videos_gantt; mkdir -p "$OUT"

go () {  # key env_label arm arm_label lat cadence trace sr ep
  local K=$1 EL=$2 ARM=$3 AL=$4 LAT=$5 PER=$6 TRACE=$7 SR=$8 EP=$9
  local mp4=$(ls $V/${K}_${ARM}/ep*.mp4 2>/dev/null | head -1)
  [ -z "$mp4" ] && { echo "SKIP ${K} ${ARM}: no mp4"; return; }
  local outcome=$(basename "$mp4" | sed 's/.*success_//; s/\.mp4//')
  python animate_rollout_gantt.py --video "$mp4" --run "$V/${K}_${ARM}" --ep "$EP" \
     --trace "$TR/$TRACE" --latency-ms "$LAT" --cadence-ms "$PER" \
     --arm-label "$AL" --env-label "$EL" --arm-sr "$SR" --success "$outcome" \
     --out "$OUT/${K}_${ARM}_${outcome}.mp4" 2>&1 | tail -1
}

go egg   "eggplant in basket" pipe110   "PIPELINED 110 ms cadence (CPU+DSP+HTA)" 117.7 117.6 pipe110x10_20260905-165246.log  63.3 0
go egg   "eggplant in basket" pipe200   "PIPELINED 200 ms cadence (CPU+DSP+HTA)" 231.8 203.0 pipe200_gated_20260905-164357.log 41.7 0
go egg   "eggplant in basket" serial283 "SERIAL 3-way (CPU+DSP+HTA)"             283.4 283.4 ungated_20260905-164735.log       17.1 0
go egg   "eggplant in basket" cpu685    "CPU-ONLY int8 monolith (no DSP/HTA)"    684.8 684.8 mono_20260905-130719.log           1.7 0
go spoon "spoon on towel"     pipe110   "PIPELINED 110 ms cadence (CPU+DSP+HTA)" 117.7 117.6 pipe110x10_20260905-165246.log  49.6 21
go spoon "spoon on towel"     pipe200   "PIPELINED 200 ms cadence (CPU+DSP+HTA)" 231.8 203.0 pipe200_gated_20260905-164357.log 27.9 21
go spoon "spoon on towel"     serial283 "SERIAL 3-way (CPU+DSP+HTA)" 283.4 283.4 ungated_20260905-164735.log 13.8 21
go spoon "spoon on towel"     cpu685    "CPU-ONLY int8 monolith (no DSP/HTA)"    684.8 684.8 mono_20260905-130719.log           0.8 21
echo ANIM_DONE

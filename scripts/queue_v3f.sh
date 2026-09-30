#!/usr/bin/env bash
# More XPU-RT attempts on the 1.2 m/s scenes where the baseline crashes after exactly two gates (seeds 1001 and
# 1018 from the two searches), latency replayed; the first completion makes the pair, then scene runs and renders.
#   nohup bash scripts/queue_v3f.sh > results/codesign_feedback/queue_v3f.log 2>&1 &
. "$(dirname "$0")/env.sh"
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces
say(){ echo "=== $(date +%H:%M:%S) $*"; }
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --cruise_speed 1.2 --episodes 1 --max_steps 1800 --keep_video"
gpu_room() { local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); n=$(pgrep -fc 'sweep_rate_dem[o]\.py|record_sensor_dem[o]\.py'); [ "$free" -ge 7000 ] && [ "$n" -lt 6 ]; }   # process count: two per simulator
D=$R/campaign_v2/display_v3s_c1.2f; mkdir -p $D/tmp; export TMPDIR=$D/tmp
fly() { local tag=$1; shift; until gpu_room; do sleep 60; done; rm -rf $D/${tag}_figdata; echo "=== $(date +%H:%M:%S) $tag"
  (cd "$SIM_TREE" && timeout 2400 $PY sims/scripts/record_sensor_demo.py $COMMON "$@" --gantt_schedule $WT/schedules/measured_gantt_xpu.json --save_video $D/$tag.mp4 --dump_figure_data $D/${tag}_figdata > $D/$tag.log 2>&1)
  grep -h "outcome=" $D/$tag.log | tail -n 1; }
S=""
for s in 1001 1018; do
  for k in 4 5 6 7 8 9; do
    x=$(fly xpu_s${s}_t${k} --ctrl_trace $T/xpu_a_cpsat_hard.csv --percep_latency_ms 56.8 --percep_hold_ms 0 --moment_scale 0.0055 --seed $((s + 100 * k)) --layout_seed $s --post_success_steps 100); echo "  xpu $s try $k: $x"
    if echo "$x" | grep -q "outcome=success"; then S=$s; K=$k; break 2; fi
  done
done
if [ -n "$S" ]; then
  src=$R/campaign_v2/display_v3s_c1.2; [ -d $src/ros_s${S}_figdata ] || src=$R/campaign_v2/display_v3s_c1.2b
  mv $D/xpu_s${S}_t${K}_figdata $D/xpu_s${S}_figdata; cp -r $src/ros_s${S}_figdata $D/ros_s${S}_figdata; echo "PAIR_SEED=$S"
  say "scene runs for seed $S at 1.2 m/s"; CELL=tall${S}s LAYOUT_SEED=$S CRUISE=1.2 bash scripts/scene_runs.sh
  say "render"; for P in 0 1; do PAPER=$P CELL=tall1008 XPU_DIR=$D/xpu_s${S}_figdata ROS_DIR=$D/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall${S}s DISPLAY_CRUISE=1.2 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'; done
else echo "no completion on the two-gate seeds"; fi
echo QUEUE_V3F_DONE

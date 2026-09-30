#!/usr/bin/env bash
# Every seed of the 36 Hz census flown as a one-episode display run, for all three arms.
#
# The four seeds the census named as separating were flown first; across those eight baseline
# flights the ROS 2 arms crashed after 3, 1, 1, 1 and 3, -, 1, 1 gates, and one of them completed.
# None crashed after exactly two, which is what panel A wants (verify_showdown_figure.py:506 takes
# one or two: the drone must enter the course and lose it before the third gate).
#
# Rather than draw seeds until one obliges -- which would be selecting the display flight on the
# baseline's outcome -- this flies the WHOLE twelve-seed population for every arm and records what
# each one did. The figure then draws a pair from a population that is on disk in full, and panel
# A's tally is that population.
#
#   scripts/display_all12_ac36.sh          env SEEDS ARMS MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; . "$WT/scripts/env.sh"
SEEDS="${SEEDS:-1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011}"
W="$WT/sims/models/warehouse/nav_fused_v12_cnn.pt"
LOG="$R/display_all12_ac36.log"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
gpu_room(){ local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
            [ "$free" -ge "${NEED_MB:-10000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }

# arm: <dir>:<prefix>:<trace>:<latency>:<extra flags>
ARMS="${ARMS:-pairs_ac36:xpu:xpu_p36free.csv:25.8:--post_success_steps 100|pairs_ac36:ros:ros_vanilla4x236.csv:32.3:|pairs_ns4:ros:ros_vanilla4x236ns4.csv:37.4:}"

fly(){  # fly <dir> <tag> <trace> <lat> <seed> <extra>
  local dir="$1" tag="$2" tr="$3" lat="$4" s="$5" extra="$6"
  local D="$R/campaign_free36/display/$dir"; mkdir -p "$D/tmp"
  [ -f "$D/${tag}_figdata/figure_data.npz" ] && { say "$dir/$tag: already on disk"; return; }
  exec 8>"$R/gpu_admit.lock"; flock 8; until gpu_room; do sleep 30; done
  ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & local settle=$!
  say "$dir/$tag"
  (cd "$SIM_TREE" && TMPDIR="$D/tmp" timeout 2400 "$ISAAC_PY" sims/scripts/record_sensor_demo.py \
     --headless --controller rl --weights "$W" --sim_dt 0.01 --decimation 1 --obstacle_level 8 \
     --prop_density 0.30 --cruise_speed 1.4 --episodes 1 --max_steps 1800 --keep_video \
     --moment_scale 0.0055 --percep_hold_ms 27.8 --ctrl_trace "$T/$tr" --percep_latency_ms "$lat" \
     --seed "$s" --layout_seed 1000 $extra \
     --gantt_schedule "$WT/schedules/measured_gantt_xpu.json" \
     --save_video "$D/$tag.mp4" --dump_figure_data "$D/${tag}_figdata" > "$D/$tag.log" 2>&1 8>&-)
  kill $settle 2>/dev/null; flock -u 8
  if grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" "$D/$tag.log"; then
    say "GPU fault in $dir/$tag: dump removed"; rm -rf "$D/${tag}_figdata"
  fi
  grep -hE "^\[ep00\]" "$D/$tag.log" 2>/dev/null | sed "s|^|  $dir/$tag |" | tee -a "$LOG"
}

for s in $SEEDS; do
  IFS='|' read -ra AS <<< "$ARMS"
  for a in "${AS[@]}"; do
    IFS=: read -r dir pre tr lat extra <<< "$a"
    fly "$dir" "${pre}_s${s}" "$tr" "$lat" "$s" "$extra"
  done
done
say "DISPLAY_ALL12_AC36_DONE"

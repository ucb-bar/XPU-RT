#!/usr/bin/env bash
# Progress for the close_drawer seed extension, n=10 -> n=30. Counts REDUCED
# cells only: a directory appears the moment a job starts, so counting those
# reports work that has not happened.
set -u; cd "$(dirname "$0")"; source dr30_hosts.sh
while true; do
  n=$(for h in "${HOSTS[@]}"; do
        $SSH ubuntu@$h 'cd /home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3 2>/dev/null && for x in drawer_*_rng1[12][0-9]/; do [ -f "$x/energy2.json" ] && basename "$x"; done' 2>/dev/null
      done | sort -u | wc -l)
  v=$(for h in "${HOSTS[@]}"; do $SSH ubuntu@$h 'ls /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_video/vid_drawer_p105/*.mp4 2>/dev/null | wc -l' 2>/dev/null; done | paste -sd+ | bc)
  r=$(for h in "${HOSTS[@]}"; do $SSH ubuntu@$h 'pgrep -cf "[t]race_eval2.py"' 2>/dev/null; done | paste -sd+ | bc)
  echo "$(date +%H:%M) drawer=$n/120 video_mp4=${v:-0} runners=${r:-0}"
  [ "$n" -ge 120 ] && [ "${v:-0}" -ge 1 ] && { echo DR30_COMPLETE; break; }
  [ "${r:-0}" -eq 0 ] && { echo "STALLED at $n"; break; }
  sleep 420
done

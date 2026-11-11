#!/usr/bin/env bash
# Energy trace, ALL four tasks x the CORRECTED arm set. Argument: "KEY:TASK:ARM".
# Supersedes job_torque.sh, which covered only eggplant+drawer (a leftover from when
# the energy analysis was conditioned on success and drawer was the only task where
# every arm succeeded) and used the OLD mislabelled pipelined latencies.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r KEY TASK ARM <<< "$1"
case "$TASK" in
  egg)    T=widowx_put_eggplant_in_basket ;;
  spoon)  T=widowx_spoon_on_towel ;;
  coke)   T=google_robot_pick_coke_can ;;
  drawer) T=google_robot_close_drawer ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in                      # MEASURED latency / cadence, ungated
  lat0)       LAT=0;     PER=auto   ;;
  pipe110fix) LAT=385.1; PER=111.4  ;;
  p105w300)   LAT=258.7; PER=124.8  ;;
  p130w275)   LAT=272.3; PER=130.1  ;;
  p150w300)   LAT=281.6; PER=150.4  ;;
  pipe200fix) LAT=260.5; PER=219.2  ;;
  serial283)  LAT=283.4; PER=283.4  ;;
  fp32_555)   LAT=555.0; PER=555.0  ;;
  cpu685)     LAT=684.8; PER=684.8  ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque2/${KEY}
mkdir -p "$(dirname "$OUT")"
python trace_eval.py --task "$T" --latency-ms "$LAT" $EXTRA --init-rng 100 --n 24 \
       --out "$OUT" > /home/ubuntu/simpler/logs/tq2_${KEY}.log 2>&1
echo "done ${KEY} rc=$? : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/tq2_${KEY}.log | tail -1)"

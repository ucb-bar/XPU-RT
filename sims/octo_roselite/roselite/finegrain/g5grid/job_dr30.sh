#!/usr/bin/env bash
# close_drawer seed extension, n=10 -> n=30, matching what eggplant and spoon
# already have. Same six curated arms, same measured latency/cadence table, same
# trace_eval2.py and reduce_torque3.py as job_t3seed.sh -- only the task differs,
# so the new cells drop into traces_torque3/ and every analysis picks them up.
#
# Argument: "TASK:ARM:SEED".
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r TASK ARM S <<< "$1"
case "$TASK" in
  drawer) T=google_robot_close_drawer ;;
  coke)   T=google_robot_pick_coke_can ;;
  egg)    T=widowx_put_eggplant_in_basket ;;
  spoon)  T=widowx_spoon_on_towel ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in
  lat0)       LAT=0;     PER=auto   ;;
  p105w300)   LAT=258.7; PER=124.8  ;;
  p150w300)   LAT=281.6; PER=150.4  ;;
  pipe200fix) LAT=260.5; PER=219.2  ;;
  serial283)  LAT=283.4; PER=283.4  ;;
  cpu685)     LAT=684.8; PER=684.8  ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"
KEY=${TASK}_${ARM}_rng${S}
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3/${KEY}
[ -f "$OUT/energy2.json" ] && { echo "skip ${KEY}"; exit 0; }
mkdir -p "$(dirname "$OUT")" /home/ubuntu/simpler/logs
python trace_eval2.py --task "$T" --latency-ms "$LAT" $EXTRA --init-rng "$S" --n 24 \
       --out "$OUT" > /home/ubuntu/simpler/logs/dr30_${KEY}.log 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "FAIL ${KEY} rc=$rc"; exit $rc; }
python reduce_torque3.py "$OUT" >> /home/ubuntu/simpler/logs/dr30_${KEY}.log 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "FAIL-REDUCE ${KEY} rc=$rc"; exit $rc; }
echo "done ${KEY}"

#!/usr/bin/env bash
# Seed extension for the curated widowx arms: identical configuration to
# job_torque3.sh -- same MEASURED latency/cadence per arm, same trace_eval2.py,
# same --n 24, same reduce_torque3.py -- with the seed as an argument instead of
# hard-coded to 100. Cell names match the existing convention exactly
# (<task>_<arm>_rng<seed>) so the new cells drop straight into traces_torque3/
# and every existing analysis picks them up with no change.
#
# Argument: "TASK:ARM:SEED".
# no `set -u`: env.sh sources conda, whose activate scripts reference unset vars.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r TASK ARM S <<< "$1"
case "$TASK" in
  egg)   T=widowx_put_eggplant_in_basket ;;
  spoon) T=widowx_spoon_on_towel ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in                      # MEASURED latency / cadence, ungated
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
[ -f "$OUT/energy2.json" ] && { echo "skip ${KEY} (already reduced)"; exit 0; }
mkdir -p "$(dirname "$OUT")" /home/ubuntu/simpler/logs
python trace_eval2.py --task "$T" --latency-ms "$LAT" $EXTRA --init-rng "$S" --n 24 \
       --out "$OUT" > /home/ubuntu/simpler/logs/t3s_${KEY}.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "FAIL ${KEY} rc=$rc"; exit $rc; }
python reduce_torque3.py "$OUT" >> /home/ubuntu/simpler/logs/t3s_${KEY}.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "FAIL-REDUCE ${KEY} rc=$rc"; exit $rc; }
echo "done ${KEY}"

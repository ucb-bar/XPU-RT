#!/usr/bin/env bash
# ENERGY REDO. Identical configuration to job_torque2.sh -- same four tasks, same
# nine MEASURED arm latencies/cadences, same --init-rng 100, same --n 24 -- but run
# through trace_eval2.py, which additionally logs the REAL PD drive torque and the
# contact force (ENERGY_AUDIT.md showed the shipped tau is only the gravity/Coriolis
# feed-forward). traces_torque3/ is therefore cell-for-cell comparable with
# traces_torque2/.
#
# The per-tick arrays are reduced to per-episode SCALARS on the worker and then
# deleted (13 MB/cell x 36 is not worth fetching); reduce_torque3.py keeps a compact
# per-actuation series.npz (~100 kB/cell) so the stall analysis can still be redone
# off the sweep.
#
# Argument: "KEY:TASK:ARM".
# no `set -u`: env.sh sources conda, whose activate scripts reference unset vars.
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
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque3/${KEY}
[ -f "$OUT/energy2.json" ] && { echo "skip ${KEY} (already reduced)"; exit 0; }
mkdir -p "$(dirname "$OUT")" /home/ubuntu/simpler/logs
python trace_eval2.py --task "$T" --latency-ms "$LAT" $EXTRA --init-rng 100 --n 24 \
       --out "$OUT" > /home/ubuntu/simpler/logs/tq3_${KEY}.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "FAIL ${KEY} rc=$rc"; exit $rc; }
python reduce_torque3.py "$OUT" >> /home/ubuntu/simpler/logs/tq3_${KEY}.log 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "FAIL-REDUCE ${KEY} rc=$rc"; exit $rc; }
echo "done ${KEY} : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/tq3_${KEY}.log | tail -1)"

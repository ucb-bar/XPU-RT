#!/usr/bin/env bash
# Torque-aware energy trace. Argument: "KEY:TASK:LAT:CADENCE".
# Logs joint FORCE alongside velocity/COM so stalls and collisions are visible --
# tau is large exactly when omega is ~0, which every motion-based metric scores as free.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r KEY TASK LAT PER <<< "$1"
case "$TASK" in
  egg)    T=widowx_put_eggplant_in_basket ;;
  drawer) T=google_robot_close_drawer ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/traces_torque/${KEY}
mkdir -p "$(dirname "$OUT")"
python trace_eval.py --task "$T" --latency-ms "$LAT" --issue-period-ms "$PER" \
       --init-rng 100 --n 24 --out "$OUT" > /home/ubuntu/simpler/logs/tq_${KEY}.log 2>&1
echo "done ${KEY} rc=$? : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/tq_${KEY}.log | tail -1)"

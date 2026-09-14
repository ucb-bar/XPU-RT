#!/usr/bin/env bash
# one sweep job: $1 = "ARM:SEED"  (ARM = A baseline 0ms ens ON, B = 283.4ms pipelined D=2)
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
ARM=${1%%:*}; S=${1##*:}
LOG=/home/ubuntu/simpler/logs/sweep_${ARM}_rng${S}.log
if [ "$ARM" = "A" ]; then
  python latency_eval.py --latency-ms 0 --ensemble stock --init-rng $S --n 24 --tag _G5 > $LOG 2>&1
else
  python latency_eval.py --latency-ms 283.4 --pipeline --init-rng $S --n 24 --tag _G5 > $LOG 2>&1
fi
echo "done $ARM seed $S rc=$? : $(grep -h \"SUCCESS RATE\" $LOG | tail -1)"

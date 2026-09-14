#!/usr/bin/env bash
# One sweep job. Argument is "TASK:ARM:SEED".
#
#   TASK   egg    widowx_put_eggplant_in_basket   (max_episode_steps 120)
#          spoon  widowx_spoon_on_towel           (max_episode_steps  60)
#          carrot widowx_carrot_on_plate          (max_episode_steps  60)
#          cube   widowx_stack_cube               (max_episode_steps  60)
#          All four are PutOnBridgeInSceneEnv -> control_freq = 5 -> 200 ms/step,
#          so D = ceil(latency_ms / 200 ms) is the same map on every task.
#
#   ARM    A  0 ms baseline, stock ensembler ON      D=0
#          B  283.4 ms MEASURED, pipelined           D=2
#          C  684.8 ms MEASURED, serial, no ensemble D=4
#
# The run directory latency_eval.py builds does NOT contain the task name, so the
# task goes into --tag to keep runs from different tasks from colliding.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
IFS=: read -r TASK ARM S <<< "$1"

case "$TASK" in
  egg)    T=widowx_put_eggplant_in_basket ;;
  spoon)  T=widowx_spoon_on_towel ;;
  carrot) T=widowx_carrot_on_plate ;;
  cube)   T=widowx_stack_cube ;;
  *) echo "bad task $TASK"; exit 2 ;;
esac
case "$ARM" in
  A) OPTS="--latency-ms 0 --ensemble stock" ;;
  B) OPTS="--latency-ms 283.4 --pipeline" ;;
  C) OPTS="--latency-ms 684.8" ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac

LOG=/home/ubuntu/simpler/logs/gw_${TASK}_${ARM}_rng${S}.log
mkdir -p /home/ubuntu/simpler/logs
python latency_eval.py --task "$T" $OPTS --init-rng "$S" --n 24 --tag "_GW_${TASK}" > "$LOG" 2>&1
rc=$?
echo "done ${TASK} ${ARM} seed ${S} rc=${rc} : $(grep -h 'SUCCESS RATE' "$LOG" | tail -1)"

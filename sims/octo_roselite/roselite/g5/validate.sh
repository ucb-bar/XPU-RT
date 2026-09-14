#!/usr/bin/env bash
# Port validation: widowx_spoon_on_towel, octo-small-1.0, corrected unnormalization.
# Published SIMPLER figure 47.2%; local reference harness 33/72 = 45.8% (seeds 0/2/4: 11,10,12).
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval
S=$1
exec python run_eval.py --task widowx_spoon_on_towel --ckpt hf://rail-berkeley/octo-small \
     --init-rng $S --n 24 --save-video-every 0 --tag _G5VAL

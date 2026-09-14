#!/usr/bin/env bash
# Pull the sweep summaries off the g5 into roselite/g5/runs/.
set -e
G5=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/g5
mkdir -p "$G5/runs"
rsync -a --include='*/' --include='summary.json' --exclude='*' \
  -e "ssh -i $HOME/.ssh/firesim.pem -o StrictHostKeyChecking=no" \
  ubuntu@3.93.66.216:/home/ubuntu/simpler/sim_eval/roselite/runs/ "$G5/runs/"
find "$G5/runs" -name summary.json | wc -l

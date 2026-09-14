#!/usr/bin/env bash
# Record a pool of episodes per highlighted case, with per-episode video.
#
# Output goes to runs_video/ (NOT runs/) on purpose: analyze.py globs
# runs/lat*_from_arrival_ens-none_rng*, so writing these here would silently
# fold extra episodes into the published RESULTS.txt aggregation.
set -euo pipefail
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
R=/scratch2/dima/misc_sw/octo_work/sim_eval/roselite
V=$R/runs_video
cd $R

run () { echo "### $1"; python latency_eval.py --out "$V/$1" --save-video-every 1 "${@:2}"; }

run base_0ms_ens-stock  --latency-ms 0     --ensemble stock --init-rng 0 --n 12
run lat283_pipelined    --latency-ms 283.4 --pipeline       --init-rng 0 --n 12
run lat283_serial       --latency-ms 283.4                  --init-rng 0 --n 12
run lat555_serial       --latency-ms 555                    --init-rng 0 --n 12
run lat684_pipelined    --latency-ms 684.8 --pipeline       --init-rng 0 --n 6
run lat684_serial       --latency-ms 684.8                  --init-rng 0 --n 12
run lat555_pipelined    --latency-ms 555   --pipeline       --init-rng 0 --n 8
echo "### ALL DONE"

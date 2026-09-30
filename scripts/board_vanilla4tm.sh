#!/usr/bin/env bash
# The vanilla graph with control on its own 100 Hz timer in its own unpinned process (held goal), 45 and 90 Hz, three replicates; pulled and summarised.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
for rep in 1 2 3; do say "vanilla4tm r$rep"; RATES="45 90" scripts/ros_traced_matrix.sh vanilla4tm $rep 2>&1 | grep -E "^===|error|fault"; done
.venv/bin/python scripts/pull_ros_traced.py $(for hz in 45 90; do for r in 1 2 3; do echo ${hz}_vanilla4tm_r$r; done; done) 2>&1 | cut -c1-200
say "VANILLA4TM_DONE"

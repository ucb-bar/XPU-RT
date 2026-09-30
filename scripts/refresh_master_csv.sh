#!/usr/bin/env bash
# Rebuild the master runs table every half hour while the campaigns run.
. "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/.."
while true; do "$ISAAC_PY" scripts/build_master_csv.py > results/codesign_feedback/master_refresh.log 2>&1; sleep 1800; done

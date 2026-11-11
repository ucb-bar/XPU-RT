#!/usr/bin/env bash
# Run the two google tasks in parallel (one process each); sequential within a task.
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
N=${N:-24} TASKS=google_robot_close_drawer bash parity/run_parity.sh > parity/drawer.out 2>&1 &
N=${N:-24} TASKS=google_robot_pick_coke_can  bash parity/run_parity.sh > parity/coke.out   2>&1 &
wait
echo PARITY_DONE

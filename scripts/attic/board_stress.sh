#!/usr/bin/env bash
# Combined load on both runtimes: two 45 Hz cameras AND the heavier stack (ffn_block 10 Hz, dronet 30 Hz).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "ROS: two cameras + heavier stack"; for rep in 1 2; do RATES="45" scripts/ros_traced_matrix.sh x2rp3 $rep 2>&1 | grep -E "ROS_TRACED|error"; done
say "XPU-RT: two cameras + heavier stack, ffn_block on IME"
CORE_KINDS="rvv,ime,rvv_c1" BACKENDS="rvv_x60,ime_x60,rvv_x60" scripts/run_xpurt_long.sh schedules/cam2_45_alt1_rich_ime.json cam2rich 2 2>&1 | grep -E "^===|trace rows|re-pulled|FATAL|Error"
say "STRESS_DONE"

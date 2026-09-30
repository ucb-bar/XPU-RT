#!/usr/bin/env bash
# The out-of-the-box ROS 2 graph (one process per node, unpinned, serial kernels, control in the
# goal callback) across camera rates, with and without the heavier stack, three replicates each;
# then the two disclosed variants at 45 Hz (QoS depth 1; MultiThreadedExecutor in every process).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
for rep in 1 2 3; do for arm in vanilla rvanilla; do say "$arm r$rep"; RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "^===|error|fault"; done; done
say "vanilla QoS 1"; RATES="45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla 1 2>&1 | grep -E "^===|error"
say "VANILLA_DONE"

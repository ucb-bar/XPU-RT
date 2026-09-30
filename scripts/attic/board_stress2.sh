#!/usr/bin/env bash
# The other two ROS 2 deployments under the combined load (two cameras + heavier stack).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
for rep in 1 2; do for arm in x2rspin x2rmulti; do say "ROS $arm r$rep"; RATES="45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "ROS_TRACED|error|not linked"; done; done
say "STRESS2_DONE"

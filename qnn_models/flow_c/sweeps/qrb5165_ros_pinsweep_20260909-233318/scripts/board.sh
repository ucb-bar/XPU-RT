#!/usr/bin/env bash
# Board discipline for the ROS pinning sweep.
#
# Every board interaction goes through `board_run`: a `timeout -s KILL` around
# an `ssh -n` whose remote command is wrapped in `flock -w 900
# /tmp/qnn_board.lock -c`, because this board is shared with other tenants.
# The lock wait is measured and printed on stderr for every call so a slow run
# can be attributed to contention rather than to the workload.
#
# If the board wedges: on the HOST, `/opt/relay.sh off; sleep 5; /opt/relay.sh
# on`, then wait ~60 s for it to come back.
set -eo pipefail

BOARD="${BOARD:-root@qrb5165}"
LOCK="${LOCK:-/tmp/qnn_board.lock}"
LOCK_WAIT="${LOCK_WAIT:-900}"
SSH_OPTS=(-o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=15
          -o ServerAliveCountMax=8)

# board_run <timeout_s> <remote command string>
board_run() {
    local tmo="$1"; shift
    local cmd="$1"
    local t0 t1
    t0=$(date +%s.%N)
    timeout -s KILL "$tmo" ssh "${SSH_OPTS[@]}" -n "$BOARD" \
        "flock -w $LOCK_WAIT $LOCK -c '$cmd'"
    local rc=$?
    t1=$(date +%s.%N)
    echo "[board] rc=$rc wall=$(echo "$t1 - $t0" | bc)s" >&2
    return $rc
}

# Save the governor once, restore it on exit -- the board is shared, and the
# cost model this sweep is scored against was captured at `performance`.
gov_save() {
    board_run 60 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > /tmp/rospin_prev_governor 2>/dev/null; cat /tmp/rospin_prev_governor'
}
gov_perf() {
    board_run 60 'for c in $(seq 0 7); do echo performance > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null; done; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
}
gov_restore() {
    board_run 60 'g=$(cat /tmp/rospin_prev_governor 2>/dev/null || echo schedutil); for c in $(seq 0 7); do echo $g > /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null; done; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
}

# shellcheck shell=bash
# Exclusive use of the SpaceMiT K3 for timing runs. Source this file, then call
# k3_lock_acquire before the first board run and k3_lock_release when done
# (k3_lock_acquire installs an EXIT trap that releases both locks).
#
# Two locks, because other users reach the board from other hosts:
#   host  : flock on $K3_BOARD_LOCK (default /tmp/k3_board.lock), the same file
#           ModelBlaster's MODELBLASTER_BOARD_LOCK / K3_BOARD_LOCK convention uses;
#   board : a held ssh session owning flock /tmp/k3_board.lock on the K3, plus a
#           /tmp/K3_IN_USE notice saying who holds it and since when.
#
# Env: MODELBLASTER_K3_HOST (ssh alias, default k3), K3_BOARD_LOCK, K3_LOCK_NOTE.

K3_HOST="${MODELBLASTER_K3_HOST:-k3}"
K3_BOARD_LOCK="${K3_BOARD_LOCK:-/tmp/k3_board.lock}"
_K3_HOLDER_PID=""

k3_lock_acquire() {
    local note="${K3_LOCK_NOTE:-$(id -un)@$(hostname -s): XPU-RT K3 walker-feedback runs}"
    exec 8>"${K3_BOARD_LOCK}"
    if ! flock -n 8; then
        echo "host lock ${K3_BOARD_LOCK} is held by another process; not running" >&2
        return 3
    fi
    local log; log="$(mktemp)"
    ssh -o BatchMode=yes "${K3_HOST}" "exec 9>/tmp/k3_board.lock; flock -n 9 || { echo BOARD_LOCK_BUSY; exit 3; };
        echo \"IN USE: ${note} since \$(date). Lock: flock /tmp/k3_board.lock\" > /tmp/K3_IN_USE;
        echo BOARD_LOCK_OK; while sleep 30; do :; done" >"${log}" 2>&1 &
    _K3_HOLDER_PID=$!
    trap k3_lock_release EXIT
    local i
    for i in $(seq 1 30); do grep -q BOARD_LOCK "${log}" && break; sleep 1; done
    if ! grep -q BOARD_LOCK_OK "${log}"; then
        echo "board lock on ${K3_HOST} not acquired: $(cat "${log}")" >&2
        rm -f "${log}"
        return 3
    fi
    rm -f "${log}"
    echo "[$(date +%T)] host and board locks held"
}

k3_lock_release() {
    [[ -n "${_K3_HOLDER_PID}" ]] && kill "${_K3_HOLDER_PID}" 2>/dev/null
    # Killing the local ssh does not always end the remote shell, so end it
    # explicitly; its inherited lock fd closes with it.
    ssh -o BatchMode=yes "${K3_HOST}" \
        'pkill -u "$(id -un)" -f "exec 9>/tmp/k3_board.lock"; rm -f /tmp/K3_IN_USE' 2>/dev/null
    _K3_HOLDER_PID=""
    flock -u 8 2>/dev/null
    echo "[$(date +%T)] locks released"
}

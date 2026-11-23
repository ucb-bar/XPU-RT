#!/usr/bin/env bash
# Every contact with the K3 that is not the build or the run itself goes through this file, so logins stay
# few and each check is one ssh call. Nothing on the board is changed except our own lock file.
#
#   board.sh acquire "<note>"   check the board is free and take our lock, in ONE login. Free means:
#                               no /tmp/K3_IN_USE notice of someone else, the shared flock
#                               /tmp/k3_board.lock not held, no lock file of someone else, no other
#                               harness/walker/bench job, nobody logged in, 1-min load <= K3TL_MAX_LOAD.
#                               If busy: retry every 3 min for up to K3TL_WAIT_S, then give up (exit 5).
#   board.sh release            remove our lock (only if it is ours)
#   board.sh show               print the same state, change nothing
#
# Env: K3TL_HOST      ssh alias of the board (default k3w)
#      K3TL_OWNER     name written into the lock (default: "<login name>, k3_timing_loop"); give each loop
#                     that runs at the same time its own, so two loops never share a lock
#      K3TL_LOCK_FILE lock file on the board (default ~/mb_k3rd/BOARD_LOCK; line: owner | date | epoch | note)
#      K3TL_MAX_LOAD  (default 4.5), K3TL_WAIT_S (default 1800)
# An ssh failure is never read as "no lock": the remote script must print ACQUIRED or BUSY.
set -u
HOST="${K3TL_HOST:-k3w}"
OWNER="${K3TL_OWNER:-$(id -un), k3_timing_loop}"
LOCK="${K3TL_LOCK_FILE:-~/mb_k3rd/BOARD_LOCK}"
MAX_LOAD="${K3TL_MAX_LOAD:-4.5}"
WAIT_S="${K3TL_WAIT_S:-1800}"
SSHO=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
cmd="${1:-show}"; note="${2:-k3_timing_loop run}"

remote() {   # remote <mode>: state report; mode=acquire also writes the lock when free
  { printf 'OWNER=%q\nNOTE=%q\nMAXL=%q\nMODE=%q\nL=%s\n' "$OWNER" "$note" "$MAX_LOAD" "$1" "$LOCK"
    cat <<'EOS'
busy=""
echo "date: $(date -u '+%F %T UTC')"
echo "kernel: $(uname -rv)"
echo "rtprio: $(ulimit -r)"
echo "home: $HOME"
nwho="$(who | wc -l)"
echo "who: $nwho session(s)"; who | sed 's/^/  /'
load="$(cut -d' ' -f1 /proc/loadavg)"
echo "load: $(cut -d' ' -f1-3 /proc/loadavg) (limit $MAXL)"
if [ -e /tmp/K3_IN_USE ]; then
  c="$(cat /tmp/K3_IN_USE)"; echo "K3_IN_USE: $c"
  case "$c" in *"$OWNER"*) : ;; *) busy="$busy [/tmp/K3_IN_USE: $c]" ;; esac
else
  echo "K3_IN_USE: none"
fi
if ( exec 7>>/tmp/k3_board.lock; flock -n 7 ); then
  echo "flock /tmp/k3_board.lock: free"
else
  echo "flock /tmp/k3_board.lock: HELD"; busy="$busy [flock /tmp/k3_board.lock is held]"
fi
cur="$(cat "$L" 2>/dev/null)"
echo "BOARD_LOCK: ${cur:-none}"
if [ -n "$cur" ]; then case "$cur" in "$OWNER"*) : ;; *) busy="$busy [BOARD_LOCK: $cur]" ;; esac; fi
jobs="$(ps -eo user:12,pid,etime,args | grep -E '_harness|xpurt|walker|_bench|mb_k3|/mb_|chrt -f|k3_board' | grep -v -E 'grep|ps -eo' || true)"
echo "jobs: $( [ -n "$jobs" ] && echo "$jobs" | wc -l || echo 0)"; [ -n "$jobs" ] && echo "$jobs" | sed 's/^/  /'
[ -n "$jobs" ] && busy="$busy [a board job is running]"
[ "$nwho" -gt 0 ] && busy="$busy [someone is logged in]"
awk -v l="$load" -v m="$MAXL" 'BEGIN{exit !(l > m)}' && busy="$busy [load $load > $MAXL]"
if [ "$MODE" != acquire ]; then
  echo "STATE: ${busy:-free}"
elif [ -z "$busy" ]; then
  mkdir -p "$(dirname "$L")" && echo "$OWNER | $(date -u '+%F %T UTC') | epoch=$(date +%s) | $NOTE" > "$L" && echo "ACQUIRED: $(cat "$L")"
else
  echo "BUSY:$busy"
fi
EOS
  } | ssh "${SSHO[@]}" "$HOST" bash -s 2>&1
}

remote_release() {   # remove our lock, only if it is ours
  { printf 'OWNER=%q\nL=%s\n' "$OWNER" "$LOCK"
    cat <<'EOS'
cur="$(cat "$L" 2>/dev/null)"
if [ -z "$cur" ]; then echo "RELEASED (no lock was present)"
elif [ "${cur#"$OWNER"}" != "$cur" ]; then rm -f "$L" && echo "RELEASED"
else echo "NOT OURS, left alone: $cur"
fi
EOS
  } | ssh "${SSHO[@]}" "$HOST" bash -s 2>&1
}

case "$cmd" in
  show) remote show ;;
  acquire)
    deadline=$(( $(date +%s) + WAIT_S ))
    while :; do
      out="$(remote acquire)"
      echo "--- $(date '+%F %T') attempt on $HOST"; echo "$out"
      grep -q '^ACQUIRED:' <<<"$out" && exit 0
      (( $(date +%s) >= deadline )) && { echo "GAVE UP after ${WAIT_S}s: board not free (or unreachable)"; exit 5; }
      if grep -q '^BUSY:' <<<"$out"; then sleep 180; else echo "(no answer from the board; retrying)"; sleep 30; fi
    done ;;
  release)
    for i in 1 2 3 4 5 6; do
      out="$(remote_release)"
      echo "$(date '+%F %T') $out"
      grep -q -e '^RELEASED' -e '^NOT OURS' <<<"$out" && exit 0
      sleep 20
    done
    echo "RELEASE FAILED: board unreachable; our lock may still be there"; exit 6 ;;
  *) echo "usage: board.sh acquire <note> | release | show" >&2; exit 2 ;;
esac

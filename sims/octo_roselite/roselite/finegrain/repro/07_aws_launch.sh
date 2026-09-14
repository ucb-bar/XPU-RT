#!/usr/bin/env bash
# usage: 07_aws_launch.sh [push|launch|verify|all] [w0 w1 ...]
#          all      = push, then launch, then verify   (the normal path)
#          push     = ship the harness (g5fine/push.sh) and check md5s match
#          launch   = g5fine/relaunch.sh, one worker at a time
#          verify   = count runners on every worker, right now
#
# Thin, verifying wrapper over the two scripts the sweep actually used:
#   g5fine/push.sh      rsync finegrain_eval.py + job.sh + octo15_inference.py
#   g5fine/relaunch.sh  scp joblist_w$w.txt, detach `xargs -P3`, CHECK the count
# Nothing here reimplements them; it adds the pre/post checks that were done by
# hand and the one that was skipped and cost a sweep.
#
# THE RULE THAT MATTERS:  NEVER TRUST THE EXIT CODE OF THE SSH THAT DETACHES THE
# REMOTE xargs.  It can hang past its own timeout while the launch succeeded, and
# it can return 0 while nothing started.  The only evidence is the process count:
#     pgrep -f 'finegrain_ev[a]l' | wc -l   ==  3
# Trusting the exit code once produced two workers running SIX runners each,
# writing into the same run dirs.  relaunch.sh already refuses to launch onto a
# box that reports a non-zero count; this wrapper re-checks afterwards too.
#
# EVERY pgrep/pkill PATTERN IS BRACKETED ('finegrain_ev[a]l') so the pattern can
# never match the command line that is running it.  Neither the xargs parent nor
# the `bash .../job.sh` wrappers contain the string, so the count is exactly the
# number of python runners.
#
# PRECONDITIONS: instances running, g5fine/hosts.sh refreshed
# (repro/05_aws_instances.sh hosts --write), joblists built (repro/06_...).
#
# Time: push ~1 min/box; launch ~90 s/box (it waits for the count); verify ~10 s.
# Produces: runs on the workers under
#   /home/ubuntu/simpler/sim_eval/roselite/finegrain/runs/<task>_<arm>_rng<seed>/
# plus /home/ubuntu/simpler/logs/fine_sweep.log per box.
# It worked if `verify` shows exactly 3 for every worker with a non-empty joblist.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
CMD="${1:-all}"; shift || true
[ "$CMD" = -h ] || [ "$CMD" = --help ] && { sed -n '2,34p' "$0"; exit 0; }
P="${P:-3}"

[ -f "$G5FINE/hosts.sh" ] || { echo "no $G5FINE/hosts.sh -- run 05_aws_instances.sh hosts --write"; exit 1; }
# shellcheck disable=SC1091
source "$G5FINE/hosts.sh"
WANT=("$@"); [ "${#WANT[@]}" -eq 0 ] && WANT=(w0 w1 w2 w3 w4 w5)

age=$(( $(date +%s) - $(stat -c %Y "$G5FINE/hosts.sh") ))
[ "$age" -gt 86400 ] && echo "WARNING: hosts.sh is $((age/3600)) h old.  Public IPs change on every
         stop/start -- re-run 05_aws_instances.sh hosts --write before pushing."

count_runners() {  # $1 = ip
  timeout 40 $SSH -n ubuntu@"$1" "pgrep -f 'finegrain_ev[a]l' | wc -l" 2>/dev/null | tr -cd '0-9'
}

do_verify() {
  local bad=0
  echo "=== runner counts (want $P where a joblist is non-empty) ==="
  for w in "${WANT[@]}"; do
    h=${HOSTS[$w]:-}
    [ -z "$h" ] && { echo "  $w: no IP in hosts.sh"; bad=1; continue; }
    jl="$G5FINE/joblist_$w.txt"; jn=0; [ -f "$jl" ] && jn=$(wc -l < "$jl")
    n=$(count_runners "$h")
    exp=$P; [ "$jn" -eq 0 ] && exp=0
    flag=""; [ "${n:-x}" = "$exp" ] || flag="   <-- EXPECTED $exp, CHECK THIS BOX"
    [ -n "$flag" ] && bad=1
    printf '  %-3s %-15s jobs=%-3d runners=%-3s%s\n' "$w" "$h" "$jn" "${n:-?}" "$flag"
    # remaining work, cheap and honest: lines already reported done
    d=$(timeout 40 $SSH -n ubuntu@"$h" \
        "grep -c '^done ' /home/ubuntu/simpler/logs/fine_sweep.log 2>/dev/null" 2>/dev/null | tr -cd '0-9')
    a=$(timeout 40 $SSH -n ubuntu@"$h" \
        "grep -c ALLDONE /home/ubuntu/simpler/logs/fine_sweep.log 2>/dev/null" 2>/dev/null | tr -cd '0-9')
    printf '        completed=%s/%s  ALLDONE=%s\n' "${d:-?}" "$jn" "${a:-0}"
  done
  return $bad
}

case "$CMD" in
  push)   bash "$G5FINE/push.sh"
          echo "--- md5 must be identical on every box AND equal to the local one ---" ;;
  launch) WORKERS="${WANT[*]}" P="$P" bash "$G5FINE/relaunch.sh"
          echo "--- relaunch.sh already verified the count; re-verifying ---"; do_verify ;;
  verify) do_verify ;;
  all)    bash "$G5FINE/push.sh"
          WORKERS="${WANT[*]}" P="$P" bash "$G5FINE/relaunch.sh"
          do_verify ;;
  *) sed -n '2,34p' "$0"; exit 2 ;;
esac

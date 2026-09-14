#!/usr/bin/env bash
# usage: 05_aws_instances.sh status
#        05_aws_instances.sh start   [w0 w1 ...]   # ONE AT A TIME, then refresh hosts
#        05_aws_instances.sh stop    [w0 w1 ...]   # STOP.  never terminate.
#        05_aws_instances.sh hosts   [--write]     # re-read public IPs
#
# The six g5.xlarge sweep workers.  Credentials live on the MANAGER
# (ubuntu@3.88.218.39, ~/.aws), NOT on this box, so every aws call is issued
# there over ssh -- see awsq() in common.sh.  VERIFIED working 2026-09-06.
#
# WHY ONE AT A TIME.  A single six-instance start-instances call FAILS with
# InsufficientInstanceCapacity: g5.xlarge capacity in these AZs is thin and the
# API is all-or-nothing per call.  Starting them individually gets all six.
# This script therefore never batches, and reports each one's outcome.
#
# WHY hosts MUST BE REFRESHED.  Public IPs CHANGE ON EVERY STOP/START.  Every IP
# in g5fine/hosts.sh (and in g5wide/WORKERS.md, SEED_ALLOCATION.md) is stale the
# moment the boxes are stopped.  Pushing or fetching against a stale map either
# times out or -- worse -- reaches somebody else's instance.
#
# NEVER TERMINATE.  Terminating deletes the root volumes, and with them the
# validated environment (Miniforge3, setuptools==80.10.2, the pinned repos, the
# passed per-worker validation gate).  Rebuild cost is ~40 min/box plus a fresh
# gate run; restart from `stopped` is ~2 min.  Cost of keeping them stopped:
# 6 x 200 GB gp3 = 1.2 TB at ~$0.08/GB-month = ~$96/month.  That overtakes the
# entire sweep's compute ($11.08 for 4,752 episodes) in under four days idle.
# Prune deliberately, per g5wide/WORKERS.md, not as a side effect of this script.
#
# Time: status/hosts ~10 s; start ~30-60 s per box plus ~2 min to boot; stop ~1 min.
# Produces: on `hosts --write`, a rewritten g5fine/hosts.sh (previous copy kept
# as hosts.sh.bak-<timestamp>).  Everything else only prints.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
CMD="${1:-status}"; shift || true
[ "$CMD" = -h ] || [ "$CMD" = --help ] && { sed -n '2,32p' "$0"; exit 0; }

# name -> id, honouring an optional worker-name filter on the command line
ids_for() {
  local want=("$@") i out=()
  for i in "${!WORKER_NAMES[@]}"; do
    if [ "${#want[@]}" -eq 0 ]; then out+=("${WORKER_IDS[$i]}"); else
      local w; for w in "${want[@]}"; do
        [ "$w" = "${WORKER_NAMES[$i]}" ] && out+=("${WORKER_IDS[$i]}")
      done
    fi
  done
  printf '%s\n' "${out[@]}"
}
name_of() { local i; for i in "${!WORKER_IDS[@]}"; do
    [ "${WORKER_IDS[$i]}" = "$1" ] && { echo "${WORKER_NAMES[$i]}"; return; }; done; echo "?"; }

describe() {
  awsq "ec2 describe-instances --instance-ids ${WORKER_IDS[*]} \
        --query 'Reservations[].Instances[].[InstanceId,State.Name,Placement.AvailabilityZone,PublicIpAddress]' \
        --output text"
}

case "$CMD" in
  status)
    echo "worker  instance             state       az            public ip"
    describe | sort | while read -r id st az ip; do
      printf '%-7s %-20s %-11s %-13s %s\n' "$(name_of "$id")" "$id" "$st" "$az" "$ip"
    done
    ;;

  start)
    # ONE AT A TIME.  Never `--instance-ids id1 id2 ...` -- that call fails whole.
    for id in $(ids_for "$@"); do
      w=$(name_of "$id")
      out=$(awsq "ec2 start-instances --instance-ids $id --output text" 2>&1)
      if [ $? -eq 0 ]; then echo "$w $id: start requested"; else
        echo "$w $id: START FAILED -- $out"
        echo "     InsufficientInstanceCapacity here means the AZ has no g5.xlarge"
        echo "     right now.  Retry this ONE id in a few minutes; do not batch."
      fi
      sleep 5
    done
    echo "--- waiting for running state ---"
    for _ in $(seq 1 20); do
      pend=$(describe | awk '$2!="running"' | wc -l)
      [ "$pend" -eq 0 ] && break
      sleep 15
    done
    "$0" status
    echo
    echo "NOW REFRESH THE IP MAP:  $0 hosts --write"
    ;;

  stop)
    for id in $(ids_for "$@"); do
      w=$(name_of "$id")
      awsq "ec2 stop-instances --instance-ids $id --output text" >/dev/null \
        && echo "$w $id: stop requested" || echo "$w $id: STOP FAILED"
      sleep 3
    done
    sleep 20; "$0" status
    echo "Instances are STOPPED, not terminated.  Root volumes (and the validated"
    echo "environment on them) are intact and still cost ~\$96/month in total."
    ;;

  hosts)
    tmp=$(mktemp)
    {
      echo "# Worker IP map. PUBLIC IPs CHANGE ON EVERY STOP/START -- regenerate with"
      echo "#   repro/05_aws_instances.sh hosts --write"
      echo "# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "declare -A HOSTS=("
      describe | sort | while read -r id st az ip; do
        w=$(name_of "$id")
        [ "$ip" = "None" ] || [ -z "$ip" ] && \
          echo "  # [$w]=?  $id  $az  -- $st, NO PUBLIC IP" && continue
        printf '  [%s]=%-16s # %s  %s\n' "$w" "$ip" "$id" "$az"
      done
      echo ")"
      echo 'KEY=$HOME/.ssh/firesim.pem'
      echo 'SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"'
    } > "$tmp"
    cat "$tmp"
    if [ "${1:-}" = "--write" ]; then
      [ -f "$G5FINE/hosts.sh" ] && cp -p "$G5FINE/hosts.sh" "$G5FINE/hosts.sh.bak-$(date +%Y%m%d-%H%M%S)"
      cp "$tmp" "$G5FINE/hosts.sh"
      echo; echo "wrote $G5FINE/hosts.sh (previous copy kept as hosts.sh.bak-*)"
    else
      echo; echo "(dry run -- pass --write to install this as $G5FINE/hosts.sh)"
    fi
    rm -f "$tmp"
    ;;

  *) sed -n '2,32p' "$0"; exit 2 ;;
esac

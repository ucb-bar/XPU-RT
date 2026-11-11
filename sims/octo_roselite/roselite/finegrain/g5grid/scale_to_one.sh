#!/usr/bin/env bash
# Scale the sweep farm from 25 GPUs back to 1, once the plane sweep is finished
# AND validated.  STOP, never terminate: stopped instances keep their EBS volumes,
# so series.npz and summary.json survive on the workers and the farm can be
# restarted.  A terminate would destroy them, so this script has no path to one.
#
#   ./scale_to_one.sh check     preconditions only, touches nothing
#   ./scale_to_one.sh archive   pull the full per-cell record local (run before stopping)
#   ./scale_to_one.sh run       check, then stop 24 of 25
#
# KEEP defaults to dima-smolvla-gpu -- the one box in the fleet that is not named
# for a sweep and predates it.  Override with KEEP=i-... if a different box should
# be the survivor.
set -u; cd "$(dirname "$0")"; source drawer_hosts.sh
MGR=ubuntu@3.88.218.39
MSSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $HOME/.ssh/firesim.pem"
KEEP="${KEEP:-i-043560448065532ff}"          # dima-smolvla-gpu, 107.20.17.186
FG=/home/ubuntu/simpler/sim_eval/roselite/finegrain
MODE="${1:-check}"
ARCH=plane3_archive

fleet_cells() {   # distinct plane3 cells that carry energy2.json, fleet-wide
  for h in "${DR[@]}"; do
    $SSH ubuntu@$h "cd $FG/plane3 2>/dev/null && for x in */; do [ -f \"\$x/energy2.json\" ] && basename \"\$x\"; done" 2>/dev/null &
  done; wait
}
fleet_unreduced() {   # simulated but never reduced -- invisible to every energy2.json count
  for h in "${DR[@]}"; do
    $SSH ubuntu@$h "cd $FG/plane3 2>/dev/null && for x in */; do [ -f \"\$x/summary.json\" ] && [ ! -f \"\$x/energy2.json\" ] && basename \"\$x\"; done" 2>/dev/null &
  done; wait
}

if [ "$MODE" = archive ]; then
  # energy2.json alone is enough for the figures, but summary.json is the authoritative
  # per-episode record and series.npz carries the per-actuation series the stall and
  # contact analyses need. ~100 kB/cell: cheap insurance against ever losing the volumes.
  i=0
  for h in "${DR[@]}"; do
    mkdir -p "$ARCH/h$i"
    rsync -a --include='*_g[0-9]*_rng*/' --include='*_g[0-9]*_rng*/energy2.json' \
          --include='*_g[0-9]*_rng*/summary.json' --include='*_g[0-9]*_rng*/series.npz' \
          --include='*_g[0-9]*_rng*/drive_config.json' --exclude='*' -e "$SSH" \
      ubuntu@"$h":$FG/plane3/ "$ARCH/h$i/" 2>/dev/null &
    i=$((i+1))
  done; wait
  echo "archived cells: $(find $ARCH -name energy2.json | wc -l)  ($(du -sh $ARCH | cut -f1))"
  exit 0
fi

echo "=== preconditions ==="
fail=0
cells=$(fleet_cells | sort -u | grep -cE '^(egg|spoon|coke|drawer)_g[0-9]+_[0-9]+_rng[0-9]+$')
unred=$(fleet_unreduced | sort -u | wc -l)
local_cells=$(find plane3_runs -name energy2.json 2>/dev/null -printf '%h\n' | xargs -r -n1 basename | sort -u | wc -l)
arch_cells=$(find $ARCH -name series.npz 2>/dev/null | wc -l)
cur=$(find ../traces_torque3 -name energy2.json 2>/dev/null | wc -l)

chk() { printf '  %-46s %s\n' "$1" "$2"; [ "$3" = ok ] || fail=1; }
[ "$cells" -eq 1760 ] && chk "plane cells reduced on the workers" "$cells/1760" ok \
                      || chk "plane cells reduced on the workers" "$cells/1760  NOT DONE" bad
[ "$unred" -eq 0 ] && chk "simulated-but-unreduced cells" "0" ok \
                   || chk "simulated-but-unreduced cells" "$unred  RUN recover_plane3.sh run FIRST" bad
[ "$local_cells" -eq 1760 ] && chk "plane cells mirrored locally" "$local_cells/1760" ok \
                            || chk "plane cells mirrored locally" "$local_cells/1760  RUN fetch_plane3.sh" bad
[ "$arch_cells" -eq 1760 ] && chk "series.npz archived locally" "$arch_cells/1760" ok \
                           || chk "series.npz archived locally" "$arch_cells/1760  RUN scale_to_one.sh archive" bad
[ "$cur" -eq 360 ] && chk "curated sweep mirrored locally" "$cur/360" ok \
                   || chk "curated sweep mirrored locally" "$cur/360" bad

if [ $fail -ne 0 ]; then
  echo; echo "REFUSING to scale down: preconditions not met. Nothing was changed."
  exit 1
fi
echo "  all preconditions met"
[ "$MODE" = check ] && { echo; echo "check only; pass 'run' to stop 24 instances"; exit 0; }

IDS=$(awk -v k="$KEEP" '$1!=k{print $1}' instances.txt | tr '\n' ' ')
n=$(echo $IDS | wc -w)
[ "$n" -eq 24 ] || { echo "expected 24 instances to stop, got $n -- refusing"; exit 1; }
echo; echo "=== stopping $n instances, keeping $KEEP ==="
$MSSH "$MGR" "aws ec2 stop-instances --instance-ids $IDS \
   --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text"
echo "--- state ---"
$MSSH "$MGR" "aws ec2 describe-instances --instance-ids $(awk '{print $1}' instances.txt | tr '\n' ' ') \
   --query 'Reservations[].Instances[].[InstanceId,State.Name,Tags[?Key==\`Name\`]|[0].Value]' \
   --output text | sort -k2"

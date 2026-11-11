#!/usr/bin/env bash
# Return the fleet to one GPU after the seed extension. STOP, never terminate.
#
# Preconditions are the ones that matter for THIS sweep -- the plane sweep's own
# archive was verified before its scale-down and has not been touched since.
# The one that has bitten this project before is the unreduced-cell check: every
# count keys on energy2.json, so a cell that simulated but never reduced is
# invisible and would be lost silently.
set -u; cd "$(dirname "$0")"; source seed30_hosts.sh
MGR=ubuntu@3.88.218.39
MSSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $HOME/.ssh/firesim.pem"
KEEP="${KEEP:-i-043560448065532ff}"          # dima-smolvla-gpu
FG=/home/ubuntu/simpler/sim_eval/roselite/finegrain
MODE="${1:-check}"

unred=$(for h in "${S30[@]}"; do
    $SSH ubuntu@$h "cd $FG/traces_torque3 2>/dev/null && for x in {egg,spoon}_*_rng1[12][0-9]/; do [ -f \"\$x/summary.json\" ] && [ ! -f \"\$x/energy2.json\" ] && basename \"\$x\"; done" 2>/dev/null
  done | sort -u | wc -l)
done_remote=$(for h in "${S30[@]}"; do
    $SSH ubuntu@$h "cd $FG/traces_torque3 2>/dev/null && for x in {egg,spoon}_*_rng1[12][0-9]/; do [ -f \"\$x/energy2.json\" ] && basename \"\$x\"; done" 2>/dev/null
  done | sort -u | wc -l)
local_new=$(ls -d ../traces_torque3/{egg,spoon}_*_rng1[12][0-9] 2>/dev/null | wc -l)

echo "=== preconditions ==="
fail=0
chk() { printf '  %-42s %s\n' "$1" "$2"; [ "$3" = ok ] || fail=1; }
[ "$unred" -eq 0 ] && chk "simulated-but-unreduced cells" "0" ok \
                   || chk "simulated-but-unreduced cells" "$unred  REDUCE FIRST" bad
[ "$done_remote" -eq 240 ] && chk "new cells reduced on the workers" "240/240" ok \
                           || chk "new cells reduced on the workers" "$done_remote/240" bad
[ "$local_new" -eq 240 ] && chk "new cells mirrored locally" "240/240" ok \
                         || chk "new cells mirrored locally" "$local_new/240  RUN seed30_fetch.sh" bad
[ $fail -ne 0 ] && { echo; echo "REFUSING: nothing changed."; exit 1; }
echo "  all preconditions met"
[ "$MODE" = check ] && { echo; echo "check only; pass 'run' to stop"; exit 0; }

IDS=$(awk -v k="$KEEP" '$1!=k{print $1}' instances.txt | tr '\n' ' ')
echo; echo "=== stopping $(echo $IDS | wc -w) instances, keeping $KEEP ==="
$MSSH "$MGR" "aws ec2 stop-instances --instance-ids $IDS \
   --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text" | tail -3
$MSSH "$MGR" "aws ec2 describe-instances --instance-ids $(awk '{print $1}' instances.txt | tr '\n' ' ') \
   --query 'Reservations[].Instances[].State.Name' --output text" | tr '\t' '\n' | sort | uniq -c

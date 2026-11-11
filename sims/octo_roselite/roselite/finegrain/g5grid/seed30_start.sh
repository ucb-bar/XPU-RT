#!/usr/bin/env bash
# Bring the sweep fleet back for the n=10 -> n=30 seed extension on the two
# widowx scenes. Starts every worker EXCEPT dima-smolvla-gpu, which is already up.
# Public IPs are reassigned on restart, so this re-queries them and rewrites
# prop_hosts.sh rather than trusting the stale drawer_hosts.sh list.
set -u; cd "$(dirname "$0")"
MGR=ubuntu@3.88.218.39
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $HOME/.ssh/firesim.pem"
KEEP=i-043560448065532ff                      # dima-smolvla-gpu, already running
IDS=$(awk -v k="$KEEP" '$1!=k{print $1}' instances.txt | tr '\n' ' ')
echo "starting $(echo $IDS | wc -w) instances"
$SSH "$MGR" "aws ec2 start-instances --instance-ids $IDS \
   --query 'StartingInstances[].[InstanceId,CurrentState.Name]' --output text"
echo "--- waiting for running ---"
for i in $(seq 1 40); do
  n=$($SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS \
        --query 'Reservations[].Instances[?State.Name==\`running\`].InstanceId' --output text" 2>/dev/null | wc -w)
  echo "  running=$n/$(echo $IDS | wc -w)"
  [ "$n" -ge "$(echo $IDS | wc -w)" ] && break
  sleep 20
done
$SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS \
   --query 'Reservations[].Instances[].PublicIpAddress' --output text" \
  | tr '\t' '\n' | grep -v '^None$' | sort > prop_ips.txt
{ echo "SSH=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i \$HOME/.ssh/firesim.pem\""
  echo "PR=( $(tr '\n' ' ' < prop_ips.txt) )"; } > prop_hosts.sh
echo "hosts: $(wc -l < prop_ips.txt)"

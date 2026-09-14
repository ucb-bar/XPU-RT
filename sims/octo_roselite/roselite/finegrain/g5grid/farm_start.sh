#!/usr/bin/env bash
# Bring the sweep fleet up and write <tag>_hosts.sh with the CURRENT public IPs.
# An instance in state `stopping` cannot be started (IncorrectInstanceState), so
# this waits for `stopped` first -- a earlier attempt silently returned 0 hosts
# for exactly that reason.
#   ./farm_start.sh <tag>
set -u; cd "$(dirname "$0")"
TAG="${1:?usage: farm_start.sh <tag>}"
MGR=ubuntu@3.88.218.39
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $HOME/.ssh/firesim.pem"
KEEP=i-043560448065532ff
IDS=$(awk -v k="$KEEP" '$1!=k{print $1}' instances.txt | tr '\n' ' '); N=$(echo $IDS | wc -w)
for i in $(seq 1 30); do
  s=$($SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[?State.Name==\`stopped\`].InstanceId' --output text" 2>/dev/null | wc -w)
  echo "  stopped=$s/$N"; [ "$s" -ge "$N" ] && break; sleep 20
done
$SSH "$MGR" "aws ec2 start-instances --instance-ids $IDS --output text" 2>&1 | grep -c STARTINGINSTANCES || echo "START FAILED -- see above"
for i in $(seq 1 45); do
  r=$($SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[?State.Name==\`running\`].InstanceId' --output text" 2>/dev/null | wc -w)
  echo "  running=$r/$N"; [ "$r" -ge "$N" ] && break; sleep 20
done
$SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[].PublicIpAddress' --output text" \
  | tr '\t' '\n' | grep -vE '^(None)?$' | sort > ${TAG}_ips.txt
{ echo "SSH=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i \$HOME/.ssh/firesim.pem\""
  echo "HOSTS=( $(tr '\n' ' ' < ${TAG}_ips.txt) )"; } > ${TAG}_hosts.sh
echo "hosts: $(wc -l < ${TAG}_ips.txt)"

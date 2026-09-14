#!/usr/bin/env bash
# Wait for the fleet to reach `running`, then capture the CURRENT public IPs.
# Errors are NOT silenced here: the previous helper redirected start-instances to
# /dev/null and reported "hosts: 0" with no reason, which cost a cycle.
set -u; cd "$(dirname "$0")"
MGR=ubuntu@3.88.218.39
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i $HOME/.ssh/firesim.pem"
KEEP=i-043560448065532ff
IDS=$(awk -v k="$KEEP" '$1!=k{print $1}' instances.txt | tr '\n' ' '); N=$(echo $IDS | wc -w)
for i in $(seq 1 60); do
  r=$($SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[?State.Name==\`running\`].InstanceId' --output text" 2>&1 | tr '\t' '\n' | grep -c '^i-')
  echo "running=$r/$N"
  [ "$r" -ge "$N" ] && break
  sleep 20
done
$SSH "$MGR" "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[].PublicIpAddress' --output text" \
  | tr '\t' '\n' | grep -vE '^(None)?$' | sort > dr30_ips.txt
{ echo "SSH=\"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i \$HOME/.ssh/firesim.pem\""
  echo "HOSTS=( $(tr '\n' ' ' < dr30_ips.txt) )"; } > dr30_hosts.sh
echo "HOSTS_READY $(wc -l < dr30_ips.txt)"

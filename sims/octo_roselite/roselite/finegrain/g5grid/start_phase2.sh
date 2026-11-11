#!/usr/bin/env bash
# Wait for the fleet to finish stopping, then start N workers for phase 2.
# g5 capacity is intermittent, so start them ONE AT A TIME and keep whatever we get.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MGR=ubuntu@3.88.218.39
SSH="ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $HOME/.ssh/firesim.pem"
WANT=${WANT:-8}
ALL=$(awk '{print $1}' "$HERE/instances.txt")
echo "waiting for 'stopping' to clear..."
for _ in $(seq 1 60); do
  n=$($SSH $MGR "aws ec2 describe-instances --instance-ids $(echo $ALL|tr '\n' ' ') \
        --query 'Reservations[].Instances[].State.Name' --output text" 2>/dev/null \
        | tr '\t' '\n' | grep -c stopping)
  [ "${n:-1}" -eq 0 ] && { echo "all settled"; break; }
  echo "  $n still stopping"; sleep 30
done
got=0; : > "$HERE/phase2_hosts.txt"
for id in $ALL; do
  [ "$got" -ge "$WANT" ] && break
  out=$($SSH $MGR "aws ec2 start-instances --instance-ids $id --output text" 2>&1)
  if echo "$out" | grep -q "InsufficientInstanceCapacity"; then echo "  $id: no capacity"; continue; fi
  if echo "$out" | grep -qi "error"; then echo "  $id: $(echo "$out"|tail -1)"; continue; fi
  got=$((got+1)); echo "$id" >> "$HERE/phase2_hosts.txt"; echo "  $id: starting ($got/$WANT)"
done
echo "started $got"
[ "$got" -eq 0 ] && exit 1
echo "waiting for running + IP..."
sleep 45
$SSH $MGR "aws ec2 describe-instances --instance-ids $(tr '\n' ' ' < "$HERE/phase2_hosts.txt") \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' --output text" \
  | tee "$HERE/phase2_ips.txt"
echo PHASE2_FLEET_READY

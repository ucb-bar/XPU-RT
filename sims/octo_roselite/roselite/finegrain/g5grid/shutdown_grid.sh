#!/usr/bin/env bash
# STOP (never terminate) all 25 sweep workers and confirm the reported state.
# AWS is reached through the manager box, which holds the credentials.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MGR=ubuntu@3.88.218.39
IDS=$(awk '{print $1}' "$HERE/instances.txt" | tr '\n' ' ')
echo "stopping: $(echo $IDS | wc -w) instances"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i ~/.ssh/firesim.pem "$MGR" \
  "aws ec2 stop-instances --instance-ids $IDS --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text"
echo "--- verify ---"
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i ~/.ssh/firesim.pem "$MGR" \
  "aws ec2 describe-instances --instance-ids $IDS --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text | sort"

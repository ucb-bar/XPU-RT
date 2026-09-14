# Worker IP map. PUBLIC IPs CHANGE ON EVERY STOP/START -- re-read them with
#   aws ec2 describe-instances --instance-ids ... --query '...PublicIpAddress'
# and update this file before pushing or fetching.
declare -A HOSTS=(
  [w0]=34.207.184.153    # i-043560448065532ff  us-east-1a (original box)
  [w1]=34.238.167.244    # i-05289cfa068e00c74  us-east-1b
  [w2]=44.193.210.194    # i-09db19b86bffe6acb  us-east-1c
  [w3]=18.234.186.173    # i-078a22ff12e6c885d  us-east-1d
  [w4]=44.192.29.49      # i-039d68dfbf56a4f61  us-east-1f
  [w5]=75.101.198.89     # i-0f7e65981bc8d4d7c  us-east-1b
)
KEY=$HOME/.ssh/firesim.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"

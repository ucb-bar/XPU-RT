# 25 workers: 6 pre-existing + 19 launched from ami-0f0c145b970259898 for the grid sweep.
# PUBLIC IPs CHANGE ON EVERY STOP/START -- refresh before any push or fetch.
declare -A HOSTS=(
  [n0]=34.207.184.153  [n1]=34.238.167.244  [n2]=44.193.210.194
  [n3]=18.234.186.173  [n4]=44.192.29.49    [n5]=75.101.198.89
  [n6]=3.95.65.244     [n7]=54.163.0.135    [n8]=54.221.20.80
  [n9]=100.48.13.133   [n10]=107.21.186.246 [n11]=3.81.103.157
  [n12]=3.81.22.165    [n13]=32.194.92.141  [n14]=35.173.251.120
  [n15]=35.175.241.234 [n16]=54.242.51.96   [n17]=13.221.65.238
  [n18]=3.235.222.171  [n19]=3.238.248.63   [n20]=32.192.78.80
  [n21]=44.222.131.105 [n22]=44.223.46.41   [n23]=98.83.36.43
  [n24]=98.92.242.7
)
KEY=$HOME/.ssh/firesim.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"

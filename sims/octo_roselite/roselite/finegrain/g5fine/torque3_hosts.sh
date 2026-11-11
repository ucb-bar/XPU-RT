# 25 g5 workers for the traces_torque3 sweep -- the same fleet as
# g5grid/drawer_hosts.sh, VERIFIED reachable and idle (0 MiB GPU, 0 runners)
# on 2026-09-08 before launch. PUBLIC IPs CHANGE ON EVERY STOP/START: re-verify
# with `bash progress_torque3.sh` before any push or fetch.
declare -A HOSTS=(
  [t0]=100.31.104.122  [t1]=100.31.255.109  [t2]=107.20.17.186
  [t3]=13.218.230.149  [t4]=32.192.89.100   [t5]=32.196.226.163
  [t6]=32.198.58.152   [t7]=32.198.63.194   [t8]=3.235.251.19
  [t9]=3.238.242.163   [t10]=34.204.191.149 [t11]=34.228.224.82
  [t12]=3.80.53.207    [t13]=3.84.127.79    [t14]=3.85.139.208
  [t15]=44.192.67.103  [t16]=44.202.89.166  [t17]=52.90.138.57
  [t18]=52.91.236.121  [t19]=54.162.113.36  [t20]=54.167.17.209
  [t21]=54.211.202.237 [t22]=54.235.20.107  [t23]=54.91.40.181
  [t24]=98.92.32.44
)
KEY=$HOME/.ssh/firesim.pem
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20"

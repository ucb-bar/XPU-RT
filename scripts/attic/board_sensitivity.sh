#!/usr/bin/env bash
# Sensitivity rows on the idle board: QoS depth 1, 200 Hz control, background CPU hogs, and a
# five-second run of the headline XPU-RT arm. Rebuilds the traced node first (qos flag).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while pgrep -f "ros_traced_matri[x]|run_xpurt_lon[g]" >/dev/null; do sleep 10; done
say "rebuild traced node"
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o ffn/*.o dronet/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_rich -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls -la ros_mb_chain_traced ros_mb_chain_traced_pool ros_mb_chain_traced_rich | wc -l'
say "ROS: QoS depth 1";        for arm in ship spin p3 multi; do QOS=1 SUFFIX=_q1 RATES="25 45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|error"; done
say "ROS: 200 Hz control";     for arm in ship spin p3 multi; do CTRL_HZ=200 SUFFIX=_c200 RATES="45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|error"; done
say "ROS: 2 background hogs";  for arm in spin p3 multi; do HOGS=2 SUFFIX=_hog2 RATES="45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|error"; done
say "XPU-RT: 200 Hz control";  scripts/run_xpurt_long.sh schedules/best45_alt2_c200.json best45alt2c200 2 2>&1 | grep -E "^===|trace rows|re-pulled"
say "XPU-RT: 2 background hogs"; HOGS=2 scripts/run_xpurt_long.sh schedules/best45_alt2.json best45alt2 2 2>&1 | grep -E "^===|trace rows|re-pulled"
say "XPU-RT: five seconds";    scripts/run_xpurt_long.sh schedules/best45_alt2_5s.json best45alt2long 2 2>&1 | grep -E "^===|trace rows|re-pulled"
say "SENSITIVITY_DONE"

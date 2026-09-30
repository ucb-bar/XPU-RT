#!/usr/bin/env bash
# The multi-threaded executor under the combined load (two cameras share one YOLO callback group).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "rebuild traced node (shared YOLO group)"
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o ffn/*.o dronet/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_rich -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls ros_mb_chain_traced ros_mb_chain_traced_pool ros_mb_chain_traced_rich | wc -l'
for rep in 1 2; do say "ROS x2rmulti r$rep"; RATES="45" scripts/ros_traced_matrix.sh x2rmulti $rep 2>&1 | grep -E "ROS_TRACED|error|not linked|fault"; done
say "STRESS3_DONE"

#!/usr/bin/env bash
# After the XPU-RT runs: rebuild the traced node with the chained control mode, run the chained
# arms at 15/25/45 Hz, then resume the ROS replicates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while pgrep -f "xpurt_runs_no[w]" >/dev/null; do sleep 10; done
say "rebuild traced node (timer + chained)"
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/model.o yolo4/kernels.o yolo4/weights.o yolo4/buffers.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls -la ros_mb_chain_traced ros_mb_chain_traced_pool'
say "chained arms"
for arm in cship cspin cp3; do RATES="15 25 45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|MATRIX_DONE|error|incomplete"; done
say "replicates"
for rep in 2 3; do for arm in cship cspin cp3 spin nproc yproc p3 ship multi; do
  RATES="15 25 45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error|incomplete"
done; done
say "BOARD_QUEUE3_DONE"

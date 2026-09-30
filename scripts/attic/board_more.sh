#!/usr/bin/env bash
# Two cameras on both runtimes, and the transformer block on the matrix engine (XPU-RT).
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "SENSITIVITY_DONE" results/codesign_feedback/xpurt_long/sensitivity.log 2>/dev/null; do sleep 30; done
say "rebuild traced node (two-camera)"
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o ffn/*.o dronet/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_rich -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls ros_mb_chain_traced ros_mb_chain_traced_pool ros_mb_chain_traced_rich | wc -l'
say "ROS: two cameras"; for rep in 1 2; do for arm in x2spin x2p; do RATES="45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "ROS_TRACED|error"; done; done
say "XPU-RT: two cameras"; scripts/run_xpurt_long.sh schedules/cam2_45_alt1.json cam2alt1 2 2>&1 | grep -E "^===|trace rows|re-pulled"
say "XPU-RT: ffn_block on IME"; CORE_KINDS="rvv,ime,rvv_c1" BACKENDS="rvv_x60,ime_x60,rvv_x60" scripts/run_xpurt_long.sh schedules/rich45_alt2_ime.json rich45alt2ime 2 2>&1 | grep -E "^===|trace rows|re-pulled|FATAL|Error"
say "MORE_DONE"

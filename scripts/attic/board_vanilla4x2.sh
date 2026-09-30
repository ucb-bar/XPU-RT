#!/usr/bin/env bash
# Rebuild the traced node with frame alternation, then the pipelining-by-hand deployment across camera rates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q VANILLA4_Q1_DONE results/codesign_feedback/ros_traced/board_vanilla4_qos1.log 2>/dev/null; do sleep 60; done
say "rebuild traced node (alternating camera)"
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o ffn/*.o dronet/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_rich -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls ros_mb_chain_traced ros_mb_chain_traced_pool ros_mb_chain_traced_rich | wc -l'
for rep in 1 2 3; do say "vanilla4x2 r$rep"; RATES="45 60 90" scripts/ros_traced_matrix.sh vanilla4x2 $rep 2>&1 | grep -E "^===|error|fault"; done
.venv/bin/python scripts/pull_ros_traced.py $(for hz in 45 60 90; do for r in 1 2 3; do echo ${hz}_vanilla4x2_r$r; done; done) 2>&1 | cut -c1-200
say "VANILLA4X2_DONE"

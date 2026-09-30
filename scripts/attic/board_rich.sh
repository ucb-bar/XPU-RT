#!/usr/bin/env bash
# The heavier stack on the board: YOLO 45 Hz + nav + control + ffn_block 10 Hz + dronet 30 Hz.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "HIGHRATE_DONE" results/codesign_feedback/xpurt_long/highrate.log 2>/dev/null; do sleep 30; done
say "stage ffn_block and dronet kernels (same builds XPU-RT runs)"
B=ModelBlaster/build/k1_xpurt
ssh k1 'mkdir -p /root/ros_mb/ffn /root/ros_mb/dronet'
for pair in ffn_block:ffn dronet:dronet; do m=${pair%%:*}; d=${pair##*:}
  scp -q $B/$m/int8/rvv_x60/{model,kernels,weights,buffers}.c $B/$m/int8/rvv_x60/{model,kernels,weights}.h ModelBlaster/kernels/rvv/mb_rvv_vxrm_compat.h k1:/root/ros_mb/$d/
done
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb
  for d in ffn dronet; do for f in model kernels weights buffers; do gcc -O2 -march=rv64gcv_zvfh -c $d/$f.c -o $d/$f.o -I$d || echo "COMPILE FAIL $d/$f"; done; done
  source /opt/ros/jazzy_prebuilt/setup.bash; R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o ffn/*.o dronet/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_rich -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  ls -la ros_mb_chain_traced ros_mb_chain_traced_pool ros_mb_chain_traced_rich'
say "XPU-RT, heavier stack"
scripts/run_xpurt_long.sh schedules/rich45_alt2.json rich45alt2 3 2>&1 | grep -E "^===|trace rows|retrying|Error"
scripts/run_xpurt_long.sh schedules/rich25_p4.json rich25p4 2 2>&1 | grep -E "^===|trace rows|retrying|Error"
say "ROS 2, heavier stack"
for arm in rspin rp3 rmulti; do RATES="25 45" scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|MATRIX_DONE|error|incomplete"; done
for arm in rspin rp3 rmulti; do RATES="25 45" scripts/ros_traced_matrix.sh $arm 2 2>&1 | grep -E "MATRIX_DONE|error|incomplete"; done
say "RICH_DONE"

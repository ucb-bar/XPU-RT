#!/usr/bin/env bash
# The whole K1 measurement campaign, in one sequence, so nothing ever runs on the board while
# something else is being measured there (a native compile during a sweep shows up in the
# sweep). Steps:
#
#   0  wait for any ROS sweep already running to finish
#   1  stage the 4-way sharded YOLO (same IR, same curated RVV kernels as the XPU-RT build) and
#      its worker pool; time it standalone on 1 and 4 harts against the golden output
#   2  link the pool-enabled traced ROS node
#   3  XPU-RT: the one-second coupled chain, three SCHED_OTHER runs and one FIFO variant, then
#      the single-frame coupled schedule with the sampler running
#   4  ROS pinned arms (spin / smte / p3 / p8), replicate 1
#   5  replicates 2 and 3 of every ROS arm
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
HOST="${MODELBLASTER_K1_HOST:-k1}"
RT="results/codesign_feedback/ros_traced"; XL="results/codesign_feedback/xpurt_long"
LONG_SCHED="schedules/scheduled_wh_coupled_chain_long_greedy_periodic_profiled.json"
say() { echo "=== $(date +%H:%M:%S) $*"; }

say "step 0: wait for the running sweep"
while pgrep -f "ros_traced_matrix.sh" >/dev/null; do sleep 15; done

say "step 1: sharded YOLO on the board, standalone timing"
G=ModelBlaster/build/k1_ros_shard4/yolov8_nano_64x96/int8/generated
scp -q $G/*.c $G/*.h $G/test_input.bin $G/test_golden.bin ModelBlaster/kernels/rvv/mb_rvv_vxrm_compat.h "$HOST:/root/ros_mb/yolo4/"
scp -q ModelBlaster/harness_linux/src/main.c "$HOST:/root/ros_mb/yolo4/harness_main.c"
S=ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/rvv_x60
ssh "$HOST" 'mkdir -p /root/ros_mb/yolo1'; scp -q $S/test_input.bin $S/test_golden.bin $S/test_io.h "$HOST:/root/ros_mb/yolo1/" 2>/dev/null
# the generated test_io.S embeds the host's absolute paths; point it at the board's copies
sed "s#$PWD/$G/#/root/ros_mb/yolo4/#g" $G/test_io.S | ssh "$HOST" 'cat > /root/ros_mb/yolo4/test_io.S'
sed "s#$PWD/$S/#/root/ros_mb/yolo1/#g" $S/test_io.S | ssh "$HOST" 'cat > /root/ros_mb/yolo1/test_io.S'
ssh "$HOST" 'cd /root/ros_mb
  for f in model kernels weights buffers; do gcc -O2 -march=rv64gcv -DMODELBLASTER_USE_POOL -DMODELBLASTER_PLATFORM_LINUX -pthread -c yolo4/$f.c -o yolo4/$f.o -Iyolo4 -Ipool || echo "COMPILE FAIL $f"; done
  (cd yolo4 && gcc -c test_io.S -o test_io.o)
  gcc -O2 -DMODELBLASTER_USE_POOL -DMODELBLASTER_PLATFORM_LINUX -pthread yolo4/harness_main.c yolo4/model.o yolo4/kernels.o yolo4/weights.o yolo4/buffers.o yolo4/test_io.o pool/modelblaster_pool.o -o yolo4/harness -Iyolo4 -Ipool -lm || echo "LINK FAIL harness4"
  if [ -f yolo1/test_io.S ]; then (cd yolo1 && gcc -c test_io.S -o test_io.o) && cp yolo4/harness_main.c yolo1/ && gcc -O2 -DMODELBLASTER_PLATFORM_LINUX yolo1/harness_main.c yolo/model.o yolo/kernels.o yolo/weights.o yolo/buffers.o yolo1/test_io.o -o yolo1/harness -Iyolo -Iyolo1 -lm || echo "LINK FAIL harness1"; fi
  ls -la yolo4/harness yolo1/harness 2>&1'
mkdir -p "$RT/yolo_standalone"
for cfg in "4core:0-3:yolo4" "1core_sharded_build:0:yolo4" "1core:0:yolo1"; do
  IFS=: read -r name cpus dir <<<"$cfg"
  ssh "$HOST" "cd /root/ros_mb && [ -x $dir/harness ] && MODELBLASTER_CPU=$cpus MODELBLASTER_ITERS=20 ./$dir/harness" > "$RT/yolo_standalone/$name.txt" 2>&1
  echo "  $name: $(grep -E 'MODELBLASTER_VERIFY' "$RT/yolo_standalone/$name.txt" | head -1)  walls: $(grep -c ITER_WALL "$RT/yolo_standalone/$name.txt")"
done

say "step 2: pool-enabled traced node"
ssh "$HOST" 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -5
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/model.o yolo4/kernels.o yolo4/weights.o yolo4/buffers.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -5
  ls -la ros_mb_chain_traced_pool'

say "step 3: XPU-RT long runs"
scripts/run_xpurt_long.sh "$LONG_SCHED" long 3 2>&1 | grep -E "^===|trace rows|DONE"
MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh "$LONG_SCHED" long 1 2>&1 | grep -E "^===|trace rows|DONE"
scripts/run_xpurt_long.sh schedules/cmp_coupled_cpsat_board.json coupled 3 2>&1 | grep -E "^===|trace rows|DONE"

say "step 4: ROS pinned arms, replicate 1"
for arm in spin smte p3 p8; do scripts/ros_traced_matrix.sh $arm 1 2>&1 | grep -E "ROS_TRACED|MATRIX_DONE|error"; done

say "step 5: replicates"
for rep in 2 3; do for arm in ship multi spin smte p3 p8; do scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error"; done; done
say "BOARD_CAMPAIGN_DONE"

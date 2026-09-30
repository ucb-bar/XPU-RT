#!/usr/bin/env bash
# Rebuild the ROS 2 baseline node on a K1 from the repo, not from whatever the board happens to hold.
#
# Every ROS rung is one program with different flags (board/k1_ros_mb/ros_mb_chain_traced.cpp): the
# arms in scripts/ros_traced_matrix.sh select nodes, executor, pool width, pinning, QoS depth and
# whether control is chained or on its own timer. This pushes that source, the worker pool
# (ModelBlaster/runtime/modelblaster_pool, the traced version) and the samplers, then builds the two
# binaries the matrix runs: with the pool (MB_WITH_POOL) and without.
#
# The per-network kernels are NOT pushed here -- they are generated from the IR
# (docs/Baselines/ros_baseline_reproduction.md step 1) and staged by scripts/board_campaign.sh, which also
# records their sha256 per run.
#
#   scripts/board_deploy_ros_node.sh          env MODELBLASTER_K1_HOST=k1
set -eu
cd "$(dirname "$0")/.."
HOST="${MODELBLASTER_K1_HOST:-k1}"; SRC=board/k1_ros_mb
ssh "$HOST" 'mkdir -p /root/ros_mb/pool'
scp -q "$SRC"/ros_mb_chain_traced.cpp "$SRC"/ros_mb_chain.cpp "$SRC"/ros_ctrl_starve.cpp \
       "$SRC"/cpu_sampler.c "$SRC"/cpu_hog.c "$SRC"/rdtime_now.c "$HOST:/root/ros_mb/"
scp -q ModelBlaster/runtime/modelblaster_pool/modelblaster_pool.c \
       ModelBlaster/runtime/modelblaster_pool/modelblaster_pool.h \
       ModelBlaster/runtime/mb_posix_compat.h "$HOST:/root/ros_mb/pool/"
ssh "$HOST" 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash
  R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector"
  gcc -O2 -march=rv64gcv -c pool/modelblaster_pool.c -o pool/modelblaster_pool.o -Ipool
  gcc -O2 cpu_sampler.c -o cpu_sampler && gcc -O2 cpu_hog.c -o cpu_hog && gcc -O2 rdtime_now.c -o rdtime_now
  for variant in "pool -DMB_WITH_POOL" "plain"; do
    set -- $variant; name=$1; shift; flags="${*:-}"
    g++ -O2 -std=c++17 $flags -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_$name.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -5 || true
  done
  echo "objects built; link with the generated kernels as scripts/board_campaign.sh step 2 does"'
echo "DEPLOY_DONE — verify with: scripts/board_source_snapshot.sh --verify"

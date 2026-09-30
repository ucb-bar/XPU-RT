#!/usr/bin/env bash
# Re-measure the baseline once the census is off the GPU, with the navigation pool's slices recorded.
#
# WHY IT WAITS. The census replays the cadence cut from the current 30_cp3n4_r* runs. Re-measuring
# those runs while it flies would leave the flights and the trace they came from describing different
# executions. The tracing change affects only what is written down -- the pool the navigation network
# was already running is unchanged -- so the cadence is expected to match; this checks that rather
# than assuming it, and leaves the old runs in place under their own tags either way.
set -u
cd "$(dirname "$0")/.."
C=results/codesign_feedback/campaign_free30/campaign.csv
until [ -f "$C" ] && [ "$(( $(wc -l < "$C") - 1 ))" -ge 96 ]; do sleep 120; done
echo "=== census complete, re-measuring the baseline $(date +%H:%M:%S)"

exec 9>results/codesign_feedback/board.lock; flock 9
scp -q results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && source /opt/ros/jazzy_prebuilt/setup.bash && R=/opt/ros/jazzy_prebuilt
  INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC 2>&1 | grep -E "error" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/model.o yolo4/kernels.o yolo4/weights.o yolo4/buffers.o nav4/model.o nav4/kernels.o nav4/weights.o nav4/buffers.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool_nav4 -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/model.o yolo4/kernels.o yolo4/weights.o yolo4/buffers.o nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS 2>&1 | grep -E "error|undefined" | head -3
  echo relinked'
flock -u 9

for r in 1 2 3; do SUFFIX=_d RATES="30" scripts/ros_traced_matrix.sh cp3n4 $r; done
.venv/bin/python scripts/pull_ros_traced.py 30_cp3n4_d_r1 30_cp3n4_d_r2 30_cp3n4_d_r3

echo "=== cadence comparison: the re-measured runs against the ones the census flew"
.venv/bin/python - <<'PY'
import json, csv, statistics, os
for tag in ("30_cp3n4_r1","30_cp3n4_d_r1","30_cp3n4_r2","30_cp3n4_d_r2","30_cp3n4_r3","30_cp3n4_d_r3"):
    d=f"results/codesign_feedback/ros_traced/{tag}"
    if not os.path.isdir(d): print(f"  {tag}: missing"); continue
    s=json.load(open(f"{d}/summary.json"))
    rows=list(csv.DictReader(open(f"{d}/chain.csv")))
    g=sorted(float(x["e2e_goal_ms"]) for x in rows if x.get("e2e_goal_ms"))
    print(f"  {tag:<18} gap_mean {s.get('gap_mean_ms') or 0:6.2f}  camera->goal {statistics.median(g):6.2f} ms")
PY
echo "RETRACE_CP3N4_DONE $(date +%H:%M:%S)"

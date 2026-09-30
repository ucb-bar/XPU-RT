#!/usr/bin/env bash
# The explicit best-effort XPU-RT placements, plus YOLO-alone timings on the E cluster and on all
# eight harts, inserted after the tiled runs and before the ROS replicates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "resume replicates" results/codesign_feedback/xpurt_long/tiled_runs.log 2>/dev/null; do sleep 10; done
pkill -f "^bash scripts/xpurt_tiled_run[s].sh"; sleep 1; pkill -f "^bash scripts/ros_traced_matri[x].sh"; sleep 1
ssh k1 'pkill -f cpu_sampler; pkill -f ros_mb_chain_traced; true'
say "YOLO alone on the E cluster and on 8 harts"
mkdir -p results/codesign_feedback/ros_traced/yolo_standalone
for cfg in "4core_E:4-7" "8core:0-7"; do IFS=: read -r name cpus <<<"$cfg"
  ssh k1 "cd /root/ros_mb && MODELBLASTER_CPU=$cpus MODELBLASTER_ITERS=20 ./yolo4/harness" > results/codesign_feedback/ros_traced/yolo_standalone/$name.txt 2>&1
  echo "  $name: $(grep -c ITER_WALL results/codesign_feedback/ros_traced/yolo_standalone/$name.txt) walls, $(grep -o 'max_abs_err=[^ ]*' results/codesign_feedback/ros_traced/yolo_standalone/$name.txt | head -1)"
done
say "best25 p4";    scripts/run_xpurt_long.sh schedules/best25_p4.json best25p4 3 2>&1 | grep -E "^===|trace rows|DONE|Error"
say "best45 alt4";  scripts/run_xpurt_long.sh schedules/best45_alt4.json best45alt4 3 2>&1 | grep -E "^===|trace rows|DONE|Error"
say "best45 p4";    scripts/run_xpurt_long.sh schedules/best45_p4.json best45p4 3 2>&1 | grep -E "^===|trace rows|DONE|Error"
say "best25 p4 FIFO"; MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh schedules/best25_p4.json best25p4 1 2>&1 | grep -E "^===|trace rows|DONE"
say "resume replicates"
for rep in 2 3; do for arm in cship cspin cp3 spin nproc yproc p3 ship multi; do
  RATES="15 25 45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error|incomplete"
done; done
say "BEST_RUNS_DONE"

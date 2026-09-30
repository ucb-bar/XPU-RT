#!/usr/bin/env bash
# VitFly's CNN front end on the K1: curated kernels, coverage, cross-compiled harness, timed
# on the board when idle -- the candidate heavier nav head.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
eval "$(bash scripts/setup_spacemit_toolchain.sh 2>/dev/null)"
cd ModelBlaster; OUT=build/k1_vitfly/int8; PY=../.venv/bin/python
mkdir -p $OUT/rvv_x60
say "skeleton + kernels"
$PY -m modelblaster.pipeline.generate_skeleton --ir $OUT/graph.json --weights $OUT/weights.npz --io $OUT/io.npz --out-dir $OUT/rvv_x60 --backend rvv_x60 --platform linux 2>&1 | tail -1
CROSS="$CROSS" $PY -m modelblaster.pipeline.generate_kernels --ir $OUT/graph.json --out-dir $OUT/rvv_x60 --target rvv_x60 --backend reference --quant int8 --global-curated-dir "$(pwd)/kernels" 2>&1 | grep -cE "falling back"
say "coverage"; $PY scripts/check_kernel_coverage.py $OUT/rvv_x60 2>&1 | grep -E "^(OK|FAIL)|dispatches" | tee build/k1_vitfly/coverage.txt
cd $OUT/rvv_x60
for f in model kernels buffers weights; do ${CROSS}gcc -O2 -march=rv64gcv_zvfh -c $f.c -o $f.o -I. -I../../../../kernels/rvv 2>&1 | grep -E "error" | head -3; done
sed "s#$(pwd)/#./#g" test_io.S > test_io_rel.S && ${CROSS}gcc -c test_io_rel.S -o test_io.o
cp ../../../../harness_linux/src/main.c harness_main.c
${CROSS}gcc -O2 -march=rv64gcv_zvfh -DMODELBLASTER_PLATFORM_LINUX harness_main.c model.o kernels.o weights.o buffers.o test_io.o -o vitfly_harness -I. -lm 2>&1 | grep -E "error|undefined" | head -5
ls -la vitfly_harness | awk '{print $5}'
cd ../../../../..
say "waiting for the board"
while pgrep -f "ros_traced_matri[x]|run_xpurt_lon[g]|board_mor[e]|board_sensitivit[y]" >/dev/null; do sleep 30; done
ssh k1 'mkdir -p /root/vitfly'; scp -q ModelBlaster/$OUT/rvv_x60/vitfly_harness k1:/root/vitfly/
for cfg in "1core:0" "4core:0-3"; do IFS=: read -r n c <<<"$cfg"
  ssh k1 "cd /root/vitfly && MODELBLASTER_CPU=$c MODELBLASTER_ITERS=20 ./vitfly_harness" > results/codesign_feedback/ros_traced/yolo_standalone/vitfly_$n.txt 2>&1
  echo "  vitfly $n: $(grep -o 'max_abs_err=[^ ]*' results/codesign_feedback/ros_traced/yolo_standalone/vitfly_$n.txt | head -1)  walls(ms): $(grep -oE 'ITER_WALL \[[0-9]+\] === [0-9]+' results/codesign_feedback/ros_traced/yolo_standalone/vitfly_$n.txt | awk '{printf "%.1f ", $4/24000}' | cut -c1-60)"
done
say "VITFLY_DONE"

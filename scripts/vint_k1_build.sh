#!/usr/bin/env bash
# ViNT, all-int8, through the same ModelBlaster pipeline as every other network: kernels for the
# K1, curated coverage, cross-compiled objects, a native harness, then a timing run on the board
# once it is idle. Every step's output is kept so "how much of it runs on curated kernels" is a
# number, not a claim.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
MB=ModelBlaster; OUT=$MB/build/k1_vint_i8/int8; PY=.venv/bin/python
while pgrep -f "extract_graph_expor[t]" >/dev/null; do sleep 15; done
tail -3 $MB/build/k1_vint_i8/extract.log
[ -f $OUT/generated/graph.json ] || { say "extraction failed"; exit 1; }
eval "$(bash scripts/setup_spacemit_toolchain.sh 2>/dev/null)"
cd $MB
mkdir -p build/k1_vint_i8/int8/rvv_x60 && cp build/k1_vint_i8/int8/generated/{graph.json,weights.npz,io.npz} build/k1_vint_i8/int8/
say "skeleton + kernels"
../$PY -m modelblaster.pipeline.generate_skeleton --ir build/k1_vint_i8/int8/graph.json --weights build/k1_vint_i8/int8/weights.npz --io build/k1_vint_i8/int8/io.npz --out-dir build/k1_vint_i8/int8/rvv_x60 --backend rvv_x60 --platform linux 2>&1 | tail -1
CROSS="$CROSS" ../$PY -m modelblaster.pipeline.generate_kernels --ir build/k1_vint_i8/int8/graph.json --out-dir build/k1_vint_i8/int8/rvv_x60 --target rvv_x60 --backend reference --quant int8 --global-curated-dir "$(pwd)/kernels" 2>&1 | grep -cE "falling back"
say "coverage"; ../$PY scripts/check_kernel_coverage.py build/k1_vint_i8/int8/rvv_x60 2>&1 | grep -vE "^(The curated|OWN kernel|If a fallback|exemption|A build targeting|    kernels/|\(or an alias)" | tee build/k1_vint_i8/coverage.txt
say "cross-compile"
cd build/k1_vint_i8/int8/rvv_x60
for f in model kernels buffers; do ${CROSS}gcc -O2 -march=rv64gcv_zvfh -c $f.c -o $f.o -I. -I../../../../kernels/rvv 2>&1 | grep -E "error" | head -3; done
${CROSS}gcc -O1 -march=rv64gcv_zvfh -c weights.c -o weights.o -I. 2>&1 | grep -E "error" | head -2
sed "s#$(pwd)/#./#g" test_io.S > test_io_rel.S && ${CROSS}gcc -c test_io_rel.S -o test_io.o
cp ../../../../harness_linux/src/main.c harness_main.c
${CROSS}gcc -O2 -march=rv64gcv_zvfh -DMODELBLASTER_PLATFORM_LINUX harness_main.c model.o kernels.o weights.o buffers.o test_io.o -o vint_harness -I. -lm 2>&1 | grep -E "error|undefined" | head -5
ls -la vint_harness 2>/dev/null && file vint_harness | cut -c1-80
cd ../../../../..
say "waiting for the board"
while pgrep -f "ros_traced_matri[x]|run_xpurt_lon[g]|board_ric[h]|board_highrate[2]" >/dev/null; do sleep 30; done
say "timing on the board"
ssh k1 'mkdir -p /root/vint'; scp -q $MB/build/k1_vint_i8/int8/rvv_x60/vint_harness k1:/root/vint/
mkdir -p results/codesign_feedback/ros_traced/yolo_standalone
for cfg in "1core:0" "4core:0-3"; do IFS=: read -r n c <<<"$cfg"
  ssh k1 "cd /root/vint && MODELBLASTER_CPU=$c MODELBLASTER_ITERS=5 ./vint_harness" > results/codesign_feedback/ros_traced/yolo_standalone/vint_$n.txt 2>&1
  echo "  vint $n: $(grep -o 'max_abs_err=[^ ]*' results/codesign_feedback/ros_traced/yolo_standalone/vint_$n.txt | head -1)  walls: $(grep -oE 'ITER_WALL \[[0-9]+\] === [0-9]+' results/codesign_feedback/ros_traced/yolo_standalone/vint_$n.txt | awk '{printf "%.0f ", $4/24000}')"
done
say "VINT_DONE"

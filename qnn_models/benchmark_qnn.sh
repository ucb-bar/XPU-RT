#!/usr/bin/env bash
#
# Benchmark a QNN model across CPU, GPU, and DSP backends on QRB5165.
#
# Usage:
#   bash benchmark_qnn.sh <model_dir> <model_name> [iters]
#
# Example:
#   bash benchmark_qnn.sh /root/models/dronet dronet 50
#   bash benchmark_qnn.sh /root/models/mobilenet_v2 mobilenet_v2 50
#   bash benchmark_qnn.sh /root/models/yolov8s yolov8s 20
#
set -euo pipefail

MODEL_DIR="${1:?Usage: benchmark_qnn.sh <model_dir> <model_name> [iters]}"
MODEL_NAME="${2:?Usage: benchmark_qnn.sh <model_dir> <model_name> [iters]}"
ITERS="${3:-50}"

QNN_SDK_ROOT="${QNN_SDK_ROOT:-/root/qairt}"
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/target:${LD_LIBRARY_PATH:-}
export ADSP_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/hexagon-v66;/dsp/cdsp;/dsp"
QNN_NET_RUN=$QNN_SDK_ROOT/bin/target/qnn-net-run
MODEL_FP="$MODEL_DIR/${MODEL_NAME}.dlc"
MODEL_Q8="$MODEL_DIR/${MODEL_NAME}_quantized.dlc"
INPUT_RAW="$MODEL_DIR/input.raw"

# Generate multi-iteration input list
MULTI_INPUT="$MODEL_DIR/input_list_multi.txt"
python3 -c "
for _ in range($ITERS):
    print('$INPUT_RAW')
" > "$MULTI_INPUT"

# Backends: name:library:model
# CPU/GPU use float32 DLC; DSP requires int8 quantized DLC
BACKENDS=(
    "CPU:$QNN_SDK_ROOT/lib/target/libQnnCpu.so:$MODEL_FP"
    "GPU:$QNN_SDK_ROOT/lib/target/libQnnGpu.so:$MODEL_FP"
    "DSP:$QNN_SDK_ROOT/lib/target/libQnnDsp.so:$MODEL_Q8"
)

echo "========================================="
echo "${MODEL_NAME} QNN Benchmark (${ITERS} iterations)"
echo "Model (float32): $MODEL_FP"
echo "Model (int8):    $MODEL_Q8"
echo "========================================="
echo ""

for entry in "${BACKENDS[@]}"; do
    IFS=: read -r NAME LIB DLC <<< "$entry"
    OUTDIR="$MODEL_DIR/output_${NAME,,}"

    echo "--- $NAME backend ---"
    rm -rf "$OUTDIR"

    START_NS=$(date +%s%N)
    if $QNN_NET_RUN \
        --dlc_path "$DLC" \
        --backend "$LIB" \
        --input_list "$MULTI_INPUT" \
        --output_dir "$OUTDIR" 2>&1; then
        END_NS=$(date +%s%N)
        ELAPSED_MS=$(( (END_NS - START_NS) / 1000000 ))
        AVG_MS=$(python3 -c "print(f'{$ELAPSED_MS / $ITERS:.2f}')")
        FPS=$(python3 -c "print(f'{1000 / ($ELAPSED_MS / $ITERS):.1f}')")
        echo "  Total: ${ELAPSED_MS} ms | Avg: ${AVG_MS} ms/iter | FPS: ${FPS}"
        echo "RESULT:${MODEL_NAME}:${NAME}:${AVG_MS}"
    else
        echo "  FAILED"
        echo "RESULT:${MODEL_NAME}:${NAME}:FAILED"
    fi
    echo ""
done

echo "========================================="
echo "Benchmark complete"
echo "========================================="

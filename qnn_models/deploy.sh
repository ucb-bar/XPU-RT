#!/usr/bin/env bash
#
# Full pipeline: export ONNX → convert to QNN DLC → deploy to QRB5165 → benchmark.
#
# Usage:
#   ./deploy.sh                              # all models, full pipeline
#   ./deploy.sh --model=dronet               # single model
#   ./deploy.sh --model=mobilenet_v2         # single model
#   ./deploy.sh --model=yolov8s              # single model
#   ./deploy.sh --export-only                # just export ONNX(es)
#   ./deploy.sh --convert-only               # just convert ONNX → DLC
#   ./deploy.sh --run-only                   # just benchmark (models must be on board)
#   ./deploy.sh --iters=100                  # set iteration count
#
# Prerequisites:
#   - conda env "xpurt" with pytorch, torchvision, ultralytics, onnx2tf, matplotlib
#   - Docker with sudo access (for QNN conversion)
#   - SSH key access to board (root@10.44.120.201)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="xpurt"
BOARD_USER="root"
BOARD_IP="10.44.120.201"
QNN_SDK="/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326"
DOCKER_IMAGE="qnn-convert"
RESULTS_FILE="$SCRIPT_DIR/benchmark_results.json"

# --- Parse args ---
EXPORT=true
CONVERT=true
DEPLOY=true
RUN=true
ITERS=50
SELECTED_MODEL=""

for arg in "$@"; do
    case $arg in
        --export-only)  CONVERT=false; DEPLOY=false; RUN=false ;;
        --convert-only) EXPORT=false; DEPLOY=false; RUN=false ;;
        --run-only)     EXPORT=false; CONVERT=false; DEPLOY=false ;;
        --model=*)      SELECTED_MODEL="${arg#*=}" ;;
        --iters=*)      ITERS="${arg#*=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Model definitions
#   input_name: ONNX input tensor name
#   hw: "H W C" for input shape (NHWC)
#   export: python script to export ONNX
#   converter: onnx (standard path) or tflite (for models with SiLU + residuals)
declare -A MODEL_INPUT_NAME MODEL_HW MODEL_EXPORT MODEL_CONVERTER
MODEL_INPUT_NAME[dronet]="input"
MODEL_HW[dronet]="112 112 3"
MODEL_EXPORT[dronet]="export_onnx.py"
MODEL_CONVERTER[dronet]="onnx"

MODEL_INPUT_NAME[mobilenet_v2]="input"
MODEL_HW[mobilenet_v2]="224 224 3"
MODEL_EXPORT[mobilenet_v2]="export_mobilenet.py"
MODEL_CONVERTER[mobilenet_v2]="onnx"

MODEL_INPUT_NAME[yolov8s]="images"
MODEL_HW[yolov8s]="640 640 3"
MODEL_EXPORT[yolov8s]="export_yolo.py"
MODEL_CONVERTER[yolov8s]="tflite"

if [ -n "$SELECTED_MODEL" ]; then
    ALL_MODELS=("$SELECTED_MODEL")
else
    ALL_MODELS=(dronet mobilenet_v2 yolov8s)
fi

# Validate model names
for m in "${ALL_MODELS[@]}"; do
    if [ -z "${MODEL_EXPORT[$m]:-}" ]; then
        echo "Unknown model: $m (valid: dronet, mobilenet_v2, yolov8s)"
        exit 1
    fi
done

# --- Ensure Docker image exists ---
if $CONVERT; then
    if ! sudo docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        echo "Building Docker image $DOCKER_IMAGE..."
        sudo docker build -f "$SCRIPT_DIR/Dockerfile.qnn-convert" -t "$DOCKER_IMAGE" "$SCRIPT_DIR"
    fi
fi

docker_run() {
    sudo docker run --rm \
        -v "$QNN_SDK":/qnn:ro \
        -v "$SCRIPT_DIR":/workspace \
        "$DOCKER_IMAGE" "$@"
}

# ===========================================================================
# Process each model
# ===========================================================================
for MODEL in "${ALL_MODELS[@]}"; do
    echo ""
    echo "###############################################"
    echo "# Model: $MODEL"
    echo "###############################################"
    echo ""

    ONNX_FILE="$SCRIPT_DIR/${MODEL}.onnx"
    DLC_FILE="$SCRIPT_DIR/${MODEL}.dlc"
    QUANTIZED_DLC="$SCRIPT_DIR/${MODEL}_quantized.dlc"
    CAL_DIR="$SCRIPT_DIR/calibration_data_${MODEL}"
    CAL_LIST="$SCRIPT_DIR/calibration_list_${MODEL}.txt"
    INPUT_NAME="${MODEL_INPUT_NAME[$MODEL]}"
    read -r H W C <<< "${MODEL_HW[$MODEL]}"
    BOARD_MODEL_DIR="/root/models/${MODEL}"
    CONVERTER="${MODEL_CONVERTER[$MODEL]}"

    # --- Step 1: Export ONNX ---
    if $EXPORT; then
        echo "=== [$MODEL] Step 1: Exporting ONNX ==="
        conda run -n "$CONDA_ENV" --no-capture-output \
            python3 "$SCRIPT_DIR/${MODEL_EXPORT[$MODEL]}" \
                --output "$ONNX_FILE"
        echo ""
    fi

    # --- Step 2: Convert ONNX → DLC ---
    if $CONVERT; then
        echo "=== [$MODEL] Step 2: Converting to DLC ==="

        if [ "$CONVERTER" = "onnx" ]; then
            # Standard path: onnxsim → snpe-onnx-to-dlc
            docker_run bash -c "\
                pip install -q onnxruntime onnx-simplifier 'numpy<2' && \
                python3.10 -c \"
import onnx
from onnxsim import simplify
model = onnx.load('/workspace/${MODEL}.onnx')
model_simp, check = simplify(model)
assert check, 'Simplification failed'
onnx.save(model_simp, '/workspace/${MODEL}_simplified.onnx')
print('Simplified ONNX saved')
\" && \
                python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
                    --input_network /workspace/${MODEL}_simplified.onnx \
                    --input_layout ${INPUT_NAME} NCHW \
                    --output_path /workspace/${MODEL}.dlc"

            sudo chown "$(id -u):$(id -g)" \
                "$SCRIPT_DIR/${MODEL}_simplified.onnx" "$DLC_FILE"

        elif [ "$CONVERTER" = "tflite" ]; then
            # YOLOv8s path: ONNX → TFLite (via onnx2tf) → DLC
            # The QNN ONNX converter has a C++ shape inference bug with
            # SiLU activations + residual blocks; the TFLite path avoids it.
            echo "  Converting ONNX → TFLite (onnx2tf)..."
            conda run -n "$CONDA_ENV" --no-capture-output \
                python3 "$SCRIPT_DIR/onnx2tf_convert.py" \
                    --input "$ONNX_FILE" \
                    --output-dir "$SCRIPT_DIR/${MODEL}_saved_model"

            echo "  Converting TFLite → DLC..."
            docker_run bash -c "\
                apt-get update -qq && \
                apt-get install -y -qq libllvm14 >/dev/null 2>&1 && \
                pip install -q 'numpy<2' decorator attrs scipy psutil pytest tflite >/dev/null 2>&1 && \
                python3.10 /qnn/bin/x86_64-linux-clang/snpe-tflite-to-dlc \
                    --input_network /workspace/${MODEL}_saved_model/${MODEL}_float32.tflite \
                    --output_path /workspace/${MODEL}.dlc"

            sudo chown "$(id -u):$(id -g)" "$DLC_FILE"
        fi

        echo "DLC saved: $DLC_FILE ($(du -h "$DLC_FILE" | cut -f1))"
        echo ""

        # Generate calibration data if missing (NHWC float32, 10 samples)
        if [ ! -d "$CAL_DIR" ]; then
            echo "  Generating calibration data..."
            mkdir -p "$CAL_DIR"
            python3 -c "
import numpy as np
for i in range(10):
    data = np.random.randn(1, $H, $W, $C).astype(np.float32)
    data.tofile('$CAL_DIR/input_{}.raw'.format(i))
"
            : > "$CAL_LIST"
            for i in $(seq 0 9); do
                echo "/workspace/calibration_data_${MODEL}/input_${i}.raw" >> "$CAL_LIST"
            done
        fi

        # Quantize DLC → INT8
        echo "=== [$MODEL] Step 2b: Quantizing DLC → INT8 ==="
        docker_run python3.10 /qnn/bin/x86_64-linux-clang/qairt-quantizer \
            --input_dlc "/workspace/${MODEL}.dlc" \
            --output_dlc "/workspace/${MODEL}_quantized.dlc" \
            --input_list "/workspace/calibration_list_${MODEL}.txt" \
            --act_bitwidth 8 \
            --weights_bitwidth 8 \
            --bias_bitwidth 8

        sudo chown "$(id -u):$(id -g)" "$QUANTIZED_DLC"
        echo "Quantized DLC: $QUANTIZED_DLC ($(du -h "$QUANTIZED_DLC" | cut -f1))"
        echo ""
    fi

    # --- Step 3: Deploy to board ---
    if $DEPLOY; then
        echo "=== [$MODEL] Step 3: Deploying to $BOARD_IP ==="
        ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p $BOARD_MODEL_DIR"
        scp "$DLC_FILE" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/${MODEL}.dlc"
        scp "$QUANTIZED_DLC" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/${MODEL}_quantized.dlc"
        scp "$SCRIPT_DIR/benchmark_qnn.sh" "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/benchmark_qnn.sh"

        # Generate dummy input on board
        ssh "${BOARD_USER}@${BOARD_IP}" "\
            python3 -c \"
import numpy as np
data = np.random.randn(1, $H, $W, $C).astype(np.float32)
data.tofile('$BOARD_MODEL_DIR/input.raw')
with open('$BOARD_MODEL_DIR/input_list.txt', 'w') as f:
    f.write('$BOARD_MODEL_DIR/input.raw\n')
\""

        echo "Deployed to ${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/"
        echo ""
    fi

    # --- Step 4: Benchmark on board ---
    if $RUN; then
        echo "=== [$MODEL] Step 4: Benchmarking ==="
        # Always push the latest benchmark script
        ssh "${BOARD_USER}@${BOARD_IP}" "mkdir -p $BOARD_MODEL_DIR"
        scp "$SCRIPT_DIR/benchmark_qnn.sh" \
            "${BOARD_USER}@${BOARD_IP}:${BOARD_MODEL_DIR}/benchmark_qnn.sh"
        ssh "${BOARD_USER}@${BOARD_IP}" \
            "bash ${BOARD_MODEL_DIR}/benchmark_qnn.sh ${BOARD_MODEL_DIR} ${MODEL} ${ITERS}" \
            2>&1 | tee "/tmp/benchmark_${MODEL}.log"
    fi
done

# --- Step 5: Collect results and generate plot ---
if $RUN; then
    echo ""
    echo "###############################################"
    echo "# Generating benchmark plot"
    echo "###############################################"

    # Parse RESULT: lines from benchmark logs into JSON
    python3 -c "
import json, glob, re

results = {}
for logfile in sorted(glob.glob('/tmp/benchmark_*.log')):
    with open(logfile) as f:
        for line in f:
            m = re.match(r'RESULT:(\S+):(\w+):([\d.]+|FAILED)', line.strip())
            if m:
                model, backend, val = m.groups()
                if model not in results:
                    results[model] = {}
                results[model][backend] = float(val) if val != 'FAILED' else None

with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to $RESULTS_FILE')
print(json.dumps(results, indent=2))
"

    conda run -n "$CONDA_ENV" --no-capture-output \
        python3 "$SCRIPT_DIR/plot_benchmarks.py" --results "$RESULTS_FILE"

    echo ""
    echo "Done. Plot saved to $SCRIPT_DIR/plots/qnn_benchmark.png"
fi

# QNN Model Benchmarking Pipeline

End-to-end pipeline for exporting neural network models to ONNX, converting
them to Qualcomm AI Engine Direct (QNN) DLC format, deploying to a QRB5165
board, and benchmarking across CPU / GPU / DSP backends.

## Quick Start

```bash
# Full pipeline — all 3 models:
./deploy.sh

# Single model:
./deploy.sh --model=dronet
./deploy.sh --model=mobilenet_v2
./deploy.sh --model=yolov8s

# Re-run benchmarks only (models already on board):
./deploy.sh --run-only --iters=50

# Partial pipeline:
./deploy.sh --export-only       # just export ONNX
./deploy.sh --convert-only      # just convert + quantize (Docker required)
```

## Conversion Flow

DroNet and MobileNetV2 use the standard ONNX-to-DLC path.  YOLOv8s requires
an intermediate TFLite step because the QNN SDK v2.45 ONNX converter has a
C++ shape-inference bug on models with SiLU activations + residual blocks.

```
                          DroNet / MobileNetV2
                          ~~~~~~~~~~~~~~~~~~~~
  PyTorch model
       │
       ▼
  ┌─────────────┐   export_onnx.py
  │  ONNX model │   export_mobilenet.py
  │  (NCHW fp32)│
  └──────┬──────┘
         │  onnx-simplifier (Docker)
         ▼
  ┌─────────────┐
  │ Simplified  │
  │    ONNX     │
  └──────┬──────┘
         │  snpe-onnx-to-dlc --input_layout <name> NCHW (Docker)
         ▼
  ┌─────────────┐
  │ Float32 DLC │──────────────────► CPU / GPU backend
  │  (NHWC)     │
  └──────┬──────┘
         │  qairt-quantizer --act_bitwidth 8 (Docker)
         ▼
  ┌─────────────┐
  │  INT8 DLC   │──────────────────► DSP backend (Hexagon v66)
  │  (NHWC)     │
  └─────────────┘


                             YOLOv8s
                             ~~~~~~~
  PyTorch model (ultralytics)
       │
       ▼
  ┌─────────────┐   export_yolo.py (includes onnxslim)
  │  ONNX model │
  │  (NCHW fp32)│
  └──────┬──────┘
         │  onnx2tf (host conda env)      ◄── workaround: QNN ONNX converter
         ▼                                     fails on SiLU + residual blocks
  ┌─────────────┐
  │  TFLite     │
  │ (NHWC fp32) │
  └──────┬──────┘
         │  snpe-tflite-to-dlc (Docker)
         ▼
  ┌─────────────┐
  │ Float32 DLC │──────────────────► CPU / GPU backend
  │  (NHWC)     │
  └──────┬──────┘
         │  qairt-quantizer --act_bitwidth 8 (Docker)
         ▼
  ┌─────────────┐
  │  INT8 DLC   │──────────────────► DSP backend (Hexagon v66)
  │  (NHWC)     │
  └─────────────┘
```

After conversion, models are scp'd to the board and benchmarked with
`qnn-net-run` across all backends.  Results are collected into
`benchmark_results.json` and plotted to `plots/qnn_benchmark.png`.

## Prerequisites

### 1. Host Machine

**Conda environment** (`xpurt`):

```bash
conda create -n xpurt python=3.11
conda activate xpurt
pip install torch torchvision          # ONNX export
pip install ultralytics                # YOLOv8s export
pip install onnx2tf tensorflow-cpu     # ONNX → TFLite (YOLOv8s path)
pip install matplotlib                 # plotting
```

**Docker** with sudo access (or add your user to the `docker` group):

```bash
# The Dockerfile is built automatically on first run.
# To build manually:
sudo docker build -f Dockerfile.qnn-convert -t qnn-convert .
```

**Qualcomm AI Engine Direct SDK** (QNN / QAIRT) v2.45.0:

```
/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326/
├── bin/x86_64-linux-clang/
│   ├── snpe-onnx-to-dlc        # ONNX → DLC converter
│   ├── snpe-tflite-to-dlc      # TFLite → DLC converter
│   └── qairt-quantizer          # DLC quantizer (float32 → INT8)
└── lib/
    ├── python/                  # Python bindings (used inside Docker)
    └── x86_64-linux-clang/      # Host-side native libraries
```

To use a different SDK version or path, edit `QNN_SDK=` in `deploy.sh`.

### 2. QRB5165 Board

The board must be reachable via SSH without a password prompt:

```bash
ssh root@10.44.120.201    # should connect without prompting
```

**Board-side QNN runtime** must be pre-installed at `/root/qairt/`:

```
/root/qairt/
├── bin/target/
│   └── qnn-net-run              # on-device inference tool
├── lib/target/
│   ├── libQnnCpu.so             # CPU backend
│   ├── libQnnGpu.so             # GPU backend (Adreno 650)
│   └── libQnnDsp.so             # DSP backend (Hexagon v66 stub)
└── lib/hexagon-v66/
    ├── libQnnDspV66Skel.so      # DSP skel library (runs on CDSP)
    ├── libQnnDspV66.so
    └── libCalculator_skel.so
```

The DSP skel libraries can be copied from the host SDK:

```bash
# One-time setup for DSP backend:
ssh root@10.44.120.201 'mkdir -p /root/qairt/lib/hexagon-v66'
scp $QNN_SDK/lib/hexagon-v66/unsigned/libQnnDspV66Skel.so \
    $QNN_SDK/lib/hexagon-v66/unsigned/libCalculator_skel.so \
    $QNN_SDK/lib/hexagon-v66/unsigned/libQnnDspV66.so \
    root@10.44.120.201:/root/qairt/lib/hexagon-v66/
```

To change the board IP or user, edit `BOARD_IP=` / `BOARD_USER=` in `deploy.sh`.

## File Inventory

| File | Purpose |
|------|---------|
| `deploy.sh` | Main entry point — orchestrates the full pipeline |
| `benchmark_qnn.sh` | On-board benchmark script (scp'd to board, runs `qnn-net-run`) |
| `Dockerfile.qnn-convert` | Docker image providing Ubuntu 22.04 + Python 3.10 + QNN SDK deps |
| `dronet.py` | DroNet model definition (PyTorch) |
| `export_onnx.py` | Export DroNet to ONNX |
| `export_mobilenet.py` | Export MobileNetV2 to ONNX |
| `export_yolo.py` | Export YOLOv8s to ONNX |
| `onnx2tf_convert.py` | ONNX → TFLite via onnx2tf (YOLOv8s workaround) |
| `plot_benchmarks.py` | Generate benchmark comparison plot from results JSON |

### Generated artifacts (not checked in)

| Artifact | Generated by |
|----------|-------------|
| `*.onnx`, `*_simplified.onnx` | Export + simplification step |
| `*.dlc`, `*_quantized.dlc` | DLC conversion + quantization step |
| `calibration_data_*/` | Calibration data for INT8 quantization |
| `*_saved_model/` | TFLite intermediate (YOLOv8s only) |
| `benchmark_results.json` | Parsed benchmark output |
| `plots/qnn_benchmark.png` | Benchmark comparison plot |

## Hardware Notes

**QRB5165 (SM8250 / KONA)**:
- **CPU**: Kryo 585 (Cortex-A77 + A55) — best for small models
- **GPU**: Adreno 650 — competitive at medium model sizes
- **DSP**: Hexagon v66 (CDSP) — requires INT8 quantized models, wins at large model sizes
- **HTP**: Not available (requires Hexagon v68+, SM8350+)

## Known Issues

- **QNN SDK v2.45 ONNX converter bug**: The C++ `infer_output_shapes` function
  produces garbage tensor dimensions on models with SiLU (Sigmoid Linear Unit)
  activations combined with residual connections (e.g., YOLOv8, YOLOv5).  The
  workaround is to convert ONNX → TFLite (natively NHWC) → DLC.

- **DSP v66 datatype restrictions**: The Hexagon v66 DSP only supports INT8
  (uFxp_8) tensors — not float32, not sFxp_32 biases.  Quantization must use
  `--bias_bitwidth 8` (not 32).

- **FastRPC for DSP**: The DSP backend communicates via FastRPC.  The skel
  libraries must be deployed to a writable path listed in `ADSP_LIBRARY_PATH`
  (e.g., `/root/qairt/lib/hexagon-v66`).

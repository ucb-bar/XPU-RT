"""Block diagram of the QNN SDK stack on QRB5165 v66.

Common layers at top span all four columns (CPU / GPU / HTA / DSP).
The bottom layers diverge into per-backend toolchains.
Output: plots/qnn_sdk_stack.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(__file__).parent.parent


def block(ax, x, y, w, h, text, fc, ec="black", fontsize=10,
          fontweight="normal", lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    ))
    ax.text(x + w / 2, y + h / 2, text,
             ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, wrap=True)


def arrow(ax, x0, y0, x1, y1, color="#666"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=10,
        color=color, linewidth=0.8,
    ))


def main():
    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 18)
    ax.axis("off")

    # Color palette
    C_APP   = "#cfe2f3"   # application
    C_IR    = "#fce5cd"   # IR / frontend
    C_TOOL  = "#fff2cc"   # tool / finalizer
    C_RT    = "#d9ead3"   # runtime API
    C_STUB  = "#ead1dc"   # backend stub
    C_KER   = "#f4cccc"   # op-package / kernels
    C_COMP  = "#d9d2e9"   # compiler
    C_IR2   = "#b6d7a8"   # lower IR
    C_HW    = "#cccccc"   # hardware

    full_w = 15.5
    col_w  = 3.7
    gap    = 0.1
    cols_x = [0.25 + (col_w + gap) * i for i in range(4)]
    col_centers = [x + col_w/2 for x in cols_x]

    # --- Common (full-width) layers, top to bottom ---
    # 1. Application
    block(ax, 0.25, 16.6, full_w, 0.9,
          "Application\nONNX / PyTorch / TensorFlow model "
          "(e.g. smolvlm_vision.onnx after MatMul→Conv1×1 rewrite + v3 slicing)",
          C_APP, fontsize=12, fontweight="bold")

    # 2. Frontend converter
    block(ax, 0.25, 15.4, full_w, 0.9,
          "Frontend conversion  —  snpe-onnx-to-dlc  →  fp32 DLC\n"
          "qairt-quantizer (calibration data → int8 DLC)",
          C_IR, fontsize=12, fontweight="bold")

    # 3. DLC IR
    block(ax, 0.25, 14.2, full_w, 0.9,
          "Vendor IR  —  DLC (Deep Learning Container)\n"
          "typed QNN op-graph: QnnOp_t + Qnn_Tensor_t  (sequential, AOT)",
          C_IR, fontsize=12, fontweight="bold")

    # 4. Runtime API
    block(ax, 0.25, 13.0, full_w, 0.9,
          "Runtime API  —  libQnnInterface.so + libQnnSystem.so + libQnnModelDlc.so\n"
          "contextCreateFromBinary  ·  graphRetrieve  ·  graphExecute  ·  contextFree",
          C_RT, fontsize=12, fontweight="bold")

    # --- Per-backend columns ---
    headers = ["CPU\n(Kryo 585)", "GPU\n(Adreno 650)",
               "HTA\n(Hexagon Tensor Accel)", "DSP / cDSP\n(Hexagon 698, HVX)"]
    for x, h in zip(cols_x, headers):
        block(ax, x, 11.5, col_w, 0.7, h, "#ffffff",
              ec="black", fontsize=13, fontweight="bold", lw=1.5)

    # 5. Finalizer (per-backend)
    finalizers = [
        "qnn-context-binary-generator\n--backend libQnnCpu.so\n→ .bin (thin wrapper)",
        "qnn-context-binary-generator\n--backend libQnnGpu.so\n→ .bin (embeds OpenCL kernels)",
        "qnn-context-binary-generator\n--backend libQnnHta.so\n→ .bin (HTA bytecode + NHWC4)",
        "qnn-context-binary-generator\n--backend libQnnDsp.so\n→ .bin (Hexagon skel state)",
    ]
    for x, t in zip(cols_x, finalizers):
        block(ax, x, 10.3, col_w, 1.0, t, C_TOOL, fontsize=10)

    # 6. Backend stub (host)
    stubs = [
        "libQnnCpu.so\n(host-side dispatch into\nref C++ kernels)",
        "libQnnGpu.so\nhost → OpenCL queue\n→ Adreno driver",
        "libQnnHta.so\nFastRPC stub\n→ HTA via QURT",
        "libQnnDsp.so\nFastRPC stub\n→ cDSP via QURT",
    ]
    for x, t in zip(cols_x, stubs):
        block(ax, x, 9.0, col_w, 1.05, t, C_STUB, fontsize=10.5)

    # 7. Op-package / kernels
    kernels = [
        "Built-in QNN ref kernels\n(C++, some ARM NEON)\nall ops supported",
        "QNN OpenCL kernels\nper op family\nauto-tuned per Adreno gen",
        "HTA op-package (firmware)\nclosed-source bytecode\nlimited op set:\nConv / Pool / Add / Sigmoid",
        "libQnnDspOpPackage.so\nHVX/HMX intrinsic kernels\nOP_PACKAGE_NOT_FOUND ←\nmissing op in v66:\nPow, Reciprocal, ScatterND…",
    ]
    for x, t in zip(cols_x, kernels):
        block(ax, x, 7.0, col_w, 1.8, t, C_KER, fontsize=10)

    # 8. Compiler / JIT
    compilers = [
        "(no runtime compile)\nshipped pre-built\nfor aarch64",
        "Adreno OpenCL JIT\nLLVM-based shader\ncompiler (proprietary)",
        "(no runtime compile)\nfinal bytecode embedded\nin .bin",
        "(no runtime compile)\nhexagon-clang used at\nop-package build time",
    ]
    for x, t in zip(cols_x, compilers):
        block(ax, x, 5.5, col_w, 1.25, t, C_COMP, fontsize=10.5)

    # 9. Lower IR
    lower_ir = [
        "LLVM IR → AArch64 asm",
        "OpenCL → Adreno LLVM IR\n→ Adreno IL",
        "HTA bytecode\n(undocumented)",
        "LLVM IR → Hexagon asm\nHVX intrinsics + HMX",
    ]
    for x, t in zip(cols_x, lower_ir):
        block(ax, x, 4.3, col_w, 1.0, t, C_IR2, fontsize=10.5)

    # 10. ISA
    isas = [
        "AArch64 / Armv8.2-A",
        "Adreno A640 ISA",
        "HTA fixed-function\n(accelerator-private)",
        "Hexagon v66 + HVX\n(1024-bit vector,\n4 vector contexts)",
    ]
    for x, t in zip(cols_x, isas):
        block(ax, x, 3.1, col_w, 1.0, t, C_HW, fontsize=10.5)

    # 11. Hardware
    hardware = [
        "Cortex-A77 (4) + A55 (4)\n8 cores @ 2.84/1.8 GHz",
        "Adreno 650\n≈1 TFLOPS fp16\n344 GOPS int8 (uchar4+mad24)",
        "HTA accelerator\nint8 Conv-centric\nfirmware cap: ~32 simul ctx",
        "Hexagon 698 cDSP\n~5 TOPS int8 nominal\nfirmware cap: ~30 simul,\n~45 cumulative (leaky)",
    ]
    for x, t in zip(cols_x, hardware):
        block(ax, x, 1.6, col_w, 1.3, t, "#a6a6a6", fontsize=10.5,
              fontweight="bold")

    # --- Arrows: vertical down the common stack ---
    for y0, y1 in [(16.6, 16.3), (15.4, 15.1), (14.2, 13.9)]:
        arrow(ax, 8, y0, 8, y1)

    # Arrow from runtime API down to the dispatcher box
    arrow(ax, 8, 13.0, 8, 12.75)

    # Dispatcher box — sits *above* the fan-out arrows, white background so
    # it overlays the arrow trunk without crossings.
    block(ax, 4.5, 12.20, 7.0, 0.55,
          "qnn-context-binary-generator  —  dispatches to a backend-specific path",
          "#ffffff", ec="black", fontsize=11, fontweight="bold", lw=1.2)

    # Arrows from the dispatcher box to each column header
    for cx in col_centers:
        arrow(ax, 8, 12.20, cx, 12.2)

    # Vertical arrows within each column
    for cx in col_centers:
        for y0, y1 in [(11.5, 11.3), (10.3, 10.1), (9.0, 8.8),
                       (7.0, 6.8), (5.5, 5.3), (4.3, 4.1),
                       (3.1, 2.9)]:
            arrow(ax, cx, y0, cx, y1)

    # --- Right-margin tier labels ---
    tier_labels = [
        (17.05, "Source"),
        (15.85, "Frontend"),
        (14.65, "Vendor IR"),
        (13.45, "Runtime API"),
        (11.85, "Backend"),
        (10.8,  "Finalizer / AOT"),
        (9.5,   "Host stub"),
        (7.9,   "Library / op-package"),
        (6.1,   "Compiler / JIT"),
        (4.8,   "Lower IR"),
        (3.6,   "ISA"),
        (2.25,  "Hardware"),
    ]
    for y, lbl in tier_labels:
        ax.text(15.85, y, lbl, ha="left", va="center",
                fontsize=9, color="#666", style="italic")

    ax.set_title("QNN SDK on QRB5165 v66 — software stack across the four backends",
                  fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    out = REPO / "plots/qnn_sdk_stack.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

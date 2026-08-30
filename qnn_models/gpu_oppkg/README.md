# `flowc.gpu` — an int8 OpenCL op package for the Adreno 650

The QNN GPU backend on the QRB5165 runs float graphs and refuses quantized
ones. This directory contains a QNN GPU op package that supplies the missing
8-bit kernels in OpenCL, the harness that validates them against the CPU and
DSP backends, and the measurements.

Headline, all measured on the board at `10.44.120.201` (QAIRT 2.45, Adreno 650):

* **`mlp_control` int8 and `dronet` int8 now run end to end on the GPU.**
  Neither composes at all with the stock package.
* **The int8 kernels are numerically right.** `mlp_control`'s int8 output is
  bit-exact against a double-precision emulation of QNN's quantization
  semantics on all 8 test inputs (0 LSB error, 100 % exact), and within 2 LSB
  of the Hexagon DSP. On `dronet`, our GPU output tracks the CPU int8
  reference at least as closely as Qualcomm's own DSP kernels do, at every one
  of the 21 intermediate tensors.
* **The GPU is not the fastest place to put these networks.** `dronet` int8:
  GPU 3.01 ms vs DSP 0.73 ms vs CPU 2.58 ms. The value here is coverage — a
  lane that previously could not run int8 at all — not peak throughput.

---

## 1. What was missing

Full detail with logs in [`INVENTORY.md`](INVENTORY.md) /
[`inventory.json`](inventory.json). The short version: the failure is not a
few ops, it is the datatype. The SDK's own GPU op-definition supplement
documents 101 ops for this backend and `UFIXED_POINT_8` appears in exactly one
of them (as `Dequantize`'s input). Every int8 network therefore dies on its
first node:

| Network | int8 stock | fp32 | fp16 | int8 with this package |
|---|---|---|---|---|
| mlp_control | ✗ `/mlp/mlp.0/Gemm` | ✓ | ✓ | **✓ whole net** |
| dronet | ✗ `/conv_modules.0/Conv` | ✓ | ✓ | **✓ whole net** |
| yolov8n | ✗ `pad_0` | ✓ | ✓ | ✗ (Pad, Concat, StridedSlice, Resize, Softmax still missing) |
| fused_full | ✗ `/vision_cnn/…/Conv` | ✓ | — | ✗ stops at `Concat`; also needs Lstm, Convert |
| fused_split vision_conv tile | ✗ | ✓ | — | **✓ tile runs** |

A second measured constraint shaped everything: **a custom op package cannot
take over nodes that a DLC composed**, because the GPU backend dispatches on
`Qnn_OpConfig_t::packageName` and the DLC path hard-codes `qti.aisw`.
Registering our package under that same name is rejected (`Duplicate operation
name: FullyConnected`). The working route is the converter's C++ model output
with the per-node package name rewritten — same weights, same encodings, one
string changed. `tools/make_cut_model.py` and the `sed` in §4 do that.

## 2. What is implemented

| QNN op type | int8 (uFxp8 / sFxp8) | fp32 / fp16 | Validated against |
|---|---|---|---|
| `FullyConnected` | ✓ | ✓ | CPU + DSP + fp64 emulation (exact) |
| `ElementWiseNeuron` (ELU, ReLU, ReLU1/6, ReluMinMax, Sigmoid, Tanh, HardSwish, HardSigmoid), and the standalone `Relu`/`Relu6`/`Sigmoid`/`Tanh`/`Elu`/`HardSwish` types | ✓ | ✓ | fp64 emulation (256/256 exact), DSP (≤2 LSB) |
| `Conv2d`, `DepthWiseConv2d` | ✓ | — (stock covers float) | CPU (95.3 % exact, max 1 LSB) |
| `PoolMax2d` | ✓ | — | in-network on dronet |
| `Batchnorm` | ✓ | — | in-network on dronet |
| `ElementWiseBinary` / `Add` / `Multiply` / `Subtract` (with broadcasting) | ✓ | — | in-network on dronet |
| `Reshape`, `Transpose` | ✓ | — | in-network on dronet (bit-exact copies) |

Deliberately refused rather than guessed: per-axis (per-channel) weight
encodings, `group` values other than 1 or depthwise, GELU/Softplus (several
in-the-wild definitions), `PoolAvg2d`, and any tensor the backend hands us as
image memory instead of a buffer. Each of those returns
`QNN_OP_PACKAGE_ERROR_UNSUPPORTED_FEATURE` with a log line, so a graph fails
loudly instead of computing something plausible and wrong.

### Kernel design, and why (Adreno 650 specifics)

* **Per-node specialised source.** Shapes, quantization offsets, scales and
  activation tables are emitted as literals into the OpenCL source at
  graph-build time, not passed as arguments — the backend compiles each node's
  program once at finalize, so the constant folding is free at run time.
* **Integer accumulation.** All int8 reductions accumulate in `int`/`int4`,
  never float: `(q+offset)` products are exact in int32 for every shape in
  these networks, and that is what makes bit-exact agreement with the CPU
  reference possible. A float accumulator starts dropping integers past 2²⁴.
* **`FullyConnected`** is a GEMV here (batch 1, K,N ∈ {16…256}), so a tiled
  GEMM would leave the machine idle: one work item per output element, K-loop
  unrolled by 4 over `vload4` (a uchar4 load is a dword — Adreno's natural
  byte-load granularity), local size 64 = one wave.
* **`ElementWiseNeuron` at int8 is a table lookup.** The activation of a byte
  has only 256 possible answers, so the host evaluates the function 256 times
  in double precision and bakes the result into the kernel as a `__constant`
  table. One byte load, one cached table read, one byte store per element — no
  `exp`/`tanh` on the ALU at all — and the result is the correctly rounded
  value of the reference formula by construction. The same kernel therefore
  serves ELU, ReLU, Sigmoid, Tanh and HardSwish identically; only the table
  changes.
* **`Conv2d`** decomposes as (Cout/4, Wout, Hout·N): one work item owns four
  output channels of one output pixel. That makes the filter read a contiguous
  `vload4` (the QNN filter layout is `[Kh][Kw][Cin][Cout]`, so Cout is the
  fastest axis) and shares each activation load across four MACs. Work-group
  8×8×1 = one wave; neighbours share filter rows and overlapping input patches,
  so the 32 KB L1 does the reuse that a `__local` tile would otherwise need a
  barrier for. Padding is skipped, not materialised — for a quantized tensor
  the pad value is the zero point, so `(q+offset) == 0` and an out-of-range tap
  contributes exactly nothing.
* **Buffers, not images.** Every kernel indexes plain `__global` buffers, and
  every op *claims* its NATIVE outputs as linear buffers
  (`QnnGpu_OutputClaim_t`). This is not a preference: an unclaimed int8 NATIVE
  tensor fails allocation in the GPU backend — graph finalize dies with
  `GPU_ERROR_INVALID_TYPE(10012) "Tensor memory error"` before a kernel runs,
  because Adreno has no image channel format for 8-bit fixed point. If a tensor
  still arrives as image memory the op refuses it.

### Two backend behaviours worth knowing (both cost real debugging time)

1. **The program cache is keyed on the kernel name.** Two nodes whose kernels
   share a name run the *same compiled program* — whichever was compiled first.
   With per-node specialised source this is silent, catastrophic and
   deterministic: layers 2/4/6 of `mlp_control` executed layer 0's arithmetic
   (K=16 instead of 256, layer 0's scales) on their own buffers. Predicting the
   wrong outputs under that model reproduced the observed bytes at 100 % on all
   three layers, which is how it was identified. Every kernel now carries an
   FNV-1a hash of its own source in its name (`uniqueKernelName`).
2. **An OpenCL build error is reported as a graph-finalize failure.** A `1f`
   literal (a `double` printed with default stream formatting, then suffixed)
   produced `Could not build cl::Program (-11)` and then error 6022. Float
   literals now go through `litf()`.

## 3. Building

Host-side syntax check (any x86 box with the SDK):

```bash
cd qnn_models/gpu_oppkg
make QNN_SDK_ROOT=/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326 BUILD_DIR=build_host
```

On the board (native g++ 9.4, aarch64-oe-linux — no cross toolchain needed):

```bash
tar cz src Makefile | ssh root@10.44.120.201 \
  "flock -w 900 /tmp/qnn_board.lock -c 'mkdir -p /root/gpu_oppkg/pkg && cd /root/gpu_oppkg/pkg && tar xz && make QNN_SDK_ROOT=/root/qairt'"
# -> /root/gpu_oppkg/pkg/build/libQnnGpuOpPackageFlowC.so
```

The timing/validation harness (`runtime/opbench.cpp`) builds the same way:

```bash
g++ -std=c++11 -O2 -pthread -I/root/qairt/include/QNN opbench.cpp -o opbench -ldl -lpthread
```

`-pthread` is not optional — the GPU backend spawns threads and aborts with
`Enable multithreading to use std::thread` otherwise.

## 4. Running a network through the package

```bash
# 1. converter -> C++ model (quantized), on the host, in the qnn-convert image
python3.10 $QNN/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network dronet_simplified.onnx --input_list calib_list.txt \
    --input_layout input NCHW --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8 \
    -o model/dronet_ref.cpp

# 2. point the nodes at our package (weights/.bin unchanged)
tools/make_flowc_model.sh model/dronet_ref.cpp     # -> model/dronet_flowc.{cpp,bin}

# 3. build the model library on the board with the SDK's converter/jni sources
bash /root/gpu_oppkg/build_model.sh dronet_flowc

# 4. run it on the GPU with the package registered
./opbench modellib_dronet_flowc/libs/aarch64-ubuntu-gcc9.4/libdronet_flowc.so \
          /root/qairt/lib/target/libQnnGpu.so 50 \
          --op-package /root/gpu_oppkg/pkg/build/libQnnGpuOpPackageFlowC.so:FlowCGpuOpPackage_interfaceProvider \
          --input input=io/dronet_q0.raw --dump-dir dump_dr_gpu
```

`qnn-net-run` works the same way with `--op_packages <lib>:FlowCGpuOpPackage_interfaceProvider`;
add `--debug --use_native_input_files --use_native_output_files` to dump every
intermediate tensor as raw int8, which is how the per-layer tables below were
produced.

Reproduce the accuracy tables from the dumps kept in `validation/out/`:

```bash
python3 tools/emulate_mlp_int8.py --compare validation/out/out_gpu_fix     # 100.00%, 0 LSB
python3 tools/emulate_mlp_int8.py --compare validation/out/out_dsp_ref     #  78.12%, 2 LSB
python3 tools/emulate_mlp_int8.py --compare validation/out/out_native_Cpu  #   3.12%, 201 LSB + warning
python3 tools/compare_dumps.py validation/out/dbg_dr_gpu/Result_0 \
        validation/out/dbg_dr_cpu/Result_0 --label-a GPU --label-b CPU \
        --order model/dronet_ref_net.json
```

## 5. Validation — what the kernels actually compute

Reference chain: the QNN **CPU** backend (`qti.aisw` int8 kernels) and the
**DSP/HVX** backend on the identical graph and identical input bytes, plus a
double-precision NumPy emulation of QNN's quantization semantics
(`real = (q + offset) · scale`).

**`mlp_control` int8, 8 distinct inputs, final output (uint8):**

| comparison | exact | max abs err |
|---|---|---|
| **GPU (this package) vs fp64 emulation** | **100.00 %** | **0 LSB** |
| GPU vs DSP (HVX int8) | 78.1 % | 2 LSB (0.24 dequantized) |
| DSP vs fp64 emulation | 78.1 % | 2 LSB |
| CPU int8 vs fp64 emulation | 3.1 % | 201 LSB |

That last row is not our bug and it matters: **the QNN CPU backend's int8 path
for this graph is broken.** It returns the *same* output for all 8 distinct
inputs, and dumping intermediates shows every `ElementWiseNeuron` output as
0xFF. The DSP and our GPU both vary correctly with the input. (Consistent with
`boards/qrb5165_v66/benchmark_results.json`, where `CPU_int8` is `null` for
these nets.) So for `mlp_control` the CPU is *not* a usable reference and the
DSP is.

**Two-op graph (`FullyConnected` + `ELU`), full 256-element activation** — the
model truncated after node 1 by `tools/make_cut_model.py`, so this isolates the
two kernels from any downstream accumulation:

| comparison | exact | max abs err |
|---|---|---|
| **GPU vs fp64 emulation** | **100.00 % (256/256)** | **0 LSB** |
| DSP vs fp64 emulation | 82.0 % | 2 LSB |

**`fused_split/vision_conv` tile (4 int8 convolutions), 1536 output bytes:**

| comparison | exact | max abs err |
|---|---|---|
| GPU vs CPU int8 | 95.3 % | 1 LSB |
| GPU vs DSP int8 | 91.3 % | 2 LSB |
| DSP vs CPU int8 | 90.2 % | 2 LSB |

Our conv is *closer* to the CPU reference than the vendor's DSP kernel is.

**`dronet` int8, per intermediate tensor** (GPU vs CPU, next to DSP vs CPU on
the same tensors — the error growth with depth is int8 rounding, not a bug):

| tensor | n | GPU=CPU | max | DSP=CPU | max |
|---|---|---|---|---|---|
| conv_modules.0 Conv | 100352 | 99.86 % | 1 | 99.83 % | 1 |
| maxpool1 | 23328 | 99.84 % | 1 | 99.79 % | 1 |
| relu_modules.1 | 6272 | 98.09 % | 1 | 95.26 % | 2 |
| Add | 6272 | 94.53 % | 1 | 85.83 % | 2 |
| relu_modules.3 | 3136 | 83.51 % | 3 | 73.18 % | 3 |
| Add_1 | 3136 | 77.36 % | 2 | 65.24 % | 2 |
| Add_2 | 2048 | 55.71 % | 3 | 47.31 % | 3 |
| Transpose(nchw) / Flatten | 2048 | 55.71 % | 3 | 47.31 % | 3 |
| relu_modules.6 | 2048 | 77.20 % | 4 | 73.63 % | 6 |
| `steer` (scalar) | 1 | — | 6 | — | 11 |
| `collision` (scalar) | 1 | — | 1 | — | 2 |

At every tensor our GPU output is at least as close to the CPU reference as the
DSP's is; the Transpose and Reshape rows are byte-identical to their input, as
they must be.

**`dronet` end outputs over 5 real calibration frames** (uint8; CPU int8 as
reference, which is trustworthy for this graph — unlike mlp_control, its output
does vary with the input):

| | `steer` mean / max err | `collision` mean / max err |
|---|---|---|
| GPU (this package) vs CPU | **4.4 / 7 LSB** (0.016 of a ~0.58 range) | 0.8 / 1 LSB |
| DSP (HVX) vs CPU | 5.4 / 11 LSB (0.025) | 1.0 / 2 LSB |

Raw bytes in `validation/out/o5_{gpu,cpu,dsp}/`. Note this is int8-vs-int8
disagreement, not accuracy against float: all three are quantized approximations
of the same fp32 network, and they round differently at each of dronet's 22
requantization points.

**Float kernels** (same graph, our package vs the stock GPU package):

| | max abs err vs stock GPU |
|---|---|
| fp32 FullyConnected+ELU | 1.9e-6 (relative 9.3e-7) |
| fp16 FullyConnected+ELU | 1.95e-3 (1 ULP at that magnitude) |

## 6. Performance

`runtime/opbench.cpp`, wallclock around `QnnGraph_execute`, warm-up excluded —
the same measurement `qnn_models/runtime/profile_segments.cpp` takes, so these
are comparable to `flow_c/measurements/qrb5165_v66.json`, with two caveats
stated below. Median of 200 (mlp) / 100 (tile) / 50 (dronet) iterations, two
independent repetitions.

| graph | GPU int8 (this package) | DSP int8 | CPU int8 | flow_c cell for reference |
|---|---|---|---|---|
| `mlp_control` (7 ops) | **258 / 263 µs** | 396 / 408 µs | 30 µs (wrong numerically) | dsp 403.8, cpu 28.5 |
| `fused_split/vision_conv` (4 convs) | **812 / 955 µs** | 447 / 453 µs | 163 / 165 µs | hta 931, dsp 459.4, cpu 12179.1 |
| `dronet` (22 ops) | **3007 / 3018 µs** | 725 / 762 µs | 2576 / 2618 µs | hta 2030.4, dsp 645.4, cpu(fp32) 6998.2 |

Caveats, stated rather than smoothed over:

* These were taken under the board's **`schedutil`** governor (the brief forbids
  changing it), while the `flow_c` cells were taken under `performance`; the
  flow_c notes measure that as a 10–36 % pessimism on accelerator cells. The
  DSP columns above (396–453 µs vs cells of 404–459) show the two paths agree
  closely for DSP anyway.
* They go through the **model-library** path (compose + finalize at run time),
  not a pre-built context binary, because a context binary cannot carry ops
  from a package the loader has not registered. Compose+finalize is *not* in
  the numbers, but it is large for the GPU: 0.16 s for `mlp_control`, 0.57 s
  for `dronet`, dominated by OpenCL program builds — a one-off bring-up cost
  the DSP (0.02–0.03 s) does not pay.
* Our CPU int8 number for the vision_conv tile (164 µs) is far below the
  flow_c cell (12179 µs) for the same tile. Different artifact and different
  path (converter model-lib here, sub-DLC + context binary there); I did not
  chase the discrepancy and do not claim to explain it.

Float, same `mlp_control` graph on the GPU: stock package 170 µs (fp32) /
229 µs (fp16), ours 1222 µs / 1226 µs. **Our float kernels are ~7× slower than
Qualcomm's.** They exist so the package is complete and to cross-check the int8
path; for float graphs the stock package is the right choice.

Where the GPU int8 path is worth taking, on these numbers: it is faster than
the DSP for the tiny MLP (258 vs 400 µs — both are dispatch-bound), and slower
for everything convolutional. What it buys the scheduler is a *fourth lane*
that can take int8 work at all, which the registry previously had to exclude.

## 7. What I could not make work, and why

* **`yolov8n` int8 on GPU.** Needs Pad, Concat, StridedSlice, Resize and
  Softmax at int8 on top of what is here. Worse, this network has no working
  ONNX→QNN converter route on 2.45 (documented in `qnn_models/README.md`; the
  in-tree DLC came via TFLite), and the package-rename trick needs the
  converter's C++ output — so even with the kernels written, the artifact route
  would have to be rebuilt first. The gap list above is derived from the DLC's
  op census, not measured end to end; that is flagged in `inventory.json`.
* **`fused_full` int8 on GPU.** Measured: the six convolutions and six reshapes
  compose and build, then it stops at `Concat`
  (`Operation does not exist: flowc.gpu Concat`). Beyond Concat it also needs
  `Lstm` (three of them) and `Convert` (uFxp8↔sFxp16). Concat is an afternoon;
  a correct quantized LSTM is not. The `vision_conv` tile — the piece the
  scheduler actually wants to place — does work.
* **Speed.** The conv kernel is a direct convolution with no `__local` tiling,
  no im2col/GEMM path and no channel blocking beyond vec4. On dronet it lands
  4× off the DSP. I chose breadth-plus-correctness over tuning; the honest
  statement is that this package makes the GPU *usable* for int8, not
  competitive with the DSP on convolutions.
* **Per-channel weight quantization** is rejected, not implemented. Every DLC
  in this repo uses per-tensor encodings, so nothing here needed it, but a
  `--use_per_channel_quantization` model would be refused at validate.
* **Context binaries.** The GPU int8 graphs cannot be cached into a context
  binary for the existing `profile_seg` harness, because the op package must be
  registered before the context exists. That is why `opbench` composes from a
  model library instead.
* **The debug hooks are still in the source** (`FLOWC_FC_MODE`,
  `FLOWC_DEBUG_CONST`, `FLOWC_CLAIM_DT`, `FLOWC_PROBE_OFF`). They are how the
  program-cache bug was found — `probe_w`/`probe_in` dump the raw device
  buffers an op sees — and they are documented in the source. Default behaviour
  is the real kernel.

## 8. File map

```
gpu_oppkg/
├── src/                    the op package (C++11, no SDK example code vendored)
│   ├── FlowCGpuCommon.{hpp,cpp}   accessors, quant helpers, output claims, kernel naming
│   ├── FlowCGpuPackage.cpp        QnnOpPackage interface provider + op registry
│   ├── OpFullyConnected.cpp       int8 GEMV + fp32/fp16
│   ├── OpNeuron.cpp               int8 LUT activations + fp32/fp16
│   ├── OpConv2d.cpp               int8 Conv2d / DepthWiseConv2d
│   └── OpMisc.cpp                 int8 PoolMax2d, Batchnorm, ElementWiseBinary, Reshape, Transpose
├── Makefile                board/host build of libQnnGpuOpPackageFlowC.so
├── runtime/opbench.cpp     compose-from-model-lib + register op package + time + dump
├── tools/
│   ├── census.py               op census from qairt-dlc-to-json output
│   ├── make_flowc_model.sh     rewrite a model .cpp onto the flowc.gpu package
│   ├── make_cut_model.py       truncate a model .cpp after node N (bisection)
│   ├── emulate_mlp_int8.py     fp64 reference for mlp_control int8 + diff a run
│   └── compare_dumps.py        per-tensor int8 diff of two --debug dumps
├── inventory.json, INVENTORY.md, inventory/   the measured gap + logs + per-DLC JSON
├── model/                  converter output per network (`*_ref.*`; the `*_flowc.*`
│                          twins are derived -- regenerate with make_flowc_model.sh)
└── validation/             fixed inputs, and the raw outputs every table above came from
```

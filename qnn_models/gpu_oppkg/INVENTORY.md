# What the Adreno 650 GPU backend cannot run, measured

Every row below is the exit status of `qnn-context-binary-generator --backend
libQnnGpu.so` on the board at `10.44.120.201`, QAIRT 2.45. The failure text is
copied from the run's own log; the raw lines are in
[`inventory/logs/gpu_compose_logs.txt`](inventory/logs/gpu_compose_logs.txt),
the machine-readable form is [`inventory.json`](inventory.json).

## Compose matrix

| Network | int8 (stock GPU pkg) | fp32 | fp16 | int8 with `flowc.gpu` op package |
|---|---|---|---|---|
| `mlp_control` | ✗ `/mlp/mlp.0/Gemm` | ✓ | ✓ ¹ | **✓ whole network** |
| `dronet` | ✗ `/conv_modules.0/Conv` | ✓ | ✓ | **✓ whole network** |
| `yolov8n` | ✗ `pad_0` | ✓ | ✓ | ✗ (needs Pad, Concat, StridedSlice, Resize, Softmax) ² |
| `fused_full` (FusedSensorNet) | ✗ `/vision_cnn/vision_cnn.0/Conv` | ✓ | — ³ | ✗ — stops at `_Concat`; also needs Lstm, Convert |
| `fused_split/vision_conv` tile | ✗ | ✓ | — | **✓ tile composes and runs** |

¹ no fp16 DLC for `mlp_control` exists in-tree; measured through a
converter-produced fp16 model library instead.
² derived from the op census of the quantized DLC, not measured end to end:
yolov8n has no working ONNX→QNN route on this SDK (see `qnn_models/README.md`),
so no package-renamed model library could be built for it.
³ no fp16 artifact exists for `fused_full`.

The int8 failure is always the *first* op of the graph, and always the same
pair of errors:

```
GPU ERROR: GPU_ERROR_INVALID_TYPE(10012)
GPU ERROR: GPU_ERROR_OP_PACKAGE_FAILED(10023)
   - OpPackage (qti.aisw) validation failure for operation <first op>
```

## What that actually means

It is not a handful of missing ops. The SDK's own GPU op-definition supplement
(`docs/QAIRT-Docs/QNN/OpDef/GpuOpDefSupplement.html`) documents 101 ops for this
backend; `QNN_DATATYPE_UFIXED_POINT_8` appears in exactly one of them, as the
*input* of `Dequantize`. The stock GPU package is a float-only package, so
**every** op of an int8 graph has to be supplied by an op package — which is why
this directory exists.

`benchmark_results.json` already showed GPU and GPU_fp16 end-to-end numbers for
dronet/yolov8n and nothing for int8; that is confirmed here from the compose
logs rather than inferred.

## Op census per network (int8)

From `qairt-dlc-to-json` on each quantized DLC (`inventory/*.qnn.json`,
summarised by `tools/census.py` into `inventory/census.json`):

| Network | nodes | op types |
|---|---|---|
| `mlp_control` | 7 | FullyConnected ×4, ElementWiseNeuron(ELU) ×3 |
| `dronet` | 29 (22 after converter fusion) | Conv2d ×10, ElementWiseNeuron ×8, Batchnorm ×3, ElementWiseBinary ×3, FullyConnected ×2, Pool ×1, Reshape ×1, Transpose ×1 |
| `fused_full` | 30 | Conv2d ×6, ElementWiseNeuron ×6, Reshape ×6, FullyConnected ×3, Lstm ×3, Convert ×3, Transpose ×2, Concat ×1 |
| `yolov8n` | 272 | ElementWiseBinary ×69, Conv2d ×64, ElementWiseNeuron ×58, Transpose ×25, StridedSlice ×18, Concat ×17, Reshape ×8, Pad ×7, Pool ×3, Resize ×2, Softmax ×1 |

## The dispatch rule that shapes everything else

A custom GPU op package **cannot take over a node that a DLC composed**.
Measured:

* Registering `flowc.gpu` (implementing `FullyConnected`) and composing
  `mlp_control_quantized.dlc` on the GPU: the package loads, prints its device
  banner, and is *never consulted* — the log still ends in
  `OpPackage (qti.aisw) validation failure for operation /mlp/mlp.0/Gemm`.
  Dispatch is by `Qnn_OpConfig_t::packageName`, which the DLC path hard-codes to
  `qti.aisw`.
* Registering the same package *as* `qti.aisw` is refused outright:
  `GPU_ERROR_OP_PACKAGE_FAILED(10023) - Duplicate operation name: FullyConnected`.

The route that does work — and the one every measurement in this directory uses
— is the converter's C++ model output (`qnn-onnx-converter -o model.cpp`), whose
per-node `packageName` string is rewritten to `flowc.gpu` before the model
library is compiled. Same weights (`.bin` is byte-identical), same
quantization encodings, one string changed.

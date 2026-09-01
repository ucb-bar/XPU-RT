# The 12 lone-Tanh vision trampolines: settled by measurement

Answers the experiment proposed in `REPRODUCTION.md` §17.3. Measured 2026-09-01
on QRB5165 v66 (`root@10.44.120.201`), QAIRT 2.45.0.260326, all 8 CPU cores on
`performance` (saved and restored to `schedutil` afterwards).

## Verdict

**A lone `Tanh` composes and runs on both DSP and HTA. The registry is right; the
carve-out is stale.** It was never a Tanh problem: the op the v66 DSP and HTA
reject is the *fused GELU* the converter builds out of the whole
`Pow→Mul→Add→Mul→Tanh→Add→Mul→Mul` cluster — `ElementWiseNeuron` with
`operation = 1` (GELU), not `operation = 8` (TANH). Reproduced verbatim below.

`qnn_models/flow_c/cores/qrb5165_qnn.json` needs **no correction**. It already
says exactly the right thing on all four cores:

| core   | `tanh` | `tanh_s8` | `gelu` | `gelu_s8` |
|--------|--------|-----------|--------|-----------|
| hta0   | yes    | yes       | no     | no        |
| dsp0   | yes    | yes       | no     | no        |
| cpu0   | yes    | yes       | no     | yes       |

Both halves of that table are now confirmed on the board.

## What the trampolines actually are

§17 states the lone-Tanh trampolines are `[1,12,1024,1024]`. They are not — that
is the *attention* trampoline shape. All 12 lone-Tanh segments
(`vision_slices_v3/cpu_seg_{01,03,...,23}.onnx`) are a single `Tanh` at
**`[1,1024,3072]`** = 3.146 M elements, one quarter the probe's 12.58 M. The
archived CPU costs sum to 109.99 ms, so the 110.0 ms headline is right even
though the shape quoted for it is not.

## 1. The probe: does a lone Tanh compose?

`expert_rewrite/tanh_probe.onnx`, `[1,12,1024,1024]`, converted with
`--input_layout x NCHW` (DLC declares `[1,1024,1024,12]`), quantized int8 on 4
float32 random-normal calibration raws.

`qnn-context-binary-generator` returned **rc=0 on both `libQnnDsp.so` and
`libQnnHta.so`** (context binaries 1 928 B and 33 565 376 B). Context creation
and execution both succeed, so the DSP's tiny binary is not a hollow compose.

`profile_seg`, 20 iters, `--gap-us 3000`, milliseconds:

| backend | precision | loop median | gap median |
|---------|-----------|-------------|------------|
| CPU     | fp32      | 36.470      | 36.510     |
| CPU     | int8      |  8.477      |  8.512     |
| DSP     | int8      | 13.763      | 13.447     |
| HTA     | int8      | 32.223      | 31.467     |

The CPU-fp32 figure divided by 4 is 9.12 ms — within 0.6 % of the 9.17 ms the
real trampolines cost, which is a clean cross-check that the trampolines are
memory-bound fp32 `tanhf` and that the probe scales.

## 2. The real trampoline: `vision_slices_v3/cpu_seg_01.onnx`

Converted `-d val_371 1,1024,3072 --input_layout val_371 NONTRIVIAL`, quantized
int8. Composed rc=0 on DSP, HTA and CPU. `profile_seg`, 50 iters, `--gap-us
3000`, milliseconds:

| backend | precision | loop median | gap median | ctx size |
|---------|-----------|-------------|------------|----------|
| CPU (shipped `ctx_cpu_seg_01__Cpu.bin`) | fp32 | 9.153 | **9.171** | 1 516 B |
| CPU (rebuilt from the same ONNX)        | fp32 | 9.140 |   9.143   | 1 528 B |
| CPU                                     | int8 | 2.066 | **2.408** | 1 580 B |
| DSP                                     | int8 | 3.289 | **5.086** | 1 936 B |
| HTA                                     | int8 | 4.862 | **5.286** | 3 152 576 B |

The rebuilt fp32 CPU number reproduces the archived 9168.93 µs to 0.3 %, so the
`performance` governor is not what makes the accelerated numbers look good.

**Numerics** (`qnn-net-run`, real board, input
`profile_inputs/cpu_seg_01_val_371.raw`, outputs dequantized to fp32):

| comparison            | cosine     | RMSE      | max abs   |
|-----------------------|------------|-----------|-----------|
| DSP int8 vs CPU fp32  | 0.99982041 | 2.66e-3   | 7.46e-3   |
| CPU int8 vs CPU fp32  | 0.99998081 | 1.67e-3   | 4.06e-3   |
| DSP int8 vs CPU int8  | 0.99994069 | 2.76e-3   | 3.91e-3   |

Output range is ±0.47 with an int8 step of 3.91e-3, so DSP lands within ~2 LSB
of the fp32 reference — this is int8 quantization error, not a broken kernel.

## 3. Better than moving it: fold it into the DSP dispatch that precedes it

Every one of `dsp_seg_01..23` (the odd ones are exactly the 12 that precede a
lone Tanh) is placed as `HTA-bundle-DSP`, whose last phase
`dsp_seg_XX_tramp_p2` already runs on DSP and already emits `val_371` — the
tensor the CPU Tanh consumes. Appending the `Tanh` to that phase deletes the
trampoline dispatch outright and keeps `val_371` on the DSP.

Built `dsp_seg_01_tramp_p2 + Tanh` (`Reshape,Transpose,Pow,Mul,Add,Mul,Tanh`,
outputs `val_364` + `val_372` instead of `val_364` + `val_371` — identical
output bytes), quantized on the existing `trampolines/calibration/` raws. The
Tanh stayed a **separate** `ElementWiseNeuron operation: 8`; the GELU fusion did
**not** re-fire, because the trailing `Add→Mul→Mul` still lives in `dsp_seg_02`.
Composed rc=0 on DSP.

A/B against the shipped `ctx_dsp_seg_01_tramp_p2__Dsp.bin`, 4 alternating pairs
of 60 iters with 3 s settles. The board showed intermittent DSP interference
(occasional medians of 190–400 ms), so `min_us` is the honest statistic here:

| run | base `min` | +Tanh `min` |
|-----|-----------|-------------|
| 1   | 17.747    | 20.860      |
| 2   | 17.855    | 20.910      |
| 3   | 17.577    | 20.826      |
| 4   | 17.668    | 19.243      |
| median | **17.71** | **20.84**  |

**Marginal cost of the folded Tanh: ≈3.1 ms** (two earlier independent pairs
gave 3.19 and 3.79 ms; call it 3.1–3.9 ms).

## 4. What moving all 12 recovers

Baseline 12 × 9.171 = **110.1 ms**, 3.3 % of the ~3330 ms single-inference path.
Per-Tanh figures are gap medians, the statistic the cost model is built from.

| option | per Tanh | 12 × | recovered | notes |
|--------|----------|------|-----------|-------|
| status quo, CPU fp32 dispatch | 9.171 | 110.1 | — | |
| CPU **int8** dispatch         | 2.408 |  28.9 | **81.2 ms** (2.4 % of pipeline) | still 12 CPU dispatches; needs an int8 boundary tensor |
| DSP int8 dispatch             | 5.086 |  61.0 | 49.0 ms | frees the CPU lane |
| HTA int8 dispatch             | 5.286 |  63.4 | 46.6 ms | 3.2 MB context each |
| **fold into `tramp_p2` (DSP)** | +3.13 |  37.6 | **72.4 ms** (2.2 % of pipeline) | also deletes 12 dispatches and 12 tensor round trips |

The largest raw number is CPU-int8 (81 ms) — most of the 9.17 ms is fp32 element
traffic, not tanh math, so simply quantizing the trampoline recovers 4.4× on the
CPU it already runs on. The fold is worth more than its 72 ms suggests: it
removes 12 dispatches from the graph and keeps `val_371` DSP-resident, and it
does not depend on the runtime learning to hand an int8 buffer to a CPU
dispatch. Standalone DSP/HTA dispatches are the weakest options: they pay the
~0.5 ms FastRPC round trip *and* the 3.1 MB in / 3.1 MB out transfer for one
elementwise op.

## 5. Root cause of the stale carve-out, reproduced

`SMOLVLA_DSP_SLICING_PLAN.md:151` records: *"confirmed the 12 Tanh blockers
(`ElementWiseNeuron` Param[0]=1 rejected on v66 DSP)"*. From
`$QNN_SDK/include/QNN/QnnOpDef.h`:

    QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU          0
    QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU         1     <-- what was rejected
    ...
    QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH         8     <-- what was blamed

Rebuilt the 8-op GELU cluster as a standalone graph (nodes `node_Pow_557,
node_Mul_559, node_Add_560, node_Mul_562` from `dsp_seg_01.onnx` + the `Tanh` +
`node_Add_565, node_Mul_567, node_Mul_568` from `dsp_seg_02.onnx`; input
`val_364 [1,1024,3072]`, output `gelu`), quantized on the real
`dsp_seg_02_val_364_*.raw` calibration. The converter collapses all eight nodes
into **one** `ElementWiseNeuron` with `operation: 1`. Composing it, rc=14 both
ways:

    DSP:  QnnDsp <I> NATIVE OpValidator::validateOpConfig _elementwiseneuron_0:qti.aisw:ElementWiseNeuron
          QnnDsp <E> Param[0] has incorrect Value 1.
          Exception encountered: Validate OpConfig failed:
            QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE: Op configuration failed validation

    HTA:  unsupported elementwise neuson op 1
          failed to create IHtaOp type name ElementWiseNeuron from package qti.aisw
          QNN_GRAPH_ERROR_INVALID_OP_CONFIG

That is the original error, verbatim, and it is GELU. Cutting at the `Tanh` is
still doing real and necessary work — it is what stops the fusion pattern from
matching, which is why `dsp_seg_01` and `dsp_seg_02` compile on DSP at all. But
the *placement* of the Tanh node on the CPU never followed from anything
measured. `build_v3_bundles.py:18` hard-codes it (`cpu_seg_XX always contributes
1, CPU-only`) and writes 1e9 µs into the DSP and HTA `results.csv`, so the
scheduler was never allowed to consider the alternative.

## 6. Caveats before shipping any of this

* **`profile_inputs/cpu_seg_01_val_371.raw` is not representative.** Its range is
  ±0.47, so a trampoline quantized on it gets an input encoding of ±0.549. The
  real activations in `trampolines/calibration/dsp_seg_01_tramp_p2/` drive
  `val_364` to ±3.14 and saturate the Tanh (output encoding −0.997…0.989).
  Timing is unaffected; accuracy is not. Requantize from the trampoline
  calibration, not from `profile_inputs/`.
* The fold must keep the GELU cut. Ending `tramp_p2` at the `Tanh` is verified
  safe; pulling the trailing `Add→Mul→Mul` in with it would re-arm the fusion and
  put `operation: 1` back in the graph.
* The 12 attention trampolines (`cpu_seg_{00,02,...,22}`, 434.7 ms) are
  untouched by any of this and remain the larger prize.

## Reproducing

Nothing was committed and no board artifacts were left behind. Working files
were built under the session scratchpad; the two graph edits are one-liners:

    # fold the Tanh into the preceding DSP phase
    m = onnx.load('vision_slices_v3/trampolines/dsp_seg_01_tramp_p2.onnx')
    m.graph.node.append(helper.make_node('Tanh', ['val_371'], ['val_372'],
                                         name='node_Tanh_563'))
    # then swap graph output val_371 -> val_372 ([1,1024,3072] float)

    # rebuild the fused GELU cluster to see the rejection
    nodes = [node_Pow_557, node_Mul_559, node_Add_560, node_Mul_562]   # dsp_seg_01
          + [Tanh(val_371)->val_372]
          + [node_Add_565, node_Mul_567, node_Mul_568]                 # dsp_seg_02
    # input val_364 [1,1024,3072], output 'gelu', initializers
    #   val_365, val_367, val_370, clone, val_375

Convert / quantize / compose exactly as in `build_trampoline_dlcs_dsp.sh`;
profile with `/root/models/smolvlm_vision_v3/profile_seg <ctx.bin> <backend.so>
<iters> --gap-us 3000`.

## Documents this contradicts

* `README.md:120` — "**Tanh boundaries**: single GELU activation op, not
  supported on DSP." The op not supported on DSP is fused GELU; Tanh is
  supported, on DSP and HTA both.
* `REPRODUCTION.md:215` and `:265` — "HTA and DSP cannot run Tanh". They can.
* `REPRODUCTION.md:953–963` (§17.3) — the lone Tanhs are `[1,1024,3072]`, not
  `[1,12,1024,1024]`; and the answer to "registry optimistic or carve-out
  stale?" is *carve-out stale*.
* `SMOLVLA_DSP_SLICING_PLAN.md:151` — `Param[0]=1` is GELU, not Tanh.
* `build_v3_bundles.py:18` and the 1e9 µs cells for `cpu_seg_*` in the DSP/HTA
  `results.csv` — an assumption, never a measurement.

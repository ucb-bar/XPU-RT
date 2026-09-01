# Sweeping the whole model for HTA gains from operator changes

Question: beyond `expert_decode`, is there anywhere else in SmolVLA where an
operator-level rewrite (MatMul<->Conv, transpose removal, dtype changes) buys
HTA eligibility or speed?

## 1. What HTA actually supports -- the consolidated constraint list

Assembled from three independent investigations, every line a verbatim
validator message from `qnn-context-binary-generator` on the board:

| constraint | evidence |
|---|---|
| no `Transpose`, at any position | `QnnHtaunsupported op Transpose` |
| `Reshape` only when input dims == output dims | `HTA op Reshape supports only equal Input and Output dimensions` |
| no two-dynamic-operand `MatMul`, **at any rank** | `unsupported op MatMul` -- identical for rank-4 batched, rank-3 batched, rank-3 single-head, rank-2 per-head |
| no `Bool_8` tensors | `validateOpConfig exits with 6000` / `INVALID_ARGUMENT` |
| no GELU (`ElementWiseNeuron` op 1); Tanh (op 8) is fine | `unsupported elementwise neuson op 1` |
| no `ReduceSum` | probe |
| `Softmax` cannot be the first layer | `Cannot natively support HTA op Softmax as first layer` |

The practical consequence: **HTA can express constant-weight
FullyConnected/Conv and elementwise ops, and essentially nothing else.** A
MatMul composes only when one operand is constant, because the converter then
rewrites it to FullyConnected.

Also established: the `Runtimes` column from `snpe-dlc-info` is worthless for
HTA. It reported `A D G C` for all 1117 decode ops including ops HTA then
refused. Only a real compose settles support.

## 2. Component-by-component

| # | component | ops | CPU | HTA | verdict |
|---|-----------|-----|-----|-----|---------|
| 1 | `smolvlm_vision_v3` (49 seg) | -- | 3172.2 | hybrid | **already exploits Conv1x1** (see 3) |
| 4 | `expert_prefill` | 1197 | 583.8 | refused | see 4 |
| 5 | `expert_decode` | 1096 | 269.95 (measured) | refused | see 4 |
| 6 | `smolvlm_text` | **1** (`Gather`) | 6.4 | excluded | nothing to rewrite |
| 7 | `state_projector` | **1** (`Gemm`) | 1.3 | 2.6 | HTA runs it, loses |
| 8 | `action_in_projector` | **2** | 4.7 | 6.7 | HTA runs it, loses |
| 9 | `action_out_projector` | **2** | 2.1 | 3.4 | HTA runs it, loses |
| 10 | `time_in_projector` | **2** | 5.8 | 6.7 | HTA runs it, loses |
| 11 | `time_out_projector` | **2** | 5.4 | 6.6 | HTA runs it, loses |

**Components 6-11 are closed.** They are one- and two-op graphs -- a single
embedding `Gather`, a `Gemm`, or a `MatMul + Add`. There is no operator change
available because there are no operators to change. HTA already composes five
of the six and is slower than the Kryo on every one. Their combined CPU cost is
25.7 ms, under 1% of the pipeline, and a perfect HTA mapping would *lose*
6.7 ms.

## 3. Vision already runs on the Conv1x1 trick

`REPRODUCTION.md:267` records the context inventory actually on the board:

    ctx_dsp_seg_00..24__*        Cpu, Dsp   (25)  whole segments -- NO Hta
    ctx_dsp_seg_NN_*_conv1x1__*  Hta only   (50)  extracted Conv1x1 kernels

**HTA never runs a whole vision segment.** It runs only the 50 extracted
Conv1x1 kernels; the HTA column in the profile CSV is synthesized from those
sub-model timings and attributed to segments that cannot execute there. So the
one component that "works on HTA" works precisely because its linears were
rewritten to Conv1x1 -- which is the finding in section 4, discovered there
empirically rather than by design.

The 24 CPU trampolines are separately exhausted by two parallel studies: the 12
attention blocks cannot reach any accelerator (0 ms recoverable, 93 ms
available from a CPU-side Split/Concat rewrite), and the 12 GELU blocks can
reach HTA once GELU is decomposed to Tanh but HTA loses to CPU int8
(5.286 vs 2.408 ms).

## 4. The finding: QNN's FullyConnected kernel is ~13x slower than its Conv2d kernel

`rewrite_matmul_to_conv1x1.py` hypothesised this in its docstring -- *"the QNN
Conv2d implementation has heavily-optimized VTCM tiling that the generic MatMul
path lacks ... may route through a faster kernel"* -- speculatively, with no
number attached. Here is the number.

The same SwiGLU block (`[50,720]x[720,2048]` x2 + `[50,2048]x[720]`), built two
ways and measured back-to-back in one board session, three interleaved repeats,
performance governor, gap median:

| form | HTA | DSP | CPU |
|------|-----|-----|-----|
| FullyConnected (`probe_mlp2d`)   | 32.07 / 32.17 / 32.33 | 19.65 / 19.58 / 19.34 | 1.71 / 1.72 / 1.66 |
| **Conv1x1 (`probe_mlpconv`)**    | **2.54 / 2.77 / 2.65** | **1.31 / (3.00) / 1.29** | 1.71 / 1.73 / 1.72 |
| speedup | **12.1x** | **14.9x** | 1.00x |

Both graphs are 6 ops and bit-exact against each other in onnxruntime
(`max|diff| 0.000e+00`). The conv build is `X[1,720,1,50] -> 3x Conv(1x1) +
Sigmoid + 2x Mul`, converted WITHOUT `--preserve_io layout` so the converter
declares the input NHWC `[1,1,50,720]` and inserts no layout ops at all: 3
Conv2d, 1 ElementWiseNeuron, 2 Eltwise_Binary, zero Transpose, zero Reshape.

Three consequences:

1. **The earlier "HTA is 18.8x slower than CPU" verdict was measuring QNN's
   FullyConnected path, not HTA's capability.** In conv form HTA is 1.54x off
   the Kryo rather than 18.8x, and **DSP at 1.31 ms actually beats the CPU's
   1.70 ms** -- the first time any accelerator has won on expert-shaped work.
2. It explains vision. HTA runs *only* extracted Conv1x1 kernels there, and
   that is why the one component that works on HTA works.
3. The CPU is indifferent (1.70 vs 1.72), so the rewrite is free to apply.

### But applying it naively to the whole graph backfires

`rewrite_matmul_to_conv1x1.py` on the full decode converts all 112
FullyConnected -- and wraps each in a `[B,M,K] -> [B,K,M,1]` round trip:

    ops        976 -> 1424
    Transpose   64 -> 288      (DLC: 227 -> 514)
    Reshape    112 -> 336
    numerics   4.2e-07 rel vs original (fp32 accumulation order, not a defect)

Measured:

| build | CPU int8 | DSP | HTA |
|-------|----------|-----|-----|
| `decode_flat_q` (MatMul form)  | **111.63 ms** | refused | refused |
| `decode_conv_q` (Conv1x1 form) | **132.24 ms** | failed  | refused (Transpose) |

**20.6 ms worse.** At decode's GEMM size the 448 added layout ops cost more
than the faster kernel saves. The probe won because it was built *natively* in
conv layout with no round trips; the whole-graph rewrite cannot be, because the
attention between the linears wants the other layout.

### What the finding is actually worth on decode

The right structure is extraction -- exactly what vision does: pull the conv
blocks out as native-layout sub-models on the accelerator, leave attention and
norms on CPU. The ceiling, from measured per-block numbers:

    MLP block   CPU 1.70 ms  ->  DSP conv 1.31 ms   saves 0.39 ms
    x16 layers                                       6.24 ms / step
    x10 denoising steps                             62.4 ms of 1116 ms  (5.6%)

before subtracting 320 CPU<->DSP handoffs. Real but modest, because CPU int8 is
already fast and the MLP is only ~24% of decode's *wall* time even though it is
67% of its MACs.

### The untested follow-up that matters more

**`expert_prefill` has never been tried in conv form.** It is 1197 ops with the
same 112 FullyConnected, it measured 583.8 ms on CPU and 1384.5 ms on DSP
(2.37x slower) -- and that DSP number was produced entirely through the slow
FullyConnected path. Prefill's GEMMs are 113 tokens deep against decode's 50,
so the layout overhead amortises over 2.26x more arithmetic. If the 14.9x DSP
kernel speedup carries, the prefill DSP verdict could invert.

### Why DSP refused the full conv graph

Not the convs -- `RmsNorm`:

    [ ERROR ] Validate OpConfig failed:
              QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND: Could not find specified op package
    last validated: rms_norm_node_:qti.aisw:RmsNorm

DSP validated all 514 Transposes and all 112 Conv2d without complaint and died
on RmsNorm, for which Hexagon v66 ships no kernel (`libQnnDspV66Skel.so` has no
RmsNorm and no registerable op package exists in the SDK or on the board -- the
R5 triage established this and it is confirmed here).

This is consistent with how the prefill DSP build succeeded: 2 tiles, a DSP
trunk plus a **CPU RmsNorm tail**. Any expert-on-DSP plan has to carve RmsNorm
out to CPU; it is not optional and it is not fixable by a rewrite, because the
converter fuses the Pow/ReduceMean/Sqrt/Reciprocal pattern into RmsNorm at
CONVERT time and offers no pass to disable it.

## 5. Verdict

| avenue | status |
|--------|--------|
| components 6-11 (text, 5 projectors) | **closed** -- 1-2 op graphs, HTA already runs 5 and loses all 5 |
| vision Conv1x1 | **already adopted** -- HTA runs only extracted conv kernels |
| vision trampolines | **exhausted** by two parallel studies (0 ms accelerator-recoverable) |
| decode transposes | **done, no effect** (120->64, 111.92 -> 111.63 ms) |
| **MatMul -> Conv1x1 kernel choice** | **NEW, 12-15x on HTA/DSP, first accelerator win on expert work** |
| conv rewrite applied whole-graph | **negative** -- layout round trips cost 20.6 ms |
| conv rewrite by extraction, decode | untested; ceiling ~62 ms of 1116 (5.6%) before handoffs |
| **conv rewrite by extraction, prefill** | **untested and the most promising thing left** |

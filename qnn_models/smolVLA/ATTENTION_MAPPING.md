# SmolVLA vision attention: can the 12 CPU trampolines be accelerated?

Follow-up to `REPRODUCTION.md` §17.2. Target of the investigation: the 12
even-index v3 vision trampolines (`vision_slices_v3/cpu_seg_{00,02,...,22}.onnx`),
each `Softmax + MatMul + Transpose + Reshape` on `[1,12,1024,1024]` scores
against `[1,12,1024,64]` values, **36.23 ms each, 434.7 ms total, 13.1% of the
~3330 ms single-inference path**.

Board: QRB5165 v66 (SM8250), QAIRT 2.45.0.260326, `performance` governor on all
8 cores, `profile_seg` wallclock around `QnnGraph_execute`, 25-40 iterations,
`--gap-us 3000`. All 12 blocks are structurally identical (verified), so one
block is measured and multiplied by 12.

## Answer in one paragraph

**No accelerator on this SoC can run these blocks, and none would want to.** HTA
has no `MatMul` kernel at any rank — not rank-4 batched, not rank-3, not the
per-head 2D form — so no reshape, split or per-head decomposition makes them HTA
-eligible. DSP and GPU *can* run the block unmodified but are 5.5x and 5.6x
slower than the CPU (197 ms and 202 ms vs 36 ms), and the int8 form the
accelerators require returns **exactly zero** on real activations because QNN
hard-codes the Softmax output encoding to scale 1/256 while every real attention
weight is below one such step. Accelerator-recoverable time: **0 ms**.

A CPU-side graph rewrite does recover time, though. The head-merge `Transpose` is
6.5 ms of the 36.2, and merging heads does not require a permute at all.
Replacing it with `Split + Concat` is bit-exact and takes the block from
**36.15 ms to 28.40 ms — 7.75 ms each, 93 ms over the 12, 2.8% of the pipeline.**
Tool: `rewrite_attention_tail.py`.

## 1. Why they fail — the exact errors

`cpu_seg_00.onnx` -> fp32 DLC -> `qairt-quantizer --act_bitwidth 8` ->
`qnn-context-binary-generator --log_level verbose` on the board.

**HTA**, unmodified block:

    QnnHta [ ERROR ] QnnHtaCannot natively support HTA op Softmax as first layer
    QnnBackend_validateOpConfig exits with 6005
    QNN_GRAPH_ERROR_INVALID_OP_CONFIG

Give Softmax a legal predecessor and the next op fails instead:

    QnnHta [ ERROR ] QnnHtaunsupported op MatMul          (p8_relu_seg00)

**DSP**: composes with no error at all, unmodified. **This block was never
rejected by the DSP backend** — it was marked infeasible by the registry.
`qnn_models/flow_c/cores/qrb5165_qnn.json` gives `dsp0` 58 capabilities and
`matmul_s8` is not among them, so the scheduler never tried it and
`gen/profile/DSP/.../results.csv` carries the `1000000000.00` sentinel for all 24
cpu segments. The registry entry is wrong; the conclusion it produced happens to
be right anyway (see §4).

**GPU**: composes and runs the *fp32* DLC. Every int8 DLC is refused with
`GPU_ERROR_INVALID_TYPE(10012)`.

**HTP**: not available on this SoC at all —
`<E> Unsupported SnapdragonModel by HTP backend`, `failed sg_htpSingletonProvider
initialize: 4000`. `libQnnHtp.so` and `libQnnHtpV68Stub.so` ship on the board but
v68 does not run on a v66 DSP. Closes that door explicitly.

## 2. Is it shape or op? — op, and it is unconditional

Every probe below is float32-in, int8-quantized, composed on HTA:

    probe                                     shapes                                  HTA result
    p2b_matmul_r4_bare   MatMul               [1,12,1024,1024] x [1,12,1024,64]       unsupported op MatMul
    p2_matmul_r4         MatMul+Transp+Resh   same                                     unsupported op MatMul
    p3_matmul_r3         MatMul               [12,1024,1024] x [12,1024,64]           unsupported op MatMul
    p4b_matmul_1head_r3  MatMul               [1,1024,1024] x [1,1024,64]             unsupported op MatMul
    p4_matmul_r2         MatMul               [1024,1024] x [1024,64]                 unsupported op MatMul

The message is byte-identical in all five. **The rank-4 batched form is not the
problem; HTA simply has no two-dynamic-operand matmul kernel.** The per-head 2D
decomposition the brief hypothesised would be accepted is rejected with the same
error as the batched one.

Corroboration: `q16_matmul_const_rhs`, a `MatMul` with a *constant* rhs,
composes on HTA — because the converter rewrites it to `FullyConnected`
(visible in `snpe-dlc-info`). So HTA's matmul-shaped capability is the
static-weight path only, and attention's `scores @ V` has two activations.

Other HTA constraints found along the way, each from its own probe:

    op          probe                  verdict
    Softmax     p1 / p5 / p5b          supported, but NOT as the graph's first layer
                p6_relu_softmax        composes once a Relu precedes it
                q10 / q11              Add(+0) and Mul(x1) prologues do NOT work -- the
                                       converter constant-folds them away and Softmax is
                                       first again (confirmed in the DLC)
    Transpose   q12_relu_transpose     unsupported op Transpose  (even as a non-first layer)
    Reshape     q13_relu_reshape       "HTA op Reshape supports only equal Input and Output
                                        dimensions" -- a shape-changing reshape is refused
    ReduceSum   q15_mul_reducesum      unsupported op ReduceSum
    Conv 1x1    p9_conv1x1_only        supported

So **three of the block's four ops are unconditionally HTA-illegal and the fourth
needs a prologue.** Only Softmax could ever execute there.

`snpe-dlc-info`'s `Runtimes` column reports `A D G C` for all four ops of
`cpu_seg_00` and is wrong for HTA on three of them. Its own footnote says it
"assumes a processor target of Snapdragon 855" — it is a generic table, not a v66
HTA query. Only a real compose settles op support. (The repo's own
`cores/qrb5165_qnn.json` `hta0` list — no matmul, no softmax, no transpose, no
reshape — is exactly right.)

## 3. Every rewrite direction, and why each dies

    direction                                 verdict
    per-head decomposition, 12x [1024,1024]   HTA rejects MatMul identically (p4). Built the
      x [1024,64]                              full graph anyway (rw1_perhead): bit-exact, and
                                               36.58 ms on CPU -- no faster than the batched form.
    reshape/squeeze the batch dim to 3D or 2D  same rejection at every rank (p3, p4, p4b).
    MatMul -> Conv1x1 for the scores x V       structurally impossible: Conv needs a constant
      product                                  kernel and V is an activation. The converter's own
                                               behaviour proves the boundary -- constant-rhs
                                               MatMul becomes FullyConnected, dynamic-rhs stays
                                               MatMul and is refused.
    split Softmax off so at least the MatMul   the MatMul cannot go to HTA at all. It can go to
      moves                                    DSP, where it is 5.0x slower (27.65 vs 5.51 ms).
    Mul + ReduceSum expansion of scores x V    HTA rejects ReduceSum; and the intermediate is
                                               805M elements. Measured for ONE head on CPU:
                                               611.78 ms. Dead twice over.
    Softmax alone on HTA behind a Relu         composes (p6) -- and runs at 153.40 ms, against
                                               39.64 ms for the same Relu+Softmax graph on CPU.
                                               3.9x slower like-for-like.
    zero-Transpose formulation (rw1_perhead)   composes nowhere new: HTA still stops at Softmax-
                                               as-first-layer, then would stop at MatMul.

The zero-Transpose graph is worth calling out because removing the permute is a
natural first instinct: **it does not unlock HTA.** `Transpose` is a wall, but
`MatMul` is a taller one behind it.

## 4. What each unit actually costs

Median ms, `performance` governor, int8 DLCs except where noted:

    piece                                        CPU      DSP       HTA        GPU
    Softmax [1,12,1024,1024]                    25.48   180.38   reject(1st)     168.05 (fp32)
    MatMul  [1,12,1024,1024]x[1,12,1024,64]      5.51    27.65   unsupported
    MatMul + Transpose + Reshape (tail incl.)   12.04    28.24   unsupported      71.62 (fp32)
    Transpose [1,12,1024,64] -> [1,1024,12,64]   6.41     1.46   unsupported
    Reshape [1,1024,12,64] -> [1,1024,768]       0.96     1.42   unsupported
    Relu + Softmax (HTA-legal softmax)          39.64   183.78      153.40
    whole block cpu_seg_00                      35.90   197.23   unsupported     201.68 (fp32)

    reference points
    Conv1x1 12->12 over 1024x1024               10.00    35.79       44.46
    FullyConnected [1024,1024]x[1024,64]         0.73     3.69      636.88

The CPU wins every piece except the `Transpose`. HTA loses even on ops it
supports at these shapes: a 1x1 conv over the score tensor is 4.4x slower than
CPU, and a `FullyConnected` of exactly the attention shape is **873x** slower —
HTA is built for compute-dense spatial convolution, and a 1024-deep skinny
GEMM over a 12.6M-element activation is its worst case.

Best conceivable heterogeneous split (Softmax CPU + MatMul CPU + tail DSP) is
25.48 + 5.51 + 0.6 = 31.6 ms against 35.90 — a 4.3 ms compute gain that needs
three graph launches instead of one and two extra tensor handoffs (50 MB and
3 MB). The 3 MB round trip through rpcmem alone costs more than the gain.

**Recoverable by accelerator mapping: 0 ms. Moving the block to DSP would cost
+161 ms each, +1934 ms over the 12.**

## 5. The accuracy trap that would have killed it anyway

Accelerators need the quantized DLC. Ran `cpu_seg_00_q.dlc` through
`qnn-net-run` on the board with **real activations** (captured from
`smolvlm_vision.onnx` via onnxruntime) and compared against the fp32 reference:

    path                              relL2 vs fp32   result
    CPU, fp32 DLC (what ships today)     0.0000       exact
    CPU, int8 DLC                        1.0000       all 786432 outputs exactly 0.0
    DSP, int8 DLC                        1.0000       all 786432 outputs exactly 0.0

The cause is visible in the DLC: QNN's quantizer **hard-codes the Softmax output
encoding**, ignoring calibration data —

    val_353 encoding : bitwidth 8, min 0.0, max 0.996093750000, scale 0.00390625

while real SigLIP attention over 1024 keys has max weight 4.55e-3 and mean
9.77e-4. **99.24% of all weights fall below half a quantization step; 100% below
one step.** Everything rounds to bin 0 and the block emits zeros. (Simulated
independently in numpy before measuring: relL2 0.79, cos 0.697 with round-half
-up; the board's round-toward-zero gives exact zeros. The V-side int8
quantization is harmless on its own — relL2 0.0036.)

The obvious fix does not exist here: `--act_bitwidth 16` produces the right
encoding (scale 1.53e-5) but composes on **nothing** —

    DSP: Input[0] has incorrect Datatype 0x416
    CPU: OpConfig validation failed for Softmax
    HTA: Cannot natively support HTA op Softmax as first layer

A `--quantization_overrides` encoding for that one tensor would work in 8-bit,
but it is moot: the fast unit is the CPU and the CPU runs the block in fp32.

## 6. What does recover time: kill the head-merge Transpose

`[1,1024,12,64]` flattened over its last two axes is head-major along the
feature axis. So merging heads needs no permute at all —

    before:  MatMul -> Transpose(perm=[0,2,1,3]) -> Reshape([1,1024,768])
    after:   MatMul -> Split(axis=1, 12x1) -> 12x Reshape([1024,64])
                    -> Concat(axis=1) -> Reshape([1,1024,768])

Bit-exact: `max|diff| = 0.0` against the original in onnxruntime for all 12
segments, and `max|diff| = 2.2e-8` against the shipped DLC's own output on the
board with real activations.

Measured on the board, fp32 DLCs (the form that actually ships), 3 interleaved
repeats of 40 iterations:

    graph                    run medians (ms)          mean
    cpu_seg_00 (baseline)    36.14, 36.21, 36.12      36.15
    rw2 concat tail          28.42, 28.34, 28.43      28.40   <- -7.75 ms, -21.4%
    rw1 per-head matmuls     36.34, 36.99, 36.41      36.58   (no gain)

**7.75 ms per block x 12 = 93.0 ms, 2.8% of the 3330 ms pipeline**, for a graph
rewrite with zero accuracy cost, no new segment boundary, and no backend change.
The per-head matmul decomposition contributes nothing on its own; the entire gain
is the permute.

Applied by `rewrite_attention_tail.py`:

    python rewrite_attention_tail.py --in-dir vision_slices_v3 \
                                     --out-dir vision_slices_v3/attn_tail --check

It matches on structure (`Softmax,MatMul,Transpose,Reshape` with
`perm=[0,2,1,3]`), hits exactly the 12 even-index segments, and skips the 12 lone
-Tanh ones. To land it, the rewritten ONNX must be re-converted to fp32 DLC and
the `cpu_seg_*` context binaries rebuilt; `pipeline_vision_v3.sh` stage 3c is the
place.

## 7. What is left after this

    before   attention x12   434.7 ms   13.1% of pipeline
    after    attention x12   341.7 ms   10.3% of pipeline

The remaining 341.7 ms is 71% softmax (25.5 ms of every 28.4 ms block). That is
not a mapping problem either. A hand-written NEON fp32 softmax over the same
tensor was benchmarked on the board against QNN's:

    scalar expf, 8 threads        26.90 ms
    NEON polynomial exp, 8 threads 22.94 ms      (max rel err 1.3e-3)
    QNN CPU backend                25.48 ms

QNN's kernel is within 11% of a decent hand-vectorised one. There is no custom
-kernel win hiding here. Softmax over 12.6M elements twelve times is simply what
SigLIP-so400m at 1024 tokens costs on this CPU; the levers left are model-level
(fewer tokens, or flash-attention-style fusion that never materialises the
1024x1024 score matrix), not mapping-level.

## 8. Corrections to earlier documents

1. §17.2 attributes the CPU placement to "batched attention is not a simple
   linear". The real reason is narrower and harder: **HTA has no dynamic-operand
   MatMul kernel at any rank**, plus no Transpose and no shape-changing Reshape.
   `rewrite_matmul_to_conv1x1.py`'s docstring (`attn_qk: batched, skip`) reaches
   the right decision for the wrong reason — a Conv1x1 rewrite is impossible not
   because the matmul is batched but because both operands are activations.
2. `cores/qrb5165_qnn.json` `dsp0` is missing `matmul` / `matmul_s8`. The DSP
   composes these blocks fine. Adding the capability would make the scheduler
   consider DSP and then correctly reject it on measured cost (197 vs 36 ms);
   leaving it out reaches the same placement by accident. Worth fixing so the
   `1000000000.00` sentinels in `gen/profile/DSP/.../results.csv` stop reading as
   "the backend refused" when it did not.
3. `snpe-dlc-info`'s `Runtimes` column must not be used to decide HTA
   eligibility. It claims `A` for `MatMul`, `Transpose` and `Reshape` here and is
   wrong on all three. Its `MACs per inference` is also wrong for batched matmul
   (reports 147k for a 805M-MAC op).

## 9. Reproducing

Probe ONNX generators, DLCs, board scripts and raw results are in the
investigation scratch dir; the two artefacts kept in the repo are this document
and `rewrite_attention_tail.py`. The board-side sequence for any op-support
question is:

    # 1. convert + quantize (calibration raws must be FLOAT32 for every input)
    snpe-onnx-to-dlc --input_network probe.onnx -d NAME d,d,d --input_layout NAME NONTRIVIAL ...
    qairt-quantizer --input_dlc probe.dlc --output_dlc probe_q.dlc --input_list cal.txt \
                    --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8
    # 2. the only authoritative support test
    qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnHta.so \
                    --dlc_path probe_q.dlc --binary_file out --output_dir . --log_level verbose
    # 3. cost
    profile_seg out.bin libQnnHta.so 30 --gap-us 3000

Serialise every board interaction behind the board-side lock —
`flock -w 3600 /tmp/qnn_board.lock ...` — not just a host-side one. A parallel
session was profiling the same DSP during this work and the first measurement
pass had to be discarded.

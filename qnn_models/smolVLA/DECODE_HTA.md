# expert_decode: quantization, and what HTA actually refuses

Everything here is measured on the QRB5165 (SM8250, Hexagon v66, HTA) with
QAIRT 2.45.0.260326. Prior work had never converted, quantized, composed or
measured `smolvlm_expert_decode` on any accelerator -- the only numbers on
record were CPU 149.6 ms and GPU 915.6 ms. This closes that gap.

## 1. The graph

1096 ONNX ops -> 1117 DLC ops, 16 identical Llama-style decoder layers.

    hidden        720
    Q heads       15 x 64 = 960
    KV heads       5 x 64 = 320      (GQA 3:1)
    MLP           2048               (SwiGLU)
    seq            50                (action chunk)
    KV cache      113 -> 163 after concat
    params        98,222,080         (6.26 M/layer x 16, confirmed by quantizer)

The cache is read-only -- decode has a single output `expert_output_embeds` and
no `present_*` -- so all 10 flow-matching steps re-run against a frozen prefix.

Per-layer sequence:

    RMSNorm    Pow ReduceMean Add Sqrt Reciprocal Mul Mul
    QKV        MatMul[720,960] MatMul[720,320] MatMul[720,320]
    RoPE       Div Sin Cos Split Mul Mul Sub Transpose Cast ScatterND (x2, on Q and K)
    KV concat  Concat [1,113,5,64]+[1,50,5,64] -> [1,163,5,64]   (x2)
    GQA expand Unsqueeze Expand Reshape  (x2)  5 -> 15 heads
    attention  MatMul [1,15,50,64]x[1,15,64,163] -> [1,15,50,163]
               Mul(scale) Where(mask) Softmax
               MatMul [1,15,50,163]x[1,15,163,64] -> [1,15,50,64]
    out proj   Reshape MatMul[960,720]  Add
    RMSNorm
    SwiGLU     MatMul[720,2048] Sigmoid Mul  MatMul[720,2048] Mul  MatMul[2048,720]  Add

## 2. Where the compute is

328.99 M MAC/layer, 5.26 GMAC = 10.5 GFLOP per decode step. CPU's 149.6 ms is
~70 GFLOP/s, so decode is genuinely compute-bound, not op-overhead-bound.

| block            | MAC/layer | share  | ops                     |
|------------------|-----------|--------|-------------------------|
| SwiGLU MLP       | 221.2 M   | 67.2%  | 3 FullyConnected x16 = 48 |
| QKV + out proj   |  92.2 M   | 28.0%  | 4 FullyConnected x16 = 64 |
| attention proper |  15.6 M   |  4.8%  | 32 MatMul + 16 Softmax  |

## 3. Quantization: solved

    decode.dlc          393.7 MB  fp32
    decode_q.dlc         98.9 MB  int8   "Quantization completed successfully"

Calibration is `gen_decode_calibration.py`. Decode has 35 inputs and 32 of them
are the KV cache, which does NOT have to be synthesized: `expert_prefill` emits
`present_key_N`/`present_value_N` with exactly decode's `past_*` shapes, so the
generator runs the real prefill and pipes its cache through. 32/35 inputs and
~96% of calibration tensor volume are real activations.

Layout/dtype rules that the quantizer reports only as bogus batch-size errors:

    raws must be FLOAT32 for every input regardless of the DLC's declared dtype
    perform_axes_to_spatial_first_order transposes declared dims:
        [1,50,720]   -> [1,720,50]
        [1,50,163]   -> [1,163,50]
        [1,113,5,64] -> [1,5,64,113]
    keep_int64_inputs=False: int64 position_ids are declared Int_32

## 4. The static Runtimes column is not trustworthy

`snpe-dlc-info` reports a per-op `Runtimes` column. For decode it says
`A D G C` -- HTA, DSP, GPU, CPU -- for **all 1117 ops**, including ops the HTA
backend then refuses outright. It is the converter's optimistic static table,
not the backend's validator. Only a real `qnn-context-binary-generator` compose
settles support.

## 5. What HTA actually refuses

Three composes, three distinct verbatim failures, each on the first offending op.

**(a) Bool_8 tensors.** Original graph, `attention_mask` declared `Bool_8`:

    Tensor ID 1  attention_mask
    Tensor ID 2  attention_mask_ncf
    Tensor ID 3  attention_mask_ncf_perm
    QnnBackend_validateOpConfig exits with 6000
    QNN_GRAPH_ERROR_INVALID_ARGUMENT

Retyping the mask to float32 (rewrite R3) clears this.

**(b) Transpose, entirely.** Rewritten graph, every tensor Float_32:

    QnnHta [ ERROR ] QnnHtaunsupported op Transpose
    QnnBackend_validateOpConfig exits with 6005
    QNN_GRAPH_ERROR_INVALID_OP_CONFIG

The HTA backend has no Transpose implementation. Decode has 227 of them, and
the one it died on was inserted by the CONVERTER (`attention_mask.ncf`,
perm [0,2,1]) to satisfy `perform_axes_to_spatial_first_order` -- not written
by us. `--preserve_io layout` suppresses the boundary ones.

This is very likely the real reason the v3 vision slicing carved 24 attention
trampolines to CPU: conv-shaped segments run on HTA, and every segment needing
a head permute cannot.

**(c) Non-identity Reshape.** MLP-only probe, Transposes suppressed:

    QnnHta [ ERROR ] QnnHtaHTA op Reshape supports only equal Input and Output dimensions

HTA's Reshape is identity-only. The converter emits Reshapes to flatten
[1,50,720] -> [50,720] for FullyConnected, so any rank-3 linear hits this.
Writing the block natively 2D removes them.

## 6. A correctness bug found and fixed: rewrite_fold_rotary

The rewrite chain was previously recorded as bit-exact. On decode it is not.
Checked against the original on real calibration data:

| rewrite                  | rel max abs diff vs original |
|--------------------------|------------------------------|
| R1 ScatterND -> Concat   | 0.000e+00                    |
| R2 Where -> mask arith   | 0.000e+00                    |
| R3 bool -> float32 mask  | 0.000e+00                    |
| R4 rotary fold           | **2.6e-02  -- WRONG**        |

`rewrite_fold_rotary.py` folded Sin/Cos at `arange(n_pos)`. That is right for
prefill, whose 113 tokens start at position 0, and wrong for decode, whose 50
action tokens sit AFTER the cached prefix at positions 113..162. The script
read the sequence LENGTH from the model but always assumed offset 0, and its
own docstring warned the fold is valid only at the folded sequence.

Fixed by adding `--start`:

    python3 rewrite_fold_rotary.py --in smolvlm_expert_decode_f32mask.onnx \
                                   --out smolvlm_expert_decode_norot113.onnx --start 113

    norot     (arange(0,50))     rel max abs diff 2.586e-02   bit-exact False
    norot113  (arange(113,163))  rel max abs diff 0.000e+00   bit-exact True

`smolvlm_expert_decode_norot.onnx` is invalid and should not be used.

## 7. Getting the MLP onto HTA -- and what it cost

The SwiGLU block is 67.2% of decode's MACs and, unlike attention, contains no
permute. It is the natural partial-offload candidate, so it was pushed all the
way through.

Two converter-level fixes were needed, neither of them a model change:

    --preserve_io layout        suppresses the boundary .ncf Transposes
                                (probe went 14 ops -> 12, zero Transpose)
    write the block natively 2D  X[50,720] instead of [1,50,720], so the
                                converter needs no flattening Reshape
                                (12 ops -> 6, zero Reshape)

`probe_mlp2d.onnx` is 6 ops -- 3 FullyConnected, 1 ElementWiseNeuron (Sigmoid),
2 Eltwise_Binary -- and is bit-exact against the rank-3 probe cut from the real
model (`max|diff| 0.000e+00`). All three backends then composed it, rc=0.

One SwiGLU block, 50 iters, performance governor, `profile_seg --gap-us 3000`,
gap-phase median:

| backend  | median   | init     | vs CPU        |
|----------|----------|----------|---------------|
| CPU int8 |  1.70 ms |  2.99 ms | --            |
| DSP int8 | 19.27 ms | 29.00 ms | 11.3x slower  |
| HTA int8 | 31.98 ms | 16.95 ms | **18.8x slower** |

**HTA runs the block and is 18.8x worse than the Kryo 585.** The op-support
walls were all removable; the performance gap is not. This is the same verdict
the prefill trunk reached on DSP (2.37x slower), reached again on the one part
of the graph that was structurally best suited to the accelerator, and by a far
wider margin. HTA is a convolution engine and a [50,720]x[720,2048] GEMM with
M=50 leaves it almost entirely idle.

### The actual win is int8 on the CPU

The interesting number in that table is not the accelerator columns. One
SwiGLU block at 1.70 ms x 16 layers = 27.2 ms, against the 100.5 ms those same
blocks account for in the 149.6 ms fp32 CPU decode -- a 3.7x speedup with no
accelerator involved.

That independently matches the trampoline probe run in parallel, which measured
CPU fp32 9.171 ms -> CPU int8 2.408 ms on an unrelated block: 3.8x. Two
different graphs, two different shapes, the same ~3.7-3.8x. The QNN CPU backend
running fp32 is the thing that was actually costing the pipeline, not the
absence of an accelerator.

## 8. Full decode, end to end

Same context binary generator, same `profile_seg`, 20 iters, performance
governor on all 8 cores, gap-phase median:

| build                     | backend | median      | note                       |
|---------------------------|---------|-------------|----------------------------|
| `decode.dlc` fp32         | CPU     | **269.95 ms** | 393.7 MB context          |
| `decode_q.dlc` int8       | CPU     | **112.77 ms** | 98.8 MB context, **2.39x** |
| `decode_q.dlc` int8       | DSP     | refused     | `Input[0] has incorrect Datatype 0x508` (Bool_8 mask) |
| `decode_q.dlc` int8       | HTA     | refused     | unsupported op Transpose   |
| `decode_n113_q.dlc` int8  | CPU     | **111.92 ms** | full rewrite chain, bit-exact |
| `decode_n113_q.dlc` int8  | HTA     | refused     | unsupported op Transpose (227 internal) |
| `decode_n113_q.dlc` int8  | DSP     | refused     | rc=14, message not captured before cleanup |

The fully-rewritten, bit-exact `norot113` build measures 111.92 ms against the
original graph's 112.77 ms -- within noise. **The five rewrites buy nothing on
the CPU.** They exist to unlock accelerators, and the accelerators do not want
the graph: even with `--preserve_io layout` suppressing the boundary
transposes, 227 internal Transposes remain in the head permutes and RoPE, and
HTA refuses on the first one.

**The recorded 149.6 ms CPU baseline for decode does not reproduce.** Measured
here under a performance governor the fp32 QNN CPU path is 269.95 ms, 1.80x
the figure in `expert_triage.json` and `REPRODUCTION.md:137`. That row is
labelled `smolvlm_expert_decode_coarse`, so it may be a different granularity
or a different execution path; either way any claim resting on 149.6 ms should
be re-derived. Against the honest fp32 number, int8 on the CPU saves 157.2 ms
per denoising step -- 1571.8 ms across the 10 steps of one action chunk.

## 9. Verdict for expert_decode

    quantization    SOLVED     98.9 MB int8, real prefill KV as calibration
    HTA op support  3 WALLS    Bool_8 tensors; Transpose (all); non-identity Reshape
    HTA composition ACHIEVED   for the SwiGLU block, once written 2D with
                               --preserve_io layout (6 ops, no Transpose/Reshape)
    HTA performance NEGATIVE   31.98 ms vs CPU 1.70 ms -- 18.8x slower
    DSP performance NEGATIVE   19.27 ms -- 11.3x slower
    CPU int8        POSITIVE   269.95 -> 112.77 ms end to end, 2.39x
    transpose -47%  NO EFFECT  111.92 -> 111.63 ms (tensors too small to matter)
    rewrites R1-R5  NO EFFECT  on CPU (111.92 vs 112.77 ms), and do not
                               unlock HTA (227 internal Transposes remain)

The accelerators are exhausted for this graph. HTA's constraints are not
incidental: with no Transpose, no shape-changing Reshape, and (per the parallel
attention study) no two-dynamic-operand MatMul at all, the only thing HTA can
express is constant-weight FullyConnected/Conv -- and on decode's [50,720]x
[720,2048] shape it is 18.8x off the Kryo anyway.

**The win that is actually available is int8 on the CPU, worth 2.39x.**

### Caveat that must be resolved before shipping int8

This is a SPEED result. Accuracy is not established, and there is a specific
known hazard: the parallel attention study found that QNN hard-codes the
Softmax output encoding to scale 1/256, which flushed 100% of SigLIP's
attention weights (mean 9.8e-4 over 1024 keys) to exactly zero. Decode's
softmax runs over 163 keys, mean weight ~6.1e-3, so it sits around 1.6
quantization steps rather than below one -- less catastrophic, but far too
coarse to assume is harmless across 16 layers and 10 denoising steps.

Before any int8 decode is trusted, run it against the fp32 reference on real
inputs and check the action outputs directly. The calibration set here is also
only partly real (32/35 inputs from a true prefill; `expert_embeds` and
`attention_mask` synthetic), which is adequate for composition and timing and
not for an accuracy claim.

## 10. Can the transposes be preprocessed away?

Partly, and it is worth knowing exactly which ones, because "227 transposes"
sounds irreducible and is not.

The rewritten decode ONNX has 120 Transposes (the DLC's 227 includes
converter-inserted layout ops, which `--preserve_io layout` already removes).
They fall into two groups:

| perm         | n  | from -> to        | shape            | what it is        |
|--------------|----|-------------------|------------------|-------------------|
| (3,1,2,0)    | 56 | Sub/Add/Concat -> Concat | [1,50,15,32] | RoPE half wrapping |
| (3,2,1,0)    | 16 | Concat -> MatMul  | [64,50,15,1]     | undoing the above |
| (0,2,1,3)    | 32 | Reshape/MatMul    | [1,*,15,64]      | head permute      |
| (0,2,3,1)    | 16 | Reshape -> MatMul | [1,*,15,64]      | K transpose (QK^T)|

**The first 72 are removable, and were only ever there to serve ScatterND.**
ONNX ScatterND indexes axis 0, so the exporter wrapped each rotary half in a
Transpose to bring axis 3 to the front. `rewrite_scatternd_to_concat.py`
replaced the ScatterND with `Concat(axis=0)` -- but a Concat can join on any
axis, so the wrapping became dead weight:

    A -T(3,1,2,0)-> [32,50,15,1] -.
                                   Concat(axis=0) -> [64,50,15,1] -T(3,2,1,0)-> [1,15,50,64]
    B -T(3,1,2,0)-> [32,50,15,1] -'

      ==  Concat(axis=3) -> [1,50,15,64] -T(0,2,1,3)-> [1,15,50,64]

`rewrite_collapse_transposes.py` does this as two standard passes rather than a
special case -- push-transpose-through-concat
(`Concat([T(X_i,p)],axis=a) == T(Concat([X_i],axis=p[a]),p)`) then
fuse-consecutive-transposes (`T(T(X,p),q) == T(X,r)`, `r[k]=p[q[k]]`) plus
identity-drop and dead-node collection.

    Transpose 120 -> 64   (56 removed, 47%)
    total ops 1032 -> 976
    bit-exact vs the original on 3 real samples: 0.000e+00

**The remaining 64 are genuine attention head permutes** -- [batch,seq,head,dim]
<-> [batch,head,seq,dim]. Two further options exist:

* the 16 output merges (`MatMul -> Reshape`) can become `Split(axis=1) +
  Concat`, which the parallel attention study verified bit-exact and measured
  21% faster on CPU for the vision blocks (36.15 -> 28.40 ms);
* the 48 Q/K/V permutes fall only to per-head decomposition -- 15 independent
  2D matmuls per attention instead of one batched op. No weight preprocessing
  removes them: the permute interleaves the sequence axis (which comes from the
  activation's rows) with the head axis (which comes from the weight's
  columns), so permuting weights offline cannot express it.

### None of this unlocks HTA

Worth stating plainly, because it is the natural hope. The parallel attention
study established that HTA has **no two-dynamic-operand MatMul kernel at any
rank** -- rank-4 batched, rank-3 batched, rank-3 single-head and rank-2 per-head
all return byte-identical `unsupported op MatMul`. A MatMul only composes when
one operand is constant, because the converter then rewrites it to
`FullyConnected`.

Decode's attention has 32 matmuls whose operands are both activations. That is
what attention *is*. Transpose sits in front of that wall, not instead of it,
so removing every transpose in the graph would still leave HTA unable to take
the attention -- and per-head decomposition, the one rewrite that removes the
last 48, was tested against HTA by the parallel study and refused.

The transpose work is therefore worth doing for the CPU int8 path, which is the
one that is actually winning, and worth nothing for HTA.

### Measured: removing them buys nothing

Same conditions as section 8 (20 iters, performance governor, gap median):

| build                        | ONNX Transpose | DLC Transpose | CPU int8   |
|------------------------------|----------------|---------------|------------|
| `decode_n113_q` (norot113)   | 120            | 227           | 111.92 ms  |
| `decode_flat_q` (collapsed)  |  64            | 164           | **111.63 ms** |

0.29 ms apart -- 0.26%, inside the run-to-run spread. **Deleting 47% of the
transposes is free and worthless on this graph.** QNN's CPU backend evidently
folds them into its tensor reads rather than materialising a permuted copy.

This does not contradict the parallel attention study's 36.15 -> 28.40 ms win
from the same class of rewrite; it locates it. Those vision tensors are
[1,12,1024,1024] and [1,1024,12,64] -- megabytes per permute. Decode's are
[1,50,15,64], 48000 elements, 192 KB. Two orders of magnitude smaller, and the
permute cost scales with the tensor. Transpose elimination pays where the
tensors are big, and decode's are not.

Consequence: the remaining 16 output merges are not worth converting to
Split+Concat either, and `rewrite_collapse_transposes.py` should be regarded as
a correctness-preserving tidy-up rather than an optimisation.

HTA still refuses the collapsed graph (`unsupported op Transpose`) -- 64 head
permutes remain, and behind them the 32 dynamic-operand MatMuls it cannot do
at any rank.

## 11. Native-conv extraction on decode: composes everywhere, wins nothing

The same treatment that produced prefill's one accelerator win was applied to
decode. Layer-0 linears extracted as native conv-layout blocks
(`build_nativeconv_blocks.py --model smolvlm_expert_decode.onnx --seq 50`),
converted without `--preserve_io layout`, calibrated on the real activations
tapped from the full decode.

All three blocks compose on cpu, dsp AND hta -- the structural result carries
over. 50 iters, 3 interleaved repeats, performance governor, gap median (us):

| block | shapes | CPU | DSP | HTA | best |
|-------|--------|-----|-----|-----|------|
| `ncd_qkv`   | 720->960, 720->320 x2 | **593.9** | 919.9 | 1768.7 | CPU |
| `ncd_oproj` | 960->720              | **381.0** | 690.1 | 2524.0 | CPU |
| `ncd_mlp`   | 720->2048 x2, 2048->720 | 1700.5 | **1679.0** | 2512.1 | DSP, tie |
| per layer   |                       | **2675.4** | 3289.0 | 6804.8 | 2653.9 |

**No accelerator wins decode.** The CPU takes qkv and o_proj outright, and the
MLP is a statistical tie -- DSP 1679.0 vs CPU 1700.5 is 1.3%, inside DSP's own
spread on that block (1198 / 1679 / 3015 across three repeats). Best-of-backend
saves 21.5 us per layer, 0.3 ms over 16 layers, which is noise.

### Why prefill wins and decode does not

Same block, same rewrite, opposite verdict:

    prefill MLP   960->2560, 113 tokens   cpu 4415  hta 2452   HTA 1.80x FASTER
    decode  MLP   720->2048,  50 tokens   cpu 1700  hta 2512   HTA 1.48x SLOWER

The crossover for the MLP block on HTA sits between 50 and 113 tokens. Decode's
GEMMs are 2.26x shorter and its hidden and MLP dims are smaller (720/2048
against 960/2560), so there is not enough arithmetic to amortise HTA's
dispatch. This is the same size-dependence that made HTA lose both projection
blocks on prefill, just applied to the whole component.

**Decode's gain remains what section 8 measured: int8 on the CPU, 2.39x.**
Nothing in the accelerator direction adds to it.

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

## 6. Re-examining the text encoder and projectors (correction)

Section 2 dismissed these on the strength of the profile CSVs. Two things were
wrong with that, in opposite directions.

**First, they run more often than section 2 implied.** `REPRODUCTION.md:915`
records `projectors x10  159.0 ms  4.8%` -- `action_in`, `time_in` and
`time_out` (4.68 + 5.85 + 5.37 = 15.90 ms) fire once per denoising step. So the
CSV-based cost is 159 ms + 9.9 ms one-shot, not the 25.8 ms single-pass figure.

**Second, and decisively: the CSV numbers for small components are wrong.**
`state_projector` still has context binaries on the board, so it is a free
control. Re-measured with the current harness:

| | recorded | CPU warm | CPU cold |
|---|----------|----------|----------|
| state_projector CPU | 1.33 ms | **17.4 us** | 64.6 us |
| state_projector DSP | 29.04 ms | 419.1 us | 781.6 us |
| state_projector HTA | 2.64 ms | 542.3 us | 2465.0 us |

The CPU figure is overstated **76x** and the DSP figure 37x. Only the HTA
number reproduces, and only against the *cold* measurement. Rebuilding and
measuring the two per-step projectors that dominate the x10 total:

| projector | recorded | CPU warm | CPU cold |
|-----------|----------|----------|----------|
| `time_in`  | 5.85 ms | **0.57 ms** | 2.10 ms |
| `time_out` | 5.37 ms | **0.31 ms** | 1.38 ms |

3-10x overstated. The pattern -- every tiny component landing at 1-6 ms
regardless of its actual arithmetic (`action_in` is 0.026 MMAC and "costs"
4.68 ms) -- is consistent with the coarse profiling harness charging fixed
setup per component rather than measuring steady-state execute.

`action_in_projector` and `action_out_projector` could not be quantized for
this check: their `[1,1,36]` and `[1,1,720]` shapes are degenerate for
`qairt-quantizer`. Their real cost is therefore still unmeasured, but by
analogy it is almost certainly well under the recorded 4.68 / 2.15 ms.

### Consequence: there is no opportunity here, and less to win than believed

Taking the measured numbers and being generous to the unmeasured `action_in`:

    recorded per step   15.90 ms   ->  x10 = 159.0 ms   (4.8% of pipeline)
    measured per step   ~1-4.5 ms  ->  x10 = ~10-45 ms  (0.3-1.4%)

So roughly 115-150 ms of the pipeline budget that was attributed to projectors
does not exist. The components are already close to free.

There is also no accelerator lever. HTA's *warm* floor is 542 us -- already
above what the CPU spends on every one of these (17-570 us warm). The text
encoder is a single `Gather` over a 47.3 M-parameter embedding table with zero
MACs; no accelerator has anything to do with a table lookup, which is why it is
excluded rather than merely slow.

The one genuine, small effect: these ARE the short dispatches that suffer the
power-collapse penalty documented in `HTA_DISPATCH_FLOOR.md` -- `time_in` goes
569 us warm to 2100 us cold, 3.7x, and the same ratio shows on the CPU. Placing
the three per-step projectors adjacent in the schedule, or fusing them into one
graph, would recover a few ms per chunk. Single-digit ms, worth doing only if
it falls out of other scheduling work.

**Action item beyond this file:** `plot_smolvla_gains.py` and
`REPRODUCTION.md:915` both carry the 159.0 ms figure and should be corrected.

## 7. Are there other blocks to cast to Conv1x1? No -- the lever is exhausted

A Conv1x1 computes, at each spatial position, a **constant-weight linear map
over the channel axis**. That is the whole expressible set, and it is worth
scanning the model against it rather than guessing.

Constant-weight MatMul/Gemm remaining, by component:

| component | ops | constW MatMul | dynamic MatMul | Conv |
|-----------|-----|---------------|----------------|------|
| prefill trunk (original) | 1108 | 112 | 32 | 0 |
| prefill trunk (convbar)  | 1588 | **0** | 32 | 112 |
| decode (original)        | 1096 | 112 | 32 | 0 |
| vision, 25 dsp_seg orig  | -- | 49 | -- | -- |
| vision, 25 dsp_seg conv1x1 | -- | **0** | -- | -- |
| 24 cpu_seg trampolines   | -- | **0** | 12 | -- |
| projectors (`*_conv` variants exist) | 2 -> 1 | 1 -> **0** | 0 | 1 |
| text encoder             | 1 | **0** | 0 | 0 |

**There is not a single constant-weight linear left un-converted anywhere in
SmolVLA.** Prefill and vision are done; decode's 112 are covered by the
`ncd_*` blocks; the projectors already have `_conv` variants on disk.

The 88 MatMuls that remain (32 prefill + 32 decode + 24 vision trampolines) are
all **two-dynamic-operand attention products**. A Conv1x1 needs its weights to
be constant, and in `QK^T` and `A·V` both operands are activations, so these
are not castable in principle -- not a tooling gap.

### RoPE and the other tempting candidates

Three ops ARE expressible and all three are pessimisations, because they are
currently *free memory movement* and a Conv1x1 turns them into arithmetic:

| op | today | as Conv1x1 | cost |
|----|-------|------------|------|
| RMSNorm scale `x * w[960]` | 108,480 mul (diagonal) | 104,140,800 MAC (dense) | **960x worse** |
| GQA expand 5->15 heads | 36,160 element copy | 34,713,600 MAC | was free |
| RoPE `rotate_half` (permute+negate) | 108,480 element shuffle | 6,942,720 MAC (block form) | was free |

RoPE deserves the specific answer since it is the natural thing to ask about.
After the rotary fold it is `x (*) COS + rotate_half(x) (*) SIN`. The
`rotate_half` half IS a signed channel permutation and therefore a legal
block-diagonal Conv1x1 -- but the `(*) COS` and `(*) SIN` halves are **not**,
because COS and SIN vary per *position*, and a Conv1x1's weights are constant
across the spatial axis by construction. So RoPE is at best half-castable, and
the half that can be cast is the half that is currently free.

Not expressible at all: Softmax (nonlinear, normalises across keys), the
attention output merge `Transpose(0,2,1,3)` (permutes seq against head, not a
channel map), and the text encoder's `Gather` (a table lookup with zero MACs).

### Why the lever does not generalise

Conv1x1 was worth 13x on the accelerators because those ops were **already
matmuls being routed through a bad kernel** -- QNN's `FullyConnected` path. It
was a kernel-selection win, not a mathematical restructuring win. Casting an op
that is *not* already a matmul does not inherit that win; it converts free work
into paid work and then charges the accelerator's dispatch floor on top.

The rule that falls out: cast to Conv1x1 exactly when the op is already a
constant-weight linear. Everywhere that was true in SmolVLA, it has been done.

## 8. The DSP: structurally better behaved, and still loses

Worth separating from HTA, because the two fail differently.

| | floor (warm) | marginal | power-collapse penalty |
|---|--------------|----------|------------------------|
| CPU | 14.5 us | 144 GMAC/s | 4.44x |
| **DSP** | **401.9 us** | **422 GMAC/s** | **1.37x** |
| HTA | 542.6 us | 493 GMAC/s | 4.68x |

DSP has a **lower dispatch floor than HTA**, a nearly flat power-collapse
penalty, 2.9x the CPU's throughput, and far broader op support -- it accepts
Transpose, non-identity Reshape and two-dynamic-operand MatMul, all of which
HTA refuses outright.

It still loses almost everywhere:

| block | CPU | DSP | HTA |
|-------|-----|-----|-----|
| nc_qkv prefill   | **1193.1** | 3075.5 | 2455.3 |
| nc_oproj prefill | **925.1** | 932.6 | 1773.3 |
| nc_mlp prefill   | 4414.9 | 3820.5 | **2452.2** |
| ncd_qkv decode   | **593.9** | 919.9 | 1768.7 |
| ncd_oproj decode | **381.0** | 690.1 | 2524.0 |
| ncd_mlp decode   | 1700.5 | **1679.0** | 2512.1 |

DSP beats the CPU on 2 of 6 -- both the largest blocks, both marginally -- and
loses to HTA wherever HTA runs at all. It is squeezed from both sides: too slow
to beat the Kryo on small blocks, too slow to beat HTA on large ones.

### The one thing only the DSP can do -- and it is catastrophic

HTA has no dynamic-MatMul kernel at any rank, so attention is HTA-impossible by
construction. DSP takes it. The expert attention core was extracted from the
prefill trunk (13 ops: 2 dynamic MatMul, Softmax, 3 Transpose, the mask
arithmetic; 35.4 MMAC, 276K softmax elements) and measured:

| backend | warm | cold |
|---------|------|------|
| **CPU** | **1871.4 us** | 2126.9 us |
| **DSP** | **40366.5 us** | 40272.3 us |
| HTA | refused: `unsupported op Transpose` | -- |

**21.6x slower than the CPU** -- worse even than the 5.5x the vision attention
study measured on its far larger blocks. 35.4 MMAC in 40.4 ms is 0.88 GMAC/s,
**480x below DSP's own 422 GMAC/s marginal rate**, so the batched MatMuls are
not the problem; the softmax and the rank-4 batched layout are. Note also the
warm/cold ratio is 1.00 -- at 40 ms the dispatch floor is irrelevant.

### Consequence

Every part of the expert is now measured on every backend it can reach:

    linears     HTA wins prefill MLP only (1.80x); DSP marginal on 2 blocks
    attention   HTA impossible; DSP 21.6x worse
    norms/RoPE  cheap elementwise, nothing to gain, and casting them to
                Conv1x1 is a pessimisation (section 7)

The ~12.07 ms/layer prefill remainder (attention + RoPE + norms) is CPU-only
and will stay that way. The accelerator search on the experts is complete.

## 9. Applied: the two measured vision wins, and the blocker that hid them

Both wins below had been measured by the parallel studies and neither had been
applied. `vision_slices_v3/attn_tail/` did not exist; the trampolines still
shipped fp32.

### 1. The 12 lone-Tanh trampolines, fp32 -> int8

`quantize_tanh_trampolines.py`. The difficulty is entirely calibration:
`profile_inputs/cpu_seg_NN_*.raw` is range +-0.505 std 0.100 while the real
distribution entering the Tanh is +-2.9 std 0.413. Rather than substitute a
nearby tensor, the script extracts the GELU pre-tanh chain from each producing
`dsp_seg_NN`

    val_364 = fc1(x)  ->  ^3  ->  *0.044715  ->  +x  ->  *sqrt(2/pi) = tanh input

and runs it on the real fc1 activations from `trampolines/calibration/`, which
is exact and does not depend on reading the GELU constants by hand.

That choice is worth 37x in accuracy. Simulated against the true distribution:

| calibration | max abs err | mean | cosine | clipped |
|-------------|-------------|------|--------|---------|
| **derived (this change)** | **0.0144** | 0.00486 | 0.999792 | **0.00%** |
| shipped `profile_inputs` | 0.5275 | 0.03125 | 0.983025 | **21.89%** |

The shipped raws would clip 22% of the distribution and produce 0.53 max error
on a [-1,1] output -- unusable. Measured on the board from the DLCs this script
produces (cpu_seg_01 / cpu_seg_11, gap median):

    CPU 2389.7 / 2494.5 us     DSP 3806.1 / 3732.0     HTA 5067.8 / 4890.2
    against the shipped fp32 9171 us  ->  3.75x

**110.1 ms -> 29.3 ms, saves 80.8 ms.** TANH_PROBE.md predicted 81.2 ms
independently.

### 2. The 12 attention tails, head-merge Transpose -> Split+Concat

`rewrite_attention_tail.py --in-dir vision_slices_v3 --out-dir
vision_slices_v3/attn_tail --check`. All 12 rewrote at
`max|diff| = 0.000e+00`; the 12 odd segments correctly skip as not attention
tails. Worth **93.0 ms** (36.15 -> 28.40 ms each) per ATTENTION_MAPPING.md.

### 3. The blocker: build_v3_bundles.py pinned every dispatch to one backend

`emit_results_csvs` wrote

    cost = m["cost_us"] if m["preferred_hw"] == csv_hw else _INFEASIBLE_US

so every backend other than the one this script chose got the 1e9 sentinel.
The placement was an assumption the scheduler could not revisit, and the
docstring stated it as fact: *"cpu_seg_XX always contributes 1, CPU-only"*.
That is why the trampolines' DSP/HTA numbers could not be expressed even once
they existed.

`add()` now takes `alt_costs_us`, and the emitter offers a dispatch on every
backend it has a MEASURED cost for, sentinel only where none exists. The odd
trampolines carry all three; the even ones stay CPU-only deliberately, because
the attention study measured 0 ms accelerator-recoverable for them.

Net for vision: **173.8 ms** (80.8 + 93.0), about 32% of the 544.7 ms the
trampolines cost and ~8% of vision's realizable total. Both wins are CPU-side.

### Correction: regenerating the bundle CSVs silently dropped the HTA bundles

Running `build_v3_bundles.py` to pick up the change above **regressed** the
emitted `results.csv` files, and they were committed that way before the
regression was caught. Before: 98 unique modules including the HTA bundle
decomposition (`dsp_seg_NN_conv1`, `_conv2`, `_tramp_p0/p1/p2`) and 24 real HTA
rows. After: 50 mono modules and 12 real HTA rows.

Cause: `segment_perf.json` in the working tree carries

    "Hta": {"status": "ok", "mean_us": 12392.27, "note": "sum of 2 conv ops"}

-- `mean_us` but **no `convs` list**. `_build_bundle` needs the per-conv
breakdown and returns `None` without it, so every segment fell back to a
DSP/CPU mono placement and every HTA bundle dispatch vanished. The committed
CSVs had been generated from a richer `segment_perf.json` than the one on disk.

The CSVs are restored from the prior commit. `_build_bundle` now warns loudly
when `Hta.mean_us` is present but `Hta.convs` is not, because "this segment has
no HTA path" and "the profile data lost its breakdown" were indistinguishable
and the second silently throws away the entire HTA placement.

**Consequence for the trampoline win:** `TRAMPOLINE_ALT_COSTS_US` is wired in
and correct, but it only reaches the cost model when the CSVs are regenerated,
which needs a `segment_perf.json` carrying `Hta.convs`. Until then the 80.8 ms
is real and measured but not yet expressed in the scheduler's inputs. The
int8 DLCs themselves exist under `vision_slices_v3/tanh_int8/`.

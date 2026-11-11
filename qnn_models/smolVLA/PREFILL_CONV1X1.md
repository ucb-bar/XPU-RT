# Conv1x1 on expert_prefill

Follow-up to the finding in `HTA_OPPORTUNITY_SWEEP.md` §4: QNN's `Conv2d`
kernel is 12-15x faster than its `FullyConnected` kernel for identical
arithmetic on HTA and DSP. Applied whole-graph to `expert_decode` it lost
20.6 ms, because the `[B,M,K] -> [B,K,M,1]` round trips cost more than the
kernel saved at 50 tokens.

Prefill is the case where that ratio should move the other way: 113 tokens
instead of 50, so the same per-op layout overhead amortises over 2.26x more
arithmetic. And prefill's DSP number -- 1384.5 ms, 2.37x SLOWER than its
583.8 ms CPU baseline -- was produced entirely through the slow
FullyConnected path.

## Setup

Target is `smolvlm_expert_prefill_trunk.onnx`, the exact artifact that
produced 1384.5 ms. It matters that it is the trunk and not the whole prefill:
DSP has no `RmsNorm` kernel on v66 (`QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND`),
and the trunk is the version whose RmsNorm stays *decomposed* into
Pow/ReduceMean/Sqrt/Reciprocal, which is why DSP accepts it at all.

    smolvlm_expert_prefill_trunk.onnx        1108 ops  MatMul 144  Transpose 144  Reshape  96
    smolvlm_expert_prefill_trunk_conv.onnx   1556 ops  Conv  112  MatMul 32
                                                       Transpose 368  Reshape 320

112 of the 144 MatMuls become Conv1x1; the 32 that remain are the batched
attention, which have two dynamic operands and cannot.

Numerics, onnxruntime, all 33 outputs x 2 real samples:

    worst rel max|diff| = 5.881e-07     (fp32 accumulation order, not a defect)

Calibration is 8 samples, `expert_rewrite/prefill_calib`, float32 raws in the
converter's declared layout -- `vlm_embeds` 433920 B `[1,960,113]`,
`attention_mask` 51076 B, `position_ids` 452 B. These match the extents
recorded in `REPRODUCTION.md` §14 exactly.

Both variants converted and quantized with **identical flags** so the A/B
isolates the kernel choice.

## Quantization

    pf_trunk.dlc        629.7 MB fp32   ->  pf_trunk_q.dlc        157.9 MB int8
    pf_trunk_conv.dlc   630.9 MB fp32   ->  pf_trunk_conv_q.dlc   158.5 MB int8

`pf_trunk_q` at 157.9 MB matches the size recorded in `REPRODUCTION.md` §14
exactly, confirming it is the same artifact that produced the 1384.5 ms DSP
number rather than a lookalike.

One trap worth recording: the trunk DLC declares only **two** inputs. The
rotary fold left `position_ids` dead, so the three-input calibration list is
rejected with `Graph contains 2 inputs, but only found input data for 3 inputs
from user` -- the same failure mode decode hit. Build the input list from the
DLC's own APP_WRITE tensors, never from the ONNX graph inputs.

## Measurement 1: the baseline reproduces, and CPU already improves

10 iters, 3 interleaved repeats, performance governor, `profile_seg --gap-us 3000`,
gap-phase median:

| build | DSP | CPU |
|-------|-----|-----|
| `pf_trunk_q` (MatMul, 112 FullyConnected) | **1411.9 / 1411.0 / 1411.3 ms** | 384.6 / 384.5 / 342.3 ms |
| `pf_trunk_conv_q` (Conv1x1, 112 Conv2d)   | FAILED -- `OP_PACKAGE_NOT_FOUND` | **323.0 / 323.5 / 324.0 ms** |

Two results before the DSP question is even settled.

**The DSP baseline reproduces.** 1411.3 ms against the 1384.5 ms on record, a
2% gap across three tight repeats (std under 0.5 ms). Unlike decode's 149.6 ms
CPU figure, which was 1.8x optimistic, this one stands.

**Conv1x1 is 1.19x faster than MatMul on the CPU here** -- 323.5 vs 384.5 ms --
where on decode it was 0.84x, i.e. 20.6 ms *worse*. Same rewrite, same backend,
opposite sign. That is the amortisation crossover made visible: the per-op
layout round trips are a fixed cost per linear, and 113 tokens buys 2.26x more
arithmetic to pay them off with than 50 tokens does. It also means the rewrite
is not merely an accelerator-enabling trick; past some sequence length it is
simply a better graph on any backend.

## The obstacle: the conv rewrite re-triggers RmsNorm fusion

DSP refused the conv build with the exact error the trunk was constructed to
avoid:

    Validate OpConfig failed:
    QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND: Could not find specified op package

Comparing full DLC op lists explains it:

| op | `pf_trunk_q` | `pf_trunk_conv_q` |
|----|--------------|-------------------|
| Eltwise_Binary | 402 | 306 |
| Reduce         | 32  | -- |
| Eltwise_Unary  | 32  | -- |
| **RmsNorm**    | **none** | **32** |

160 decomposed ops collapsed into 32 `qti.aisw:RmsNorm`, the one op Hexagon v66
ships no package for.

This is *not* the rewrite deleting the anti-fusion barriers -- both ONNX files
still carry all 32 `Mul`-by-1.0 barriers that
`rewrite_block_rmsnorm_fusion.py` inserted (verified by scanning initializers).
The converter folded them away in the conv case and not in the plain case. The
barrier script's own docstring warns about exactly this: *"Pair this with an
ir_optimizer_config that skips RemoveNoOps and SquashConstantInput, or the
converter will simply fold the barrier away again."* Neither build passed such
a config; the plain trunk happened to survive without one and the conv graph,
whose extra Reshape/Transpose give those passes more to chew on, did not.

Fix under test: `expert_rewrite/ir_opt_keep_barriers.yaml`, skipping only
`RemoveNoOps` and `SquashConstantInput` and leaving every other optimization on.

## Defeating the fusion: a barrier that cannot be folded

`--ir_optimizer_config` skipping `RemoveNoOps` and `SquashConstantInput` did
**not** help -- the rebuilt DLC still carried 32 `RmsNorm`. And no RmsNorm pass
is exposed in `--dump_ir_optimizer_config_template`, so it cannot be disabled
directly.

The reason the existing barrier is fragile is that it is a *no-op*: a `Mul` by
constant 1.0 is something the converter is always entitled to delete. The fix
is a barrier that changes the arithmetic and therefore cannot legally be
removed, while still being exact. `rewrite_rmsnorm_scale_barrier.py` scales the
variance by 4 and undoes it after the reciprocal:

    m4  = mean * 4.0
    a   = m4 + 4*eps      = 4*(mean + eps)
    s   = sqrt(a)         = 2*sqrt(mean + eps)
    rec = 1/s             = 0.5 * r
    out = rec * 2.0       = r

4.0, 2.0 and the scaled epsilon are exact powers of two, so no rounding is
introduced anywhere. Rather than adding a node it **repurposes the existing
Mul-by-1.0 barrier**, rewriting its constant from 1.0 to 4.0 -- the same node
that was removable becomes unremovable.

The real chain is `Pow -> ReduceMean -> Mul(1.0 barrier) -> Add -> Sqrt ->
Reciprocal -> Mul -> Mul`, so a matcher looking for `Add` fed directly by
`ReduceMean` finds nothing; it has to look through the barrier.

    RMSNorm chains given a scale barrier: 32  (scale 4.0, undo 2.0)
    convbar vs trunk: worst rel max|diff| = 5.881e-07

That 5.881e-07 is *identical* to the conv-only variant's error, i.e. the scale
barrier contributed exactly nothing on top of the conv rewrite's accumulation
order -- which is what the power-of-two choice is for.

## Measurement 2: fusion defeated, and the CPU win grows

The barrier holds. `pf_convbar_q` DLC op list:

    560 Transpose   498 Eltwise_Binary   289 Reshape   112 Conv2d
     32 Split   32 Resize   32 Reduce   32 MatMul   32 Eltwise_Unary   32 Concat

`Reduce` and `Eltwise_Unary` are back at 32 each and **`RmsNorm` is gone**,
with all 112 Conv2d intact. Note this contradicts the guidance in
`rewrite_block_rmsnorm_fusion.py`: the prescribed `--ir_optimizer_config`
skipping `RemoveNoOps`/`SquashConstantInput` did *not* work (32 RmsNorm
survived). Making the barrier non-removable is what works.

CPU, 10 iters, 3 interleaved repeats, performance governor, gap median:

| build | CPU | vs MatMul baseline |
|-------|-----|--------------------|
| `pf_trunk_q`     MatMul / decomposed norm | 384.6 / 384.5 / 342.3 ms | -- |
| `pf_trunk_conv_q` Conv1x1 / fused RmsNorm | 323.0 / 323.5 / 324.0 ms | 1.19x |
| `pf_convbar_q`    Conv1x1 / decomposed norm | **297.4 / 297.7 / 297.9 ms** | **1.29x** |

Three tight repeats each, cleanly separated. **87 ms off the prefill trunk on
the CPU alone, from two rewrites that are numerically exact to 5.9e-07.** The
barrier build is also faster than the fused-RmsNorm build (297.6 vs 323.5),
so QNN's CPU RmsNorm kernel is slower than the decomposed chain it replaces --
worth knowing independently.

## DSP: op support cleared, resource limit hit

    convbar / Dsp : FAILED rc=15
      QnnDsp <E> Error from rpc for graph pf_convbar in context 1

This is a **different class of failure** from everything before it. Not
`validateOpConfig`, not `OP_PACKAGE_NOT_FOUND` -- op support is now fully
cleared. It is an RPC failure during graph construction on the DSP, i.e. a
resource limit. The graph grew from 1108 ops / 225 Transposes (which DSP
accepted and ran at 1411 ms) to 1588 ops / 560 Transposes.

Verbose log, all of it:

    260.4ms [INFO]  validateOpConfig node_Add_7328:qti.aisw:ElementWiseBinary   <- last op, near graph end
    264.3ms [INFO]  rpcMemoryAlloc 158233936 isInit 1                           <- 158 MB context allocated OK
   2005.0ms [ERROR] QnnDsp <E> Error from rpc for graph pf_convbar in context 1
   2006.0ms [ERROR] QnnDsp <E> fail on dspGraph->finalizeInit()
   2006.0ms [ERROR] Finalize Graph for Idx = 0 failed with error = 6022

**Every op validated.** Validation walked to the end of the graph and passed;
the 158 MB context allocated; the failure is `finalizeInit()` with error 6022.

My first reading of this was a per-graph capacity limit -- 1108 ops finalize,
1588 do not. **That reading is wrong**, and the split experiment below is what
disproved it.

## Verdict

| claim | result |
|-------|--------|
| DSP baseline 1384.5 ms reproduces | **yes** -- 1411.3 ms, 2% |
| Conv1x1 helps at 113 tokens where it hurt at 50 | **yes** -- 1.29x vs 0.84x |
| Conv1x1 makes the prefill trunk faster on CPU | **yes** -- 384.5 -> 297.6 ms, **87 ms** |
| rewrites are numerically safe | **yes** -- 5.881e-07, fp32 accumulation order |
| RmsNorm fusion can be defeated | **yes** -- but not by the documented method |
| prefill trunk runs faster on DSP | **no** -- graph too large to finalize |

**The headline is the CPU result.** 87 ms off the prefill trunk, 1.29x, from
two exact rewrites and no accelerator involved. Combined with the decode
finding (int8 CPU, 2.39x), the pattern across this whole effort is consistent:
every real win on this SoC has come from giving the Kryo a better graph, and
none from moving work to HTA or DSP.

**The DSP question is now open rather than closed.** Before this, the expert
prefill on DSP was 2.37x slower than CPU and the reason was assumed to be the
silicon. It is not -- it is the FullyConnected kernel, which is 14.9x off the
Conv2d kernel for identical arithmetic. The conv form cannot be *finalized* as
one graph, but the Flow C tile model exists precisely to split graphs. Splitting
the conv trunk into 2-4 DSP graphs is the obvious next experiment, and it now
has a concrete target: 112 conv ops that individually measure ~15x faster than
what produced the 1411 ms.

## Reproduce

    python3 rewrite_matmul_to_conv1x1.py smolvlm_expert_prefill_trunk.onnx \
            -o smolvlm_expert_prefill_trunk_conv.onnx
    python3 rewrite_rmsnorm_scale_barrier.py --in  smolvlm_expert_prefill_trunk_conv.onnx \
                                             --out smolvlm_expert_prefill_trunk_convbar.onnx
    # convert + quantize with the 2-input calibration list (position_ids is dead)
    # calibration: expert_rewrite/prefill_calib, float32 raws, [1,960,113] layout

## Splitting the conv trunk across DSP graphs

Error 6022 at `finalizeInit()` is a per-graph capacity limit, and Flow C's
whole model is tiles, so the question is whether N smaller conv graphs finalize
and what they cost together.

The trunk splits cleanly: 16 layers of exactly 99 nodes, residual stream at
`add_5, add_11, ... add_89` (node indices 102, 201, ... 1488). A cut at layer
boundary L needs

    graph A:  vlm_embeds, attention_mask  ->  present_{key,value}_0..L-1, add_<L>
    graph B:  add_<L>,    attention_mask  ->  present_{key,value}_L..15,  add_95

Both halves need `attention_mask`; the only new cross-tile tensor is the
residual stream itself, `[1,113,960]` -- 108 KB at int8. The 32 KV outputs
cross to the host in either arrangement, so splitting adds exactly one
108 KB handoff per cut, not a redistribution of the KV traffic.

Boundary activations for calibration are tapped from the full trunk in
onnxruntime (`add_23`, `add_47`, `add_71`, 8 samples each) rather than
synthesized, so each sub-graph is calibrated on the real distribution its
predecessor produces.

### The split result, and what it actually proves

Cut at layer 8, both halves quantized cleanly and symmetrically (794 ops,
56 Conv2d, 79 MB each):

| build | ops | DSP | CPU |
|-------|-----|-----|-----|
| `pf_trunk_q`  MatMul | 1108 | **finalizes, 1411 ms** | 384.5 ms |
| `pf_convbar_q` conv  | 1588 | fails 6022 | 297.6 ms |
| `s2A_q` conv half    |  794 | **fails 6022** | 149.2 / 149.7 / 149.0 ms |
| `s2B_q` conv half    |  794 | **fails 6022** | 148.9 / 147.3 / 148.8 ms |

**794 conv ops fail where 1108 MatMul ops succeed.** So the DSP failure is not
about graph size at all -- halving it changed nothing -- and my capacity
reading above was wrong. Something about the Conv2d formulation in this graph
is what v66 will not finalize.

The CPU column is a clean control: 149.2 + 148.9 = 298.1 ms against the
unsplit conv trunk's 297.6 ms. Splitting is free on CPU, which confirms the
cut itself is sound and the halves are doing the same total work.

Note this is not a general conv problem on the DSP: `probe_mlpconv`, three
Conv1x1 at [1,720,1,50] -> 2048, composes and runs on DSP at 1.31 ms. The
difference is that the probe was built natively in conv layout with zero
surrounding Transposes, while the rewritten trunk wraps every conv in a
rank-3 -> rank-4 -> rank-3 Transpose/Reshape dance (560 Transposes).

### Down to a single layer: it is the conv formulation, not the size

One decoder layer, extracted twice from the two trunks so the only difference
is whether its seven linears are `Conv2d` or `FullyConnected`:

| probe | ops | DSP | CPU |
|-------|-----|-----|-----|
| `L1mm`   73 ops, 9 MatMul        | 73  | **88.54 / 88.55 / 88.47 ms** | 24.14 / 22.14 / 25.18 ms |
| `L1conv` 103 ops, 7 Conv + 2 MatMul | 103 | **FAILED -- rpc / 6022** | 18.51 / 18.51 / 18.50 ms |

**A single layer in conv form will not finalize on the DSP**, while the same
layer in MatMul form composes and runs. Graph size is conclusively not the
variable: 73 ops succeed, 103 fail, 794 fail, 1108 succeed, 1588 fail --
the split is entirely along MatMul vs Conv2d.

Two useful cross-checks fall out:

* `L1mm` on DSP x 16 layers = 1416 ms against the whole trunk's measured
  1411 ms. The per-layer decomposition is faithful and the DSP trunk time is
  purely layer-serial.
* `L1conv` vs `L1mm` on CPU is 18.51 vs 24.14 ms = **1.30x**, matching the
  whole-trunk 1.29x. The conv win on CPU is uniform per layer, not an artifact
  of some particular block.

### Why, and where the DSP path actually lies

This is not a general Conv2d problem on v66. `probe_mlpconv` -- three Conv1x1
at `[1,720,1,50]` into 2048 channels -- composes on DSP and runs at 1.31 ms.
The difference is layout:

    probe_mlpconv   built NATIVELY in conv layout, input declared NHWC
                    [1,1,50,720], zero Transpose, zero Reshape       -> DSP OK
    L1conv          matmul-layout graph with each conv wrapped in a
                    rank-3 -> rank-4 -> rank-3 Transpose/Reshape pair -> DSP FAILS

So the DSP will run these convs; it will not run them wrapped in the layout
dance that `rewrite_matmul_to_conv1x1.py` produces when applied to a
transformer written in `[B, M, K]` form.

**The viable path is therefore extraction, not whole-graph rewriting** -- which
is exactly what vision already does: `REPRODUCTION.md` §267 records 50
*extracted* Conv1x1 kernels as the only thing HTA ever runs, with the
surrounding graph left on another backend. The equivalent for the experts is to
pull the 112 linears out as native-conv-layout sub-models, run those on DSP,
and leave attention, norms and the residual stream on the CPU. That is a
tiling exercise for Flow C, not another rewrite.

## Final verdict

| question | answer |
|----------|--------|
| does Conv1x1 help prefill? | **yes, on CPU: 384.5 -> 297.6 ms, 1.29x, 87 ms** |
| is it numerically safe? | yes -- 5.881e-07, fp32 accumulation order |
| does the DSP baseline reproduce? | yes -- 1411.3 vs 1384.5 ms recorded |
| can the RmsNorm fusion be defeated? | yes -- with a non-removable scale barrier, not the documented config |
| does splitting into N DSP graphs help? | **no -- conv fails on DSP at every granularity, down to one layer** |
| is the DSP conv path dead? | not dead, but it needs native-conv-layout extraction, not this rewrite |

## Native-conv-layout extraction

The split experiment showed the DSP rejects convs *wrapped in the matmul-layout
transpose dance*, but runs them fine when the graph is natively conv-shaped
(`probe_mlpconv`, 1.31 ms). So the experts' 112 linears are extracted as
native-conv sub-models instead of rewritten in place.

Per layer the seven linears form three blocks, cut at their real activation
boundaries in `smolvlm_expert_prefill_trunk.onnx`:

| block | input | convs | shapes |
|-------|-------|-------|--------|
| `nc_qkv`   | `mul_1` (post-RMSNorm)        | 3 | 960->960, 960->320, 960->320 |
| `nc_oproj` | `_unsafe_view_2` (attn output)| 1 | 960->960 |
| `nc_mlp`   | `mul_14` (post-RMSNorm)       | 3 | 960->2560, 960->2560, 2560->960 |

All built directly as `[1, C, 1, 113]` with `Conv(1x1)` and real weights lifted
from the trunk, and converted **without** `--preserve_io layout` so the
converter declares the input NHWC `[1,1,113,C]` and emits no layout ops at all.

Calibration is the genuine activation each block sees, tapped from the full
trunk in onnxruntime over the same 8 samples (`mul_1` range -5.31..4.49,
`mul_14` range -5.18..4.58) rather than synthesized.

### Result: it composes everywhere, and HTA finally wins a block

All three blocks composed on **DSP and HTA** -- the first time anything
expert-shaped has reached HTA at all. 50 iters, 3 interleaved repeats,
performance governor, gap median (us):

| block | CPU | HTA | DSP | best |
|-------|-----|-----|-----|------|
| `nc_qkv`   3 convs, 960->960/320/320 | **1193** | 2455 | 3075 | CPU |
| `nc_oproj` 1 conv, 960->960          | **925**  | 1773 | 932  | CPU (DSP ties) |
| `nc_mlp`   960->2560 x2, 2560->960   | 4415     | **2452** | 3820 | **HTA, 1.80x** |
| per-layer total                       | 6533     | 6680 | 7827 | **4570 mixed** |

Correctness, each block against an independent numpy reference on its real
tapped input:

    nc_mlp   vs numpy SwiGLU   max|diff| 5.364e-07
    nc_qkv Q vs numpy matmul   max|diff| 1.192e-06
    nc_qkv K / V               max|diff| 9.537e-07
    nc_oproj vs numpy matmul   max|diff| 3.129e-07

**HTA runs the MLP block 1.80x faster than the Kryo** (2452 vs 4415 us), and
its three repeats are tight (2381-2485) while DSP's are not (2495-4192, and
similarly noisy on qkv). The pattern is size-driven: HTA wins the block with
the big GEMMs (960x2560 twice plus 2560x960) and loses both projection blocks,
which is what you would expect from an engine that needs enough arithmetic to
amortise its dispatch.

### What a tiled expert is worth

Best-of-backend per layer is 4570 us against 6533 us CPU-only, i.e. **1963 us
per layer saved, 31.4 ms over 16 layers**, all of it from moving the MLP to HTA.

Against the handoff: the MLP tile needs `[1,113,960]` in and out, 108 KB each
way at int8. At the measured 8.6 us dispatch latency plus DDR movement that is
roughly 0.23 ms round trip per layer, ~3.7 ms over 16 -- so **net ~27.7 ms**.

Put in context, the extracted linears are 6.53 ms x 16 = 104.5 ms of the
297.6 ms CPU conv trunk, i.e. only **35%** of it; the other 65% is attention,
norms, RoPE and layout, none of which any accelerator will take. So the
achievable saving is ~27.7 ms on 297.6 ms, **about 9%**.

### Honest assessment

The structural result is real and new: native-conv extraction is the form in
which the experts *do* reach both accelerators, HTA included, and it produces
the first measured accelerator win on expert work. The performance result is
modest -- 9% of the prefill trunk -- because the linears are a minority of the
wall time once the CPU is running them in conv form, and only the largest of
the three blocks is worth exporting.

For comparison, the two pure-CPU rewrites in this document are worth 87 ms on
the same graph. The ranking is unchanged: give the Kryo a better graph first,
export the MLP to HTA second.

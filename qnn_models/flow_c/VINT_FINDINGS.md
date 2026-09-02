# ViNT on QRB5165 (SM8250, v66) — applying the SmolVLA findings

Board `root@10.44.120.201`, QAIRT 2.45.0.260326, `performance` governor on all 8 cores
(restored to `schedutil` afterwards). Statistic is `gap_median_us` from
`profile_seg <ctx.bin> <backend.so> <iters> --gap-us 3000`, which is what the runtime
actually pays per dispatch; `loop_median` is quoted where it differs materially.
Everything below was measured this session. Nothing here is committed.

Artifacts: `qnn_models/flow_c/gen/vint_work/` (gitignored). Reusable rewrites:
`rewrite_gelu_to_sigmoid.py`, `rewrite_gemm_to_conv1x1.py` (both in this directory).

---

## 0. Headline

The ViNT decoder can be moved off the CPU entirely. Three ops out of 177 were blocking it.

| decoder variant | backend | gap_median | vs baseline |
| --- | --- | ---: | ---: |
| body, exact GELU (baseline) | CPU fp32 | 12315.7 us | 1.00x |
| body, GELU -> `x*Sigmoid(1.702x)` | CPU fp32 | 9498.6 us | 1.30x |
| body, sigmoid-GELU, int8 | **DSP** | 8386.5 us | 1.47x |
| body, sigmoid-GELU + Conv1x1, int8 | **DSP** | **6243.6 us** | **1.97x** |
| waypoint tail (8 ops) | CPU fp32 | 74.9 us | |
| whole decoder tile, as shipped | GPU fp16 | 16205.8 us | 0.76x |

3 interleaved repeats of 60 iters each; per-repeat spread was 1.5% (CPU) to 3.4% (DSP).

End to end, on the `vint_par` structure (goal_enc ‖ obs_enc, then decoder):

| config | goal_enc | obs_enc | decoder | makespan |
| --- | --- | --- | --- | ---: |
| as shipped (re-measured) | CPU int8 5.9 | DSP int8 11.3 | CPU fp32 12.6 | 23.9 ms |
| + sigmoid-GELU | CPU int8 5.9 | DSP int8 11.3 | CPU fp32 9.5 | 20.8 ms |
| + decoder on DSP + Conv1x1 | CPU int8 5.9 | DSP int8 11.3 | DSP int8 6.2 (+0.07 tail) | **17.6 ms** |

**1.36x end to end.** The shipped 23.955 ms makespan claim in `vint_par.json` reproduces
(re-measures at 23.9 ms); the encoder half of that binding is already optimal and is
left alone.

---

## 1. Inventory

`qnn_models/flow_c/gen/onnx/vint.onnx`: 1931 nodes, opset 17, inputs `obs_img`
[1,18,64,85] fp32 and `goal_img` [1,3,64,85] fp32; outputs `output` (distance, [1,1])
and `3138` (5 waypoints, [1,5,4]).

**359.8 MMAC per inference**, measured by running the graph in onnxruntime with the real
shapes and summing Conv/Gemm/MatMul. The `88.7M MACs` in the `vint.json` comment is not
this number.

| tile | ONNX nodes | QNN ops | MMAC | weight bytes (fp32) |
| --- | ---: | ---: | ---: | ---: |
| `vint_goal_enc` (tile0, ONNX 6-536) | 534 | 241 | 39.5 | 4.7 MB |
| `vint_obs_enc` (tile1, ONNX 540-1069) | 531 | 241 | 230.0 | 4.7 MB |
| `vint_decoder` (tile2, ONNX 538 + 1071-1930) | 866 | 177 | 90.3 | 58.5 MB |

Encoder QNN op mix (each, identical): 74 Eltwise_Binary, 65 ElementWiseNeuron,
65 Conv2d, 17 Pool, 16 DepthWiseConv2d, 1 Transpose, 1 StridedSlice (tile0) /
1 Split (tile1), 1 FullyConnected, 1 Concat.

Decoder QNN op mix: 55 Reshape, 33 Transpose, **23 FullyConnected**, 17 Eltwise_Binary,
12 Gather, 9 ElementWiseNeuron, **8 MatMul (two dynamic operands — attention)**,
8 LayerNorm, 4 Softmax, 2 StridedSlice, 2 ScatterNd, 1 L2Norm, 1 CumulativeSum, 1 Concat.

Constant-weight vs dynamic: whole net has **25 constant-weight** MatMul/Gemm and
**8 dynamic** (QK^T and AV of the 4 attention layers). The encoders contain zero
MatMul — they are pure Conv — so Finding 1 has nothing to rewrite there. Every
constant-weight linear in ViNT lives in the decoder.

Decoder MMAC: 8 FFN linears 58.7 (7.34 each, [7,512]x[512,2048] and [7,2048]x[2048,512]),
4 qkv 22.0, 4 out_proj 7.3, `output_layers.0` 1.8, attention 0.2, rest < 0.5.
Arithmetic intensity is **1.5 MAC per weight byte** — the decoder is weight-bandwidth-bound,
not compute-bound. That single fact drives most of what follows.

> **Trap:** `snpe-dlc-info`'s "Total MACs per inference" reports a FullyConnected's MACs as
> its weight count, i.e. it assumes one token. It says `14M` for a decoder that does 90.3M.
> It is right about Conv (says 36M for a tile that does 39.5M).

---

## 2. Re-measured cells vs `measurements/qrb5165_v66.json`

| cell | recorded | re-measured | ratio |
| --- | ---: | ---: | ---: |
| `vint_par/vint_goal_enc` dsp | 5809.1 | 8015.6 | **1.38x worse** |
| `vint_par/vint_goal_enc` cpu int8 | 7075.3 | 5891.3 | 0.83x |
| `vint_parallel3/vint_goal_enc` cpu@fp32 | 13594.6 | 10546.7 | 0.78x |
| `vint_par/vint_obs_enc` dsp | 9532.6 | 11617.2 | **1.22x worse** |
| `vint_par/vint_obs_enc` cpu int8 | 15549.4 | 15392.4 | 0.99x |
| `vint_parallel3/vint_obs_enc` cpu@fp32 | 73844.3 | 66194.7 | 0.90x |
| `vint_parallel3/vint_decoder` cpu@fp32 | 14422.6 | 12592.8 | 0.87x |
| `vint_parallel3/vint_decoder` cpu int8 | 42431.4 | 35669.7 | 0.84x |
| `vint/vint_encoders` dsp | 14213.0 | 16360.2 | **1.15x worse** |
| `vint/vint_encoders` cpu@fp32 | 84163.0 | 77490.4 | 0.92x |
| `vint/vint_decoder` gpu fp16 | 16425.0 | 16371.1 | 1.00x |
| `vint/vint_encoders` gpu fp16 | 55854.0 | 55677.8 | 1.00x |

Finding 5 holds, and in both directions: the **DSP** cells are 15-38% optimistic, the
**CPU** cells 8-22% pessimistic. The practical consequence is that the goal encoder's
DSP-vs-CPU ordering **inverts** — CPU int8 5.89 ms beats DSP 8.02 ms, where the table
says the opposite. (`vint_par` already places it on the CPU, so no binding changes; but
anything reading the table to make that choice would get it wrong.)

GPU cells are the one thing that reproduces exactly, and they are also the most stable
of any backend (std 130-190 us against 600-1500 us for CPU and DSP).

---

## 3. Which findings applied, and what they were worth

### Finding 4 (int8 on the CPU) — **inverts on the decoder**

- Encoders, CPU: fp32 66.2 -> int8 15.4 ms (**4.3x**, holds).
- Decoder, CPU: fp32 12.6 -> int8 35.7 ms (**2.8x slower**).

The quantized decoder DLC is clean — 176 of 177 op outputs are `uFxp_8`, no conversion
ops. The cause is the QNN **CPU** backend's int8 kernels for non-conv ops. Per-op chain
probes on a [1,7,512] tensor (chains of K vs K+64 ops, slope = per-op cost):

| op | CPU fp32 | CPU int8 | DSP int8 | HTA int8 |
| --- | ---: | ---: | ---: | ---: |
| Eltwise_Binary | 2.69 us | 5.64 us | 4.45 us | ~14 us |
| ElementWiseNeuron | 2.47 us | **23.9 us** | 7.37 us | ~14 us |

The decoder is 154 of 177 ops non-linear; the encoders are conv-dominated. That is the
whole difference. So: **int8 on the ViNT decoder is only usable on an accelerator**, which
is what motivated everything in §4.

### Finding 1 (FullyConnected vs Conv1x1) — reproduces; magnitude scales with token count

Probe = the decoder's own FFN stack, 4x [512->2048->512], 33.6 MB of weights, three op
forms at identical arithmetic and identical weight bytes, validated equivalent in
onnxruntime (rel 1e-6):

| N | form | CPU fp32 | CPU int8 | DSP int8 | HTA int8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | FullyConnected | 3487 | 1236 | 2663 | 6054 |
| 7 | Conv1x1 `[1,C,1,S]` | 10111 | 1425 | **1337** | **2431** |
| 7 | Conv1x1 `[N,C,1,1]` | 30202 | 2045 | 6135 | 5857 |
| 56 | FullyConnected | 9034 | 2702 | 6903 | 41275 |
| 56 | Conv1x1 `[1,C,1,S]` | 21888 | 2721 | 5728 | **2469** |
| 56 | Conv1x1 `[N,C,1,1]` | 209535 | 6755 | 28027 | 41676 |

- FC -> Conv1x1: **HTA 2.49x at N=7, 16.7x at N=56**; **DSP 1.99x at N=7, 1.21x at N=56**;
  CPU int8 0.87-1.00x (no gain, matching SmolVLA's 1700 vs 1720).
- The gap is N-dependent because HTA's FullyConnected cost scales with the token count
  (6.05 -> 41.3 ms for 8x N) while Conv1x1 is nearly flat (2431 -> 2469). ViNT has only
  **7 tokens**, so it sits at the small end of the effect.
- **`[N,C,1,1]` (token as batch) is a 2-9x loss on every backend.** Layout matters more
  than the rewrite — confirmed here in a second failure mode (no compose failure, just
  catastrophic slowness) distinct from SmolVLA's rank-3 finalize failure.

### Finding 2 (dispatch floor) — reproduced, and one correction

- DSP floor confirmed at **393-422 us** (3-op graph); HTA floor **531-537 us**.
- The HTA fit `873 + 158*ops + 1.393*MMAC` predicts the new probes well:
  `convs_n7` predicted 2219 vs measured 2431; `convs_n56` predicted 2791 vs 2469.
- **Correction: the 158 us/op term is Conv-specific.** HTA elementwise/activation ops cost
  **14.0 us/op**, DSP 4.5-7.4 us/op. Do not apply 158 us/op to a graph of cheap ops.
- Applying the fit to ViNT: encoders 241 ops -> **~39 ms**; decoder 177 ops -> **~29 ms**.
  Against DSP 8.0/11.6 ms and CPU 5.9/15.4 ms, **HTA loses by 3-5x on op count alone**,
  before op support is even consulted. HTA is dead for ViNT for a quantitative reason,
  not just the `StridedSlice`/`Split` blockers recorded in the bindings.

### Finding 3 (op support) — extended, plus a DSP map that did not exist

Single-op probes, int8, each op wrapped between two `Mul`s so it is neither first nor last:

| op | DSP | HTA |
| --- | --- | --- |
| Clip, Concat | OK | OK |
| Transpose (converter elides it between elementwise ops) | OK | OK |
| Gather | OK | unsupported op Gather |
| StridedSlice | OK | unsupported op StridedSlice |
| Reshape (rank-changing) | OK | unsupported op Transpose |
| LayerNorm | OK | unsupported op Transpose |
| Softmax (not first) | OK | unsupported op Transpose |
| **MatMul, two dynamic operands, rank 4** | **OK** | unsupported op Transpose |
| ReduceL2 | OK | unsupported op ReduceSum |
| CumulativeSum | **Param[0] has incorrect Value 1** | unsupported op CumulativeSum |
| ScatterNd | **Param[0] has incorrect Value 1** | unsupported op Transpose |
| fused GELU (ElementWiseNeuron type 1) | **Param[0] has incorrect Value 1** | (same, per Finding 3) |

Two things worth carrying forward:

1. **The DSP does support two-dynamic-operand MatMul.** Attention is HTA-impossible by
   construction but *not* DSP-impossible. The `vint_par.json` note
   "`dsp@int8`: Param[0] has incorrect Value 1" was not about the transformer at all.
2. **`Erf` alone is not convertible** ("Converter does not support 'Erf' op type"). The
   converter only accepts `Erf` inside the Div/Erf/Add/Mul/Mul GELU pattern, which it then
   fuses into exactly the op the DSP rejects. Fusing is not optional, so GELU must be
   expressed some other way.

---

## 4. The actual change: getting the decoder onto the DSP

Bisected with 8 prefix subgraphs of the decoder (compose-only, DSP): p1-p6 compose,
p7 fails. p7 is the first FFN and the delta p6->p7 contains the first fused GELU.
Three ops out of 177 block the whole tile.

**Cut 1 — the waypoint tail.** `onnx.utils.extract_model` at
`/action_predictor/action_predictor.0/Gemm_output_0` splits the decoder into

- `dec_body`: 169 QNN ops, all the arithmetic;
- `dec_tail`: 8 QNN ops (2 StridedSlice, 2 ScatterNd, 1 Reshape, 1 L2Norm,
  1 CumulativeSum) operating on a [1,20] tensor — the cumulative-sum integration of
  waypoint deltas. Bit-exact recombination. Costs **74.9 us** on the CPU.

This removes the `CumulativeSum` and both `ScatterNd`.

**Cut 2 — GELU.** `rewrite_gelu_to_sigmoid.py` replaces the exact erf-GELU with
`x * Sigmoid(1.702*x)` (Mul / Sigmoid / Mul — Sigmoid is accepted everywhere; the ViNT
encoders already run 65 of them on the DSP). 4 sites. After this the body composes on
CPU fp32, CPU int8 **and DSP int8**.

**Cut 3 — Conv1x1.** `rewrite_gemm_to_conv1x1.py` rewrites all 23 rank-2 constant-weight
linears as 1x1 Conv over `[1,K,N,1]`. (Run onnxruntime `ORT_ENABLE_BASIC` first so the
12 MatMul+Add linears become rank-2 Gemm; see §5.) Validated bit-exact-to-fp32-roundoff
(4.8e-7) in onnxruntime.

The converter does **not** elide the wrapper: 177 -> 323 QNN ops, Transpose 33 -> 124,
Reshape 54 -> 109 — exactly the hazard Finding 1 describes. But unlike the rank-3
in-place case, **the v66 DSP finalizes it happily**, and it is still 1.34x faster than the
FullyConnected form. The rank-2 -> rank-4 wrapper is benign where the rank-3 one was fatal.

### Accuracy

Against the exact-GELU fp32 ONNX in onnxruntime, on the 8 real calibration samples:

| variant | waypoint err (frac. of output range) | distance 20.197 -> |
| --- | ---: | --- |
| sigmoid-GELU only, CPU fp32 | mean 0.35%, max 0.51% | 20.080 |
| + int8 DSP, FullyConnected | mean 1.48%, max 2.38% | 20.022 |
| + int8 DSP, Conv1x1 | mean 1.72%, max 2.51% | 20.022 |

int8 costs ~1.5-1.7% of the waypoint range; the GELU approximation costs 0.35% and is
not the dominant term. Whether 1.7% on waypoints is acceptable is a policy-level call
that has not been made here — the fp32-CPU sigmoid-GELU variant (1.30x, 0.35% error) is
the conservative fallback.

---

## 5. What does NOT apply, and why

**ONNX-level constant folding — worth exactly zero.** onnxruntime `ORT_ENABLE_BASIC`
collapses the decoder 866 -> 206 nodes and each encoder 534 -> 241, bit-exact
(max_abs_diff 0.0), removing all 299 `Constant`, 102 `Unsqueeze`, 54 `Shape`, 41 `Gather`
and the `Range`/`Mod`/`Equal`/`Where` machinery that torch.export leaves behind. The
resulting DLCs are **op-for-op identical** to the unfolded ones (decoder 177 ops both
ways, encoder 241 both ways, same histogram, same reported MACs). `snpe-onnx-to-dlc`
already performs the identical folding. It is still useful as a *preprocessing* step for
the Conv1x1 rewrite, because it turns MatMul+Add into rank-2 Gemm, but it buys no time.

**`build_nativeconv_blocks.py` / native-conv extraction.** ViNT's export does not name its
linears `linear`/`linear_1`.../`linear_6`, so the script would need reworking — but the
bigger reason is that at **7 tokens** the FC->Conv1x1 gap on DSP is only 2.0x (against
15x at N=56). The in-place rewrite already captured 1.34x of that; a full native-layout
rebuild of the decoder (attention included) could recover at most ~1.3x more on a 6.2 ms
tile. Not worth it. Native-conv extraction pays when the sequence dimension is large;
ViNT's is 7.

**HTA, anywhere in ViNT.** Ruled out by the op-count term (§3, Finding 2): 241 ops
-> ~39 ms for an encoder the DSP does in 11.6, 177 ops -> ~29 ms for a decoder the DSP does
in 6.2. The op-support blockers (`StridedSlice`, `Split`, `Transpose`, `Gather`,
`ReduceSum`, `CumulativeSum`, no dynamic MatMul) are real but redundant — even a fully
supported ViNT would lose on HTA by 3-5x. ViNT is a many-op, low-MMAC-per-op network,
which is the shape HTA is worst at.

**Encoder changes of any kind.** The encoders are pure Conv, so Finding 1 has nothing to
rewrite; they are already int8 and Finding 4 already holds there (4.3x). The re-measured
batch sweep (1/2/3/6 frames per dispatch) gives marginal costs of **0.99 ms/frame on DSP**
and **1.88 ms/frame on CPU int8**, with intercepts 5.7 and 4.1 ms. Every split of the
6 obs frames across the two lanes is worse than the current one, because the decoder
depends on both encoders and the CPU also has to do the goal frame:

| split | DSP lane | CPU lane | encoder stage |
| --- | ---: | ---: | ---: |
| 6 obs on DSP, goal on CPU (**shipped**) | 11.3 | 5.9 | **11.3 ms** |
| 5 obs DSP, goal + 1 obs CPU | 10.6 | 11.9 | 11.9 ms |
| 4 obs DSP, goal + 2 obs CPU | 9.6 | 11.8 | 11.8 ms |
| 3 obs DSP, goal + 3 obs CPU | 8.7 | 13.7 | 13.7 ms |
| goal on DSP, 6 obs on CPU | 8.0 | 15.4 | 15.4 ms |

The shipped `vint_par` encoder split is optimal on re-measured numbers.

**The GPU** — confirmed, and the contrast with SmolVLA is real, but it is not a win.

- Decoder: **16.21 ms fp16** (reproduces the recorded 16.425 exactly) against a CPU fp32
  cell of 12.32 ms — **1.32x slower**, not the 4-6x SmolVLA saw on every component. The
  reason ViNT does better on the GPU is the same reason its decoder is slow everywhere:
  it is weight-bandwidth-bound (57.6 MB fp32 for 90 MMAC), and the GPU has its own
  bandwidth to spend. The decoder body in fp32 on the GPU is 18.12 ms, so most of the
  fp16 build's advantage is halved weight bytes, consistent with that reading.
- Encoders: **55.68 ms fp16** against 16.07 ms on DSP — a **3.5x loss**, fully in line
  with SmolVLA.
- The `_gpu_note` in `vint.json` ("the GPU is the faster of the two in situ ... the only
  one whose cost does not depend on its neighbours") is a statement about CPU contention
  from co-scheduled networks, not about ViNT. With the decoder on the DSP at 6.2 ms the
  argument is moot: the decoder no longer touches the contended CPU at all, and it is
  2.6x faster than the GPU option. **The GPU should be dropped from the ViNT bindings.**

---

## 6. Incidental measurements worth keeping

**CPU weight-streaming model.** Fitting the FFN probe across N (8.4 / 58.7 / 469.8 MMAC at
constant 33.6 MB of weights):

```
CPU fp32 :  2438 us + 14.04 us/MMAC     intercept => 33.6 MB / 2.44 ms = 13.8 GB/s
CPU int8 :   739 us +  4.18 us/MMAC     intercept =>  8.4 MB / 0.74 ms = 11.4 GB/s
```

The intercept is weight streaming and it dominates at ViNT's 7 tokens (70% of the fp32
cost). Marginal rates are 71 GMAC/s fp32 and 239 GMAC/s int8 — the latter well above
Finding 2's 144 GMAC/s Conv1x1 figure, so FullyConnected is not a bad CPU kernel; the
decoder was simply never compute-bound.

**DSP fits from the same probe** (Conv1x1 form): `709 us + 10.7 us/MMAC` (93 GMAC/s);
FullyConnected form `2122 us + 9.2 us/MMAC` — same slope, 3x the fixed cost, i.e. QNN's
DSP FullyConnected penalty is ~215 us per op against ~39 us per Conv1x1 op.

**The QNN CPU backend emits no per-op profile events.** `profile_per_op` with
`QNN_PROFILE_LEVEL_DETAILED` returns `unique_op_events: 0` on `libQnnCpu.so`, so CPU
time cannot be attributed per op. Chain probes (§3) are the workaround.

**The converter deletes Transposes** that sit between elementwise ops, and folds `Pad`
into `Conv`, and folds `Reshape`/`Transpose` triples around `GlobalAveragePool`. A
Transpose that survives into the DLC (33 in the decoder) is one that is structurally
required by a MatMul or a rank change, not one the converter failed to notice.

---

## 7. Reproduce

```bash
WORK=qnn_models/flow_c/gen/vint_work
D=qnn_models/slicing_study/gen/vint/vint_par_enc__f1a55807fc80

# 1. split off the waypoint tail (bit-exact)
python3 -c "import onnx; onnx.utils.extract_model('$D/tile2.onnx','$WORK/dec/dec_body.onnx',
  ['t_compress_goal_enc_Gemm_output_0','t_compress_obs_enc_Gemm_output_0'],
  ['output','/action_predictor/action_predictor.0/Gemm_output_0'], check_model=False)"

# 2. GELU -> x*Sigmoid(1.702x)
python3 qnn_models/flow_c/rewrite_gelu_to_sigmoid.py $WORK/dec/dec_body.onnx $WORK/dec/dec_body_g.onnx

# 3. ORT BASIC fold (turns MatMul+Add into rank-2 Gemm), then Gemm -> Conv1x1
#    (the fold itself is DLC-neutral; it only exposes the Gemms to the rewrite)
python3 -c "import onnxruntime as ort
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
so.optimized_model_filepath='$WORK/dec/dec_body_gf.onnx'
ort.InferenceSession('$WORK/dec/dec_body_g.onnx',so,providers=['CPUExecutionProvider'])"
python3 qnn_models/flow_c/rewrite_gemm_to_conv1x1.py $WORK/dec/dec_body_gf.onnx $WORK/dec/dec_body_gc.onnx

# 4. convert / quantize (docker qnn-convert, QNN_SDK mounted at /qnn)
snpe-onnx-to-dlc --input_network dec_body_gc.onnx --output_path dec_body_gc.dlc \
  -d t_compress_goal_enc_Gemm_output_0 1,512 -d t_compress_obs_enc_Gemm_output_0 6,512
qairt-quantizer --input_dlc dec_body_gc.dlc --output_dlc dec_body_gc_q.dlc \
  --input_list body_list.txt --act_bitwidth 8 --weights_bitwidth 8

# 5. on the board
qnn-context-binary-generator --model libQnnModelDlc.so --backend libQnnDsp.so \
  --dlc_path dec_body_gc_q.dlc --binary_file dec_body_gc_q__dsp8 --output_dir ctx
profile_seg ctx/dec_body_gc_q__dsp8.bin libQnnDsp.so 60 --gap-us 3000
```

Contexts left on the board in `/root/vint_dec/ctx/`: `dec_body__cpu32`,
`dec_body_g__cpu32`, `dec_body_g_q__cpu8`, `dec_body_g_q__dsp8`, `dec_body_gc__cpu32`,
`dec_body_gc_q__dsp8`, `dec_body__gpu`, `dec_tail__cpu32`. Everything else pushed this
session (`/root/vint_probe`, `/root/vint_opcost`, `/root/vint_opsup`, `/root/vint_bisect`)
has been deleted; the governor is back on `schedutil`.

---

## 8. HTA on the decoder: the measured first-blocker (addendum)

§5 rules HTA out for the decoder on the op-count term. That conclusion stands, but
it was an inference; here is the actual rejection, from composing both decoder
variants on `libQnnHta.so` with `--log_level verbose`:

    dec_body_gc_q (with Conv1x1)   FAILED
    dec_body_g_q  (without)        FAILED
      QnnHtaHTA op Reshape supports only equal Input and Output dimensions
      last tensors: t_compress_obs_enc_Gemm_output_0 -> _Reshape_output_0

It dies on the **first Reshape**, at the encoder->decoder boundary, before reaching
any conv or any attention. Identical with and without Conv1x1, so the rewrite
neither causes nor fixes it.

**The 1x1 convs are not and cannot be the problem.** A `Conv`'s weight is an
initializer by definition, and `dec_body_gc.onnx` carries 23 Conv with
**constant weights and zero dynamic** — the rewrite only ever fires on
constant-weight MatMuls, because that is the only case a conv can express.

What *is* a variable-tensor problem is the **8 dynamic MatMuls** the decoder
retains (attention `QK^T` and `A.V`). Those cannot reach HTA at any rank; the
SmolVLA study got byte-identical `unsupported op MatMul` for rank-4 batched,
rank-3 batched, rank-3 single-head and rank-2 per-head. Both operands are
activations and HTA has only a constant-weight kernel path.

So the ordering is: **Reshape blocks first, dynamic MatMul blocks irrecoverably
behind it.** Fixing the Reshape would only move the failure to the attention.
The registry's exclusion list for this tile names both (`matmul_s8` alongside
`layer_norm_s8`, `cat2_c1_s8`, `softmax_s8`).

Caveat on the op-count argument in §5: the ~39 ms estimate used a flat
158 us/op, and §6 of this same document corrects that figure as Conv-specific
(elementwise is ~14 us/op). With ViNT's actual op mix the estimate is much lower
and should not be trusted to better than a factor of two. **The op-support
blockers above are the hard evidence; the cost argument is supporting only.**

# Slicing granularity on QRB5165 (v66): how fine is worth it

Every number below was measured on the physical board at `10.44.120.201`
under the `performance` governor with `qnn_models/runtime/profile_segments.cpp`
(`/root/qnn_runtime/profile_seg`), CPU cells unmasked. Nothing is projected.
Each row is reproducible from one line of `experiments.jsonl`, which carries
the boundary tensor names, the IR op ranges they resolve to, the sha256 of
every ONNX / DLC / quantized DLC produced, the full command lines, the
governor state, and the compose result per backend.

```
python3 qnn_models/slicing_study/slice_experiment.py --network <net> --name <label> --cut <tensor> ...
python3 qnn_models/slicing_study/analyze.py [network]      # regenerates the tables below
```

*Statistic.* Each cell is the **median of `profile_seg`'s per-sweep median**
over however many sweeps that experiment has (`sweeps=` in `analyze.py`
output; 1 or 2). The median rather than the mean because two other workers
share this board and a CPU sweep occasionally catches a contention spike —
`fused_k2_fc` t0's CPU cell moved 0.443 → 0.610 ms between sweeps, and its
*mean* moved 0.438 → 1.508 ms. DSP cells reproduce to ±7–19 %, HTA to ±4 %,
CPU int8 to ±11 %.

---

## 1. The per-dispatch overhead that bounds everything

`slice_experiment.py --overhead-probe` builds synthetic graphs of exactly one
1×1 `Conv` over a 1×1×H×W input — one MAC per pixel, one channel — and
measures them at 200 iterations. At 1×1 the graph is a dispatch and nothing
else, so that column *is* the per-dispatch cost.

| probe input | boundary bytes (int8) | HTA | DSP | CPU int8 | CPU fp32 |
|---|---:|---:|---:|---:|---:|
| 1×1 | 1 | **0.540 ms** | **0.367 ms** | **0.0026 ms** | 0.0026 ms |
| 64×64 | 4 096 | 1.047 | 0.416 | 0.020 | 0.026 |
| 256×256 | 65 536 | 2.558 | 1.191 | 0.218 | 0.311 |
| 512×512 | 262 144 | 6.746 | 2.852 | 0.854 | 1.240 |
| 1024×1024 | 1 048 576 | 22.254 | 10.238 | 3.404 | 5.205 |

(medians; `experiments.jsonl`, label `overhead_probe`.)

The real cuts agree with the intercept and give a better slope, because the
probe's single-channel convolution wastes the vector units. Taking the DSP
sum over tiles at each granularity minus the whole-network DSP cell, divided
by the number of cuts:

| network / cut | cuts | boundary bytes crossing | Δ DSP total | per cut |
|---|---:|---:|---:|---:|
| `mlp_k2` | 1 | 128 | 0.341 ms | 0.341 |
| `mlp_k4` | 3 | 448 | 1.043 | 0.348 |
| `dronet_k2_mid` | 1 | 3 136 | 0.371 | 0.371 |
| `dronet_k3_blocks` | 2 | 9 408 | 0.755 | 0.378 |
| `dronet_k4_blocks` | 3 | 11 456 | 1.136 | 0.379 |
| `dronet_k5_blocks` | 4 | 34 784 | 1.470 | 0.368 |
| `fused_k3_heads` | 2 | 576 | 0.635 | 0.318 |
| `yolons_k2_neck` | 1 | 716 800 | 4.380 | **4.380** |
| `yolons_k2_prod` | 1 | 716 800 | 4.659 | **4.659** |
| `yolons_k2_dfl` | 1 | 1 209 600 | 6.416 | **6.416** |
| `yolons_k3_neck_dfl` | 2 | 1 926 400 | 9.686 | 4.843 |

**The model that falls out, and the one number to plan with:**

```
extra DSP dispatch  ~=  0.37 ms  +  5.4 ns x (bytes of boundary tensor)
extra HTA dispatch  ~=  0.54 ms  +  (same order, but see below)
extra CPU dispatch  ~=  0.003 ms +  ~3 ns/byte
```

* **DSP fixed cost is 0.34–0.38 ms**, dead flat over five networks and four
  granularities — the FastRPC round trip, as expected.
* **The per-byte term is what actually kills fine slicing on big feature
  maps.** yolov8n's backbone/neck boundary is 716.8 KB; that one cut costs
  4.4–4.7 ms on the DSP, twelve times the fixed dispatch. A cut is only
  cheap if it lands on a *narrow* tensor.
* **HTA's dispatch is ~1.5× the DSP's and far less predictable.** The 1×1
  probe ranges 0.26 (min) / 0.54 (median) / 2.96 ms (p99) over 200 calls, and
  dronet's HTA totals imply 0.93–2.2 ms per extra dispatch depending on the
  sweep. Treat HTA dispatch as ≥0.5 ms with a long tail.
* **CPU dispatch is free.** 2.6 µs. dronet on CPU int8 costs *less* sliced
  into five tiles (6.889 ms) than whole (7.466 ms). Slicing never costs
  anything on the CPU lane — it only ever costs accelerator lanes.

---

## 2. dronet — 24 IR ops, HTA-capable after the offline BN rewrite

Source `qnn_models/dronet_full_hta.onnx` (29 ONNX nodes; BN folded into the
convs, FC head as 1×1 conv, trailing view dropped). Cuts taken at the
residual-block sums.

| k | cut | critical path | DSP total | HTA total | CPU-int8 total | backends unlocked |
|---:|---|---:|---:|---:|---:|---|
| **1** | — | **0.659 ms** (dsp) | 0.659 | 1.653 | 7.466 | hta, dsp, cpu |
| 2 | `backbone_relu_4d` (op 26) | 0.651 (dsp→cpu) | 1.038 | 2.498 | 6.784 | all, both tiles |
| 2 | `/Add_1_output_0` (op 17) | 1.030 | 1.030 | 2.395 | 6.729 | all, both tiles |
| 3 | `/Add_output_0`, `/Add_1_output_0` | 1.243 | 1.414 | 4.541 | 6.710 | all, all tiles |
| 4 | + `/Add_2_output_0` | 1.228 | 1.795 | 8.268 | 6.812 | all, all tiles |
| 5 | + `/maxpool1/MaxPool_output_0` | 1.308 | 2.129 | 5.365 | 6.889 | all, all tiles |

**Recommendation: one tile (`bindings/dronet.json`).** dronet is small enough
that its whole-network DSP cell (0.659 ms) is under two dispatch overheads.
Nothing is unlocked by cutting: *every* tile at *every* granularity composed
on all three backends, so the coverage argument that justifies slicing
yolov8n and FusedSensorNet does not apply here at all. The curve is flat and
then rises: k=2 already costs +56 %.

Two findings worth carrying forward:

* **CPU fp32 is 5× faster than CPU int8 for dronet** — 1.482 ms vs 7.466 ms
  whole-network. QnnCpu's int8 conv is a reference kernel. The existing
  `dronet/dronet_full@cpu` cell of 6998 µs is an int8 cell; the fp32 build of
  the same graph is the better CPU placement and is competitive with HTA
  (1.653 ms).
* **The one cut that is arguably worth having is `backbone_relu_4d`**, not for
  latency but as a cheap non-DSP placement: backbone on HTA (1.394 ms) plus
  the two head convs on CPU int8 (0.012 ms) is 1.406 ms, better than any
  whole-network placement that is not the DSP (HTA 1.653, CPU fp32 1.482).
  It is in the journal as `dronet_k2_head` if the scheduler ever needs it.

**Further slicing stops paying at k=1.**

---

## 3. yolov8n — where the cut actually is, and what HTA will take

Two source graphs, and the difference between them is the study's biggest
single result.

### 3a. Without the Split→Conv1×1 rewrite, HTA is unreachable at any granularity

Sliced straight from `qnn_models/yolov8n.onnx`:

| k | cut | HTA | DSP | CPU int8 |
|---:|---|---|---:|---:|
| 1 | — | ✗ `unsupported op Split` | 25.232 ms | 72.722 |
| 2 | `/model.9/cv2/act/Mul_output_0` | ✗ `unsupported op Split` **on both tiles** | 12.396 + 16.066 | 33.691 + 38.427 |

The production `yolov8n_backbone.onnx` reaches HTA only because it was built
from `yolov8n_nosplit.onnx`, where each `Split` was replaced by two 1×1
channel-selector convolutions (`optimizations.md` #14). That rewrite is a
*precondition for slicing to buy anything at all* on this network — exactly
the role the offline BN fold plays for dronet — and it is not something a
neutral slicer can discover. Everything below uses the rewritten graph
(241 nodes, `Conv` 80, no `Split`).

### 3b. With it, the HTA-able prefix runs 34 ops further than the shipped cut

| k | cut (op index) | critical path | HTA | DSP | CPU int8 | lane split at best assignment |
|---:|---|---:|---:|---:|---:|---|
| **1** | — | **25.376 ms** (dsp) | ✗ Transpose | 25.376 | 74.058 | dsp 25.376 |
| 2 | `/model.9/cv2/act/Mul_output_0` (102) — *the shipped cut* | 30.035 | 14.006 (t0) | 30.035 | 73.945 | dsp 30.035, or hta 14.006 + dsp 16.314 |
| 2 | `/model.15/cv2/act/Mul_output_0` (136) | 29.756 | **15.690 (t0)** | 29.756 | 74.042 | hta 15.690 + dsp 15.127 |
| 2 | `/model.22/Concat_1_output_0` (224) | **25.872** | ✗ Transpose | 31.792 | 75.046 | dsp 18.609 + cpu 7.263 |
| 3 | 102 + 224 | 33.437 | 14.057 (t0) | 38.861 | 74.910 | hta 14.057 + dsp 12.077 + cpu 7.303 |
| **3** | **136 + 224** | 30.489 | **15.581 (t0)** | 35.062 | 75.083 | **hta 15.581 + dsp 7.879 + cpu 7.288** |

Three things this settles:

1. **`Resize` composes on HTA.** The tile `ops[0,136]` contains both FPN
   upsample `Resize` ops, three `MaxPool`, five `Concat` and eight residual
   `Add`, and it composes and runs at 15.581 ms. The claim in
   `flow_c/measurements/qrb5165_v66.json` that the head fails because of
   "Resize/Slice/Softmax" is only right about the last two — and even they
   are not what the compose log names.
2. **The actual HTA blocker is the `Transpose` the converter emits for the
   detect head's `Reshape` ops.** Tile `ops[137,224]` contains nothing but
   `Conv, Sigmoid, Mul, Concat, Reshape` and is rejected with `unsupported op
   Transpose`. The same mechanism blocks FusedSensorNet (`Flatten`) — a
   layout-changing op is the boundary that costs HTA, and it costs it whether
   you wrote the Transpose or the converter did.
3. **The shipped cut at op 102 is 34 ops earlier than it needs to be.** Moving
   it to op 136 hands HTA the whole FPN neck at no measurable extra cost
   (15.690 vs 14.006 ms on HTA for 34 more ops) and takes 1.2 ms off the DSP
   tile.

**Recommendation: three tiles at ops 0–136 / 137–224 / 225–240
(`bindings/yolov8n.json`).** Its single-instance latency is 30.489 ms against
the 25.376 ms one-tile optimum — a 20 % latency premium — and in exchange it
puts 15.6 ms on HTA, 7.9 ms on DSP and 7.3 ms on CPU, cutting DSP occupancy
by 3.2×. In the 4-way flow_c schedule the DSP is the contended lane and
yolov8n is what bounds the makespan, so that is the trade the scheduler
wants. **If yolov8n's own latency is the objective and the DSP is otherwise
free, use one tile and do not cut at all.**

**Further slicing stops paying at k=3.** Each further cut on this network
costs 0.37 ms + 5.4 ns/byte, and every remaining boundary in the detect head
carries ≥700 KB (≥4 ms) against tiles that are only 7–15 ms to begin with.
The measured k=3 pair proves it: `102+224` and `136+224` cut the same total
1.93 MB and both pay ~9.7–13.5 ms of DSP for it.

---

## 4. FusedSensorNet (`fused_full`) — the Flatten is the boundary, again

Source `qnn_models/flow_c/gen/onnx/fused_full.onnx` (91 nodes).

| k | cut | critical path | HTA | DSP total | CPU |
|---:|---|---:|---|---:|---|
| 1 | — | 0.896 ms (cpu fp32) | ✗ Transpose | 3.434 | int8 ✗ `Reshape`; fp32 0.896 |
| 2 | `/vision_cnn/vision_cnn.7/Relu_output_0` (op 7) | 0.624 | **0.647 (t0)** | 3.765 | 0.164 + 0.460 |
| 2 | `/depth_fc/Gemm_output_0` (op 15) | 0.736 | ✗ Transpose | 4.033 | 0.610 + 0.126 |
| **3** | ops 7 + 13 | **0.447** | **0.641 (t0)** | 4.461 | 0.164 + 0.086 + 0.197 |
| 3 | ops 9 + 15 (flow_c's vision_head/depth_head/tail) | 0.465 | ✗ Transpose | 4.069 | 0.353 + 0.119 + 0.112 |

**Recommendation: three tiles at ops 0–7 / 8–13 / 14–90
(`bindings/fused_full.json`).** 0.447 ms of critical path against 0.896 ms
for the monolith — a 2.0× improvement, and the whole win is that the pieces
stop being forced onto one precision. This reproduces and slightly refines
flow_c's `fused_split`: the useful boundary is before the vision `Flatten`
(HTA takes ops 0–7 at 0.641 ms), and cutting the depth branch out separately
buys another 0.02 ms, which is noise.

Compose facts:

* Every tile containing a `Flatten` is rejected on HTA with `unsupported op
  Transpose` — ops[0,9], ops[0,15], ops[8,13] all fail, ops[0,7] passes.
  This is the same layout-op rule as yolov8n's `Reshape`.
* Every tile containing the LSTM tail is rejected at **CPU int8** with
  `validation failed for Reshape` and must be built fp32.
* **A cut can introduce a rejection of its own.** `fused_k2_fc` t1 and
  `fused_k3_heads` t2 — the tail taken from the *FC* outputs rather than the
  conv outputs — fail HTA on `unsupported op Convert`, an op that exists in
  neither the source graph nor the other tail slices. The requantization the
  converter inserts at that particular boundary is itself unsupported.

**Further slicing stops paying at k=3**: the remaining tile is a 3-layer LSTM
whose recurrence is sequential, and it is already only 0.197 ms.

---

## 5. ViNT — the cut that turns "CPU only" into "half on the DSP"

Source `qnn_models/flow_c/gen/onnx/vint.onnx` (1931 nodes, 610 of them
`Constant`). Boundary shapes here are dynamic in the source graph; the
harness freezes them from the shapes onnxruntime actually produced during
calibration capture, because `-d` alone does not get the converter past a
graph whose interior still carries symbolic dims.

| k | cut | critical path | HTA | DSP | CPU int8 | CPU fp32 |
|---:|---|---:|---|---|---:|---:|
| 1 | — | 59.775 ms (cpu int8) | ✗ StridedSlice | ✗ `Param[0] has incorrect Value 1.` | 59.775 | 98.564 |
| **2** | `/compress_obs_enc/Gemm_output_0` | **30.080** | ✗ StridedSlice (t0) / ✗ Transpose (t1) | 16.339 (t0) / ✗ (t1) | 21.975 / 41.020 | 89.922 / **13.741** |
| 3 | + `/compress_goal_enc/Gemm_output_0` | 30.633 | ✗ StridedSlice / ✗ Split / ✗ Transpose | 5.728 + 11.095 / ✗ | 6.718 + 15.570 / 41.322 | 12.389 + 73.111 / 13.810 |

**Recommendation: two tiles at the compress projections
(`bindings/vint.json`) — i.e. the cut flow_c already ships, confirmed
optimal.** It halves the critical path (59.8 → 30.1 ms) and, more
importantly, moves 16.3 ms of it off the CPU onto silicon that was doing
nothing, because the monolith composes on no accelerator at all.

**Further slicing stops paying at k=2.** Splitting the two EfficientNet
encoders apart (k=3) costs 0.55 ms and gains nothing, for a structural
reason worth recording: the two encoders are semantically parallel, but a
*contiguous* slice of a topologically ordered graph cannot express them as
independent tiles — the obs encoder tile ends up consuming the goal encoder's
output, so they serialise. flow_c's binding format already supports
non-contiguous op sets (`fused_split`'s tail is one); expressing ViNT's
encoders as two parallel tiles needs that, and would predict
max(5.728, 11.095) + 13.741 = 24.8 ms. That is the one slice this study could
not build and the obvious next experiment.

Also new: **ViNT's encoders are 4× faster at CPU int8 (21.975 ms) than at CPU
fp32 (89.922 ms)** — the opposite of what the current measurements note
records for this network — while the decoder is 3× faster at fp32 (13.741 vs
41.020 ms). Precision has to be chosen per tile, not per network.

---

## 6. mlp_control — the floor, and it behaves like one

Source `qnn_models/flow_c/gen/onnx/mlp_control.onnx`, 7 ops, ~70k MACs.
Calibration is synthetic (seeded `N(0,1)`, persisted under the experiment's
`calib/_synthetic/`) because no captured `obs` set exists in the tree; it
affects the quantization scales, not the timing.

| k | critical path | DSP total | CPU-int8 total | HTA |
|---:|---:|---:|---:|---|
| 1 | 0.030 ms | 0.408 | 0.030 | ✗ `unsupported elementwise neuson op 0` |
| 2 | 0.029 | 0.749 | 0.098 | ✗ same, both tiles |
| 4 | 0.026 | 1.451 | 0.046 | ✗ on the three Elu tiles; **0.541 ms on the Gemm-only tile** |

**Recommendation: one tile (`bindings/mlp_control.json`).** As expected: the
three critical paths are indistinguishable (all inside the CPU cells' own
noise), while on the DSP each of the four tiles pays a full 0.357–0.372 ms
dispatch for 0.006–0.026 ms of work. The floor result is quantitative: a cut
pays only when it moves more than 0.37 ms of accelerator work, and this
entire network is 0.030 ms.

One detail: the trailing `Gemm`-only tile *does* compose on HTA (0.541 ms),
so `elu_s8` really is the whole of HTA's objection — but that tile runs in
0.002 ms on the CPU, 250× faster, so the placement is useless.

---

## 7. What the whole study says

**The trade-off curve, in one place** (critical path, ms, best composable
backend per tile; **bold** = recommended):

| k | dronet | mlp_control | fused_full | yolov8n | ViNT |
|---:|---:|---:|---:|---:|---:|
| 1 | **0.659** | **0.030** | 0.896 | *25.376* | 59.775 |
| 2 | 0.651 – 1.030 | 0.029 | 0.624 – 0.736 | 25.872 – 30.035 | **30.080** |
| 3 | 1.243 | — | **0.447** | **30.489** | 30.633 |
| 4 | 1.228 | 0.026 | — | — | — |
| 5 | 1.308 | — | — | — | — |

*(yolov8n k=1 is italicised: it is the latency optimum but leaves both
accelerators idle and pins the network to the contended DSP lane.)*

1. **Slicing never buys speed. It buys placement.** In every one of the five
   networks the finest granularity is never the fastest, and in three of them
   (dronet, mlp_control, yolov8n) the *coarsest* is the fastest. Slicing pays
   only where the op set makes a coarse tile infeasible on the silicon you
   want (ViNT, FusedSensorNet) or where the schedule needs a lane freed
   (yolov8n).
2. **The break-even rule is one line:** a cut pays when it moves more than
   `0.37 ms + 5.4 ns × boundary_bytes` of accelerator work onto an otherwise
   idle lane (DSP; ≥0.5 ms + similar for HTA; ~0 for CPU). That is why
   dronet — 0.66 ms whole — can never be cut, and why yolov8n can be cut
   twice but not three times.
3. **Cut on narrow tensors.** The identical structural cut costs 0.37 ms on
   dronet (3 KB boundary) and 4.4 ms on yolov8n (717 KB). Granularity is
   bounded by tensor width, not by op count.
4. **A single layout-changing op disqualifies a whole tile on HTA, and the
   converter can create one you did not write.** `Reshape` (yolov8n),
   `Flatten` (FusedSensorNet) and the boundary requantization (`Convert`, in
   `fused_k2_fc`) all surface as `unsupported op Transpose`/`Convert`. The
   productive way to choose an HTA cut is: find the last op before the first
   layout change.
5. **Precision is a per-tile decision.** CPU int8 beats CPU fp32 by 4× on
   ViNT's encoders and loses by 5× on dronet and by 7× on ViNT's decoder;
   QnnCpu's int8 convolution is a reference kernel, its int8 GEMM is not.

### Compose-failure catalogue (the whole point of the exercise)

Every distinct rejection observed, with the string the compose log printed.
Full list, per tile, in `experiments.jsonl`.

| op set in the tile | backend | log says |
|---|---|---|
| anything containing `Split` (yolov8n as exported) | HTA | `unsupported op Split` |
| anything containing `Reshape` (yolov8n detect head) | HTA | `unsupported op Transpose` |
| anything containing `Flatten` (FusedSensorNet) | HTA | `unsupported op Transpose` |
| ViNT's transformer decoder | HTA | `unsupported op Transpose` |
| ViNT's obs encoder (6-frame slice) | HTA | `unsupported op StridedSlice` |
| ViNT's goal encoder / both encoders | HTA | `unsupported op StridedSlice` |
| `Elu` (mlp_control) | HTA | `unsupported elementwise neuson op 0` |
| FusedSensorNet tail taken from the FC outputs | HTA | `unsupported op Convert` (introduced by the cut) |
| ViNT decoder, and ViNT whole | DSP | `Param[0] has incorrect Value 1.` |
| FusedSensorNet LSTM tail, any variant | CPU int8 | `validation failed for Reshape` |
| yolov8n backbone/neck ops 0–136 incl. 2×`Resize`, 3×`MaxPool` | HTA | **composes** — 15.581 ms |
| mlp_control trailing `Gemm` alone | HTA | **composes** — 0.541 ms |

### Known limits of this study

* **Numerics are not checked.** `profile_seg` runs zero-filled buffers; this
  measures dispatch cost, not correctness. Boundary calibration is captured
  from the real graph so the quantization scales are meaningful, but no
  end-to-end output comparison was run.
* **Contiguous cuts only.** A tile is a contiguous run of the source graph's
  topological order plus any constant subgraph it needs. Genuinely parallel
  branches (ViNT's two encoders, FusedSensorNet's two conv branches) cannot
  be expressed as independent tiles this way; flow_c's `ops: {ranges: [...]}`
  can, and §5 quantifies what that would be worth for ViNT (~24.8 vs 30.1 ms).
* **Single-instance critical path.** The reported critical path is the longest
  path through the tile DAG with each tile at its cheapest composable
  backend, one instance in isolation. It does not model lane contention,
  which the flow_c README shows costs the CPU lane up to 5.6× in situ.
* **HTA cells are noisy** (p99/median ≈ 3× on short graphs). HTA comparisons
  between tiles under ~2 ms should be read as order-of-magnitude only.
* **`yolov8n_nosplit.onnx` was taken as given**, not rebuilt; the rewrite
  script referenced in `optimization_flow.md` (#14) is not in the tree.

---

# 8. Branch-parallel tiles: a tile is a set of ops, not a span

§7's last caveat was that a contiguous slicer cannot express independent
branches, so ViNT's two encoders and FusedSensorNet's two conv branches
were forced to serialise. `slice_experiment.py` now has two more modes:

* `--tile OUT[,OUT...]` (repeatable) — **branch mode**. Each tile is named
  by the tensors it must produce and claims the backward closure of them
  minus what earlier tiles claimed. Independent branches fall out as tiles
  whose input sets touch no other tile's outputs; the record carries
  `independent_pairs` and each tile a true `depends_on`.
* `--subgraph IN[,IN]:OUT[,OUT]` with `--batch1 N` — **subgraph mode**, the
  only way to name a tile that starts in the middle of the graph, plus a
  rewrite of the leading dimension of every tile boundary (and of the
  captured calibration) from B to N.

`analyze.py` gained the matching cost model: a **makespan with one lane
per machine kind**, minimised over every backend assignment. Two tiles on
the same kind serialise even when the DAG says they are independent —
without that, branch parallelism looks like free money, which it is not.
`ir_branch_ranges.py` runs the same closure on flow_c's IR so a manifest's
`ops.ranges` are derived rather than transcribed.

## 8.1 ViNT — the win, and it is bigger than the model predicted

ViNT's shipped cut puts **both** EfficientNet-b0 encoders in one graph, so
they serialise inside one dispatch on one lane. They do not consume each
other's output: the goal encoder takes `goal_img` plus the last `obs_img`
frame, the obs encoder takes `obs_img`. In flow_c IR space the closure
partition is exactly ops 0–295 / 296–538 / 539–604 with **no shared
ancestor**, so the branches are genuinely independent.

Cells are the median of 3 sweeps.

| slice set | tiles | makespan (1 lane/kind) | same tiles, no concurrency | best assignment |
|---|---:|---:|---:|---|
| whole network | 1 | 59.775 ms | 59.775 | cpu int8 |
| **shipped cut** (encoders / decoder) | 2 | **29.379 ms** | 29.379 | dsp → cpu fp32 |
| contiguous 3-way (§5) | 3 | 30.633 ms | 30.633 | dsp → dsp → cpu |
| **branch-parallel 3-way** | 3 | **23.201 ms** | 28.758 | goal@cpu ∥ obs@dsp → dec@cpu fp32 |

* **6.178 ms (21 %) off the critical path** versus what ships today.
* **The dispatch the split adds is free within noise.** The same three
  tiles run with no concurrency are 28.758 ms against the two-tile
  29.379 ms — a *negative* 0.621 ms, where the overhead model predicts
  +0.367 ms (0.367 fixed + 5.4 ns × 512 boundary bytes ≈ 0.370). The
  fused encoder tile's own sweep spread is 14.064–16.339 ms, so the honest
  statement is: the extra dispatch costs less than the measurement can
  resolve, and the concurrency is worth 5.557 ms.

### Verified on the board, not just in the model

The recommended manifest was run through flow_c end to end — `artifacts` →
MILP (MOSEK) `schedule` → `runtime --lane-mode kind-network` → `stage` →
`run --tuned` — and the per-tile actual start/end read out of the trace
(`flowc/run_vint_parallel/run.log`, journal label
`vint_par_enc__flowc_trace_verification`):

```
network   tile             kind   start_ms   end_ms   dur_ms
vint_par  vint_goal_enc    cpu       0.222    6.590    6.368
vint_par  vint_obs_enc     dsp       0.227    9.581    9.354     <- overlap 6.363 ms
vint_par  vint_decoder     cpu       9.632   38.100   28.468
```

The goal encoder runs **entirely inside** the obs encoder's window:
**6.363 ms of measured concurrency**, 96 % of the 6.615 ms the makespan
model predicted. Whole-run wall 40.060 ms against a 40.000 ms prediction
(1.00×).

Per-tile in-situ / isolated-cell ratios: `vint_goal_enc@cpu` 0.96×,
`vint_obs_enc@dsp` 1.01×, `vint_decoder@cpu fp32` **2.05×** (28.468 in situ
vs a 13.913 ms cell). That last one is the CPU-contention effect this
board's measurement notes already document — a multi-threaded CPU tile's
cost is a function of what runs beside it. It does not change the ranking:
the decoder pays the same inflation either way, so the split still wins in
situ by about the same 6 ms.

## 8.2 The batch split: measured, and it loses

ViNT's obs encoder is **one** EfficientNet run on a batch of 6 stacked
frames (`obs_img` 1×18×64×85, a `Split`+`Concat` making 6×3×64×85), not
six subgraphs — so a natural idea is to dispatch it six times at batch 1
and spread the frames over both accelerators. Sliced between
`/Concat_1_output_0` and `/compress_obs_enc/Gemm_output_0` with the leading
dim rewritten (journal labels `vint_obs_b1/b2/b3`, 3 sweeps each):

| batch | DSP int8 | per frame | CPU int8 | HTA |
|---:|---:|---:|---:|---|
| 1 | **4.690 ms** | 4.690 | 6.064 | ✗ `unsupported op Transpose` |
| 2 | 5.322 | 2.661 | 5.887 | ✗ same |
| 3 | 5.342 | 1.781 | 8.213 | ✗ same |
| 6 (as shipped) | 9.288 | 1.548 | 15.549 | ✗ `unsupported op Split` |

**One extra frame inside a dispatch costs 0.920 ms; a dispatch of its own
costs 4.690 ms.** The fixed part of a batch-1 encoder dispatch is 3.77 ms —
*ten times* the 0.367 ms FastRPC round trip — because at batch 1 the graph
is latency-bound, not throughput-bound: weights are re-streamed and HVX is
underfed. So:

| obs-encoder strategy | ViNT critical path |
|---|---:|
| keep the batch, split obs-vs-goal (recommended) | **23.201 ms** |
| 2 × batch-3, both on DSP | 24.597 |
| 3 + 3 across DSP ∥ CPU, goal on DSP | 24.813 |
| 3 + 3 across DSP ∥ CPU, goal on CPU | 28.741 |
| 6 × batch-1, all on DSP | 42.053 |

Six batch-1 dispatches are **28.140 ms against 9.288 ms batched — 3.03×
worse.** The 3+3 split does beat the batch-6 tile *on the encoder alone*
(max(5.342 dsp, 8.213 cpu) = 8.213 vs 9.288), but the CPU half then
collides with the decoder, which needs that lane for 13.9 ms, and the
network as a whole loses. **No batch split pays.**

**And batch-1 does not unlock HTA.** Removing the six-frame `Split` does
remove the `unsupported op StridedSlice` rejection — but the tile then
fails on `unsupported op Transpose`, from EfficientNet's static-padding
lowering. StridedSlice was only the *first* op the log named. ViNT still
composes on no accelerator but the DSP, at any batch size.

## 8.3 FusedSensorNet — branch-parallel is expressible, and does not pay

Branch mode reproduces flow_c's `fused_split` structure independently:
tiles 0 and 1 come out with `depends_on: []`, and the tail with the
non-contiguous ONNX range `[[8,9],[14,90]]` — the first tile in this study
whose ops are not one span.

| slice set | makespan | no concurrency | best assignment |
|---|---:|---:|---|
| whole network | 0.896 ms | 0.896 | cpu fp32 |
| contiguous 3-way (§4) | **0.452 ms** | 0.452 | all three on cpu |
| branch-parallel conv/depth/tail | 0.548 | 0.548 | both branches on cpu, back to back |
| branch-parallel at the FC outputs | 0.503 | 0.503 | both branches on cpu |
| contiguous 3-way at the FC outputs | 0.511 | 0.588 | cpu ∥ dsp → cpu |

**The branches are too cheap to be worth a lane.** Vision is 0.150 ms and
depth 0.048 ms on CPU int8; the cheapest accelerator placement that could
overlap them is 0.398 ms (DSP) or 0.540 ms (HTA). Moving a branch to an
accelerator to overlap it costs 3–11× the branch itself, so the best
isolated assignment runs both on the CPU lane back to back and the
concurrency column stays flat. **A branch split pays only when the
branch's runtime exceeds the dispatch it adds** — 0.367 ms on DSP,
0.540 ms on HTA. ViNT's 6.6 ms goal encoder clears that bar by 18×;
FusedSensorNet's 0.048 ms depth branch misses it by 8×.

It is still the manifest to ship, and that is a judgement, not a
measurement: both branches compose on HTA (0.607 / 0.540 ms) where the
contiguous variant's second tile does not, and the CPU cells that make the
serial answer win are the ones this board measures at ~5.6× in situ (and
which §8.1 just re-confirmed at 2.05× for ViNT's decoder). `bindings/
fused_full.json` carries the branch-parallel set with that caveat stated;
`fused_k3_convs` in the journal is the isolation optimum if the CPU lane
is free.

## 8.4 What changed in the recommendations

| network | §7 said | now | why |
|---|---|---|---|
| ViNT | 2 tiles (shipped cut) | **3 tiles, branch-parallel** | 29.379 → 23.201 ms, 6.363 ms of overlap verified in a trace |
| fused_full | 3 tiles, contiguous | 3 tiles, branch-parallel | same measured cost within noise; exposes two HTA-capable independent branches |
| dronet / mlp_control / yolov8n | unchanged | unchanged | dronet and mlp have no parallel branches; yolov8n's are the three detect-head scales, all downstream of one backbone |

### Index-space warning on the manifests

`ops.ranges` in a flow_c manifest index the **IR**, and the artifacts are
sliced from the **ONNX**; the two spaces differ (ViNT: 605 IR ops vs 1931
ONNX nodes). The ViNT and fused_full bindings therefore carry IR ranges —
ViNT's derived by `ir_branch_ranges.py`, fused_full's adopted from
flow_c's own `fused_split.json` — with the ONNX node ranges kept alongside
in `_ops_onnx_nodes`, and every binding now states its
`_provenance.op_index_space`. **`bindings/yolov8n.json` is still in ONNX
node space** and needs a remap before it can be dropped into flow_c; that
is the one manifest here that is not directly runnable.

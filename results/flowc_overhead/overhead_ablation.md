# QRB5165 overhead ablation — full experiment log

## What is swept, and why

A slicing recommendation is only worth acting on if it survives the overhead we cannot pin down. On K1 the per-dispatch cost is small, stable and visible, so a slice decision can be read straight off the compute cells. On QRB5165 it is none of those: a tile boundary crosses FastRPC, may re-quantize, and may move a large tensor between an accelerator and the CPU.

`qnn_models/slicing_study/RESULTS.md` fits the marginal cost of a cut as **0.37 ms + 5.4 ns x boundary_bytes** (DSP; >=0.5 ms fixed for HTA). That single fit was taken **on an idle board**, and it is exactly the quantity a busy board, a different governor or a context eviction would change. So rather than trust one number, this sweeps it and reports the range over which each network's best slice set stays best.

Two knobs:

* **call overhead (ms)** — charged once per tile: dispatch setup, context acquire.
* **transfer rate (ns/byte)** — charged on every byte a tile must receive across a cut.

The measured cells already contain whatever overhead each tile pays as a standalone graph. What is swept is the MARGINAL cost of having cut at all — which a finer slice set pays and a coarser one does not. So the overhead can only ever push the ranking toward coarser sets; the question is how hard it has to push.

## How to read the notation

**This is NOT the S0..S4 ladder.** That ladder (see `../flowc_stages/stage_ladder.md`) adds a degree of freedom per step. Here every cell is a *complete* re-ranking of every whole-network slice set for that network under one overhead assumption, and the cell names the winner:

    <slice-set label>  k=<number of tiles>  <makespan in ms>

`k=1` means the monolith won — i.e. at that overhead, cutting does not pay. A row where the winner changes is a **flip**: the overhead got large enough to reverse the recommendation.

Partial-coverage subgraph probes are excluded from the ranking. `vint_obs_b*` covers ops 558-1069 of a 1931-op graph — it is cheaper than any full slice set for the trivial reason that it does less, and including it makes the partial set win every row.


## vint

*3 partial-coverage probe(s) excluded.*

| call ms | @0 ns/byte | @5.4 ns/byte | @20 ns/byte |
|---:|---|---|---|
| 0 | **`vint_par_enc` k=3** · 23.2 ms | **`vint_par_enc` k=3** · 23.2 ms | **`vint_par_enc` k=3** · 23.3 ms |
| 0.1 | **`vint_par_enc` k=3** · 23.4 ms | **`vint_par_enc` k=3** · 23.4 ms | **`vint_par_enc` k=3** · 23.5 ms |
| 0.2 | **`vint_par_enc` k=3** · 23.6 ms | **`vint_par_enc` k=3** · 23.6 ms | **`vint_par_enc` k=3** · 23.7 ms |
| 0.37 | **`vint_par_enc` k=3** · 23.9 ms | **`vint_par_enc` k=3** · 24.0 ms | **`vint_par_enc` k=3** · 24.0 ms |
| 0.54 | **`vint_par_enc` k=3** · 24.3 ms | **`vint_par_enc` k=3** · 24.3 ms | **`vint_par_enc` k=3** · 24.4 ms |
| 1 | **`vint_par_enc` k=3** · 25.2 ms | **`vint_par_enc` k=3** · 25.2 ms | **`vint_par_enc` k=3** · 25.3 ms |
| 2 | **`vint_par_enc` k=3** · 27.2 ms | **`vint_par_enc` k=3** · 27.2 ms | **`vint_par_enc` k=3** · 27.3 ms |
| 4 | **`vint_par_enc` k=3** · 31.2 ms | **`vint_par_enc` k=3** · 31.2 ms | **`vint_par_enc` k=3** · 31.3 ms |
| 8 | **`vint_par_enc` k=3** · 39.2 ms | **`vint_par_enc` k=3** · 39.2 ms | **`vint_par_enc` k=3** · 39.3 ms |
| 16 | **`vint_par_enc` k=3** · 55.2 ms | **`vint_par_enc` k=3** · 55.2 ms | **`vint_par_enc` k=3** · 55.3 ms |
| 32 | **`vint_par_enc` k=3** · 87.2 ms | **`vint_par_enc` k=3** · 87.2 ms | **`vint_par_enc` k=3** · 87.3 ms |

**slicing wins at every overhead tested** (up to 32 ms/call, 20 ns/byte).


## yolov8n

| call ms | @0 ns/byte | @5.4 ns/byte | @20 ns/byte |
|---:|---|---|---|
| 0 | `yolo_k1_whole` k=1 · 25.2 ms | `yolo_k1_whole` k=1 · 25.2 ms | `yolo_k1_whole` k=1 · 25.2 ms |
| 0.1 | `yolo_k1_whole` k=1 · 25.3 ms | `yolo_k1_whole` k=1 · 25.3 ms | `yolo_k1_whole` k=1 · 25.3 ms |
| 0.2 | `yolo_k1_whole` k=1 · 25.4 ms | `yolo_k1_whole` k=1 · 25.4 ms | `yolo_k1_whole` k=1 · 25.4 ms |
| 0.37 | `yolo_k1_whole` k=1 · 25.6 ms | `yolo_k1_whole` k=1 · 25.6 ms | `yolo_k1_whole` k=1 · 25.6 ms |
| 0.54 | `yolo_k1_whole` k=1 · 25.8 ms | `yolo_k1_whole` k=1 · 25.8 ms | `yolo_k1_whole` k=1 · 25.8 ms |
| 1 | `yolo_k1_whole` k=1 · 26.2 ms | `yolo_k1_whole` k=1 · 26.2 ms | `yolo_k1_whole` k=1 · 26.2 ms |
| 2 | `yolo_k1_whole` k=1 · 27.2 ms | `yolo_k1_whole` k=1 · 27.2 ms | `yolo_k1_whole` k=1 · 27.2 ms |
| 4 | `yolo_k1_whole` k=1 · 29.2 ms | `yolo_k1_whole` k=1 · 29.2 ms | `yolo_k1_whole` k=1 · 29.2 ms |
| 8 | `yolo_k1_whole` k=1 · 33.2 ms | `yolo_k1_whole` k=1 · 33.2 ms | `yolo_k1_whole` k=1 · 33.2 ms |
| 16 | `yolo_k1_whole` k=1 · 41.2 ms | `yolo_k1_whole` k=1 · 41.2 ms | `yolo_k1_whole` k=1 · 41.2 ms |
| 32 | `yolo_k1_whole` k=1 · 57.2 ms | `yolo_k1_whole` k=1 · 57.2 ms | `yolo_k1_whole` k=1 · 57.2 ms |

**the monolith wins everywhere** — no cut ever repays itself.


## fused_full

| call ms | @0 ns/byte | @5.4 ns/byte | @20 ns/byte |
|---:|---|---|---|
| 0 | **`fused_k3_convs` k=3** · 0.5 ms | **`fused_k3_convs` k=3** · 0.5 ms | **`fused_k3_convs` k=3** · 0.5 ms |
| 0.1 | **`fused_par_fc` k=3** · 0.7 ms | **`fused_par_fc` k=3** · 0.7 ms | **`fused_par_fc` k=3** · 0.7 ms |
| 0.2 | **`fused_par_fc` k=3** · 0.9 ms | **`fused_par_fc` k=3** · 0.9 ms | **`fused_par_fc` k=3** · 0.9 ms |
| 0.37 | **`fused_par_fc` k=3** · 1.2 ms | **`fused_par_fc` k=3** · 1.2 ms | **`fused_par_fc` k=3** · 1.3 ms |
| 0.54 | `fused_k1_whole` k=1 · 1.4 ms | `fused_k1_whole` k=1 · 1.4 ms | `fused_k1_whole` k=1 · 1.4 ms |
| 1 | `fused_k1_whole` k=1 · 1.9 ms | `fused_k1_whole` k=1 · 1.9 ms | `fused_k1_whole` k=1 · 1.9 ms |
| 2 | `fused_k1_whole` k=1 · 2.9 ms | `fused_k1_whole` k=1 · 2.9 ms | `fused_k1_whole` k=1 · 2.9 ms |
| 4 | `fused_k1_whole` k=1 · 4.9 ms | `fused_k1_whole` k=1 · 4.9 ms | `fused_k1_whole` k=1 · 4.9 ms |
| 8 | `fused_k1_whole` k=1 · 8.9 ms | `fused_k1_whole` k=1 · 8.9 ms | `fused_k1_whole` k=1 · 8.9 ms |
| 16 | `fused_k1_whole` k=1 · 16.9 ms | `fused_k1_whole` k=1 · 16.9 ms | `fused_k1_whole` k=1 · 16.9 ms |
| 32 | `fused_k1_whole` k=1 · 32.9 ms | `fused_k1_whole` k=1 · 32.9 ms | `fused_k1_whole` k=1 · 32.9 ms |

slicing stops winning at **0.54 ms/call** (last held at 0.37 ms).


## dronet

| call ms | @0 ns/byte | @5.4 ns/byte | @20 ns/byte |
|---:|---|---|---|
| 0 | **`dronet_k2_head` k=2** · 0.7 ms | `dronet_k1_whole` k=1 · 0.7 ms | `dronet_k1_whole` k=1 · 0.7 ms |
| 0.1 | `dronet_k1_whole` k=1 · 0.8 ms | `dronet_k1_whole` k=1 · 0.8 ms | `dronet_k1_whole` k=1 · 0.8 ms |
| 0.2 | `dronet_k1_whole` k=1 · 0.9 ms | `dronet_k1_whole` k=1 · 0.9 ms | `dronet_k1_whole` k=1 · 0.9 ms |
| 0.37 | `dronet_k1_whole` k=1 · 1.0 ms | `dronet_k1_whole` k=1 · 1.0 ms | `dronet_k1_whole` k=1 · 1.0 ms |
| 0.54 | `dronet_k1_whole` k=1 · 1.2 ms | `dronet_k1_whole` k=1 · 1.2 ms | `dronet_k1_whole` k=1 · 1.2 ms |
| 1 | `dronet_k1_whole` k=1 · 1.7 ms | `dronet_k1_whole` k=1 · 1.7 ms | `dronet_k1_whole` k=1 · 1.7 ms |
| 2 | `dronet_k1_whole` k=1 · 2.7 ms | `dronet_k1_whole` k=1 · 2.7 ms | `dronet_k1_whole` k=1 · 2.7 ms |
| 4 | `dronet_k1_whole` k=1 · 4.7 ms | `dronet_k1_whole` k=1 · 4.7 ms | `dronet_k1_whole` k=1 · 4.7 ms |
| 8 | `dronet_k1_whole` k=1 · 8.7 ms | `dronet_k1_whole` k=1 · 8.7 ms | `dronet_k1_whole` k=1 · 8.7 ms |
| 16 | `dronet_k1_whole` k=1 · 16.7 ms | `dronet_k1_whole` k=1 · 16.7 ms | `dronet_k1_whole` k=1 · 16.7 ms |
| 32 | `dronet_k1_whole` k=1 · 32.7 ms | `dronet_k1_whole` k=1 · 32.7 ms | `dronet_k1_whole` k=1 · 32.7 ms |

slicing stops winning at **0.1 ms/call** (last held at 0 ms).


## mlp_control

| call ms | @0 ns/byte | @5.4 ns/byte | @20 ns/byte |
|---:|---|---|---|
| 0 | **`mlp_k4` k=4** · 0.0 ms | **`mlp_k4` k=4** · 0.0 ms | `mlp_k1_whole` k=1 · 0.0 ms |
| 0.1 | `mlp_k1_whole` k=1 · 0.1 ms | `mlp_k1_whole` k=1 · 0.1 ms | `mlp_k1_whole` k=1 · 0.1 ms |
| 0.2 | `mlp_k1_whole` k=1 · 0.2 ms | `mlp_k1_whole` k=1 · 0.2 ms | `mlp_k1_whole` k=1 · 0.2 ms |
| 0.37 | `mlp_k1_whole` k=1 · 0.4 ms | `mlp_k1_whole` k=1 · 0.4 ms | `mlp_k1_whole` k=1 · 0.4 ms |
| 0.54 | `mlp_k1_whole` k=1 · 0.6 ms | `mlp_k1_whole` k=1 · 0.6 ms | `mlp_k1_whole` k=1 · 0.6 ms |
| 1 | `mlp_k1_whole` k=1 · 1.0 ms | `mlp_k1_whole` k=1 · 1.0 ms | `mlp_k1_whole` k=1 · 1.0 ms |
| 2 | `mlp_k1_whole` k=1 · 2.0 ms | `mlp_k1_whole` k=1 · 2.0 ms | `mlp_k1_whole` k=1 · 2.0 ms |
| 4 | `mlp_k1_whole` k=1 · 4.0 ms | `mlp_k1_whole` k=1 · 4.0 ms | `mlp_k1_whole` k=1 · 4.0 ms |
| 8 | `mlp_k1_whole` k=1 · 8.0 ms | `mlp_k1_whole` k=1 · 8.0 ms | `mlp_k1_whole` k=1 · 8.0 ms |
| 16 | `mlp_k1_whole` k=1 · 16.0 ms | `mlp_k1_whole` k=1 · 16.0 ms | `mlp_k1_whole` k=1 · 16.0 ms |
| 32 | `mlp_k1_whole` k=1 · 32.0 ms | `mlp_k1_whole` k=1 · 32.0 ms | `mlp_k1_whole` k=1 · 32.0 ms |

slicing stops winning at **0.1 ms/call** (last held at 0 ms).


## Robustness envelope

How much call overhead each recommendation survives, against the measured 0.37 ms:

| network | slicing survives to | vs the measured 0.37 ms |
|---|---|---|
| `vint` | >32 ms/call | ~86x headroom |
| `yolov8n` | never ms/call | n/a — never slices |
| `fused_full` | 0.37 ms/call | **1.00x — inside the uncertainty** |
| `dronet` | 0 ms/call | **0.00x — inside the uncertainty** |
| `mlp_control` | 0 ms/call | **0.00x — inside the uncertainty** |

**Four of five recommendations sit inside the measurement uncertainty.** Only ViNT's survives a plausible overhead range. fused_full flips one step above the measured value, and dronet and mlp_control flip *below* it. A current, measured overhead number is therefore a prerequisite for trusting any slicing advice except ViNT's.

**The per-byte term barely matters.** Across the whole grid only the 0 ms/call rows of dronet and mlp_control move between rate columns; everywhere else the columns are identical. The fixed per-cut cost dominates at these tensor sizes, so the harder-to-measure transfer rate is the term you can afford to be wrong about — and the FastRPC/context-acquire cost is the one that decides the outcome.


# Sharding the navigation network across harts

`fused_full` (the navigation network) runs on one hart in every deployed ROS 2 arm. This page
records what happens when it is split across several, measured standalone on the K1, and why the
deployed arms keep it on one.

## The split that exists

ModelBlaster generates a sharded build with `MB_SHARD_FACTOR`. For `fused_full` the split is over
**output channels**: the conv weights are emitted as per-shard arrays
(`fused_full_vision_cnn_0_weight_q_shard_0..N_rvv_x60`) and the driver calls
`parallel_conv2d_s8_sharded(pool, in, w_shards_dK, N, C/N, ...)`. Seven convolutions shard this way
(`vision_cnn` 0/2/4/6, `depth_conv` 0/2). `kernels.c` is identical across widths — only `model.c`
differs, because the split is a dispatch decision rather than a different kernel.

The LSTM tail does not shard.

## Measured, standalone on the board

`MODELBLASTER_ITERS=30`, median of the last 25 iterations, each configuration pinned with
`MODELBLASTER_CPU`. Raw output in `results/codesign_feedback/ros_traced/nav_standalone/`.

| build | harts | median wall | vs unsharded on one hart | `max_abs_err` |
|---|---|---|---|---|
| unsharded | 1 | 97 579 | 1.00x | 0.4775 |
| shard2 | 1 | 129 988 | 0.75x | 0.4775 |
| shard2 | 2 | 88 348 | **1.10x** | 0.4775 |
| shard4 | 1 | 226 091 | 0.43x | 0.4775 |
| shard4 | 4 | 86 237 | **1.13x** | 0.4775 |

Every configuration reaches the same golden error, so the split is numerically neutral: what changes
is only how the same arithmetic is divided. The error is not zero because the comparison is against
the float reference and this network's path is int8 with an fp16 linear and LSTM; it is the same
0.4775 with and without sharding, which is the property that matters here.

Sharding a hart's worth of work across four harts returns 13 %. The convolutions parallelise; the
sequential LSTM then dominates, so the whole-network return is bounded well below the width. On the
deployed chain the navigation network is about 5 ms of a 30.7 ms camera-to-control path, so 13 % of
it is roughly 0.6 ms, near 2 % end to end, for three additional harts.

**That is why the partitioned arms give navigation one hart.** The two harts this leaves unused are
reported as unused rather than filled.

## Two properties of the generator worth knowing before rebuilding

* **The shard factor must divide the output-channel count.** `fused_full`'s convolutions have 64, 32
  and 16 output channels, so widths 2 and 4 shard and width 3 does not: asked for 3, the generator
  emits zero `conv2d_s8_sharded` calls and the plain per-op path instead. It does this without a
  diagnostic, and `ModelBlaster/scripts/check_kernel_coverage.py` still reports OK, because every op does have a
  kernel. Confirm a width took by counting `conv2d_s8_sharded(` in the generated `model.c`.
* **`CROSS` must be set when generating kernels.** Without it the curated RVV kernels fail their
  cross-compile verify and the generator falls back to the scalar reference, silently and with
  coverage still OK. The difference is large — a scalar-reference `fused_full` measures about 17x the
  curated one. The provenance is recorded per op in the generated `kernel_picks.json`, and a curated
  `fused_full` build has a roughly 20 KB `kernels.c` against roughly 8 KB for the reference one.

## Reproducing

```bash
eval "$(bash scripts/setup_spacemit_toolchain.sh)"        # defines CROSS
cd ModelBlaster
for w in 1 2 4; do
  OUT=build/k1_ros_shard$w/fused_full/int8; IR=build/k1_xpurt/fused_full/int8
  mkdir -p $OUT/generated && cp $IR/{graph.json,weights.npz,io.npz} $OUT/
  MB_SHARD_FACTOR=$w ../.venv/bin/python -m modelblaster.pipeline.generate_skeleton --ir $OUT/graph.json \
      --weights $OUT/weights.npz --io $OUT/io.npz --out-dir $OUT/generated --backend rvv_x60 --platform linux
  MB_SHARD_FACTOR=$w CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_kernels \
      --ir $OUT/graph.json --out-dir $OUT/generated --target rvv_x60 --backend reference --quant int8 \
      --global-curated-dir "$(pwd)/kernels"
  ../.venv/bin/python scripts/check_kernel_coverage.py $OUT/generated     # must print OK
  grep -c 'conv2d_s8_sharded(' $OUT/generated/model.c                     # must be 7 for w=2,4
done
```

Staging and timing on the board follow `scripts/board_campaign.sh` step 1, with `nav$w` in place of
`yolo4` and `-march=rv64gcv_zvfh` (this network has an fp16 linear).

## In the deployed chain: `cp3n4`

The standalone return above is a bound, not a chain result. `cp3n4` is `cp3` with navigation sharded
four ways over the efficiency cluster and control sharing those same harts (`--nav-pool 4
--nav-harts 4,5,6,7`, both processes `taskset`ed to `4-7`, control still in the goal callback). It
needs the node linked against the 4-way navigation build, which is why it selects
`ros_mb_chain_traced_pool_nav4` rather than the binary every other arm uses; that binary is built
once and the arms already measured are not relinked.

Placing the two on the same harts is defensible because they are consecutive links of one chain:
navigation publishes the goal and control runs in that goal's callback, so for a given frame they
are not resident at the same time.

Three replicates of each at a 30 Hz camera:

| arm | chain camera-to-goal, per replicate | median | spread | E-core busy `c4..c7` |
|---|---|---|---|---|
| `cp3` | 30.69, 30.62, 30.67 | 30.67 ms | 0.07 ms | 15 3 1 1 |
| `cp3n4` | 29.98, 32.16, 30.13 | 30.13 ms | **2.18 ms** | 12 6 6 7 |

The control cadence is the same in both (gap mean 33.40 ms, so 30 Hz; gap max 79-81 ms either way):
sharding navigation does not touch the command rate, because the rate is set by the chaining rather
than by how fast the chain runs.

The four efficiency cores are now all in use rather than three sitting idle, and they consume about
31 points of busy time against 20 for the same work. The chain does not get shorter: the means are
30.76 ms sharded against 30.66 ms unsharded, and the sharded arm's replicates spread thirty times
wider. That is the standalone number arriving as predicted -- 13 % of a roughly 4 ms stage is about
0.5 ms, which is below the variation the pool introduces at this width.

**So the deployed arms keep navigation on one hart.** Filling the idle cores costs processor time and
repeatability and returns nothing measurable at the end of the chain. Three replicates is thin
evidence for the spread specifically, which is reported as observed rather than as established.

## In the deployed chain with no idle harts: `vanilla4x2ns4`

`cp3n4` above put the nav pool on a cluster that was mostly idle. The stronger test is the
arrangement where nothing is idle: `vanilla4x2` already runs two YOLO pools across all eight harts
and keeps every one of them between 36 % and 59 % busy. `vanilla4x2ns4` adds the four-way nav pool
on top of that, so no stage of the perception→navigation chain is left on a single core.

One replicate of each at a 36 Hz camera (`HZ=36 NAVPOOL=4 scripts/ros_traced_matrix.sh vanilla4x2`,
with `NAVHARTS` naming the cluster for the pinned variants):

| arm | nav pool | nav callback | camera→goal | control cadence | busiest hart |
|---|---|---|---|---|---|
| `vanilla4x2d2` | none, nav on one hart | 4.15 ms | **32.56 ms** | 27.74 ms | 59 % |
| `vanilla4x2ns4c` | 4-way, **unpinned** | 5.76 ms | **37.35 ms** | 27.76 ms | 55 % |
| `vanilla4x2ns4a` | 4-way, harts 0–3 | 5.77 ms | 38.71 ms | 27.79 ms | 54 % |
| `vanilla4x2ns4b` | 4-way, harts 4–7 | 5.95 ms | 39.17 ms | 27.78 ms | 54 % |

The pool makes the chain **1.15× longer**, and it is not the kernel that fails to scale: the same
sharded navigation is 1.66× faster on four harts than on one when run standalone on an idle board
(226 k → 136 k cycles, the section above). YOLO's own callback degrades alongside it, 25.89 → 29.93 ms.
There is simply no idle capacity for the pool to take, so it takes YOLO's, and the chain pays the
dispatch and barrier cost on both sides.

Leaving the pool unpinned is the best of the three placements, which is the expected ordering: the
OS can put a worker wherever a hart is free that instant, where a fixed cluster cannot.

**This is the result worth having, and it is not a negative one.** It says the ROS 2 baseline drawn
in the 36 Hz showdown is not being held back by a single-threaded stage — it has been given the most
parallel arrangement of itself that we can build, and it is still bound by the control-rate floor,
because that floor comes from chaining control to perception rather than from how fast any stage
runs. See [`showdown_cam36_allcores_reproduction.md`](../Evaluation/showdown_cam36_allcores_reproduction.md) §3.

**Building it for the board.** The nav node must be linked against the 4-way build, and the compile
needs `-march=rv64gcv_zvfh` rather than plain `rv64gcv`: the sharded convolutions emit `vfloat16m1_t`,
which requires the `zvfh` extension. `modelblaster_pool_trace_arm` must be called on the nav pool as
well as the YOLO pool, or the run records no nav shard slices and the Gantt draws nav on one lane.
`--nav-harts` is optional — omitted, the pool is created unpinned, which is variant `c`.

---

## Kernel inventory

`scripts/kernel_inventory.py [--board]` reports, per build, the kernel provenance recorded in
`kernel_picks.json`, the size of the generated `kernels.c`, and how many convolutions were emitted in
their sharded form; with `--board` it also asks the K1 what is deployed and greps each `kernels.c` for
vector and matrix-engine instructions. It exists because both failure modes in §"Two properties"
above are silent and invisible in a directory listing.

What it reports today. Every kernel deployed on the board carries vector code: `ctrl` 25 intrinsic
lines, `nav` and its 1/2/4-wide variants 66, `yolo` and `yolo4` 164, `yolo4_ime256` 162 plus **15
matrix-engine lines**, `yolo4_imetab` 173 plus 8. Nothing scalar is deployed.

One build is flagged: `k1_ros_ctrl/yolov8_nano_64x96` carries the scalar reference set. It is
referenced by no script and no page here, and nothing measured was built from it.

## Where the matrix engine is and is not used

| network | `ime_x60` profile | what the unconstrained solve takes |
|---|---|---|
| `yolov8_nano_64x96` | yes | **679 of 1470** dispatches |
| `fused_full` | yes | none -- the solver costs both and RVV wins |
| `mlp_control` | none | not offered |

`mlp_control` is absent from the engine's tables by measurement rather than by omission. Its four
linears are all `M=1`, and `pipeline/ime_cost.ime_speedup_for` returns **0.12x** on each, with
`ime_useful` answering `False, 'IME not faster on any known shape -> excluded from ime table'`. The
engine's win tracks `K` and `N` -- how full a tile is -- not `M`, so a matrix-vector product leaves
most of it idle.

The detector's own table keeps the same rule per dispatch rather than per network: the measured
comparison (`artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv`) has RVV ahead on 8 of 57 shapes, and
those 8 run on RVV.

**Still unexploited:** the sliding-window family `vmadot1/2/3/n`. The ISA probe
(`ModelBlaster/artifacts/ime_isa_probe/FINDINGS.md`) shows `vmadot1` executes on cluster 0, and every
kernel here emits only `vmadot`. It exists so adjacent output positions can reuse already-loaded A
registers instead of gathering them again, which is the im2col step the fused convolution performs per
tile -- and the worst of the 8 losing shapes are the detect-head convolutions at `M=6`, where the
gather is largest relative to the work it feeds. Whether the slide form closes that gap is unmeasured.

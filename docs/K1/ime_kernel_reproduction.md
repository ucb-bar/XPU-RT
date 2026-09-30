# The K1 IME on the deployed chain: measuring it, the fused kernel, and what it is worth

The SpaceMiT K1 has an int8 matrix engine (`smt.vmadot`, legal on cluster 0 only — harts 4-7 SIGILL).
Before this it contributed nothing to the deployed 3-network chain, and the reason was not the
hardware. This is how that was established, what was written, and what it measured.

## 0. Before any of this: getting the submodule, which a clone cannot do for you

**Everything below §2 lives inside the `ModelBlaster/` submodule, and a plain clone cannot fetch
it.** This is the one step the repository cannot take on a reviewer's behalf, so it is first.

The superproject pins ModelBlaster at `ed776fd1a9457a1c39498e1067524beeff15ec34`, and that commit
*does* contain all the IME work — the fused kernel, both benches, both measured tables, the
only-if-better rule and the pool's per-slice tracing. What it is not is **pushed**: it sits on the
local branch `feat/split-linear-along-m`, nine commits ahead of that branch's tip on
`github.com/ucb-bar/ModelBlaster`, and `git ls-remote` does not have it. So

```bash
git clone --recurse-submodules <this repo>     # ModelBlaster/ comes up EMPTY
git submodule update --init ModelBlaster       # fatal: ... did not contain ed776fd1...
```

fails outright rather than checking out something older. Note what the failure is *not*: the pointer
is not stale, and advancing it would not help — there is no pushed commit that has this work, so
every candidate the pointer could name has the same problem. Only a push fixes it at the remote, and
that is the maintainer's to do.

Until then the commit travels in a git bundle, the same convention `archive_v3/` uses for the board
traces: the bundle is large and untracked, its sha256 and its exact tip are in the **tracked**
`artifact/history/MANIFEST.sha256`. With the bundle beside the clone:

```bash
git submodule init ModelBlaster
git config submodule.ModelBlaster.url "$PWD/artifact/history/modelblaster.bundle"
git submodule update ModelBlaster
git -C ModelBlaster rev-parse HEAD        # ed776fd1a9457a1c39498e1067524beeff15ec34
```

`artifact/history/README.md` has the rest, including how to re-cut the bundle and how to point the
submodule back at the real remote afterwards. `scripts/verify_modelblaster_inputs.py` checks that the
bundle's recorded tip is still the commit the superproject pins — re-cut the bundle without moving
the pointer, or move the pointer without re-cutting, and the two drift apart with nothing else to say
so. With the submodule absent that script reports the content checks as SKIP, not PASS.

## 1. Why the engine was unreachable

Three separate causes, each of which alone hides the engine:

1. **No kernel for the op the graph actually has.** Curated kernels are looked up by exact op name.
   The graph fuses Conv→BN→SiLU into `conv2d_batchnorm2d_silu_s8`; the IME library had only
   `conv2d_s8`. That is 57 of the deployed `yolov8_nano_64x96`'s 90 dispatches — every backbone
   convolution, including every one the engine could win.
2. **Fused nodes reported no shape.** A fused node carries no `shape` of its own (the numbers belong
   to the `conv2d_s8` it fused), so `shapes_from_ir` returned `{}` and every shape-keyed lookup —
   including the guard that decides IME eligibility — saw nothing to match.
3. **The cost table was measured against a different baseline and different shapes.**
   `artifacts/ime_conv/ime_vs_rvv_conv.csv` compared the IME against the *standalone* RVV conv, not
   the fused RVV kernel that is deployed, and its 50 shapes came from a larger yolov8. None of the
   deployed net's shapes were in it, so the only-if-better rule excluded them as unknown.

## 2. Measuring the engine on the deployed net's own shapes

```bash
eval "$(bash scripts/setup_spacemit_toolchain.sh)"
cd ModelBlaster
MB_IME_BENCH_NETS=yolov8_nano_64x96 MB_IME_BENCH_OUT=$PWD/artifacts/ime_conv_yolo64x96 \
    ../.venv/bin/python scripts/ime_conv_verify_bench.py        # needs the board
```

Compiles the IME kernel, the RVV conv kernel and an independent scalar oracle into one binary, checks
both against the oracle per shape, and times them with `rdcycle` on cluster 0. Result: 34 shapes, all
bit-exact, IME faster on 8 (best 1.71×), 8.3 % fewer conv cycles taking the faster per shape.

**Read that number carefully.** It is against the standalone RVV conv. The deployed build runs the
fused kernel, and against *that* the same comparison is a loss — which is what §4 measures.

The rows are appended to `artifacts/ime_conv/ime_vs_rvv_conv.csv`, the table the picker's guard and
`pipeline/ime_cost.py` read.

For the fused op the right table is the one measured against the *fused* RVV kernel:

```bash
cd ModelBlaster && ../.venv/bin/python scripts/ime_fused_conv_bench.py   # needs the board
# -> artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv
```

It compiles the IME and the deployed RVV fused kernels into one binary over the graph's own 57
fused-conv dispatches with their real quant parameters, requires the IME output to be
**byte-identical** to RVV's, and times both. That is the loop every change in §4a was measured in;
it costs ~40 s of board time against ~10 min for a full model build and harness run.

## 3. The fused kernel

`ModelBlaster/kernels/ime/ime_conv2d_batchnorm2d_silu_s8_ime_vmadot_4x4x8.c` — the im2col→`vmadot`
MAC core of the IME conv kernel with the BN and SiLU stages of the RVV fused kernel folded into its
store path. Bit-exact by construction: the MAC is integer and identical to RVV's, and the two float
stages are the same expressions in the same order.

Two things in it were found by measuring, not by reading:

* **The epilogue is tabulated only above a break-even.** Both float stages map int8 to int8, so a
  256-entry table per channel is memoization. But building it costs 256 evaluations per channel while
  a panel produces only `M = OH*OW` outputs per channel: tabulating unconditionally cost 16k
  evaluations against 6k outputs on this net's 8×12 layers — 163 ms against 47 for the whole model.
  Gated at `M >= 256` (the RVV kernel's own threshold) it is 96 ms.
* **A 1×1 fast path.** For `KH=KW=1, S=1, P=0` the im2col matrix is the input transposed, so the
  general path's four integer divisions per packed byte are pure overhead — and the 1×1 layers are
  exactly where the engine is otherwise at parity. Adding it took the win count from 0 to 11.

Register the algorithm for an op in `pipeline/reference_kernels.py` (`AlgorithmCandidate(name=
"ime_vmadot_4x4x8", target_affinity=("ime","ime_x60"), ...)`), or the picker will never look for the
file.

```bash
OUT=build/k1_ime_fused/yolov8_nano_64x96/int8; IR=build/k1_xpurt/yolov8_nano_64x96/int8
mkdir -p $OUT/generated && cp $IR/{graph.json,weights.npz,io.npz} $OUT/
CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_skeleton --ir $OUT/graph.json \
    --weights $OUT/weights.npz --io $OUT/io.npz --out-dir $OUT/generated --backend ime_x60 --platform linux
CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_kernels --ir $OUT/graph.json \
    --out-dir $OUT/generated --target ime_x60 --backend reference --quant int8 --global-curated-dir "$(pwd)/kernels"
```

The log must say `[conv2d_batchnorm2d_silu_s8/ime_vmadot_4x4x8] ... curated[ime] verify PASS`; if it
says "IME not probed — not faster on any known shape", §2 has not been run or its rows are missing.
Add `MB_SHARD_FACTOR=4` to both commands for the sharded build.

## 4. What it is worth, measured on the board

One hart and four, every run bit-exact against the golden output (`max_abs_err=0`). Both columns
are per-dispatch medians over 20 iterations of the same harness; "before" is the first fused kernel
(`ime_fused_v3_1core.txt`, `ime_shard4_4harts.txt`), "after" is the packing rewrite of §4a
(`ime_fused_v5_1core.txt`, `ime_shard4_v5_4harts.txt`), all four in
`results/codesign_feedback/ros_traced/yolo_standalone/`.

| | 1 hart | 4 harts (sharded) |
|---|---|---|
| deployed RVV | 47.15 ms | 24.26 ms |
| IME forced on every fused dispatch — **before** | 90.00 ms | 52.35 ms |
| IME forced on every fused dispatch — **after** | **32.58 ms** | **16.77 ms** |
| per-dispatch best-of — before | 45.40 ms | 24.25 ms |
| **per-dispatch best-of — after** | **32.19 ms (31.7 % faster)** | **15.71 ms (35.2 % faster)** |
| fused-conv dispatches the engine wins | 11 → **50** of 57 | 0 → **45** of 57 |

Before the rewrite the engine was only ever a per-dispatch *alternative* — forcing it everywhere
was ~2× slower than RVV. It is no longer: forced on every fused conv it now beats all-RVV outright,
at one hart and at four. The best-of is still better than either pure arm, because seven dispatches
still lose (§6).

## 4a. Where the engine's time was actually going

The first kernel's losses were not the MAC. Measured per shape against the *deployed fused* RVV
kernel with `ModelBlaster/scripts/ime_fused_conv_bench.py` (57 real dispatches, ~40 s of board time per
iteration, IME output required to be byte-identical to RVV's), three packing faults accounted for
almost all of it:

1. **B was re-packed for every 4-channel panel.** `for n0 { pack B panel; for mt { MAC } }` walks
   the weight tensor at stride `OC` taking four bytes from each `OC`-byte row — one cache line
   fetched per byte kept, `OC/4` times over. `l7.conv` (K=1152, OC=256) made 64 passes over a
   288 KB tensor to feed a MAC with M=6 rows: 1.96 ms against RVV's 0.97 ms. Packing B **once per
   dispatch**, in one linear pass over the weight in its native k-major order, is the single
   largest fix.
2. **The A gather derived its taps with two integer divisions per packed byte.** `l0.conv` packed
   49 KB of im2col at ~80 cycles a byte: 2.51 ms against RVV's 0.95 ms. Counting `k` over
   `(ic, kh, kw)` removes every division.
3. **Neither copy was vectorised**, and could not be: in the old `[m-tile][k-slab][i][q]` packing
   order a fixed tap's bytes across consecutive output pixels are 8 apart *inside* a tile and then
   jump at the tile boundary. Storing as **`[k-slab][m-tile][i][q]`** makes that stride a uniform 8
   for every pixel (crossing a tile boundary also advances by 32−24 = 8), so one (tap, output row)
   pair is one strided copy — and a whole k-slab is one 8-field **segment store** (`vsseg8e8`),
   which is what finally beats the strided scatters. B is packed `[k-slab][panel][j][q]` for the
   same reason. The MAC walks the slab axis with a stride instead of a fixed 32: one `add` in place
   of one `addi`. **The 32-byte tiles the engine sees are byte-for-byte the ones it saw before.**

A fourth, smaller one: the general path packs from a zero-padded copy of the input, so no tap needs
a bounds test and every tap of a slab spans the same run of pixels (all-IME 32.19 → 31.80 ms on the
bench). Nothing in the epilogue changed — the float expressions are still textually the RVV
kernel's, which is why all 57 shapes stayed byte-identical through every step.

Per-shape, on the bench: all-RVV 41.7 ms, all-IME 66.5 → 31.8 ms, IME wins 9 → 49 of 57.

**One thing that did not work.** On a 2×3 feature map the general path copies 2–3 bytes at a time,
so a scalar loop for `n <= 4` inside the copy helper looks free. It cost 31 of the 49 wins
(all-IME 32.38 → 48.59 ms), and not on the short copies — `detect.cv2_1_0`, whose runs are 5–6
long and never take the scalar branch, went 0.661 → 0.934 ms. The branch makes the helper too big
to inline, so every call becomes a real call with its own `vsetvl`. The helper stays branchless.

Sharding and the matrix engine compose — the engine is **per-hart on cluster 0**, not one shared
unit: the sharded IME build runs 191 / 99 / 52 ms on 1 / 2 / 4 harts before the rewrite, 16.77 ms on
4 harts after it.

## 5. Delivered: one schedule, mixed per dispatch, executed on the board

The best-of is no longer a best-of across two runs. One schedule now routes each YOLO dispatch to
the engine the measurements say is faster, and it has been executed.

**The cost model had to be fixed first.** `pipeline/ime_cost.py` keyed every `conv2d*` op to
`artifacts/ime_conv/ime_vs_rvv_conv.csv`, measured against the STANDALONE RVV conv — a kernel the
deployed build never executes (§1.3). It now holds one table per conv op-kind:

    conv2d_s8                   -> artifacts/ime_conv/ime_vs_rvv_conv.csv
    conv2d_batchnorm2d_silu_s8  -> artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv

and any other `conv2d*` op has no table, so it stays on RVV. A shape measured more than once (the
fused table has one row per DISPATCH) keeps its WORST speedup. Locked in by
`pipeline/tests/test_ime_cost.py`.

That one lookup is the difference between the engine being unreachable and being deployed:

| | (op, shape) placed on IME | fused-conv dispatches | in the ime_x60 build? |
|---|---|---|---|
| standalone table (before) | 8 | — | **no** — `ime_useful` said "not faster on any known shape" |
| fused table (after) | **23** | **49 of 57** | yes, `curated[ime]/ime_vmadot_4x4x8` |

```bash
python -m modelblaster.pipeline.apply_ime_hint --ir build/k1_xpurt/yolov8_nano_64x96/int8/graph.json \
    --network yolov8_nano_64x96 --out /tmp/graph.ime.json     # 23 placed, 12 kept on RVV
```
The 23 (op, shape) records cover exactly the 49 dispatches the fused table calls a win, and no
others — the only-if-better rule placing nothing it has not measured faster.

### Getting them into a schedule

The scheduler routes per dispatch through `scheduler.enable_impls`: every cluster-0 core-group
combination is emitted a second time as an `ime` combination whose cost is read from the net's
`ime_x60` profile, the two share machine names so they serialise against each other, and
`postprocessing` records the winner as the dispatch's `impl`. `ingest_xpurt_schedule` writes that
`impl` into the dispatch table and the walker strcmps it to pick the backend's per-model function
table — buffers are shared per model, so a network may mix engines dispatch by dispatch.

So what was missing was the `ime_x60` profile, and **the IME's win depends on the width**: at one
hart it takes 49 of 57 fused convs, at four it takes 46 — and not the same 46. `l0.conv` LOSES at
one hart (0.96x) and wins 2.94x at four, because sharding OC=16 across four harts is bad for the
RVV kernel and fine for the IME's. A width-blind speedup misplaces a dozen dispatches, so
`scripts/make_ime_profile.py` grew a `--from-runs` mode that takes the per-dispatch ratio from a
matched PAIR of whole-net board runs at that width, and applies it to the deployed profile's own
RVV baseline so both cells are on one clock:

```bash
R=results/codesign_feedback/ros_traced/yolo_standalone
scripts/make_ime_profile.py --net yolov8_nano_64x96 --gen-root gen/mb_shard/profile --topo topo_0 \
    --from-runs $R/1core.txt:$R/ime_fused_v5_1core.txt              # 49/90 cells
scripts/make_ime_profile.py --net yolov8_nano_64x96 --gen-root gen/mb_shard/profile --topo topo_0_1_2_3 \
    --from-runs $R/4core_w.txt:$R/ime_shard4_v5_4harts.txt          # 46/90 cells
```

A ratio is only an IME result for an op the picker actually swaps, so the tool asks `ime_useful`
which ops the `ime_x60` build will swap and gives cells to those only — without that guard
yolo's two `conv2d_s8` detect heads picked up "1.03x" cells that were two measurements of the
same RVV kernel. There is no 2-hart IME run, so 2-wide ime cells are absent and the solver simply
never places there (`INFEASIBLE_COST`), which is the designed behaviour for a missing ime profile.

Then the ordinary path, with one spec change (`enable_impls: true`, `data/toplevel/wh_chain36_w2pime.json`):

```bash
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain36_w2pime.json \
    --solver greedy_periodic --max-periodic-iters 1 --use-profiled --random-seed 42
cp schedules/scheduled_wh_chain36_w2pime_greedy_periodic_profiled.json schedules/fig_w2pg36ime_greedy.json
.venv/bin/python scripts/clamp_schedule_widths.py schedules/fig_w2pg36ime_greedy.json $IRS \
    --out schedules/fig_w2pg36ime_greedy_clamped.json
.venv/bin/python scripts/check_schedule_feasibility.py --schedule schedules/fig_w2pg36ime_greedy_clamped.json \
    --gen-root gen/mb_shard
( exec 9>results/codesign_feedback/board.lock; flock 9
  BACKENDS="rvv_x60,rvv_x60,ime_x60" CORE_KINDS="rvv,rvv_c1,ime" \
    scripts/run_xpurt_long.sh schedules/fig_w2pg36ime_greedy_clamped.json w2pg36ime 3 )
```

1492 of the 4768 dispatches carry `impl: ime` — 41 per frame — and every one is on cluster 0
(`find_illegal_implementations` is what makes that structural rather than lucky; harts 4-7 SIGILL).

Two checks had to learn what a width-dependent IME cell means, and both are covered by tests:

* `check_schedule_feasibility.py --gen-root` — the kernel-availability check hard-coded `gen/mb`
  while the spec is solved against `gen/mb_shard`, so every ime dispatch read as having no kernel.
* the same check asked the `topo_0` profile whether a kernel exists. Availability is per (model,
  backend, op) — the codegen commits ONE kernel per op — so it now unions every profiled width and
  decides for the dispatch's OP. It refused the legal 4-hart `l0.conv` placement purely because
  that layer's ONE-hart cell is rvv-costed. An op with no ime row at any width still fails.
* `run_xpurt_k1.sh`'s staleness list gained `pipeline/ime_cost.py` and `reference_kernels.py`. The
  list exists so a curated-kernel edit cannot be left behind in a generated `kernels.c`; a
  correction to the table that DECIDES whether that kernel is swapped in belongs in it for exactly
  the same reason.

### What it measured

36 Hz camera, three replicates per arm, `SCHED_OTHER`, both binaries cross-built the same day from
the same tree — the all-RVV arm was re-measured so this is a same-build A/B, not a comparison
against an older number. Traces in `results/codesign_feedback/xpurt_long/trace_w2pg36{base,ime}_other_run{1,2,3}.csv`,
summary in `results/codesign_feedback/ime_mixed_schedule_36hz.json`.

| camera->control, ms | run 1 | run 2 | run 3 | median | YOLO span/frame |
|---|---|---|---|---|---|
| all-RVV (`fig_w2pg36_greedy_clamped`) | 29.73 | 29.65 | 29.81 | **29.73** | 21.53 ms |
| per-dispatch IME (`fig_w2pg36ime_greedy_clamped`) | 21.98 | 22.01 | 23.08 | **22.01** | **14.53 ms** |

**22.01 ms against 29.73 — 1.35x, 7.7 ms off the chain**, and 1.37x against the 30.1 ms the figures
quote for this arm. YOLO's own span is 1.48x (21.53 -> 14.53 ms). The tail moves with it: chain p95
33.2-35.6 -> 26.4 ms, chain max 41.2-42.1 -> 28.4-28.8 ms.

**Bit-exact, every run.** `MODELBLASTER_VERIFY [yolov8_nano_64x96] max_abs_err=0` on all six runs.
`fused_full`'s 1.83e-4 is its fp16 nav head and is byte-identical between the two arms — nav is
fp16 and the K1's engine is int8, so it is never routed to the IME (§6).

The gain is larger than the 31.7 % / 35.2 % standalone best-of because the schedule collects it
twice: the dispatches get faster AND the frame's critical path shortens, so the solver stops
spilling work to cluster 1 (280 yolo dispatches land on CPU_E instead of 392, in the schedule and
in the trace alike) and hart 0 comes back from saturation -- its measured kernel fraction
(`hart_acc_*.csv`) falls from 83 % to 65 %.

### What is still not delivered

* Only `yolov8_nano_64x96` has an `ime_x60` profile built this way. `fused_full` is fp16 and
  `mlp_control`'s linears are M=1 (0.12x), so neither is a candidate without the requantization
  §6 describes — but nothing has measured the other nets at all.
* No 2-hart IME measurement, so the solver cannot consider a 2-wide IME block.
* The ime cells are a measured per-dispatch RATIO applied to the deployed RVV profile, not a
  per-dispatch measurement of the deployed mixed build. They carry `source=ime_derived(...)` and a
  `PROVENANCE.json` saying so, and `--require-source k1` will not accept them.
* Figures still quote 30.1 ms for this arm; nothing here has been wired into a figure.

## 6. Where the engine still loses, and why that part is structural

Seven fused-conv dispatches still lose at one hart, and they are all the same shape family: the
M=6 (2×3 feature map) 3×3 convolutions — `detect.cv{2,3}_2_*`, `l7.conv`, `l8.m0.*`, `l21.m0.*`,
`l19.conv` — at 0.81–0.99×. M=6 fills two 4-row tiles with two rows wasted, *and* amortizes B's
transpose (a full pass over the weight tensor, which RVV never has to make) over only those two
tiles. It is the small-M limit the matmul kernel's header already records (M=7 → 0.25×); packing
faster took it from 0.30× to 0.81× but does not remove it. At four harts the sharding splits OC, so
each hart has fewer panels to amortize its A pack over and twelve dispatches lose instead of seven.

Outside this net the same limit holds: the control network's linears are `M=1`, measured at 0.12× —
a matrix engine cannot pay there. Nav is fp16, and the K1's engine is int8 only, so it would need a
requantization the accuracy contract has to approve first.

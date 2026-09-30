# Giving the ROS 2 baseline the matrix engine

The flight figures compare XPU-RT against a ROS 2 Jazzy deployment of the same three networks on
the same SpaceMiT K1, running the same ModelBlaster-generated kernels
([`ros_baseline_reproduction.md`](ros_baseline_reproduction.md)). XPU-RT's schedule routes some
YOLO dispatches to the K1's int8 matrix engine ([`ime_kernel_reproduction.md`](../K1/ime_kernel_reproduction.md)
§5); the ROS arm's kernels are all RVV. A reviewer can fairly ask whether the comparison is
measuring the runtime or the accelerator.

This page answers that by **building the ROS arm the engine and measuring it**, in the form an
engineer with an accelerator and no per-shape profiling would actually deploy it: the IME kernel
for every op that has one. Four arms, all at the 30 Hz camera the figure uses, three replicates
each, on the board.

Related: [`ime_kernel_reproduction.md`](../K1/ime_kernel_reproduction.md) for the engine and its kernels,
`ModelBlaster/artifacts/ime_isa_probe/FINDINGS.md` for which instruction families this part
implements, [`ros_arms_catalog.md`](ros_arms_catalog.md) for every ROS arm measured here.

---

## 1. The mechanism, and why it decides the answer before any timing does

`smt.vmadot` **executes on cluster 0 (`CPU_P#0-3`) and raises SIGILL on cluster 1 (`CPU_E#0-3`)**.
It does not fall back and it does not run slowly: the process dies. That is established per hart by
the SIGILL probe in `ModelBlaster/artifacts/ime_isa_probe/FINDINGS.md`, and it is the reason the
schedules XPU-RT emits carry a structural check that refuses an `ime` dispatch on a CPU_E hart
(`find_illegal_implementations`, `tests/test_codegen_contract.py`).

ROS 2 has no such check, and the traced node never pins itself — placement is whatever the
launcher's `taskset` gave it, and the manifest records the mask it ran under. So giving ROS the
engine forces a choice that giving it the RVV kernels does not:

* leave the deployment where it is, spread across both clusters, and the processes on cluster 1
  die the first time they run a convolution; or
* confine the whole deployment to cluster 0, and run the three networks on four harts instead of
  eight.

Both are measured below, and the second is measured **with RVV kernels too**, so the accelerator
can be separated from the half machine it costs.

---

## 2. Building an all-IME YOLO: `MB_IME_FORCE`

The kernel picker is table-guided. An (op, shape) reaches the engine only where a measured table
says the engine beats the RVV kernel that would otherwise run; `pipeline/ime_cost.ime_speedup_for`
returns nothing for an unmeasured shape, and nothing means RVV
(`ModelBlaster/pipeline/ime_cost.py`). On the deployed `yolov8_nano_64x96` that puts
`conv2d_batchnorm2d_silu_s8` — 57 of the 98 dispatches — on the engine and leaves the six
`conv2d_s8` detect heads on RVV.

`MB_IME_FORCE=1` switches the guard off: every op that HAS an IME kernel takes it, measured or not,
winner or loser. The default does not move; the switch is opt-in, and the build it produces says in
its own metadata that it is not table-guided.

```bash
cd /scratch/agustin/xpurt-dev-sync && eval "$(bash scripts/setup_spacemit_toolchain.sh)"
cd ModelBlaster
OUT=build/k1_ime_force_shard4/yolov8_nano_64x96/int8; IR=build/k1_xpurt/yolov8_nano_64x96/int8
mkdir -p $OUT/generated && cp $IR/{graph.json,weights.npz,io.npz} $OUT/
MB_SHARD_FACTOR=4 MB_IME_FORCE=1 CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_skeleton \
    --ir $OUT/graph.json --weights $OUT/weights.npz --io $OUT/io.npz \
    --out-dir $OUT/generated --backend ime_x60 --platform linux
MB_SHARD_FACTOR=4 MB_IME_FORCE=1 CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_kernels \
    --ir $OUT/graph.json --out-dir $OUT/generated --target ime_x60 --backend reference \
    --quant int8 --global-curated-dir "$(pwd)/kernels"
../.venv/bin/python scripts/check_kernel_coverage.py $OUT/generated      # must print OK
```

Drop `MB_IME_FORCE=1` from both commands (and change `OUT` to `build/k1_ime_table_shard4`) for the
table-guided build the comparison needs. The generator prints, and `kernel_picks.json` records:

```
[conv2d_s8] MB_IME_FORCE=1 -- probing IME ANYWAY; the table says: IME not faster on any known shape
*** BUILD IS NOT TABLE-GUIDED: MB_IME_FORCE=1 selected the K1 IME kernel for every op that has one ...
*** ops on the IME in this build: ['conv2d_batchnorm2d_silu_s8', 'conv2d_s8']
```

`table_guided: false`, `ime_force: true`, `ime_forced_ops` and `ime_forced_over_table_ops` go into
`kernel_picks.json` next to the kernels, so the build cannot later be mistaken for a table-guided
one. Locked in by `ModelBlaster/pipeline/tests/test_ime_force.py`.

### Is it actually all-IME

| build | IME ops | dispatches on the engine | `vmadot` sites in `kernels.c` |
|---|---|---|---|
| `k1_ros_shard4` (deployed RVV) | — | 0 / 98 | 0 |
| `k1_ime_table_shard4` (table-guided) | `conv2d_batchnorm2d_silu_s8` | 57 / 98 | 7 |
| `k1_ime_force_shard4` (`MB_IME_FORCE=1`) | + `conv2d_s8` | **63 / 98** | 13 |

63 of 98 is every dispatch in this net that has an IME implementation: the remaining 35 are
`add`, `cat`, `chunk`, `maxpool` and `upsample`, for which the library has no matrix kernel at
all. The switch removes a guard; it does not invent implementations.

### Numerics

Bit-exact, at both widths, against the same golden output the RVV build is checked against:

```bash
ssh k1 'cd /root/ros_mb && MODELBLASTER_CPU=0-3 MODELBLASTER_ITERS=20 ./yolo4_ime256/harness'
# === MODELBLASTER_VERIFY === max_abs_err=0 max_rel_err=0 n=8316
```

`results/codesign_feedback/ros_traced/yolo_standalone/ime_force_shard4_{1hart,4harts}.txt`:
`max_abs_err=0` on both. No shape in the forced build fails the golden compare. This is the
whole-net check rather than `ModelBlaster/scripts/ime_fused_conv_bench.py`, because the bench covers the 57
fused convs and the forced build's extra six dispatches are `conv2d_s8`.

### The model alone, four harts on cluster 0

20 iterations, median of the warm iterations, `ticks/24000`:

| build | 4 harts (0-3) |
|---|---|
| deployed RVV (`yolo4`) | 25.02 ms |
| table-guided IME | 17.69 ms |
| all-IME (`MB_IME_FORCE=1`) | **17.26 ms** |

Forcing the six detect-head convs onto the engine as well is worth 0.4 ms here — the tables exclude
them because they lose at the shapes they are measured at, and at four harts that exclusion is
close to free either way.

**One build detail is load-bearing.** The IME kernels must be compiled with the `ime_x60`
backend's own `-march` (`-march=rv64gcv_zvl256b_zfh_zvfh`), not the plain `-march=rv64gcv` the ROS
staging in `ros_baseline_reproduction.md` §1 uses for the RVV kernels. With `rv64gcv` the same
all-IME build measures 30.45 ms instead of 17.26
(`yolo_standalone/ime_force_shard4_4harts_marchgcv.txt`). The RVV build barely notices the flag —
24.49 ms with `zvl256b` against 25.02 without (`rvv_shard4_4harts_zvl256b.txt` and
`rvv_shard4_4harts_today.txt`, same sources, same hour) — which is why the ROS staging never needed
it, and why the RVV arms below are the board's existing binary, unchanged.

---

## 3. The four arms

All four are the paper figure's `vanilla4x2` layout at 30 Hz: five processes (camera, two
perception, nav, control), the camera alternating frames between the two perception processes,
each perception process running YOLO on a 4-hart pool, control chained to the goal topic.
The arms differ only in **which kernels are linked** and **where the processes are allowed to run**.

```bash
cd /scratch/agustin/xpurt-dev-sync
# (a) IME kernels, the deployment as it stands: pools on 0-3 and 4-7
for r in 1 2 3; do RATES=30 SUFFIX=_ime BINSUF=_ime scripts/ros_traced_matrix.sh vanilla4x2 $r; done
# (b) IME kernels, everything confined to cluster 0
for r in 1 2 3; do RATES=30 SUFFIX=_ime BINSUF=_ime scripts/ros_traced_matrix.sh vanilla4x2c0 $r; done
# (c) RVV kernels, everything confined to cluster 0 -- the control for (b)
for r in 1 2 3; do RATES=30 SUFFIX=_rvv scripts/ros_traced_matrix.sh vanilla4x2c0 $r; done
# (d) RVV kernels on both clusters is already on disk as 30_vanilla4x2_r{1,2,3}

.venv/bin/python scripts/pull_ros_traced.py 30_vanilla4x2_ime_r1 ... 30_vanilla4x2c0_rvv_r3
.venv/bin/python scripts/ros_ime_ladder.py
```

`BINSUF` selects which linked copy of the traced program the arm runs (`_ime` →
`ros_mb_chain_traced_pool_ime`, the same C linked against the all-IME YOLO); `vanilla4x2c0` is
`vanilla4x2` with both perception pools on harts 0-3 and every process `taskset -c 0-3`. Staging
and linking the IME binary on the board:

```bash
G=ModelBlaster/build/k1_ime_force_shard4/yolov8_nano_64x96/int8/generated
ssh k1 'mkdir -p /root/ros_mb/yolo4_ime256'
scp $G/{buffers,kernels,model,weights}.c $G/{kernels,model,weights,test_io}.h \
    $G/test_input.bin $G/test_golden.bin $G/kernel_picks.json \
    ModelBlaster/kernels/rvv/mb_rvv_vxrm_compat.h k1:/root/ros_mb/yolo4_ime256/
scp ModelBlaster/harness_linux/src/main.c k1:/root/ros_mb/yolo4_ime256/harness_main.c
sed "s#$PWD/$G/#/root/ros_mb/yolo4_ime256/#g" $G/test_io.S | ssh k1 'cat > /root/ros_mb/yolo4_ime256/test_io.S'
ssh k1 'cd /root/ros_mb && F="-march=rv64gcv_zvl256b_zfh_zvfh -mabi=lp64d -DMODELBLASTER_RVV_IHWOC_WEIGHTS=1"
  for f in model kernels weights buffers; do gcc -O2 $F -DMODELBLASTER_USE_POOL -DMODELBLASTER_PLATFORM_LINUX \
      -pthread -c yolo4_ime256/$f.c -o yolo4_ime256/$f.o -Iyolo4_ime256 -Ipool; done
  (cd yolo4_ime256 && gcc -c test_io.S -o test_io.o)
  gcc -O2 -DMODELBLASTER_USE_POOL -DMODELBLASTER_PLATFORM_LINUX -pthread yolo4_ime256/harness_main.c \
      yolo4_ime256/{model,kernels,weights,buffers,test_io}.o pool/modelblaster_pool.o \
      -o yolo4_ime256/harness -Iyolo4_ime256 -Ipool -lm
  source /opt/ros/jazzy_prebuilt/setup.bash; R=/opt/ros/jazzy_prebuilt
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp \
        -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools \
        -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4_ime256/{model,kernels,weights,buffers}.o \
      nav/*.o ctrl/*.o pool/modelblaster_pool.o -o ros_mb_chain_traced_pool_ime \
      -L$R/lib -Wl,-rpath,$R/lib $LIBS'
```

Only the YOLO objects change: `nav` and `ctrl` are the same objects both binaries link. Neither
of the other two networks is a candidate for this engine anyway — `fused_full` has an fp16 head
and the engine is int8 only, and `mlp_control`'s linears are M=1
([`ime_kernel_reproduction.md`](../K1/ime_kernel_reproduction.md) §6).

---

## 4. What arm (a) does: it dies

All three replicates, identically. The second perception process — the one whose pool the arm
places on harts 4-7 — is killed by SIGILL the first time it runs a convolution:

```
bash: 2141185 Illegal instruction  ./ros_mb_chain_traced_pool_ime --nodes perception2 \
      --cameras 2 --yolo-pool 4 --pool-harts 4,5,6,7 ...
percep2 exit=132        # 128 + 4 = SIGILL
```

It writes no manifest, no trace and no released rows: it does not get far enough to finish one
frame. The same five processes with both pools on cluster 0 exit 0. Raw capture in
`results/codesign_feedback/ros_with_ime/sigill_evidence_ros.txt`; the per-hart version, all eight
harts of the standalone model, in `sigill_evidence.txt` — harts 0-3 exit 0, harts 4-7 exit 132.
The kernel log records `cause: 0x2` (illegal instruction) with the instruction word in `badaddr`:

```
badaddr 0xe240342b  ->  opcode 0x2b (custom-1), funct3 3 (ss), funct7 0x71 (OPMMA)
```

which is `smt.vmadot` ss, exactly what the IME kernels emit as a `.insn`. The trap is the matrix
instruction itself, not something reached through it.

**The deployment does not stop — it halves.** The camera keeps alternating frames between a live
perception process and a dead one, so every second frame produces no goal, and the control node,
chained to the goal topic, fires half as often.

---

## 5. The ladder

30 Hz camera, 20 s per run, three replicates per arm, median across replicates, first 3 s
discarded. `late` counts released frames whose goal reached the control node more than one frame
period (33.33 ms) after release; `dropped` counts released frames that produced no goal at all.
Numbers from `results/codesign_feedback/ros_with_ime/ladder.json`.

| arm | kernels | harts | control cadence | camera→goal med | p95 | late | dropped | busy % on 0-3 / 4-7 |
|---|---|---|---|---|---|---|---|---|
| `30_vanilla4x2` | RVV | 8 | **33.33 ms (30.0 Hz)** | 31.38 ms | 32.40 | 14/509 | 35 | 38-48 / 16-34 |
| `30_vanilla4x2_ime` | IME | 8 (dies on 4-7) | 66.67 ms (15.0 Hz) | 24.07 ms | 24.59 | 1/255 | **291** | 22-27 / 1-4 |
| `30_vanilla4x2c0_rvv` | RVV | 4 (cluster 0) | 33.32 ms (30.0 Hz) | 31.10 ms | 60.21 | 120/510 | 35 | 72-77 / 0 |
| `30_vanilla4x2c0_ime` | IME | 4 (cluster 0) | 33.33 ms (30.0 Hz) | **23.96 ms** | 40.52 | 126/509 | 35 | 46-55 / 0 |

Read down the column that matters and the four arms say three separate things.

**The naive deployment is not slower, it is broken.** Its 24.07 ms camera→goal is genuinely the
fastest median in the table, and it is meaningless: it is the median of the 255 frames that
survived out of 546, taken on a machine where half the work has stopped. What the flight sees is
the control cadence, and that has gone from 33.33 ms to 66.67 — one command every other frame.

**Confined to cluster 0, the engine is a real gain on the median.** 23.96 ms against 31.10 ms for
the same four harts running RVV: 7.1 ms, 1.30×. It also buys headroom — per-core busy falls from
72-77 % to 46-55 %, which is the same 1.3-1.4× showing up as slack instead of latency. Against the
8-hart RVV arm the figures use, 23.96 against 31.38 ms is 1.31×, so on the median the engine pays
for the cluster it costs and a little more.

**It does not pay on the tail.** p95 goes 32.40 ms (RVV, 8 harts) → 40.52 ms (IME, 4 harts), and
late frames 14/509 → 126/509. Four harts is where both `vanilla4x2c0` arms hurt: two perception
processes sharing one 4-hart pool region contend whenever their frames overlap, and the engine
narrows that tail (60.21 → 40.52 p95, and control gap max 64-76 ms → 47-60 ms) without removing
it. The 8-hart RVV arm has no such contention, and that is what keeps its p95 inside the frame
period.

---

## 6. What this does to the fairness question

Stated neutrally: **the accelerator is available to ROS, and taking it changes what ROS can be
given.**

* There is no version of "ROS also gets the IME" that leaves the rest of the deployment alone.
  The instruction is legal on four of the eight harts, so a ROS deployment that uses it is a
  4-hart deployment. Arm (a) is what happens if that is not noticed, and it is a crash rather than
  a regression.
* Given the pinning, blanket IME improves the ROS chain's median by 1.30× over RVV on the same
  four harts and 1.31× over RVV on all eight — and worsens its p95 by 8.1 ms relative to the
  8-hart arm, because it is now sharing four harts between two perception processes.
* The comparison the figures make is therefore not "runtime with an accelerator against runtime
  without one". Both arms can reach the engine. What differs is that XPU-RT's schedule places
  the ime dispatches on cluster 0 and the rest elsewhere in one plan, so it uses the engine
  *and* all eight harts; ROS's placement is a `taskset` on a whole process, so using the engine
  at all means giving up the other cluster for every node in the deployment.
* The honest caveat in the other direction: nothing here has been run through a flight. These are
  board timings. And blanket IME is not the best the engine can do for ROS either — the
  table-guided build is 0.4 ms slower than the forced one on the model alone at four harts, and
  the per-dispatch mixed schedule ([`ime_kernel_reproduction.md`](../K1/ime_kernel_reproduction.md) §5)
  is a lever ROS has no way to pull at all, because a ROS node links one kernel per op.

---

## 7. Files

| | |
|---|---|
| the switch | `ModelBlaster/pipeline/ime_cost.py` (`FORCE_ENV`, `force_requested`, `FORCE_NOTE`), `ModelBlaster/pipeline/generate_kernels.py` |
| its tests | `ModelBlaster/pipeline/tests/test_ime_force.py` |
| the builds | `ModelBlaster/build/k1_ime_force_shard4/`, `ModelBlaster/build/k1_ime_table_shard4/` |
| model-alone runs | `results/codesign_feedback/ros_traced/yolo_standalone/{ime_force,ime_table,rvv}_shard4_*.txt` |
| crash evidence | `results/codesign_feedback/ros_with_ime/sigill_evidence{,_ros}.txt` |
| the arms | `results/codesign_feedback/ros_traced/30_vanilla4x2{_ime,c0_ime,c0_rvv}_r{1,2,3}/` |
| the ladder | `scripts/ros_ime_ladder.py`, `results/codesign_feedback/ros_with_ime/ladder.json` |
| the harness | `scripts/ros_traced_matrix.sh` (`BINSUF`, arm `vanilla4x2c0`) |

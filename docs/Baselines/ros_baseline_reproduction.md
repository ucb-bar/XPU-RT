# Reproducing the ROS 2 baseline on the K1

The baseline in the flight figures is **real ROS 2 Jazzy running the real ModelBlaster kernels** —
the same generated C that XPU-RT's runtime executes, on the same board. Only the orchestrator
differs. Every number the figures quote about it is read back from a raw file this page tells
you how to produce.

Related: [`measurements_and_ablations.md`](../Evaluation/measurements_and_ablations.md) for the board
measurements generally, [`K1/k1_board.md`](../K1/k1_board.md) for board access,
[`ros_with_ime.md`](ros_with_ime.md) for the same arms built with the K1 matrix engine instead
of the RVV kernels.

---

## What is measured, and what is simulated

| | |
|---|---|
| **measured on the K1** | every callback (which hart, entry/exit `rdtime`), every camera release, goal arrival and control output, per-core busy % |
| **simulated in Isaac Lab** | the flight consequence of that timing |

The drone never leaves the simulator. The simulator holds the last command until the next one
arrives, so the on-board cadence decides how often the vehicle is steered (`--sched_latency_ms`)
and the on-board chain decides how old its goal is (`--percep_latency_ms`). Captions say it this
way: **board-measured timing, simulated flight consequence.**

---

## 0. Prerequisites on the board

ROS 2 Jazzy, prebuilt, plus a native compiler:

```bash
ssh k1 'ls /opt/ros/jazzy_prebuilt/setup.bash && g++ --version | head -1'
# g++ (Bianbu 14.2.0-...) 14.2.0
```

`g++ 14.2` compiles the RVV intrinsics correctly. The headers are nested one level deeper than
usual (`include/rclcpp/rclcpp/rclcpp.hpp`), so every `include/*/` directory goes on the include
path — the build lines below do that. Nothing is pinned or given a real-time policy unless a
launcher does it explicitly; the program records the affinity mask and scheduling policy it ran
under in its manifest.

---

## 1. Stage the kernels on the board

Both arms run the *same* generated code, so the kernels are copied from the build XPU-RT uses:

```bash
cd /scratch/agustin/xpurt-dev-sync
B=ModelBlaster/build/k1_xpurt
ssh k1 'mkdir -p /root/ros_mb/{yolo,nav,ctrl,pool,yolo4}'
for pair in yolov8_nano_64x96:yolo fused_full:nav mlp_control:ctrl; do
  m=${pair%%:*}; d=${pair##*:}
  scp $B/$m/int8/rvv_x60/{model,kernels,weights,buffers}.c $B/$m/int8/rvv_x60/{model,kernels,weights}.h \
      ModelBlaster/kernels/rvv/mb_rvv_vxrm_compat.h k1:/root/ros_mb/$d/
done
ssh k1 'cd /root/ros_mb
  for d in yolo ctrl; do for f in model kernels weights buffers; do gcc -O2 -march=rv64gcv -c $d/$f.c -o $d/$f.o -I$d; done; done
  for f in model kernels weights buffers; do gcc -O2 -march=rv64gcv_zvfh -c nav/$f.c -o nav/$f.o -Inav; done'
```

`nav` needs `zvfh` because `fused_full` has an fp16 linear.

### The 4-hart YOLO

The pinned layouts run YOLO sharded four ways inside the perception node. That build comes from
the **same IR and the same curated RVV kernels**, generated with a shard factor, and is checked
against the golden output before it is used:

```bash
eval "$(bash scripts/setup_spacemit_toolchain.sh)"     # the curated-kernel verify needs CROSS
cd ModelBlaster; OUT=build/k1_ros_shard4/yolov8_nano_64x96/int8; IR=build/k1_xpurt/yolov8_nano_64x96/int8
mkdir -p $OUT/generated && cp $IR/{graph.json,weights.npz,io.npz} $OUT/
MB_SHARD_FACTOR=4 ../.venv/bin/python -m modelblaster.pipeline.generate_skeleton --ir $OUT/graph.json \
    --weights $OUT/weights.npz --io $OUT/io.npz --out-dir $OUT/generated --backend rvv_x60 --platform linux
MB_SHARD_FACTOR=4 CROSS="$CROSS" ../.venv/bin/python -m modelblaster.pipeline.generate_kernels --ir $OUT/graph.json \
    --out-dir $OUT/generated --target rvv_x60 --backend reference --quant int8 --global-curated-dir "$(pwd)/kernels"
../.venv/bin/python scripts/check_kernel_coverage.py $OUT/generated      # must print OK
```

Without `CROSS` in the environment the curated kernels fail their verify step and the generator
falls back to the scalar reference silently; the coverage check is what catches that. Step 1 of
`scripts/board_campaign.sh` stages these sources with the worker pool
(`ModelBlaster/runtime/modelblaster_pool/`), builds them with `-DMODELBLASTER_USE_POOL`, and
times the model alone on 1 and 4 harts against the golden output
(`results/codesign_feedback/ros_traced/yolo_standalone/`).

---

## 2. Build the traced nodes

`results/codesign_feedback/ros_control_jitter/ros_mb_chain_traced.cpp` is the program every ROS
arm runs; `cpu_sampler.c` is the per-core sampler; `rdtime_now.c` hands several processes one
trace origin.

```bash
scp results/codesign_feedback/ros_control_jitter/{ros_mb_chain_traced.cpp,cpu_sampler.c,rdtime_now.c} k1:/root/ros_mb/
ssh k1 'cd /root/ros_mb && gcc -O2 cpu_sampler.c -o cpu_sampler && gcc -O2 rdtime_now.c -o rdtime_now
  source /opt/ros/jazzy_prebuilt/setup.bash
  R=/opt/ros/jazzy_prebuilt; INC=""; for d in $R/include/*/; do INC="$INC -I$d"; done
  LIBS="-lrclcpp -lrcl -lrcutils -lrcpputils -lrmw -lrmw_implementation -lstd_msgs__rosidl_typesupport_cpp \
        -lrosidl_runtime_c -lrosidl_typesupport_cpp -llibstatistics_collector -ltracetools \
        -lstatistics_msgs__rosidl_typesupport_cpp -lpthread"
  g++ -O2 -std=c++17 -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced.o -I. $INC
  g++ -O2 -std=c++17 -DMB_WITH_POOL -c ros_mb_chain_traced.cpp -o ros_mb_chain_traced_pool.o -I. -Ipool $INC
  g++ -O2 ros_mb_chain_traced.o yolo/*.o nav/*.o ctrl/*.o -o ros_mb_chain_traced -L$R/lib -Wl,-rpath,$R/lib $LIBS
  g++ -O2 -pthread ros_mb_chain_traced_pool.o yolo4/{model,kernels,weights,buffers}.o nav/*.o ctrl/*.o \
      pool/modelblaster_pool.o -o ros_mb_chain_traced_pool -L$R/lib -Wl,-rpath,$R/lib $LIBS'
```

Things the build needs that are easy to miss: `libstatistics_collector`,
`statistics_msgs__rosidl_typesupport_cpp` and `rmw_implementation` are pulled in transitively
by `rclcpp` and must be named; the generated entry points take **three** arguments (`input,
output, pool`) and the node declares them that way; the three model headers cannot share a
translation unit (each defines unmangled aliases), so the node declares the entry points itself.

What the program does, and deliberately does not do:

* four `rclcpp::Node`s in one process — camera timer → perception (`yolov8_nano_64x96`) → nav
  (`fused_full`) → control (`mlp_control`) — on `std_msgs/String` topics, QoS depth 10;
* control in one of two shapes: `--ctrl-mode timer` (default) runs a free `--ctrl-hz` timer on
  the newest goal it holds; `--ctrl-mode chained` runs control inside the goal callback, once
  per perception result;
* `--executor single|multi`; **`spin()` with a stopper thread, never `spin_some()`** (which
  runs ready work on the calling thread and collapses a multi-threaded executor onto one core);
* rclcpp's default mutually-exclusive callback group per node (a generated model writes static
  buffers and must not overlap itself; different nodes still run in parallel under `multi`).
  With `--cameras 2` both cameras' perception callbacks share one such group, because there is
  one YOLO model instance in the process — two cameras that must run at once go in two
  processes (`x2p`);
* `--nodes <subset>` with a shared `--t0` runs the graph as several processes over DDS;
* `--yolo-pool 4 --pool-harts 0,1,2,3` runs YOLO on a worker pool (the pool binary);
  `--pin-main <cpu>` pins the executor thread; the launcher does any process-level `taskset`;
* it waits for every publisher/subscription to match before the camera starts, so DDS discovery
  is never counted as latency, and it drops nothing — the host applies the warm-up window.

Files written per run (`/root/ros_mb/out/<tag>/`): `trace.csv` in the XPU-RT trace contract
(one row per callback: network, instance, `worker_hart = sched_getcpu()`, entry/exit ticks),
`released.csv`, `goals.csv`, `consumed.csv`, `ctrl_gaps.csv`, `manifest.json` (args, executor
threads, `rmw`, affinity mask, scheduling policy, counts, kernel and IR hashes), and the
sampler's `cpu.csv`.

---

## 2b. Per-slice tracing: what runs inside a callback

A ROS 2 node records one row per callback, because a callback is where its abstraction ends: the
network is one library call. XPU-RT's runtime dispatches each operator itself and records all of
them, so the two runtimes' schedule rows were drawn at granularities that differ by a factor of
about ninety, and a pooled callback had to be credited to the harts its manifest declared rather
than drawn on the harts that ran it.

Two measurements already exist and are now read. The generated model keeps a per-op record for
every dispatch -- the IR dispatch id, node name, op kind, tensor shape and duration -- exposed by
`model_<net>_profile_records()`; the durations are in 24 MHz ticks, because the generated
`rdcycle` is defined as `rdtime` on this kernel. And the worker pool can record each slice it
runs: `modelblaster_pool_trace_arm()` takes a caller-supplied buffer and every slice appends the
hart from `sched_getcpu`, its rdtime span, the worker index and the invocation counter, through a
lock-free reserve-then-fill that costs two clock reads per slice. The master's own slice 0 is
recorded the same way, so the calling hart's share is not lost.

`ros_mb_chain_traced.cpp` drains both after each perception callback and writes them as extra rows
of the same trace contract, with a real `dispatch_id`, `op` and `worker_hart`. A pooled op
therefore appears once per hart that ran it, at the time it ran; a sequential op appears once on
the callback's hart, its boundaries apportioned from the measured per-op durations across the
callback's measured span.

The callback rows are unchanged, and the chain is still read from them
(`make_measured_gantt_pair.callback_rows`), so latency is unaffected: measured on the board over
the same run, 195 frames give a 60.082 ms median whether the chain is computed from callback rows
alone or from every row. `executed_units()` decides what to draw -- a slice is the measurement, so
where a slice covers a per-op row on the same hart the per-op row is dropped -- and a pool is
credited from the sampler only for traces that carry no slices of their own.

Both ROS arms drawn in the figures are measured this way: `45_vanilla4_r1`, YOLO on a 4-hart pool,
and `45_vanilla4x2_r1`, the arm given all eight cores. A 20 s run adds roughly 180k slice rows and
69k per-op rows, about 22 MB of trace. The chain median of a traced run matches the untraced run of
the same arm to within the replicate spread (242.47 and 242.43 ms; 454.56 and 455.60 ms). The
instrument is not free at the control output: on the arms whose executor thread also drains the
slice buffer, the control-gap mean rises by about 0.3 ms pooled over replicates (25.68 to 26.01 ms
for `vanilla4`, 16.64 to 16.89 ms for `vanilla4x2`, roughly 1.3%), while `p3`, whose control runs on
its own timer in its own process, is unchanged at 10.00 ms. Quote the chain from a traced run and
the control gap from the arm's untraced replicates, or re-derive both from the same set. With
slices present the ROS 2 row is drawn on the harts that ran the work, and no lane is credited from
the manifest.

Every node drains its own model's records: `emit_detail` takes the provider to call, so the nav and
control nodes are recorded at the same granularity as perception instead of one row per callback.
Over a 20 s run at 45 Hz the nav node writes 11,070 operator rows across its 738 callbacks (the
fused network's convolutions, its LSTM, its linear layers and the input cast) and the control node
5,166 (linear and ELU). The chain is still read at the callback boundary and does not move: the same
arm measures 242.43 ms before the two nodes were instrumented and 242.66 ms after, inside the spread
of its replicates. A node whose own operators are not recorded keeps its callback row, so its work
is drawn either way.

A process runs exactly the nodes named in `--nodes`, matched against the comma list as whole
names. This matters for the arms that split perception across two processes: a process launched
as `--nodes perception2` runs that node alone, subscribes to `frame2` alone, and leaves the first
camera stream to the process that owns it. The camera in those arms is launched with
`--alternate`, publishing even frames to `frame` and odd frames to `frame2`, so the two pools
divide the stream rather than share it. Check the split in any trace by counting node callbacks
against distinct frame ids: at 45 Hz over 20 s `45_vanilla4x2_r1` records 898 inferences over 898
distinct frames, 449 to each perception node.

## 3. Run the arms

`scripts/ros_traced_matrix.sh <arm> [replicate]` runs one arm across camera rates (`RATES`,
default 5…30 Hz) with the sampler around every run and pulls the files back to
`results/codesign_feedback/ros_traced/<hz>_<arm>_r<k>/`. Arms are real deployments, named by
what the launcher does to the process:

| arm | processes | executor | YOLO | control | placement |
|---|---|---|---|---|---|
| **`vanilla`** | 4 | single each | 1 hart (the generated `run_model` as-is) | **chained** (in the goal callback) | one node per stage, one process each, **nothing pinned**: ROS 2 as one writes it out of the box |
| **`vanilla4`** | 4 | single each | the model's 4-hart build (the pool pins its own workers) | chained | as `vanilla`; ROS itself still unpinned |
| **`vanilla4t`** | 1 | single | 4-hart build | timer | one process, default executor, control on its own 100 Hz timer, unpinned |
| **`vanilla4tm`** | 4 | single each | 4-hart build (pool on P#0-3) | timer | as `vanilla4` — camera, perception, nav, control as four unpinned processes (`affinity_mask 0xff`) — with control on its own 100 Hz timer in its own process, acting on the held goal (`board_vanilla4tm.sh`: 45 and 90 Hz × 3) |
| **`vanilla4x2`** | 5 | single each | two 4-hart builds: `perception` pool on P#0-3, `perception2` pool on E#4-7 | chained | pipelining by hand: the camera alternates frames between the two perception processes (`alternate: true`; both cameras' YOLO callbacks in one mutually-exclusive group), nav and control unpinned; every process `affinity_mask 0xff`, `SCHED_OTHER`. The "ROS 2 on all eight cores" arm of the composite's Gantt (`ros8`) and the flights' `ros_vanilla4x2{45,90}` traces (`ros_traced_matrix.sh vanilla4x2`, `RATES="45"` / `"90"`) |
| **`vanilla8`** | 4 | single each | one 8-hart build (`--yolo-pool 8 --pool-harts 0..7`) | chained | as `vanilla4`, with the single perception process given every hart for its pool (`ros_traced_matrix.sh vanilla8`, 45 Hz × 3) |
| **`vanilla8tm`** | 4 | single each | one 8-hart build | timer | `vanilla8` with control on its own 100 Hz timer, as `vanilla4tm` (`ros_traced_matrix.sh vanilla8tm`, 45 Hz × 3) |
| **`rvanilla`, `rvanilla4`** | 6 | single each | 1 hart / 4-hart build | chained | `vanilla` / `vanilla4` + `ffn_block` and `dronet` as two more unpinned processes |
| `ship` | 1 | single | 1 hart | timer | unpinned — ROS 2 as it ships, all nodes in one process |
| `multi` | 1 | multi | 1 hart | timer | unpinned |
| `spin` | 1 | single | 4-hart pool | timer | `taskset 0-3`, executor on P#0 |
| `smte` | 1 | multi | 4-hart pool | timer | `taskset 0-3` |
| `p3` | 3 | single each | 4-hart pool | timer | perception P#0-3, nav E#0, control E#1 |
| `p8` | 3 | single each | 4-hart pool | timer | nav E#0-1, control E#2-3 |
| `part8` | 3 | single each | 4-hart pool + a 2-hart pool for nav and one for control | timer | the machine partitioned between the three networks: YOLO on P#0-3, nav on E#0-1 with `--nav-pool 2`, control on E#2-3 with `--ctrl-pool 2` |
| `yproc` | 2 | single | 4-hart pool | timer | perception alone; nav + control on E#0 |
| `nproc` | 2 | single | 4-hart pool | timer | nav alone; perception + control on P#0-3 |
| `cship`, `cspin`, `cp3` | as above | | | **chained** | control runs per goal, no timer |
| `rspin`, `rp3`, `rmulti` | as `spin` / `p3` / `multi` | | + `ffn_block` 10 Hz + `dronet` 30 Hz | timer | the heavier stack (`--extra`); `rp3` puts the two extra nodes in a fourth process on E#2–3 |
| `x2spin`, `x2p` | 1 / 4 | single | 4-hart pool (`x2p`: one pool per camera, P#0-3 and E#0-3) | timer | two 45 Hz cameras (`--cameras 2`); `x2p` is one process per camera plus nav and control |
| `x2rspin`, `x2rmulti`, `x2rp3` | 1 / 1 / 5 | single / multi / single each | as `x2spin` / `x2p` | timer | two cameras **and** the heavier stack; `x2rp3` adds the extras as a fifth, unpinned process |

Run tags carry the sensitivity knobs as suffixes, `<hz>_<arm><suffix>_r<k>`: `_q1` = `QOS=1` (keep-last-1 instead of
the default depth 10), `_c200` = `CTRL_HZ=200` (a 200 Hz control timer), `_hog2` = `HOGS=2` (two unpinned CPU hogs
started before the run), `_smoke` = the 8 s smoke run of the vanilla graph that checked the traced binary before the
matrix (`SECS=8`; not used by any figure). Replicates are `_r1`–`_r3`, one `ros_traced_matrix.sh` invocation each.

The `vanilla*` arms are the baseline the flight figure uses: a graph written the way the
tutorials write it, launched as processes, left to the OS. Measured at a 45 Hz camera, three
replicates each (`ros_traced/summary.csv`, constants `ROS_VANILLA` in
`scripts/measured_timing.py`): serial YOLO → control every 48 ms (20 Hz), camera→goal 265 ms,
57 % of frames dropped by the default depth-10 queue; the 4-hart build → 25.7 ms (39 Hz)
chained or 30.4 ms (33 Hz) timer-driven, 120–240 ms camera→goal, five or six cores busy; with
the heavier stack → 84 ms (12 Hz). Each plateaus at 12–39 goals a second whatever the camera
rate (15–90 Hz swept). The tuned arms (pool + pinning, multi-threaded executor, hand
partitioning) are the appendix: what it takes to recover the control cadence by hand, and what
it still costs in latency. `scripts/plot_deployment_layers.py` draws the vanilla deployments
against the schedule's two solvers. `scripts/board_campaign.sh` and
`scripts/board_campaign2.sh` run the whole matrix with replicates. **Never compile on the board
while an arm is running** — a native build shows up in the sweep.

---

## 4. Read the results

```bash
scripts/pull_ros_traced.py             # per-run chain.csv + summary.json, ros_traced/summary.csv
scripts/measured_timing.py --verify    # every quoted constant re-derived from the raw files
```

`pull_ros_traced.py` builds each run's camera→goal and camera→control latencies from the raw
stamps (blank cells where the depth-10 queue dropped a frame, so saturation is visible), applies
a 3 s warm-up, and takes the per-core busy % over the run window. The sustained camera rate per
arm is the highest rate whose camera→goal median is under 1.5× the period.

The camera→goal median describes the settled chain. At 45 Hz the camera period (22.2 ms) is shorter
than one YOLO callback, so the depth-10 queue fills and the latency rises from its unqueued value to a
fixed point. The time to reach that fixed point varies between replicates of the same arm — across
`45_p3_r{1,2,3}` it ranges from 0.3 s to 11.9 s into a 20 s run — so a whole-run median taken while the
queue is still filling lies between the two regimes. Compare replicates on the settled window: the
median over the last quarter of each run's frames agrees to ~0.5 ms across replicates where the
whole-run medians differ by more.

## 4b. The spread across deployments

The same three nodes, the same kernels and the same board give very different chains depending
only on how the deployment is arranged. Measured at a 45 Hz camera, over the distinct arms in
`ros_traced/` (best replicate of each), 14 of 38 arrangements bring camera-to-goal under the
66.7 ms frame window and 24 do not, from 31.1 ms (`p3_q1`, which drops queued frames at QoS depth
1) to 623.4 ms (`x2rspin`). Arrangements that differ only in where a node is pinned land on both
sides of the window: `p3` reaches 56.4 ms and `spin` 118.9 ms. Rebuild the table with

```bash
.venv/bin/python scripts/measured_timing.py --verify        # every arm re-derived from its trace
awk -F, 'NR==1 || $2==45' results/codesign_feedback/ros_traced/summary.csv
```

Which arrangement falls inside the window is not evident from the node graph; each was measured.

Two of those arrangements change the workload rather than the placement. `vanilla4x2` and `x2p`
run a second perception node, and a generated model keeps static buffers, so the second node is a
second model instance in its own process: two copies of the 3.1 MB weights object, each taking
alternate frames at 22.5 fps from the same 45 Hz camera. That buys latency — `vanilla4x2`
measures 37.0 ms camera-to-goal against `p3`'s 56.4 ms — at twice the model memory, and it is not
a comparison against a single-instance schedule. The arms that deploy the same one-instance
workload are `vanilla`, `vanilla4`, `vanilla4t`, `vanilla4tm`, `spin`, `multi`, `smte` and `p3`.

A third arrangement gives each network its own harts rather than giving perception more of them.
`part8` keeps YOLO on a four-hart pool and builds a two-hart pool for nav and another for control,
one process per network on its own pair, so all eight harts are spoken for. Six runs at 45 Hz
measure 56.16-56.26 ms camera-to-goal (one run of the six read 30.78) against `p3`'s 56.34-56.41
over six runs in the same sessions, and nav's own execution is 4.17 ms against 4.14: unchanged.
The two small networks have no parallel work to give the harts. Their generated sources carry the
pool entry point but, for the ops these two models use, no pool path behind it -- `nav/model.c`
says so in a comment and `nm` finds no pool symbol in the objects even when they are compiled with
`MODELBLASTER_USE_POOL`; only `ctrl` gains one, and control is 0.06 ms. Perception is where the
parallel work is, which is why the four-hart pool is the arm's whole story.

The other way to reach eight harts keeps one model instance and widens its pool, which is a runtime
argument to the generated pool (`modelblaster_pool_create_on_harts`) rather than a build: `vanilla8`
and `vanilla8tm` ask for eight. Measured at 45 Hz, three runs each, the width buys nothing. Chained
control reads 242.5 ms camera-to-goal on the eight-wide pool against `vanilla4`'s 242.4 ms; on the
timer arm 241.9/246.9 ms against `vanilla4tm`'s 242.5/247.5 ms; the control gap and goal rate are
unchanged, and the runs deliver the same number of goals as their four-wide siblings (643-648
chained, 656-658 on the timer, against 646-648 and 648-656). The per-core sampler says why: on both
eight-wide arms three harts carry the pool at 85-88 % and one further hart is busy, exactly the
shape the four-wide arms show, while the remaining harts stay under 20 %. It is the kernel, not the
deployment: alone on the board YOLOv8n int8 takes 24.58 ms on four harts and 25.03 ms on eight
(§1.8g of `measurements_and_ablations.md`). A second instance, not a wider pool, is what puts the
K1's eight harts to work on this network.


## 5. Scope of each claim

| claim | holds for |
|---|---|
| control starved to ~33 Hz (15 Hz camera) / ~17 Hz (≥20 Hz camera) | one process, default executor, serial YOLO — as shipped |
| control starved to ~33 Hz at the 45 Hz design rate | one process, default executor, YOLO on 4 harts |
| control holds 100 Hz | control on its own thread or process, or the multi-threaded executor |
| control rate = camera rate, ≤ 33 Hz | control chained to the goal topic, any layout |
| chain 53–54 ms / 30 ms | serial YOLO / 4-hart YOLO, camera at or below the sustained rate |
| every layout plateaus: 18 fps as shipped, 33 fps pooled on one thread, 39 fps with control in its own process | camera 45–90 Hz, `ros_traced/{60,75,90}_*` |
| heavier stack: pooled single-thread control falls to 53 ms; multi-threaded executor drops 32 % of frames | `ros_traced/{25,45}_r{spin,p3,multi}_*` |

The as-shipped and chained rows are common deployments and the ones the flight figure uses; the
tuned rows are measured and reported as the limits of the claim.

## 6. What this does not establish

* One board, one unit. Nothing here separates silicon-to-silicon variation.
* The nodes run the deployed three-network pipeline. A different graph shape distributes
  differently across an executor.
* The flight outcome is simulated. The timing is measured; the crash is not.
* A replayed flight carries whichever parts of a deployment's timing were injected into it. The
  control cadence comes from `--ctrl_trace`, the camera-to-control latency from
  `--percep_latency_ms`, and they are separate switches: a pair flown with the trace alone differs
  only in control rate. Read the pair's own record rather than assuming, and note that dumps
  written before the recorder stored `percep_latency_ms` do not say. In the same scene and seed the
  two settings do not give the same flight: at seed 1005 the solved arm completes the course on a
  cadence-only replay and crashes at the first gate once its measured 56.8 ms is injected.
* A hand-pinned ROS 2 reaches a comparable operating point. Compared on one quantity —
  camera-to-control, what the schedule panel draws — `p3` measures 61.6 ms against the solved
  arm's 56.8 ms, both with control every 10.00 ms; `p3`'s navigation result is ready at 56.4 ms
  and waits for its free-running control timer. Over 348 flights
  paired on the same seed, cruise, density and course the two are level (55 completions each,
  41 wins each way, difference +0.0 pts, 95% CI [-5.4, +5.4]). What the board separates is not
  the point they reach but how: `p3` needs an explicit affinity mask per process, leaves three
  harts near idle and keeps nav and control on one hart each, while the solved schedule places
  every network across seven or eight. Of 38 arrangements measured, 14 land inside the frame
  window (§4b).
* Arms are compared at a fixed gain and, separately, at each arm's calibrated gain
  (`campaign_tallcal`, 0.5 / its control rate). The calibrated set injects no perception latency,
  so it isolates cadence: there the 39 Hz baseline completes 0 of 60 and the 96 Hz arm 7 of 60.

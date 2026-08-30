# Reproducing the mlp_control + dronet + yolov8n XPU-RT schedule on QRB5165 (QNN)

Board reproduction of the same three-network schedule the spike flow runs
(`docs/mlp_dronet_yolo_spike_reproduction.md`), targeting a physical
QRB5165 through Qualcomm's QNN runtime instead of spike/RISC-V: ingest the
PyTorch models with modelblaster's `extract_graph`, emit the XPU-RT
contract artifacts with modelblaster's own emitters, solve, ingest the
schedule back through modelblaster's `ingest_xpurt_schedule`, generate a
QNN lane runtime, run it on the board, and render predicted-vs-actual
Gantt charts with modelblaster's `plot_xpurt_trace.py`.

**Expected result:** `25/25 entries executed, wall=32.56 ms (predicted
makespan 32.56 ms, ratio 1.00x), tensor handoffs=3` for the MILP schedule
— dronet ×7 on HTA at a 5 ms period, mlp_control ×16 on CPU at 2 ms,
yolov8n's two tiles back-to-back on DSP. The greedy schedule reproduces at
`47/47 ... wall=77.53 ms (1.00x)`; both match their own prediction, and the
2.4× difference between them is the scheduler, not the runtime.

This flow lives in `qnn_models/flow_c/`. It reuses modelblaster as a
**library for the front end and the scheduling path** — it does not use
modelblaster's compiler. See "How this differs from the standard
modelblaster flow" below before assuming anything is shared.

## Reproduce: exact steps

```bash
export XPURT_ROOT=/path/to/your/XPU-RT
cd "${XPURT_ROOT}"

git submodule update --init zephyr-chipyard-sw
git -C zephyr-chipyard-sw submodule update --init modelblaster

# One-time: an interpreter with torch for the extract stage. modelblaster's
# own conda env is found automatically; override with FLOWC_MB_PYTHON.
cd zephyr-chipyard-sw && source scripts/install_conda.sh && cd ..

# One-time: stage the context binaries the tiles name (see Prerequisites).
cd qnn_models/flow_c
python3 flow_c.py stage --ctx-source /root/repro_perlane

# The flow.
python3 flow_c.py all
```

`all` chains `ir → artifacts → schedule → runtime → stage → run → plots`;
every stage is independently re-enterable and writes to disk, so run them
one at a time while you are learning the flow.

## Prerequisites

### Board

Reachable over SSH without a password prompt, with the QAIRT 2.45 runtime
installed at `/root/qairt`:

```bash
ssh root@10.44.120.201 'ls /root/qairt/bin/target/qnn-context-binary-generator \
                           /root/qairt/lib/target/libQnn{Cpu,Dsp,Hta}.so \
                           /root/qairt/lib/hexagon-v66/libQnnDspV66Skel.so; g++ --version | head -1'
```

The runtime is compiled **on the board** — stock `g++ 9.4` (Ubuntu 20.04)
is enough; there is no cross-compiler and no CMake in this flow. Override
the host with `--board` / `$QNN_BOARD_HOST`.

### Context binaries

Every tile in `bindings/*.json` names a context binary. Those are built
once per (DLC, backend) with the SDK tool on the board:

```bash
ssh root@10.44.120.201 'export LD_LIBRARY_PATH=/root/qairt/lib/target \
  ADSP_LIBRARY_PATH="/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp"
  /root/qairt/bin/target/qnn-context-binary-generator \
    --backend /root/qairt/lib/target/libQnnHta.so \
    --model   /root/qairt/lib/target/libQnnModelDlc.so \
    --dlc_path /root/qnn_runtime_ctx/dronet_full_hta_quantized.dlc \
    --binary_file ctx_dronet_full_hta__Hta --output_dir /root/repro_perlane'
```

The DLCs themselves come from `qnn_models/deploy.sh` (ONNX →
`snpe-onnx-to-dlc` → `qairt-quantizer`, inside the `qnn-convert` Docker
image); that step is upstream of this flow and is not re-run here.
`flow_c.py stage` links whatever `--ctx-source` holds into the board's
`ctx_dir`.

### Host

- **torch**, for `extract_graph`. Resolution order: `$FLOWC_MB_PYTHON`, this
  interpreter if it already has torch, then a conda env named by
  `$FLOWC_MB_ENV` (default `zephyr`, the env modelblaster's own
  `install_conda.sh` creates).
- **torch + onnx**, for `flow_c.py ir --export-onnx` and the artifact
  cross-check. Auto-detected across conda envs; override with
  `$FLOWC_ONNX_PYTHON`. modelblaster's env has torch but not onnx, so this
  is usually a different interpreter — that is expected, not a
  misconfiguration.
- **matplotlib**, for the Gantt charts.
- **cvxpy + mosek**, only for `--solver milp`. Not required for the flow;
  `greedy_periodic` is the default. To add them without touching an
  existing env:
  ```bash
  python3 -m venv /tmp/milpenv && /tmp/milpenv/bin/pip install cvxpy mosek matplotlib
  export MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic
  ```

## 0. Environment

```bash
export XPURT_ROOT=/path/to/your/XPU-RT
cd "${XPURT_ROOT}/qnn_models/flow_c"
```

No conda activation is needed for the flow itself — stock `python3` runs
every stage and shells out to the torch interpreter only for `ir`.

## 1. `ir` — ingest the models

```bash
python3 flow_c.py ir              # add --export-onnx to also write gen/onnx/*.onnx
```

Runs modelblaster's `extract_graph` on `models/<id>.py` for the two
PyTorch-sourced networks, adapts yolov8n's IR from the partition JSON, then
prints each tile's op range, its provenance, where it can run, and the
artifact cross-check:

```
dronet: 24 IR ops from pytorch:dronet, 1 binding(s)
  dronet_full            ir ops   0..23  ( 24)  runs on: cpu, dsp, hta
      range from: whole network (ops='all')
mlp_control: 7 IR ops from pytorch:mlp_control, 1 binding(s)
  mlp_control_full       ir ops   0..6   (  7)  runs on: cpu, dsp
      hta  blocked by elu_s8
yolov8n: 249 IR ops from onnx:yolov8n, 2 binding(s)
  yolov8n_backbone       ir ops   0..102 (103)  runs on: cpu, dsp, hta
      range from: yolov8n_HTA_split.json group 0 (hardware_target=HTA)
  yolov8n_head           ir ops 103..248 (146)  runs on: cpu, dsp
      hta  blocked by reshape, resize, slice, softmax, transpose
  artifact check: yolov8n_backbone: handoff tensor model_4_cv2_act_Mul_output_0 produced — OK
  artifact check: yolov8n_head: declares 146 IR ops ... but yolov8n_head.onnx holds 138 nodes (delta -8)
```

Both `blocked by` lines are the board's real compose failures, derived from
the registry rather than measured. The `-8` is the expected route mismatch
between the QNN-route IR and the ONNX-route artifact — see "Known
limitations".

## 2. `artifacts` — the XPU-RT contract

```bash
python3 flow_c.py artifacts
```

Writes, per network and per machine kind:

```
gen/qnn_vmfb/<net>/qrb5165_flowc/<HW>/<net>.int8/<net>.int8_dispatch_graph.json
gen/profile/<HW>/qrb5165_flowc/<net>/<net>.int8/.../topo_0/results.csv
data/toplevel/networks_flowc_3way_qrb5165.json
```

and reports the two capability-excluded cells plus the horizon it pinned
instance counts from:

```
  excluded: mlp_control/mlp_control_full on hta: ... cost set to 54227 us
  excluded: yolov8n/yolov8n_head on hta: ... cost set to 3946462 us
measured horizon (non-periodic worst case): 73.81 ms -> dronetx15, mlp_controlx37
```

The profile CSVs carry provenance (`source=qnn-mean@<date>`) from
`measurements/qrb5165_v66.json`. **The target label is `qrb5165_flowc`**,
not `qrb5165_v66`, so these binding-level artifacts never collide with the
per-op ones the older QNN flow wrote.

## 3. `schedule` — solve

```bash
python3 flow_c.py schedule                       # greedy_periodic (default)
```

For the MILP schedule, run the solver directly with the cvxpy env:

```bash
cd "${XPURT_ROOT}"
MOSEKLM_LICENSE_FILE=$HOME/mosek/mosek.lic /tmp/milpenv/bin/python \
  scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_flowc_3way_qrb5165.json \
    --solver milp --profiled --time-limit 300
# -> Status: optimal   Makespan (non-periodic): 31.45 ms
# -> schedules/scheduled_networks_flowc_3way_qrb5165_profiled.json
```

Both solvers also write a predicted-only Gantt to `plots/`.

## 4. `runtime` — dispatch table + QNN runtime sources

```bash
cd "${XPURT_ROOT}/qnn_models/flow_c"
python3 flow_c.py runtime --tag milp \
  --schedule "${XPURT_ROOT}/schedules/scheduled_networks_flowc_3way_qrb5165_profiled.json"
```

```
25 entries  predicted makespan 32.563 ms
  per network: {'dronet': 7, 'mlp_control': 16, 'yolov8n': 2}
  per lane:    {'hta(HTA)': 7, 'cpu(CPU)': 16, 'dsp(DSP)': 2}
  contexts:    4
emitted gen/runtime/flowc_3way_qrb5165_milp/dispatch_table.h + runtime_main.cpp
  lane mode kind: hta, cpu, dsp
```

`--tag` keeps two solvers' outputs side by side (separate runtime dir,
board dir, log dir and plot names). `--lane-mode kind-network` gives one
worker per (kind, network) instead of per kind.

## 5. `stage` + `run` — build and execute on the board

```bash
python3 flow_c.py stage --ctx-source /root/repro_perlane
python3 flow_c.py run --tag milp
```

`run` shells out to `qnn_models/runtime/deploy_and_run.sh`, which scp's the
two generated sources, compiles them on the board with a single `g++`
invocation, runs the binary, and saves the log:

```
[bringup] HTA  /root/qnn_runtime_ctx/ctx_dronet_full_hta__Hta.bin → 1 graph(s): dronet_full_hta
[main] 4 context(s), 25 entries, 3 lane(s):
  lane 0  hta   core 7  spin 200 us  fifo 0
  lane 1  cpu   core 5  spin 2000 us  fifo 40
  lane 2  dsp   core 6  spin 200 us  fifo 0
[summary] 25/25 entries executed, wall=32.563 ms (predicted makespan 32.563 ms,
          ratio 1.00x), tensor handoffs=3
```

Runtime knobs (all optional): `FLOWC_ITERATIONS`, `FLOWC_SPIN_US`,
`FLOWC_FIFO`, `FLOWC_AFFINITY`, `FLOWC_WARMUP`, `FLOWC_TRACE`,
`FLOWC_DUMP_TENSORS`. The defaults come from the core registry.

## 6. `plots` — predicted vs actual

```bash
python3 flow_c.py plots --tag milp
```

```
entries: 25
  predicted makespan: 32.563 ms
  actual    makespan: 32.563 ms (1.00x predicted)
predicted vs actual : plots/flowc_3way_qrb5165_milp_predicted_vs_actual.png
  zoom (first 20 ms): plots/flowc_3way_qrb5165_milp_predicted_vs_actual_zoom.png
  flat trace CSV     : runs/flowc_3way_qrb5165_milp/trace.csv
  scheduler timeline : plots/networks_flowc_3way_qrb5165_profiled.png
```

The renderer is modelblaster's `scripts/plot_xpurt_trace.py`, unmodified.
The Flow C runtime emits its trace in that script's own column schema with
microsecond ticks, so `--clock-mhz 1` is the entire adaptation. Red borders
mark entries that ran past their predicted finish — expect all
`mlp_control` bars to carry one: predicted 0.028 ms, actual ~0.06 ms, still
comfortably inside its 2 ms period.

## Worked example: adding a network (FusedSensorNet)

`fused_full` — modelblaster's FusedSensorNet (vision CNN + ToF depth conv +
3-layer LSTM + head, three inputs) — went through this flow end to end,
including the conversion stages the three original networks skipped because
their DLCs already existed. The sequence, all from `qnn_models/flow_c`:

```bash
# 1. ONNX from the same module modelblaster's codegen consumes
python3 flow_c.py ir --export-onnx        # gen/onnx/fused_full.onnx (3 inputs, opset 17)

# 2. ONNX -> DLC (multi-input: rank-4 inputs NCHW, the 1x21 state NONTRIVIAL)
sudo docker run --rm -v $QNN_SDK:/qnn:ro -v $PWD/gen/convert:/workspace qnn-convert \
  python3.10 /qnn/bin/x86_64-linux-clang/snpe-onnx-to-dlc \
    --input_network /workspace/fused_full.onnx \
    -d front_grey 1,1,60,90 --input_layout front_grey NCHW \
    -d tof_cross  1,4,8,8   --input_layout tof_cross  NCHW \
    -d lowdim     1,21      --input_layout lowdim     NONTRIVIAL \
    --output_path /workspace/fused_full.dlc

# 3. int8, calibrated from the model's own get_calibration_samples()
#    (10 samples x 3 inputs -> cal/*.raw + calibration_list.txt)
sudo docker run --rm ... qairt-quantizer --input_dlc /workspace/fused_full.dlc \
    --output_dlc /workspace/fused_full_quantized.dlc \
    --input_list /workspace/cal/calibration_list.txt \
    --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 8

# 4. context binaries on the board, one per backend that composes
# 5. measure with profile_segments.cpp, write measurements/ + bindings/
# 6. add to a workload and run the flow as usual
```

What the board said, which is the interesting part:

| Backend | Compose | Note |
|---|---|---|
| DSP (int8) | ✅ 2.58 MB ctx | 4.74 ms — the LSTMs run on HVX |
| HTA (int8) | ❌ | `QnnHta: unsupported op Transpose` |
| CPU (int8) | ❌ | CPU op package rejects the quantized `Reshape` op config |
| CPU (fp32) | ✅ 6.24 MB ctx | 1.15 ms under the lane's exec mask |

So precision became a per-(tile, backend) property in the manifest: the DSP
entry names the int8 context, the CPU entry names the fp32 one. It is also
the case that broke the assumption that capability is a property of an op
*kind* — the same int8 `Reshape` composes fine inside the yolov8n head on
CPU. The registry check stays useful as a first filter; the measurements
file is what records what actually composed.

With `fused_full` at a 20 ms period added to the 3-way workload, MOSEK
solves the 58-op model to optimal and puts it on the CPU lane beside
mlp_control. Measured over 3 reps: makespan 0.99-1.11x predicted, no
deadline misses, `dronet` 2.50 ms vs a 2.563 ms cell and `yolov8n` 15.55 vs
14.76 — but `fused_full` itself runs ~5.6x its isolated cell (6.39 ms p50
against 1.147 ms). See "Known limitations".

### The threading lesson this produced

Two networks sharing the CPU lane is what forced `--lane-mode kind-network`
into use, and running it exposed two real bugs:

1. **A real-time lane must not hold its priority across `graphExecute`.**
   A pthread inherits scheduling policy *and* affinity from its creator, so
   a `SCHED_FIFO`, core-pinned lane hands the QNN CPU backend a pool of
   real-time threads pinned to one core. With two such lanes on the same
   core the board stopped scheduling anything else and needed a power cycle.
   The runtime now raises priority and pins **for the gate only**, then
   drops to `SCHED_OTHER` and restores the lane's execute mask before
   dispatching.
2. **Spin windows must be bounded by the lane's own cadence.** A 2000 µs
   spin on a 2 ms period is a lane that spins continuously. The emitter now
   caps each lane's spin at a quarter of its minimum inter-entry gap
   (mlp_control: 2000 → 274 µs), and a shared-core guard demotes any second
   real-time lane that lands on an already-claimed core.

`deploy_and_run.sh` also gained `RUN_TIMEOUT` (default 120 s, `timeout -s
KILL` on the board side) so a wedged runtime can no longer take the board
out of ssh reach.

## Running in the state the cost model was measured in

The flow's default run leaves the board on `schedutil` and walks the
schedule once, which costs about 15% of makespan accuracy. `--tuned`
reproduces the warm-up configuration the earlier hand-written runtime used
(`XPURT_SCHEDULE_ITERATIONS` + a `performance` governor):

```bash
python3 flow_c.py run --workload dronet_mlp_yolo_fused_split.json --tag tuned --tuned
```

It pins all eight cores to `performance`, runs with `FLOWC_ITERATIONS=2` so
the reported trace is the second (warm) walk, and restores `schedutil`
afterwards. Measured on the 4-network split workload, three runs each:

| Configuration | wall (median) | ratio to prediction |
|---|---|---|
| baseline: schedutil, one walk | 37.42 ms | 1.16× |
| warm-up only (2 walks) | 34.31 ms | 1.06× |
| performance governor only | 34.52 ms | 1.07× |
| **both (`--tuned`)** | **33.23 ms** | **1.03×** |

Then re-measure the cells under the same governor and re-solve — the cells
are what actually diverge. Under `schedutil` the board idles at 710 MHz of
2419, and because the host clock gates FastRPC dispatch, *accelerator* cells
come out pessimistic by 10-36% (yolov8n's backbone on HTA measures 21.6 ms
under schedutil and 13.9 ms under performance). With cells and run in the
same state the schedule tracked 1.00× on three consecutive runs with a
0.08 ms spread, and per-tile ratios tightened from a 0.70-18.5× spread
(median 1.14×) to 0.64-10.4× (median 0.96×), with every accelerator tile
inside 0.95-1.12×.

The one tile still far off is the fp32 LSTM tail on the CPU lane: 0.35 ms
alone, 3.69 ms in-situ. That is the contention limitation in "Known
limitations", not a measurement-state problem.

## How this differs from the standard modelblaster flow

Flow A (spike/FireSim) and this flow share the front end and the scheduling
path, and diverge completely at codegen:

| Stage | Flow A — modelblaster | Flow C — this doc |
|---|---|---|
| PyTorch → IR | `extract_graph` | **same, reused verbatim** |
| Quantization | `extract_graph --quant int8` (PTQ), weights baked into C | `qairt-quantizer` on the DLC; `extract_graph`'s `weights.npz` is **discarded** |
| IR → compute | `generate_skeleton` + `generate_kernels` (curated + LLM kernels) | **not used.** The compute artifact is a QAIRT DLC built upstream by `snpe-onnx-to-dlc`; `bindings/*.json` maps IR op ranges onto its graphs |
| Dispatch unit | one IR op = one kernel call | one **binding** = one `QnnGraph_execute` |
| Core registry | `cores/*.json` validates per-op `hardware_target` | **same loader**, plus a `qnn` block per core (backend `.so`, gate spin, SCHED_FIFO) and capability-derived profile exclusions |
| Dispatch graph | `emit_dispatch_graph.emit()` | **same function**, fed a coarse IR (one synthetic op per binding) |
| Profile CSV | `profile_writer.write_profile()` from spike/FireSim cycles | **same function**, from on-board µs (`clock_mhz=1.0`) |
| Scheduler | `scripts/run_xpurt_schedule.py` | **same** |
| Schedule → table | `ingest_xpurt_schedule.load()` | **same**, plus a 6-line N-slot patch for the third lane (`CPU_X`) |
| Runtime codegen | `generate_xpurt_main.py` → Zephyr C | `flowc/emit_runtime.py` → Linux C++ |
| Runtime threading | `modelblaster_pool`: parallel-for over harts, *inside* one op | one pinned lane thread per machine kind, blocking in `QnnGraph_execute`; no intra-dispatch pool |
| Build | `west build` (Zephyr + CMake) | one `g++ -std=c++2a -O2 -pthread ... -ldl` on the board |
| Target | spike / FireSim, RISC-V harts | QRB5165: HTA, Hexagon v66 DSP, Kryo CPU |
| Trace + plot | `MODELBLASTER_XPURT_TRACE` + `plot_xpurt_trace.py` | **same schema and plotter**, µs ticks |

**No C or C++ is shared.** The emitted runtime includes only libstdc++,
POSIX, the QNN SDK headers and its own `dispatch_table.h`; the board
compile line adds only `-I$QNN_SDK_ROOT/include`. The longest run of
identical significant lines between the emitted runtime and any of
modelblaster's six C/C++ sources is 2 (an include and a brace). About 27%
of the emitted file matches `qnn_models/runtime/gen/.../runtime_main.cpp` —
this repo's own hand-written QNN runtime — which is where the tensor
helpers and context bringup come from. The one C-level relationship to
modelblaster is a schema: `flowc_sched_entry_t`'s first block is
field-for-field `xpurt_sched_entry_t`, re-declared rather than included, so
a reader of one table can read the other.

### Additions with no Flow A counterpart

- **`bindings/*.json` — the tile map.** Flow A does not need one: its
  dispatch unit is the op. Here a tile is a contiguous IR op range compiled
  into one QNN graph, with a context binary per backend. Tiling is
  therefore a compile-time decision, and re-tiling costs a conversion.
- **`flowc/lowering.py` — declared converter transforms.** A binding states
  which rewrites its compiled artifact carries (BN folded into conv, an FC
  head re-expressed as 1×1 conv, trailing views dropped), and the
  capability check runs on the result. dronet is why: against the raw
  PyTorch IR it is infeasible on every backend, and it only becomes an HTA
  graph because the ONNX was rewritten offline.
- **Capability-derived profile cells.** Instead of the `100_000 µs`
  sentinels the older coarse cost model carried, cells the registry
  excludes are generated, labelled `qnn-excluded`, and reported with the
  blocking ops named.
- **Measured-horizon instance pinning.** xpu-rt sizes periodic instances as
  `ceil(horizon / period)` from the sum of each non-periodic op's *slowest*
  machine. Fed the exclusion costs, that took the model from 47 to 5,576
  operations. Flow C computes the horizon over measured cells only and pins
  the counts through the spec's `num_instances` override.
- **Per-lane runtime policy in the registry.** `gate_spin_us` and
  `sched_fifo_prio` per core kind. Measured over 4 reps × 32 `mlp_control`
  instances: SCHED_OTHER gave exec max 2.22 ms and 3 deadline misses;
  `SCHED_FIFO 40` on the CPU lane alone gave 0.106 ms and zero, with no
  makespan change. Do not raise it on the dsp/hta lanes — they block in
  FastRPC and would starve the callbacks they wait on.

## Map to key files

### The flow

| Path | Role |
|---|---|
| `qnn_models/flow_c/flow_c.py` | CLI driver: `ir artifacts schedule runtime stage run plots all` |
| `qnn_models/flow_c/flowc/mb.py` | modelblaster import bridge (`src/` on `sys.path`), interpreter resolution, the N-slot patch |
| `qnn_models/flow_c/flowc/ir.py` | `extract_graph` subprocess, `torch.onnx.export` of the same module, graph.json adapter |
| `qnn_models/flow_c/flowc/bindings.py` | tile map, partition-derived ranges, registry capability check, artifact cross-check |
| `qnn_models/flow_c/flowc/lowering.py` | declared converter transforms the capability check runs on |
| `qnn_models/flow_c/flowc/artifacts.py` | dispatch graphs, profile CSVs, workload spec, measured horizon |
| `qnn_models/flow_c/flowc/schedule.py` | modelblaster ingest + binding join → `FlowCEntry` |
| `qnn_models/flow_c/flowc/emit_runtime.py` | `dispatch_table.h` + `runtime_main.cpp` (lanes, gates, trace) |
| `qnn_models/flow_c/flowc/plots.py` | predicted-vs-actual Gantts via modelblaster's plotter |

### Configuration and data

| Path | Role |
|---|---|
| `qnn_models/flow_c/cores/qrb5165_qnn.json` | core registry: kinds, harts (lane affinity), capabilities, per-lane `qnn` policy |
| `qnn_models/flow_c/bindings/{dronet,mlp_control,yolov8n}.json` | tile map per network: op ranges, per-backend context + graph, declared lowerings |
| `qnn_models/flow_c/measurements/qrb5165_v66.json` | measured per-(tile, backend) µs with provenance and the compose failures |
| `qnn_models/flow_c/workloads/dronet_mlp_yolo.json` | which networks, periods, machine slots, ctx dir |

### Reused from modelblaster (Python only)

| Path | Used for |
|---|---|
| `pipeline/extract_graph.py` | PyTorch → `graph.json` IR |
| `pipeline/core_registry.py` | loads `cores/qrb5165_qnn.json` |
| `pipeline/emit_dispatch_graph.py` | `<net>.int8_dispatch_graph.json` |
| `pipeline/profile_writer.py` | `results.csv` in the IREE schema |
| `pipeline/ingest_xpurt_schedule.py` | schedule → ordered entries, deps, `time_dependency` |
| `scripts/plot_xpurt_trace.py` | the predicted-vs-actual Gantt |
| `models/{dronet,mlp_control}.py` | the PyTorch definitions + trained checkpoints |

### Reused from `qnn_models/` (this repo, not modelblaster)

| Path | Used for |
|---|---|
| `qnn_models/runtime/deploy_and_run.sh` | scp + on-board `g++` + run + capture |
| `qnn_models/runtime/profile_segments.cpp` | the per-(tile, backend) measurements |
| `qnn_models/boards/qrb5165_v66/graphs/yolov8n_HTA_split.json` | the partition the yolov8n tile boundaries derive from |
| `qnn_models/{yolov8n_backbone,yolov8n_head}.onnx` | what the yolov8n sub-DLCs were compiled from |
| `qnn_models/deploy.sh` | upstream ONNX → DLC → int8 conversion (not run by this flow) |

### Generated

| Path | Contents |
|---|---|
| `qnn_models/flow_c/gen/ir/<net>/int8/graph.json` | `extract_graph` output (gitignored) |
| `qnn_models/flow_c/gen/onnx/<net>.onnx` | ONNX from the same module (gitignored) |
| `qnn_models/flow_c/gen/runtime/<workload>[_tag]/` | `dispatch_table.h` + `runtime_main.cpp` (gitignored) |
| `gen/qnn_vmfb/<net>/qrb5165_flowc/<HW>/...` | dispatch graphs |
| `gen/profile/<HW>/qrb5165_flowc/<net>/...` | profile CSVs |
| `data/toplevel/networks_flowc_3way_qrb5165.json` | workload spec |
| `schedules/scheduled_networks_flowc_3way_qrb5165*.json` | schedules |
| `runs/<workload>[_tag]/run.log`, `trace.csv` | run log and flat trace |
| `plots/flowc_3way_qrb5165*[_tag]*.png` | Gantt charts |

## Known limitations

- **What executes is not all modelblaster-sourced.** `mlp_control`'s DLC
  traces to modelblaster's trained checkpoint (all 8 ONNX initializers hash
  identically to a fresh export). `dronet`'s does not: the board's DLC was
  built from `qnn_models/dronet.onnx`, which is randomly initialized. The
  trained export exists (`flow_c.py ir --export-onnx`) but has not been
  converted, and converting it for HTA additionally needs the offline BN
  rewrite the binding declares. `yolov8n` has no modelblaster involvement
  at all.
- **Two op-id spaces.** The yolov8n IR the manifest indexes is the
  QNN/TFLite route (249 ops); the artifacts its tiles name were sliced from
  the ONNX route (233 nodes → backbone 103 + head 138 = 241, the slicer
  materialising the 8 `Split` ops at the cut). The tile *boundary* agrees
  across all three; only the op ids differ. `flow_c.py ir` reports the
  delta. Op-level correspondence would need
  `qnn_models/runtime/build_qnn_to_onnx_namemap.py`.
- **Coarse granularity is declared, not decided.** There is no automatic
  "is this network worth splitting" pass. xpu-rt's `granularity_advisor.py`
  is the only automated signal — on the greedy schedule it flags that
  yolov8n's largest dispatch (21.58 ms) exceeds mlp_control's free slot
  (1.97 ms) — and Flow C prints it without acting on it.
- **Tensor handoff is name-matched.** It fires when consecutive tiles were
  compiled from the same route (3 handoffs on the MILP schedule, where both
  yolov8n tiles sit on DSP) and silently does not when they were not
  (0 under greedy, tiles on different backends). `FLOWC_DUMP_TENSORS=1`
  prints the names. Explicit mapping in the manifest is the fix.
- **A CPU-heavy network's real cost is not its isolated cell.** `fused_full`
  measures 1.147 ms alone under its lane's exec mask and ~6.4 ms (p90 10.2 ms)
  inside the 4-way schedule. The per-lane execute mask binds the lane thread,
  but the QNN CPU op package builds its thread pool at bringup on the main
  thread with full-machine affinity, so it contends with every other lane —
  including the real-time-gated one. Same class as the per-op MILP gap in
  `runtime/HETEROGENEOUS_SCHEDULING_QRB5165.md`: CPU contention is not in the
  cost model. Networks on their own silicon match their cells to within 5%.
- **`--solver milp` needs cvxpy + a MOSEK license**; `greedy_periodic` is
  the default and needs neither.
- **The exclusion cost is still a cost.** xpu-rt's `Operation` carries a
  processing time per machine with no "forbidden" flag, so excluded cells
  get a derived, labelled sentinel. A per-op machine mask in
  `xpu-rt/workload.py` is the clean fix.

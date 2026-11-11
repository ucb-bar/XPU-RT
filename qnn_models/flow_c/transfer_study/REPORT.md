# Transfer costs and placement on the QRB5165 — full characterization

Measured, not inferred. 21 instrumented board runs across 7 scheduling points
(609 traced tile executions), plus placement data pooled from 21 points and 63
runs already on record.

    data_log.json             everything below, machine-readable
    data_log_entries.csv      609 rows, one per instrumented tile execution
    data_log_placements.csv   243 rows, one per (point, tile, lane) placement
    characterization.txt      the generated tables
    characterize.py           regenerates all of it, read-only

---

## 1. The transfer mechanism, and what the scheduler thinks it costs

A tile boundary is not a DMA or a shared ION buffer. It is a **memcpy through a
global mutex-guarded `std::unordered_map`**:

  * after every execute, `cache_put` copies **every output tensor** of the tile
    into the map, keyed `network/tensor_name`;
  * before every execute, `cache_get` looks up **every input tensor** by the
    same key and, on a hit, copies it in.

Two consequences follow, and both were confirmed in the traces:

**The cost is invisible to the cost model.** `tr.start_ms` is taken *after* the
`cache_get` loop and `tr.end_ms` *before* the `cache_put` loop, so the tile
duration the cost cells are built from contains neither. The copies are real
lane wall time that no cell accounts for.

**The cost is invisible to the scheduler.** `scripts/run_xpurt_schedule.py:244`:

    transfer_times = np.zeros((n_cores, n_cores))

The MILP consumes this matrix in constraint (3) —
`t[i] >= t[pred] + dur[pred] + max_transfer_time` — so the mechanism is fully
wired. It is only ever fed zeros. **Every cross-lane edge is modelled as free.**

## 2. What it actually costs

Across 609 instrumented executions:

    total execute time      3331.051 ms
    handoff-in (cache_get)     4.048 ms    0.12% of execute
    handoff-out (cache_put)   62.252 ms    1.87% of execute
    total handoff             66.300 ms    1.99% of execute

**~2% of execute time, and it is overwhelmingly on the write side.** That
asymmetry is structural: `cache_get` only pays a memcpy when a key hits, while
`cache_put` pays unconditionally for every output of every tile.

Cost tracks bytes, at roughly memcpy bandwidth once the tensor is big enough to
matter:

    output bytes (pow2 bucket)     n    put p50      GiB/s
                          16 B   141    0.0009 ms     0.02
                       1,024 B   156    0.0017 ms     0.57
                       2,048 B     6    0.0044 ms     0.43
                     524,288 B   144    0.3144 ms     1.55

Below ~1 KiB the cost is a fixed ~1–2 µs of map lookup and mutex, not copying.
At 700 KiB it is a real 0.31 ms at ~1.55 GiB/s — well under what the memory
system can do, because the copy is serialized behind one global mutex shared by
all three lanes.

Per tile, the only entries that matter are yolov8n's:

    network/tile@lane                       n   exec p50   put p50   out KiB   handoff/exec
    yolov8n_b/yolov8n_backbone@dsp          3     14.185    0.4298     700.0      3.0%
    yolov8n_c/yolov8n_head@dsp             12     17.961    0.3791     689.1      2.1%
    yolov8n_a/yolov8n_backbone@hta         12     14.428    0.3560     700.0      2.5%
    ...
    mlp_control/mlp_control_full@cpu      141      0.076    0.0009       0.0      2.5%

Note the two ends of the range reach the same ~2.5% for opposite reasons:
yolov8n because it copies 700 KiB, mlp_control because a 76 µs tile cannot
absorb even 1 µs of fixed overhead.

## 3. Most of the copying is wasted

`cache_put` runs unconditionally, whether or not anything downstream will ever
ask for the tensor. Classifying each tile by its role in the binding DAG:

    tile                 role          n   cached MiB  read MiB   put ms   output ever read?
    yolov8n_backbone     source       72       49.219     0.000   31.360   yes
    yolov8n_head         terminal     72       48.450     8.203   23.984   NO -- pure waste
    fused_vision_conv    source       78        0.114     0.000    5.316   yes
    mlp_control_full     isolated    183        0.002     0.000    1.175   NO -- pure waste
    fused_tail           terminal     78        0.001     0.025    0.150   NO -- pure waste
    fused_depth_conv     source       78        0.076     0.000    0.144   yes
    dronet_full          isolated     36        0.000     0.000    0.073   NO -- pure waste
    vint_encoders        source        6        0.021     0.000    0.035   yes
    vint_decoder         terminal      6        0.000     0.000    0.015   NO -- pure waste

    cache lookups          921, of which 57 hit   (6.2%)
    bytes cached         97.88 MiB
    bytes read back       8.23 MiB   (8.41% of cached)
    terminal-tile copies 25.397 ms of 66.300 ms  (38.3%), 48.5 MiB

**38.3% of all transfer time copies the output of a terminal tile — a tile no
other tile depends on, so the copy is structurally unreadable.**
`yolov8n_head` alone accounts for 24.0 ms and 48.5 MiB of it.

The picture is worse than the terminal share suggests. `yolov8n_backbone`
caches 49.2 MiB and `yolov8n_head` reads back only 8.2 MiB of it: the backbone
has several output tensors and the head consumes one. Counting both effects,
**8.41% of everything copied is ever read.**

### The fix is cheap and local

The binding manifests already carry `depends_on`, so the emitter knows at
generation time which tensors are consumed. Restricting `cache_put` to tensors
that some downstream tile actually names would remove ~38% of handoff time
outright, and gating on the specific consumed tensor name would take out most
of the rest. On these workloads that is ~0.75% of execute time — small, but it
is pure loss, it lands on the critical path, and it is concentrated exactly on
the busiest lane.

## 4. Transfer is not free, and it is not stable either

Per-run handoff totals show a contention tail:

    baseline_seed0  rep1   2.103 ms   1.07%
    baseline_seed0  rep2   1.535 ms   0.86%
    baseline_seed0  rep3  18.025 ms  10.35%   <-- same binary, same schedule

An order-of-magnitude spike on an identical rep. The single global
`g_cache_mu` serialises all three lanes' handoffs, so a 700 KiB copy on one
lane blocks the others. Comparing the same tile on an idle-lane point against a
contended one gives `fused_tail` **1.0 µs -> 1.9 µs (x1.95)**.

This matters for the earlier finding that ~10% of the makespan gap is
"scheduling". Transfer is part of that residual: unmodelled, on the critical
path, and with a heavy tail under contention.

## 5. Placement

Pooled over 21 points and 63 runs, 243 placements:

    tile                    hta    dsp    cpu   total  lanes  distribution
    mlp_control_full          0     30    148     178      2  cpu=83%  dsp=17%
    yolov8n_backbone         45     14      0      59      2  hta=76%  dsp=24%
    yolov8n_head              0     49     10      59      2  dsp=83%  cpu=17%
    dronet_full              21     30      3      54      3  dsp=56%  hta=39%  cpu=6%
    fused_vision_conv        28     18      0      46      2  hta=61%  dsp=39%
    fused_depth_conv         13      9     24      46      3  cpu=52%  hta=28%  dsp=20%
    fused_tail                0     10      36     46      2  cpu=78%  dsp=22%
    vint_encoders             0      2       0      2      1  dsp=100%
    vint_decoder              0      0       2      2      1  cpu=100%

### Capability envelope: what is reachable but never used

    tile                 eligible          used          unused
    yolov8n_backbone     cpu,dsp,hta       dsp,hta       cpu
    fused_vision_conv    cpu,dsp,hta       dsp,hta       cpu
    vint_encoders        cpu,dsp           dsp           cpu
    vint_decoder         cpu,gpu           cpu           gpu

Four tiles never touch a lane they are eligible for. Three of those are CPU and
the solver is right to avoid it — `yolov8n_backbone@cpu` is 34.3 ms against
13.2 ms on DSP. **The interesting one is `vint_decoder`: the GPU is its fastest
eligible lane (16.4 ms vs 22.6 ms on CPU) and it was never scheduled there**,
because the GPU lane is not in the machine set these points were solved
against. That is the single largest placement gain still on the table.

### Stability: half the tiles migrate

    tile                 points  modal lane  modal share   distribution
    yolov8n_backbone         59         hta         76%    hta:45  dsp:14
    yolov8n_head             59         dsp         83%    dsp:49  cpu:10
    mlp_control_full         21         cpu         95%    cpu:20  dsp:1
    dronet_full              15         dsp         60%    dsp:9  hta:5  cpu:1
    fused_vision_conv        11         hta         55%    hta:6  dsp:5
    fused_depth_conv         11         cpu         64%    cpu:7  hta:3  dsp:1
    fused_tail               11         cpu         91%    cpu:10  dsp:1

**4 of 9 tiles hold one lane in >=90% of points; 5 migrate.** The migrators are
exactly the tiles with a genuine second option of comparable cost —
`fused_vision_conv` splits 6/5 between HTA and DSP because the two cells are
1.115 ms and 0.765 ms, close enough that lane pressure decides. `dronet_full`
uses all three lanes across points.

This is the solver behaving correctly, and it is why a per-tile "preferred
backend" heuristic would lose: the right lane is a function of what else is
competing, not of the tile.

### Placement quality: the price of parallelism

Charging every placement against the tile's fastest eligible lane:

    cell                            fastest   cost ms   placed on              excess
    yolov8n/yolov8n_head                dsp    17.652   dspx49 cpux10        +345.1 ms
    vint/vint_decoder                   gpu    16.425   cpux2                 +42.8 ms
    yolov8n/yolov8n_backbone            dsp    13.188   htax45 dspx14         +37.3 ms
    dronet/dronet_full                  dsp     1.035   dspx30 htax21 cpux3   +34.2 ms
    mlp_control/mlp_control_full        cpu     0.067   cpux148 dspx30        +18.6 ms
    fused_split/fused_tail              cpu     1.627   cpux36 dspx10         +17.8 ms
    fused_split/fused_depth_conv        cpu     0.147   cpux24 htax13 dspx9   +15.8 ms
    fused_split/fused_vision_conv       dsp     0.765   htax28 dspx18          +9.6 ms
    vint/vint_encoders                  dsp    14.213   dspx2                  +0.0 ms

    total: +521.3 ms of "excess" across all placements

**This number is not waste.** A lane runs one tile at a time; spilling to a
slower lane is usually the whole point. `yolov8n_head` contributes +345 ms
purely because ten of its 59 placements went to CPU at 52.2 ms instead of DSP
at 17.7 ms — and those are the cases where the DSP was already saturated at
92.5% occupancy and the alternative was waiting.

The one line that *is* waste is `vint_decoder`: +42.8 ms against a lane that
was simply not offered to the solver.

---

## Recommendations, in order of return

1. **Give the solver a non-zero transfer matrix.** It already consumes one. Even
   a crude `bytes / 1.55 GiB/s` derived from §2 would stop it treating a 700 KiB
   boundary as free. This is a one-line change plus a size table.
2. **Stop caching terminal-tile outputs.** `depends_on` is already in the binding
   manifests; ~38% of handoff time is structurally unreadable.
3. **Add the GPU to the machine set for ViNT points.** It is `vint_decoder`'s
   fastest eligible lane and was never available to the solver.
4. **Shard the handoff mutex per network, or per key.** One global lock across
   three lanes produced a 10x tail (2.1 ms -> 18.0 ms on an identical rep).

## Reproducing

    cd qnn_models/flow_c
    python3 transfer_study/run_transfer_study.py --reps 3   # 21 board runs
    python3 transfer_study/characterize.py                  # tables + data log

The study re-emits runtimes from the sweep's already-solved schedules, so
placements are identical to the recorded sweep and the only difference is the
instrumentation added to `flowc/emit_runtime.py` (`hin_ms`, `hout_ms`,
`hin_bytes`, `hout_bytes`, `hin_hits`, `hin_tensors`, `hout_tensors`).

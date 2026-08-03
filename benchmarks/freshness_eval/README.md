# Freshness-validity evaluation

Does a dependent robotic control pipeline keep meeting its **own** deadlines
while emitting **invalid** outputs, because its inputs are stale?

    deadline_valid   consumer finished by its own deadline
    freshness_valid  input age at output <= freshness window phi
    output_valid     both

If deadline success stays high while `output_valid` falls, then local
schedulability is not a sufficient correctness criterion for a dependent chain —
which is the premise the rest of the adaptive-scheduling work rests on.

## Environment

The only environment with the full solver stack is the `merlin-dev` conda env
(python 3.11, `cvxpy`, `mosek`, `ortools`, `matplotlib`, `pandas`, `pytest`).
The repo `.venv` lacks mosek and ortools.

```bash
PY=/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python
export PYTHONPATH=/scratch2/agustin/XPU-RT:/scratch2/agustin/XPU-RT/xpu-rt
```

`mosek.lic` must be present at the repo root (gitignored — it is machine-local).
It is only needed for the MILP schedulers; the policies in this sweep are list
schedulers and do not require it.

## Required profile files

The sweep reads FireSim-measured per-dispatch costs from
`gen/profile/<backend>/firesim_gemmini_opu/<model>/<model>.int8/topo_0/results.csv`
and dispatch graphs from `gen/vmfb/<model>/...`. Both are committed. Regenerate
them from ModelBlaster with:

```bash
# dispatch graphs (ModelBlaster's own emitter)
cd /scratch2/agustin/ModelBlaster
for m in dronet mlp_control yolov8_nano_64; do
  PYTHONPATH=. $PY -m pipeline.emit_dispatch_graph \
      --ir examples/$m/int8/generated/graph.json \
      --out-root /scratch2/agustin/XPU-RT/gen/vmfb \
      --target firesim_gemmini_opu --hw gemmini
done

# per-dispatch costs
cd /scratch2/agustin/XPU-RT
$PY scripts/export_profile_db_to_results_csv.py
```

Each `results.csv` has a sibling `_provenance.json` recording both repo SHAs,
sample counts, the zero-cost dispatch ids, and the clock assumption.

## Commands

```bash
# sweep
$PY -m benchmarks.freshness_eval.run \
    --config data/toplevel/freshness_canon_300ms.json \
    --output-dir results/freshness_eval \
    --seeds 0 --bursts 0,1,2,3,4 --deltas 5,10,20,30,50 \
    --policies static_nominal,edf,heft,static_conservative \
    --cell-timeout 900

# figures
$PY -m benchmarks.freshness_eval.plot \
    --input results/freshness_eval --output figures/freshness_eval
```

The exact command used for a given run is saved to `command.txt` in its output
directory.

## Runtime

Roughly, on a contended shared host (one solve per policy x burst x seed):

| B | ops | solve |
|---|-----|-------|
| 0 | 336 | ~6 s |
| 1 | 548 | ~12 s |
| 2 | 760 | ~19 s |
| 3 | 972 | ~1 min |
| 4 | 1184 | >3 min |

The full 4-policy x 5-burst grid at one seed is roughly 25–40 minutes. `B=4` is
the tail; `--cell-timeout` records a cell that exceeds it as a failure in the
manifest rather than stalling the sweep, so the grid never has a silent hole.

Seeds are cheap to add but buy little: all four policies are deterministic, so
extra seeds are a control that confirms determinism rather than a source of
variance. The manifest reports, per `(policy, B)`, whether the schedule was
identical across seeds.

`phi` is free — the freshness window changes the verdict, not the schedule, so
it is applied post-hoc rather than costing a re-solve.

## Expected outputs

```
results/freshness_eval/
  manifest.json        config, A0, phi grid, policies, provenance, determinism, failures
  aggregate.csv        one row per (policy, B, phi, seed)
  per_invocation.csv   one row per consumer invocation per freshness edge
  intervals.csv        every invocation interval, incl. the soft interfering work
  command.txt
  git_commits.json
  work/                per-cell solver logs
figures/freshness_eval/
  plot1_deadline_vs_freshness.{png,pdf}
  plot2_diagnostic_timeline.{png,pdf}
  plot3_phi_sensitivity.{png,pdf}
```

There is no `selector_log.csv` yet — the epoch-level candidate selector is a
later phase. Its absence is expected, not a failed run.

## Timing provenance

Everything is FireSim-measured cycles from ModelBlaster's `profile_db`,
**converted to milliseconds at an assumed 1 GHz** (`mean_time = cycles / 1e6`),
which is what ModelBlaster's solver configs assume (`cycles_per_ms: 1000000`).

**That is not the clock the hardware runs at.** The Alveo U250 bitstreams close
timing at 25–30 MHz. At 25 MHz a DroNet inference is 361 ms and a 10 ms control
period is impossible, so the millisecond-denominated workload exists only under
the 1 GHz assumption. It is a documented, uniform scale factor recorded in every
manifest and every `_provenance.json`; it cancels in relative comparisons, but
no absolute millisecond claim is valid without restating it. Raw cycles are
preserved in the `cycles` column of each `results.csv`.

Backends are `gemmini` and `rvv_opu`, matching the prebuilt
`alveo_u250_firesim_shuttle_gemmini_opu` bitstream
(`FireSimGemminiAndOPUShuttleConfig`: tile 0 Shuttle + Gemmini RoCC, tile 1
Shuttle + Saturn OPU vLen=128). Measured per-network totals over the dispatches
present in the graph:

| network | gemmini | rvv_opu | ratio |
|---|---|---|---|
| `mlp_control` | 0.546 ms | 0.546 ms | **1.00x** |
| `dronet` | 17.973 ms | 241.462 ms | 13.4x |
| `yolov8_nano_64` | 67.202 ms | 1069.004 ms | 15.9x |

Eight `yolov8_nano_64` dispatches are `chunk2_c1`, one of ModelBlaster's
zero-cost alias ops (kernel call skipped, dependency semaphore still posted).
They are exported with `cycles=0` and `source=zero_cost_by_construction` so
precedence edges survive without an invented duration. Any *other* unprofiled
dispatch is a hard error.

## Interpretation guidance

**`phi` is anchored on A0, the measured uncontended input-age ceiling, not on
the producer period.** On this grid the DroNet period is 50 ms but A0 is
60.546 ms, and the uncontended age distribution is exactly
`{20.5, 30.5, 40.5, 50.5, 60.5}` ms. A window below A0 therefore reports
staleness caused by the 50 ms *sampling rate*, not by contention. `phi = A0 +
delta` guarantees every point has real headroom, and the closed-form check is a
unit test (`test_freshness.py::AnalyticCrossCheck`).

**Read `B=0` as the control.** It is the same workload with no soft work at all.
Any staleness there is structural — including the two invocations at t=0 and
t=10 that precede the first DroNet completion and are recorded as
`no_completed_producer`, not as stale.

**Makespan is a poor discriminator here** and should not be read as a contention
metric: it is pinned near 290.5 ms at every B by the last control release
(290 ms) plus 0.546 ms of work. The rates are the metric.

**`deadline_success_rate` is close to 1.0 by construction** for a 0.546 ms
consumer in a 10 ms window on a two-cluster machine. The divergence is therefore
real but *bounded from above only by the oversubscribed points* — say so rather
than presenting an open-ended gap as the finding. See limitations.

**The oracle row is a post-hoc upper bound**, the best `output_valid_rate`
available at each `(B, phi)` among the deployable policies. It is not a
deployable policy.

**Post-passes are forced off.** Both compaction and automerge rewrite the
emitted fixture, so leaving either on would make this a comparison of
policy+post-pass. The effective setting is recorded in
`manifest.post_passes`.

## Known limitations

- **Freshness is imposed, not observed.** No dataflow exists between the
  perception and control networks in either repo: XPU-RT's `edges` become MILP
  precedence constraints, ModelBlaster's `deps` become semaphore ordering edges,
  and buffers are named per `(model, tensor)` with no instance or version tag —
  all instances of a network share one output buffer. Which producer instance a
  consumer consumed is **inferred from timestamps, never recorded**, and that
  stays true on hardware. Every record carries
  `producer_instance_provenance = inferred_from_schedule_timestamps`. Do not
  describe these results as measuring runtime dataflow freshness.

- **The freshness edge is deliberately not a precedence constraint.** A
  controller that blocks waiting for perception can never consume a stale input
  — it can only miss its deadline. Declaring the same pair under both `edges`
  and `freshness_edges` raises.

- **`phi` is not application-derived.** It is anchored on a measured property of
  the workload, not on a control-stability requirement. No claim is made that
  any particular `phi` is correct for a real drone controller.

- **`latest_completed` is one consumption policy** among several; release-matched
  and versioned alternatives are not evaluated.

- **`mlp_control` has no backend differentiation.** Its `gemmini` and `rvv_opu`
  profiles are byte-identical, because `kernels/gemmini/` has no linear kernel
  and an all-linear model falls back to the same scalar reference on both. So
  "isolate control on a separate core" is not an available protection mechanism,
  and the consumer's placement is not a degree of freedom.

- **Deadline success is near-structural**, as above. The oversubscribed `B>=3`
  points are what bound the divergence region.

- **`static_conservative` uses a soft cost penalty**, not a hard reservation:
  `preferred_hw` adds a large penalty to other placements. Describe it as a
  preference. It applies one mechanism only (reserve the fast accelerator for
  the producer); pushing the soft work onto the vector unit is a second, much
  blunter mechanism that belongs to the degraded-safety candidate.

- **Trace-driven, not runtime-executed.** These are solver schedules, not
  hardware traces. Nothing here has run on the FPGA.

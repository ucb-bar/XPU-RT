# Salvage report: `feat/agentic-fusion-loop`

Audit of `ucb-bar/ModelBlaster@feat/agentic-fusion-loop` (HEAD `16b7b52`) against the
canonical `feat/k1-xpurt`, before any new fusion/sharding machinery is written.

Two findings dominate and are stated up front, because both change what should be
built next.

1. **The IR rewriters need no porting at all.** `apply_fusion_hint.py`,
   `apply_split_hint.py` and both their test files are **byte-identical**
   between the branch and canonical. Where the two diverge in adjacent code,
   canonical is strictly *ahead* (it adds `conv2d_batchnorm2d_silu_s8` and
   performs conv+BN(+act) fusion at extraction time). The salvage premise does
   not hold here; the work is bug-fixing, not porting.

2. **The headline `-48.6%` sharding result is not substantiated by its own
   artifacts.** Detail in the next section. The honest figure derivable from the
   same data is **≈ -0.25% end-to-end** (-7.3% on the one split op).

---

## The -48.6% claim

Claim, from the commit message of `16b7b52`: splitting `l0.conv` of yolov8_nano
two ways across gemmini + rvv_opu takes wall time from 489,014 to 251,398 ticks.

**The `v26_unsharded_baseline/` directory does not contain an unsharded run.**
Verified directly in its `uartlog`:

```
$ grep -E 'l0\.conv' .../v26_unsharded_baseline/uartlog
0,yolov8_nano,0,0,conv2d_s8,l0.conv,gemmini,0,0.000000,11.341962,0,16,7472   <- TRACE says l0.conv
gemmini_q31,0,l0.conv.tile_0,conv2d_s8,...;OC=8;...,7455420                  <- PROFILE says tile_0
gemmini_q31,1,l0.conv.tile_1,conv2d_s8,...;OC=8;...,7387560                  <- PROFILE says tile_1

$ grep -c 'l0\.conv,conv2d_s8.*OC=16' .../v26_unsharded_baseline/uartlog
0
```

The trace is labelled with the unsharded op; the binary that produced it
contains only the two OC=8 tiles. An unsharded `l0.conv` (OC=16) appears
nowhere in the run.

Four further corroborating facts, all from the branch's own logs:

* **Different schedules.** baseline `run.log:4` ingests **212** entries from
  `scheduled_networks_1yolo_4mlp_2dronet_firesim_only_yolov8_nano_cpsat_profiled.json`;
  sharded `run.log:34` ingests **213** from `scheduled_yolov8_l0_shard_v26.json`.
  The commit's claim of "the same schedule" is not what ran.
* **Off-by-one labels.** Matching each trace row to the same run's profile table
  by op name: sharded **205/205** agree within 2 ticks; baseline **1/202**. Every
  baseline row carries the *next* op's timing under the previous op's name --
  the signature of a 213-op library driven by a 212-entry table.
* **Half the network ran on the wrong core.** Comparing profile `backend`
  against trace `core_kind`: sharded **0/205** mismatches, baseline **108/202**.
  Total standalone compute: baseline 523.4M cycles vs sharded 287.2M. A 236M
  difference cannot come from splitting one 9.7M-cycle op; it is the
  misplacement.
* **Neither run was verified.** No `MODELBLASTER_VERIFY`, no `max_abs_err` in
  either uartlog -- the `*** PASSED ***` present is Zephyr's termination marker.
  Their outputs differ entirely (`sum=201500` vs `sum=-3.76e6`), so they did not
  compute the same function. This violates the branch's own
  `realize-and-run` rule: *"Verify must pass before reporting a measurement."*

Also: `grep` for `489014`, `251398`, `48.6` across the whole branch returns
**zero hits outside the commit message**. The number was computed by hand.

### What the artifacts do support

* **Concurrency is real.** Sharded trace rows 0 and 1: tile_0 on gemmini
  `16 -> 7478`, tile_1 on rvv_opu `14 -> 9059`. Genuinely overlapping execution
  on two different resources. (Note "within 2 ticks" is within 2000 cycles -- a
  trace tick is 1000 cycles, see the unit bug below.)
* **The honest gain.** Against the branch's own profile DB (unsharded `l0.conv`
  on gemmini = 9,758,230 cycles, identical across 3 runs) and the sharded run's
  measured tiles: critical path `max(7,461,551, 9,044,605)` = 9,044,605, i.e.
  **-7.3% on that op, ≈ -0.25% of the ~287M-cycle network.**
* **A genuinely interesting negative.** An OC=8 tile costs 7.46M against 9.76M
  for OC=16 -- **76% of the full op for half the output channels**, so a 2-way
  OC split inflates total work by ~53%. That is precisely the case an acceptance
  test on summed cycles rejects and a makespan test accepts.

### Is sharding still worth doing?

**Yes -- but on independent evidence, not this one.** Measured on the K1 this
session: DroNet's per-instance service time falls 113.7 -> 62.0 -> 32.4 ms on
1/2/4 cores, and a 4-core shard cuts worst-case lateness from 108.8 ms to
27.2 ms (achieved frequency 22.6 -> 29.0 Hz against 30 Hz required). Sharding is
real on this hardware. The `v26` result should simply not be cited for it.

---

## Component verdicts

| component | branch path | verdict | reason | current equivalent |
|---|---|---|---|---|
| `apply_fusion_hint` | `pipeline/apply_fusion_hint.py` | **KEEP** | byte-identical to canonical | same file |
| `apply_split_hint` | `pipeline/apply_split_hint.py` | **KEEP** | byte-identical to canonical | same file |
| both hint tests | `pipeline/tests/test_apply_*.py` | **KEEP** | identical; 20/20 pass | same files |
| fused KernelSpecs | `pipeline/reference_kernels.py` | **KEEP** | canonical is ahead (+`conv2d_batchnorm2d_silu_s8`) | canonical |
| split/fusion codegen | `pipeline/generate_skeleton.py` | **KEEP** | canonical is a strict superset | canonical |
| `decision_loop.py` skeleton | `scripts/decision_loop.py` | **PORT** | right stage sequence: advisor -> rank -> filter -> rewrite -> build -> verify -> measure -> accept | none |
| `decision_loop` acceptance test | `scripts/decision_loop.py:361-373` | **REPLACE** | accepts on Σ standalone cycles of ONE network -- structurally rejects every parallelism win, and discards the `wall_clock_cycles` it already measures | new K1 objective |
| `REALIZABLE_FUSE_PAIRS` | `scripts/decision_loop.py:66-70` | **DISCARD** | declared, never read | -- |
| `measure_candidate.sh` | `scripts/measure_candidate.sh` | **REPLACE** | spike-stdout coupled; hardcoded conda interpreter; `LLM_PROVIDER=bedrock` default; **mutates the source `graph.json` in place** and restores after, so a crash corrupts the repo | `scripts/run_xpurt_k1.sh` |
| `ingest_measured_cycles.py` | `scripts/ingest_measured_cycles.py` | **PORT** | clean parser; emits zero provenance | none |
| `run_xpurt_bundle.py` | `scripts/run_xpurt_bundle.py` | **REPLACE** | FireSim-specific throughout; `FORCE_REGEN=0` reuse is the direct cause of the corrupted baseline above; ELF path wrong for the very runs it produced. Keep `_extract_trace()` | `scripts/run_xpurt_k1.sh` |
| `close_xpurt_loop.py` | `scripts/close_xpurt_loop.py` | **PORT** | good orchestration; inject the two XPU-RT tool paths | none |
| `emit_measured_report.py` | `scripts/emit_measured_report.py` | **PORT** | best-designed file in scope; predicted-report overlay joined on `(network, instance, dispatch_id)` is right | none |
| `profile_db.py` | `benchmarks/profile_db.py` | **PORT + schema REPLACE** | right shape (append-only JSONL, `(run_id, dispatch_id)` idempotency); but no impl hash, no core/resource, no sample count, no p90, no artifact path -- and it **excludes hetero runs outright**, which is where the entire multi-model objective lives | `gen/profile*` trees |
| `update_pdb_*.py`, `recalibrate_pdb_*` | `scripts/` | **DISCARD** | hardcoded one-offs that write cycle counts into a `mean_time_ns` field and mutate `results.csv` in place -- the exact side-file hazard `WARNING.md` documents | one ingest path |
| `realize-hint` skill | `.claude/skills/realize-hint/` | **PORT** | sound rules, notably "never edit source `graph.json` in place" -- which `measure_candidate.sh` violates. 1 absolute path (L26) | already vendored |
| `realize-and-run` skill | `.claude/skills/realize-and-run/` | **PORT workflow / REPLACE commands** | right discipline; but step 3 duplicates step 2, step 4's `--schedule` flag does not exist (real flag is `--predicted-report`), 3 absolute paths | already vendored |
| `WARNING.md` | `artifacts/agentic_fuse_split/WARNING.md` | **KEEP verbatim** | the most valuable artifact in the whole audit | -- |
| `v26_shard_l0/` | `artifacts/runtime_optimization/` | **KEEP as data** | self-consistent (205/205) sharded measurement -- but it has no valid control | -- |
| `v26_unsharded_baseline/` | `artifacts/runtime_optimization/` | **DISCARD** | not an unsharded run | -- |
| the -48.6% claim | commit message only | **DISCARD** | see above | K1 shard measurements |
| Bedrock coupling | `pipeline/llm_client.py`, `bedrock_client.py` | **REPLACE** | no `codex` arm exists on the branch; `ConverseResult` and `extract_code_block` live *inside* `bedrock_client.py`, so any new client inherits the dependency unless they are lifted out first | canonical has `codex_client.py` |

---

## Bugs found, all present on BOTH branches

Ordered by severity. None of these is a porting question -- they are live defects
in code the K1 loop is about to depend on.

**S1 -- CRITICAL, silent numerical corruption.** A split `linear_s8` never
receives a weight or bias offset. `generate_skeleton.py:1204-1217` has no
`split_from` handling at all, while the `conv2d_s8` arm at `:1238-1247` does
exactly the right thing. `apply_split_hint.py:57-59` explicitly promises the
offset in its docstring. Generated C for a 2-way split of N=64:

```c
parallel_linear_s8(..., probe_lin0_weight_q, probe_lin0_bias_q, buf_probe_y,        1,32,32, ...);
parallel_linear_s8(..., probe_lin0_weight_q, probe_lin0_bias_q, (buf_probe_y + 32), 1,32,32, ...);
```

The two calls differ only in the *output* pointer. Weights are `[N, K]` and the
kernel indexes `weight[n*K + k]`, so both tiles compute output rows `[0,32)` and
tile 1 writes duplicates into `y[32:64]`. No crash, no build error, wrong
numbers. It escaped notice because DroNet's linears are `N=1`, so the splitter
rejects them before this path is reached and only conv splits were ever
exercised.

**F1 -- HIGH.** `_validate_fuse_group` (`apply_fusion_hint.py:81-117`) never
implements the external-re-entry check its own docstring specifies, so a
non-adjacent fuse group produces a dependency **cycle** with no error.
Reproduced on a 3-op chain fusing `[0,2]`.

**S2 -- HIGH.** `M > 1` linear splits produce overlapping writes past the parent
buffer: output `[M,N]` is row-major, so an N-tile is a strided column slice, not
contiguous. With M=4, N=64, 2 tiles, tile0 writes `[0,128)` and tile1 writes
`[32,160)`. Accepted with no error.

**id_remap -- MEDIUM.** Both rewriters renumber non-fused ops contiguously and
never emit the `old_id -> new_id` map, so any artifact keyed on `dispatch_id`
misaligns silently after a rewrite. Cheap fix: emit `out["id_remap"]`.

**S3 -- MEDIUM, latent.** Conv OC-split offset `tile*tile_oc*OH*OW` is correct
only for batch `N==1`. Unguarded.

**Unit bug -- MEDIUM.** `actual_*_cycles` in `xpurt_trace.csv` are **mtime ticks
of 1000 CPU cycles**, established empirically (trace duration x 1000 equals the
same run's profile cycle count for 205/205 ops). `emit_measured_report.py` and
`run_xpurt_bundle.py` both divide by `clock_mhz=1000` as if they were cycles, so
every microsecond figure they emit is **1000x too small**. The column name is
also wrong.

---

## The invariant to carry forward

`WARNING.md` documents a rejected fusion speedup: the IR rewrite removed ops
from the scheduler's view without any fused kernel existing to perform the
merged work, so *"the schedule counted the work as gone; the hardware would
still need to do it."* Its corrective patch then edited a `results.csv` that the
scheduler does not read, and the "honest" number came out within 0.05 ms of the
fiction -- because both runs read the same unmodified DB.

The v26 baseline is the mirror image of that failure. There, modelled work
vanished while the hardware would still do it. Here, the hardware genuinely did
the work, but under a dispatch table that did not describe it -- so the timings
are real and the **labels** are fiction. Neither is visible from the summary
numbers; both produced confident, stable, wrong results.

So the rule is stronger than "no bookkeeping-fiction speedups":

1. A rewrite may not reduce modelled work unless a kernel exists that performs
   the merged work.
2. Costs must reach the scheduler through the live ingest path, never a
   side-file.
3. A corrective patch must be independently checked to have taken effect -- an
   implausibly small delta after a large intended change is evidence of a no-op.
4. **The trace's labels must be provably generated from the same artifact the
   binary executed.** Nothing on the branch checks this, and it is what a
   provenance record (impl hash + dispatch table hash per run) would have
   caught.

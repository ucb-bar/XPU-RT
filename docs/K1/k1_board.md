# Running on the K1 board

Every command here was run on 2026-08-29 against the board. Timings are from
that session, so they tell you what "normal" looks like.

Setup lives in [`environment.md`](../environment.md); this is the operational
page. [`the_loop.md`](../Feature/the_loop.md) is the index for what the pieces mean.

## The board

SpaceMiT K1 (BananaPi M7), 8 riscv64 harts in **two 4-core L2 clusters**:

```
CPU_P  harts 0-3   cluster 0    IME (smt.vmadot) executes here
CPU_E  harts 4-7   cluster 1    IME SIGILLs here
```

VLEN=256 (`zvl256b`). `rdtime` is a fixed **24 MHz**; `rdcycle` SIGILLs from
userspace, so every "cycle" count in this repo is an rdtime tick.

## Before anything

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"     # exports CROSS
```

Not optional, and not a preference — see the two compiler traps below.

```bash
# ~/.ssh/config
Host k1
    HostName <board ip>
    User root
    IdentityFile <key>
```

Scripts read `MODELBLASTER_K1_HOST`, default `k1`. Nothing in this repo reads
or writes credentials; ssh does whatever it is already configured to do.

### Check the board is free — and do not use loadavg

```bash
ssh k1 'ps aux | grep -E "harness|xpurt" | grep -v grep'
ssh k1 'awk "/^cpu[0-9]/ {u=\$2+\$3+\$4; t=u+\$5; printf \"%s %.1f%%\n\", \$1, 100*u/t}" /proc/stat'
```

**`/proc/loadavg` has a permanent floor of exactly 2.00** on this board, from
two D-state kernel threads (`vq0`, `vq1`) that never leave uninterruptible
sleep. Any idleness check reading it concludes the board is busy, forever. An
idle board looks like `2.00 2.00 2.00` with every CPU under 1%.

The board is shared. Check before you take it.

## Profile one model

```bash
PY=$PWD/.venv/bin/python PROFILE_OUT_ROOT=$PWD/gen_mb/profile ITERS=3 \
    bash ModelBlaster/scripts/run_model_k1.sh ffn_block int8 rvv_x60 0
```

**≈6 s** end to end: extract → build → cross-compile → deploy → run → profile.
Arguments are `<model> <quant> <backend> <verbose>`.

Correctness is not a separate step. The harness golden-compares **in-binary**
on every run and the profile is only written if it passes:

```
=== MODELBLASTER_VERIFY === max_abs_err=0 max_rel_err=0 n=32768
```

Output lands as an IREE-shaped `results.csv` under
`gen_mb/profile/<impl>/<target>/<model>/<basename>/<spec>/<topo_tag>/`. The
schema outlived the IREE path because it is what `profile_loader.py` reads.

### Several harts

```bash
MB_CORES=0,1,2,3 ITERS=7 PY=$PWD/.venv/bin/python \
  PROFILE_OUT_ROOT=$PWD/gen_mb/profile \
  bash ModelBlaster/scripts/run_model_k1.sh dronet int8 rvv_x60 0
```

`MB_CORES` derives the worker-pool width, the affinity mask, the binary suffix
**and** the profile's `topo_` tag from one place, so a profile cannot claim a
core count it did not run on. They used to be set separately, and a run tagged
`topo_0_1_2_3` whose pool was actually one thread is a serial measurement filed
under a parallel name — indistinguishable afterwards.

Watch for `int("0_1_2_3") == 123` in Python: digit separators. That is how a
`topo_123` directory once appeared.

Profiling at 1/2/4/8 harts is what makes `shard_advice` possible at all — it is
a claim about how cost changes with cores, and cannot be inferred from a
single-core profile.

## Schedule, then run the schedule

```bash
.venv/bin/python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_k1_mb.json --solver greedy
```

**≈1.7 s.** Then build and run the scheduled binary:

```bash
PY=$PWD/.venv/bin/python bash ModelBlaster/scripts/run_xpurt_k1.sh \
    --schedule schedules/scheduled_<spec>_greedy_profiled.json \
    --models ffn_block --backends rvv_x60,rvv_x60 --quant int8
```

**≈4.5 s.** `--models` and `--backends` must be **parallel lists**: core kind
*k* is executed by backend *k*. The K1 registry has two kinds (`rvv`,
`rvv_c1`), so a single-backend build still passes it twice.

Outputs land beside the generated walker:

```
build/k1_xpurt/_gen/<sched>/<sched>_stdout.txt      raw board stdout
build/k1_xpurt/_gen/<sched>/<sched>_trace.csv       per-dispatch trace
```

### Check the schedule is legal first

```bash
.venv/bin/python scripts/check_schedule_feasibility.py <schedule.json>
```

Most findings are slowdowns. **One is not**: an `ime`-tagged dispatch placed on
CPU_E SIGILLs and produces no output at all. `{"cpu_p": 8}` — the runbook's own
old recommendation — maps `CPU_P#4..7` onto cluster 1 and dies. Use
`{"cpu_p": 4, "cpu_e": 4}` for anything with IME.

## Live telemetry

```bash
MB_XPURT_STREAM=1 PY=$PWD/.venv/bin/python \
  bash ModelBlaster/scripts/run_xpurt_k1.sh --schedule <s.json> \
       --models ffn_block --backends rvv_x60,rvv_x60 --quant int8

grep '"start_ticks"' build/k1_xpurt/_gen/<sched>/<sched>_stdout.txt \
    > /tmp/telemetry.jsonl

.venv/bin/python xpu-rt/streaming_feedback.py \
    --telemetry-stream /tmp/telemetry.jsonl \
    --out schedules/xpurt_feedback.json \
    --windows-from data/toplevel/<spec>.json \
    --run-id k1_$(date -u +%Y%m%dT%H%M%SZ)
```

One JSON line per dispatch **end**, written with a single `write()` so
concurrent workers cannot tear a line. The trace block only prints at exit,
which is enough to explain a run and useless for responding to one.

**Pass `--windows-from`.** The walker does not know its own deadlines — periods
live in the spec, not the binary — so without it the miss rate is reported as
*unknown*, not zero.

## Reading a run

```bash
.venv/bin/python scripts/join_k1_trace.py <trace.csv> <schedule.json>
.venv/bin/python scripts/plot_k1_trace_gantt.py --composite A=t.csv:s.json
```

The join answers "slow kernel or long queue", which is what a deadline miss
turns on.

## One full revolution

```bash
.venv/bin/python examples/feedback_loop/one_revolution.py
```

Runs everything that does not need the board and stops at the steps that do,
rather than substituting a modelled number. The measured version, ~35 s:

| step | time | result |
|---|---|---|
| profile baseline, 1 hart | 6.0 s | 644313 ticks |
| solve + `--emit-feedback` | 1.7 s | 20 dispatches with hints |
| advice, joined with the run | 0.04 s | 2 actionable shard items |
| bridge → hint | 0.02 s | d1 ×8 (6.407×), d3 ×8 (4.949×) |
| apply → annotated IR | 0.03 s | 5 dispatches, still 5 |
| build + run sharded, 8 harts | 4.5 s | 155903 ticks, `max_abs_err=0` |
| re-solve on the new costs | 0.4 s | `pdb_hash` differs |
| verdict | 0.04 s | **ACCEPT**, on term 4 (p99 response) |

## The two compiler traps

**GCC 13.2 reorders `vsetvl`.** It moves two `__riscv_vsetvl_*` calls so a
widening instruction runs under the narrow vtype:

```
vsetvli e32,m4     <- sets SEW=32
vsetvli e8,m1      <- clobbers it to SEW=8
vle8.v / vle8.v
vsext.vf4          <- ILLEGAL: widening 8->32 needs SEW=32
```

The binary SIGILLs with **no stdout at all**. This is what `CROSS` defaults to
via chipyard's riscv-tools, which is why `setup_spacemit_toolchain.sh` refuses
anything below 14. Loud, at least: it crashes rather than computing a wrong
answer, so no past measurement is invalidated by it.

**GCC 14.3 is wrong in the opposite direction, and it is worse.** It
substitutes a wrong AVL on a *chained* `vsetvl` — one whose argument is itself
a `vsetvl` result — producing a wrong answer instead of a crash. Two committed
kernels shipped that way (`lstm_s8` err=20, `avgpool2d_s8` err=68). The only
safe form is to pass the **element count** to every width:

```bash
ModelBlaster/scripts/check_rvv_avl.py     # refuses the chained form
```

## When something goes wrong

| symptom | cause |
|---|---|
| SIGILL, no stdout | GCC 13.2, or an `ime` dispatch on cluster 1 |
| wrong answer, build fine | chained `vsetvl` AVL — run `check_rvv_avl.py` |
| `undeclared identifier ..._rvv_x60` | an emitter calling `_weight_name` without `backend` |
| `fatal error: model.h` | header staging; fixed in both harnesses, re-run cmake clean |
| profile written but suspiciously fast | check the `topo_` tag matches `MB_CORES` |
| empty profile tree from `find` | `gen_mb/profile` is a **symlink**; `find` does not follow it |
| board "busy" but nothing running | you read `/proc/loadavg`; use per-CPU `/proc/stat` |
| every solver returns the same schedule | the workload is uncontended — that is the answer, not a bug |

## Measurements, and what they do not establish

* [`k1_contention.md`](k1_contention.md) — do concurrent dispatches slow each
  other down? **Null result**: distributions overlap, arms not monotonic in
  co-runner count.
* [`k1_cost_by_pred.md`](k1_cost_by_pred.md) — cost of reading what the
  previous dispatch wrote, from elsewhere. ~6% off-hart, ~10% cross-cluster —
  a **model fitted to three measured classes**, not 64 independent
  measurements, and measured on one model's chain shape.

Both are worth reading before quoting either number.

## Repeatability

Three reps gave a worst CV of **48.7%** on `linear_s8` in the run above. The
4.13× speedup is solid; its third digit is not. Use `ITERS=7` or more for
anything that goes in a figure, and interleave arms rather than blocking them —
two solo runs twenty minutes apart differed by 2.6% with nothing else on the
board.

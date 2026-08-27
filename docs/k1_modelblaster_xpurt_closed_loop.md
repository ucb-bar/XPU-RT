# SpaceMiT K1: profile → schedule → execute → advise → recompile

A runbook you can follow from a fresh checkout to a measured result on the
physical K1. Every number quoted here was measured on the board; where something
did not work, that is recorded too, because the failures cost more time than the
successes did.

---

## 0. The board

| | |
|---|---|
| Console | `/dev/ttyUSB4` — CH340 (`1a86:7523`), 115200 8N1, group `dialout` |
| Host | `ssh k1` → `10.44.98.236` (wlan0; both Ethernet ports are down) |
| Key | `/scratch2/agustin/DIMA_SLICE`, already in the board's `authorized_keys` |
| OS | Bianbu 3.0, Linux 6.6.63, glibc 2.41 |
| SoC | Spacemit X60, 8 × riscv64 |

### The SSH stanza matters

```
Host k1 spacemit
    HostName 10.44.98.236
    User root
    IdentityFile /scratch2/agustin/DIMA_SLICE
    IdentitiesOnly yes
    IdentityAgent none          # <- this line
```

**`IdentityAgent none` is not optional.** With a stale `ssh-agent` in the
environment, `ssh` hangs at `Next authentication method: publickey` and the
board's own log shows `Connection closed by authenticating user root [preauth]`
— i.e. the *client* gave up. `ssh-add -l` hanging is the tell. The same fault
breaks `git` over SSH; use `GIT_SSH_COMMAND` with the same option.

### Topology, measured not assumed

| Cluster | Cores | L2 | scalar | RVV | IME (`smt.vmadot`) |
|---|---|---|---|---|---|
| 0 | 0,1,2,3 | 512K shared 0-3 | yes | yes | **executes** |
| 1 | 4,5,6,7 | 512K shared 4-7 | yes | yes | **SIGILL** |

`/proc/cpuinfo` reports an identical ISA on all 8 harts and never mentions IME —
it is a vendor extension the kernel does not enumerate. The table above comes
from `artifacts/k1_bringup/*/ime_probe.c`, which pins each core and executes the
raw encoding under a `SIGILL` handler. Re-run it with
`ssh k1 /root/mb_k1/bin/ime_probe`.

All 8 cores share **one** cpufreq policy, already `performance` at a fixed
1.6 GHz. No governor change is needed, so none was made.

### Two facts that will bite you

**`rdcycle` raises SIGILL from userspace.** `rdtime` works, and is a fixed
**24.000 MHz** (41.7 ns tick). Any generated code that times itself with
`rdcycle` does not run slowly here — it dies. Convert tick counts with
**24 MHz**, never the 1.6 GHz core clock.

**Load average sits at 2.00 on an idle board.** Two vendor kernel threads
(`vq0`, `vq1`) are wedged in uninterruptible sleep at 0% CPU. Benign. Do not
chase it.

---

## 1. Build the compiler and the runtime

```bash
cd /scratch2/agustin/XPU-RT
git submodule update --init --recursive merlin/third_party/iree_bar   # ~5 GB
bash merlin/build_tools/SpacemiT/setup_toolchain.sh                    # 800 MB
source /scratch2/agustin/miniforge3/etc/profile.d/conda.sh && conda activate merlin-dev
unset LDFLAGS CFLAGS CXXFLAGS CPPFLAGS      # <- see below
bash setup.sh --skip-toolchain
```

**`unset LDFLAGS` is load-bearing.** Activating the conda env exports
`LDFLAGS=-L$CONDA_PREFIX/lib …`; CMake captures it into
`CMAKE_{EXE,MODULE,SHARED}_LINKER_FLAGS` at configure time, and every RISC-V link
then resolves `-lstdc++` to the **x86** one:

```
ld.lld: error: .../libstdc++.so is incompatible with elf64-littleriscv
```

If a build directory already has the poisoned flags, strip them without
rebuilding everything:

```bash
cmake . -DCMAKE_EXE_LINKER_FLAGS="-Wl,--gc-sections" \
        -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--gc-sections" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--gc-sections"
```

Also build with assertions off, or every measurement is ~26% high:

```bash
cmake . -DIREE_ENABLE_ASSERTIONS=OFF && ninja iree-benchmark-module \
    merlin_dispatch_scheduler merlin_baseline_async
```

> The build type is already `Release`; IREE re-enables assertions on top, which
> is what triggers google-benchmark's "Library was built as DEBUG" warning.
> Same dispatch, same core: **80.0 µs with assertions, 63.2 µs without.**

---

## 2. Compile models

```bash
export MERLIN_DIR=/scratch2/agustin/XPU-RT/merlin      # <- see below
bash runtime/scripts/compile_all_models.sh
for d in $(find gen/vmfb -path "*spacemit_x60*" -name "*_dispatch_graph.dot"); do
    python3 runtime/scripts/dot_dispatch_parser.py "$d" --json-out "${d%.dot}.json"
done
```

**`MERLIN_DIR` must be set explicitly** if a sibling `../merlin` exists. The
wrapper prefers the sibling over the submodule, and will silently compile with
that checkout's pip-installed `iree-compile` — producing bytecode 16.0 modules
that the runtime we just built (17.0) refuses to load.

`pydot` is required by the DOT parser and is not in the env by default.

---

## 3. Profile per dispatch, per core

```bash
python3 runtime/scripts/profile_k1.py --cpu-ids 0 --reps 10 \
        --models mlp,dronet --hw RVV,scalar                    # cluster 0
python3 runtime/scripts/profile_k1.py --cpu-ids 4 --reps 10 \
        --models mlp,dronet --hw RVV,scalar --hw-label-suffix _c1   # cluster 1
```

Writes two files per cell:
- `results.csv` — the existing IREE-shaped schema, **median** in `mean_time`, so
  `profile_loader.py` needs no change.
- `profile.jsonl` — every sample, plus median/p90/p99/min/max/CV, cpu ids,
  cluster, clock.

The cluster goes in the **hw label**, not the topo tag: XPU-RT keys profiles off
combination *size* (`topo_0` for one core, whichever core), so encoding the
cluster in the tag would make the profile unfindable.

### What the profiles say

| | |
|---|---|
| MLP, 5 dispatches | 335.7 µs — every dispatch 63-78 µs **regardless of work**; ~94% is launch overhead |
| DroNet, 19 dispatches | **113.7 ms** on one core; top-5 dispatches are 75% of it |
| cluster 1 vs cluster 0 | **0.996** median ratio on compute-bound dispatches — the clusters are equivalent for RVV |
| IME vs RVV (DroNet) | **1.079** — IME is 7.9% *slower* overall; it wins on exactly 2 of 19 dispatches |

---

## 4. Schedule

```bash
python3 scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_k1_mlp_dronet.json \
    --solver greedy --profiled
```

Two settings in that config decide what the model means:

- `"machine_combination_mode": "singletons"` — every core independently
  schedulable, so 8 dispatches genuinely run at once. The alternative,
  `"prefix"`, makes a *cluster* one resource that may be given up to 4 cores,
  and then at most 2 dispatches run concurrently on the whole board.
- `"topo_tag": "topo_0"` with `topo_tag_override` — singletons **must** be paired
  with single-core profiles. A 4-hart timing here would credit each core with the
  whole cluster's throughput.

IME is expressed as an implementation *on cluster-0 cores*, never as a separate
machine — see `xpu-rt/capabilities.py`. Modelling it as `{"ime": 4}` extra
machines produces schedules that cannot run: the IME machine is busy while the
core it executes on is still marked idle.

---

## 5. Execute on the board

```bash
tar czf - gen/vmfb/*/spacemit_x60 | ssh k1 'tar xzf - -C /root/mb_k1/'
scp schedules/scheduled_networks_k1_mlp_dronet_greedy_profiled.json k1:/root/mb_k1/schedule.json
ssh k1 'cd /root/mb_k1 && ./bin/merlin-dispatch-scheduler schedule.json local-task 1 1 0 \
    --vmfb_dir=/root/mb_k1 --cpu_p_cpu_ids=0 --cpu_e_cpu_ids=4 --visible_cores=8 \
    --variant_p=RVV --variant_e=RVV --trace_csv=/root/mb_k1/trace.csv'
scp k1:/root/mb_k1/trace.csv artifacts/k1_run/
```

**Pin to as many cores as the profiles used.** With `--cpu_p_cpu_ids=0,1,2,3`
against single-core profiles, dispatches run 3.75× faster than planned
(18.29 ms planned, 4.88 ms actual) and every calibration number is meaningless.

**Caveat:** `CPU_P#2` is parsed and recorded but **not enforced** — the runner
dispatches to a per-cluster worker pool and IREE's local-task picks the core
within it. Treat intra-cluster placement as unpinned.

---

## 6. Predicted vs actual

```bash
python3 scripts/join_k1_trace.py \
    --schedule schedules/scheduled_networks_k1_mlp_dronet_greedy_profiled.json \
    --trace artifacts/k1_run/trace.csv \
    --out-json artifacts/k1_run/predicted_vs_actual.json
```

Joins on the stable dispatch key, never array position. Result on the reference
run:

| | |
|---|---|
| service-time error, all | +14.1% median |
| service-time error, ≥1 ms | **+4.8%** median |
| **queueing share of elapsed time** | **87.6%** |
| periodic instances missing their window | 42 / 42, worst 685 ms late |

The systematic **+17-25%** on the large convolutions is not profile error: solo
profiles were taken one dispatch at a time, and the run has both clusters busy.
That gap is contention, measured.

---

## 7. Advice, and applying it

```bash
python3 scripts/emit_compile_advice.py \
    --schedule schedules/scheduled_networks_k1_mlp_dronet_greedy_profiled.json
python3 scripts/apply_compile_advice.py \
    --schedule schedules/scheduled_networks_k1_mlp_dronet_greedy_profiled.json \
    --advice artifacts/k1_run/compile_advice.json \
    --out schedules/scheduled_k1_advice_applied.json
```

`compile_advice.json` is the contract; the `rationale` string is a courtesy
field. Every item carries the evidence that produced it, and `unchanged` items
record negative results so a later round does not re-propose them.

Measured, baseline vs advice-applied, same board, same pinning:

| | before | after | delta |
|---|---|---|---|
| retargeted dispatches (50) | 21 991 µs | 16 172 µs | **−26.5%** |
| all dispatches | 1 229 482 µs | 1 231 831 µs | +0.2% |

Predicted −21.2%, measured −26.5%. **And it does not matter at system level** —
those dispatches are 1.8% of service time. Accepted per-dispatch, rejected as a
system-level win. Executing the schedule is what tells you the difference; a
kernel benchmark cannot.

---

## 8. The ModelBlaster path

```bash
export MODELBLASTER_K1_HOST=k1
export CROSS=/scratch2/agustin/XPU-RT/merlin/build_tools/riscv-tools-spacemit/\
spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-
bash ModelBlaster/scripts/run_model_k1.sh mlp_control int8 scalar 0
```

extract → skeleton (`--platform linux`) → kernels → build → deploy → run →
verify → profile, in one command. `--platform linux` is what swaps `rdcycle`
for `rdtime`; without it the binary SIGILLs on its first timed dispatch.

Reference result: `max_abs_err=0` (bit-exact), 7 dispatches, 9028 rdtime ticks
(376 µs). Note the smallest dispatch costs **62 ticks ≈ 2.6 µs** against IREE's
~63 µs floor — the per-dispatch overhead that dominates the IREE MLP is a
property of that runtime, not of the hardware.

### Kernel generation is Codex-only

```bash
export LLM_PROVIDER=codex
export CODEX_CALLS_LOG=artifacts/k1_run/codex_calls.jsonl
BACKEND=llm bash ModelBlaster/scripts/run_model_k1.sh ...
```

There is no fallback from Codex to Bedrock, by design and by test
(`ModelBlaster/tests/test_codex_provider.py`). If Codex is unavailable the
kernel step fails loudly and the caller falls back to reference/curated
kernels — deterministic artifacts already in the tree, not another model.
The call log records `provider: codex`; note it deliberately does **not** reuse
`bedrock_client._append_call_log`, which hardcodes `"provider": "bedrock"`.

#### Compare against the best kernel you already have, not the reference

The first Codex kernel (`artifacts/k1_run/codex/`) was bit-exact on the board and
**4.48× faster than the scalar reference** — a number it would be easy to report
as a win. Against `kernels/rvv/rvv_linear_s8_direct.c`, which was already in the
tree, it is **41% slower**:

| `linear_s8` total, rdtime ticks | scalar ref | curated RVV | Codex RVV |
|---|---|---|---|
| | 8095 | **1280** | 1807 |

Accept requires correctness **and** an improvement in the selected metric. It is
archived, not promoted. Always run the incumbent on the same board in the same
conditions before claiming a generated kernel is an improvement.

---

## 9. Feeding advice back into ModelBlaster

```bash
python3 scripts/advice_to_fusion_hint.py \
    --advice artifacts/k1_run/compile_advice_mlp_control.json \
    --ir <gen>/graph.json --model mlp_control \
    --out <gen>/fusion_hint.json --pair-only
python3 -m modelblaster.pipeline.apply_fusion_hint \
    --hint <gen>/fusion_hint.json --model mlp_control \
    --ir <gen>/graph.json --out <gen>/graph.fused.json
```

The scheduler never edits C: it emits advice, the adapter translates the
actionable subset into ModelBlaster's own `modelblaster.fusion_hints/v1`, and
`apply_fusion_hint` does a pure JSON-in/JSON-out graph rewrite. mlp_control goes
from 7 ops to 4 (three fused `linear_s8+elu_s8` pairs plus the tail linear).

Note the evidence only appeared **after** the previous round: once the curated
RVV linear kernel landed, the elu ops went from noise to **39.7%** of runtime.
Fusion advice that would have been wrong at round 0 is right at round 1, which
is the argument for closing the loop rather than optimising once.

---

## Artifacts

```
artifacts/k1_bringup/<ts>/     board inventory, topology, IME probe + source
artifacts/repo_reconciliation.md
artifacts/k1_progress.md       running log: what was tried, what failed, why
artifacts/k1_run/              traces, predicted-vs-actual, compile_advice.json
gen/profile/<hw>/spacemit_x60/<model>/<basename>/topo_0/{results.csv,profile.jsonl}
```

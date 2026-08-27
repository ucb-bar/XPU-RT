# K1 closed-loop bring-up — running log

Plan: `~/.claude/plans/gleaming-exploring-floyd.md`
Repo SHAs at start: XPU-RT `e70a4b4` on `feat/freshness-validity-eval`;
merlin `4582be0`; zephyr-chipyard-sw `418136f`; ModelBlaster submodules uninitialized.

---

## M0 — Board contact — **DONE** (2026-08-26)

Artifacts: `artifacts/k1_bringup/20260826-203613/`

### Attempted / commands
- Probed both candidate UARTs with a pyserial script (`ime_probe`-style listen + poke).
- `ssh k1 ...` for the whole inventory once SSH was working.
- Cross-compiled and ran `ime_probe.c` on the board.

### Passed
- **Console found:** `/dev/ttyUSB4` (CH340 `1a86:7523`) @115200 8N1 — landed on a live
  `root@k1` shell. `/dev/ttyUSB0` (CP2102N) is a different board.
- **SSH works:** `ssh k1` → `10.44.98.236` (wlan0), root, key `/scratch2/agustin/DIMA_SLICE`
  already in `/root/.ssh/authorized_keys`. Stanza added to `~/.ssh/config` (backup taken).
- **Hardware:** Spacemit(R) X60, 8×riscv64, Bianbu 3.0, Linux 6.6.63, glibc 2.41.
- **Cluster split proven** via L2 sharing: cores 0-3 → L2#0, cores 4-7 → L2#1 (512K each).
- **IME measured, not assumed:** `smt.vmadot` executes on cores 0-3, SIGILLs on cores 4-7.
  Reproducible ×3. RVV works on all 8 cores.
- **Clock:** single cpufreq policy over all 8 cores, governor already `performance` at a
  fixed 1.6 GHz. ⇒ `PROFILE_CLOCK_MHZ=1600`, and no governor change was needed or made.
- **Cross toolchain acquired:** `riscv64-unknown-linux-gnu-gcc 14.3.0` under
  `merlin/build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2`.
- **First code run on the K1** (the M2 sanity gate, done early): static probe binary
  deployed via scp and executed.

### Failed / surprises
- **SSH hung for ~20 min of debugging** at `Next authentication method: publickey`. Cause was
  a **stale ssh-agent on this host** (`ssh-add -l` itself hangs), not the board. Fixed with
  `IdentityAgent none` in the config stanza. Same root cause affects `git` over SSH.
- **Board has no compiler at all** — no gcc/g++/cc/clang/cmake/make/perf; only python3.
  Bianbu apt offers gcc 14.2 / build-essential if we ever want native builds.
  ⇒ cross-compilation is the default path.
- **SpaceMiT GCC 14.3.0 rejects `-march=...xsmtvdot`** ("unsupported non-standard
  extension"). IME through GCC therefore needs raw `.insn`; the IREE/LLVM path
  (`--iree-llvmcpu-target-cpu-features=+xsmtvdot`) is a different compiler and is untested
  so far. Relevant to M4.
- Load average is a steady 2.00 on an idle board: two vendor kernel threads (`vq0`,`vq1`)
  wedged in D-state. Zero CPU. **Benign for profiling** — do not chase.

### Blocker (needs the user)
- **No push access to `ucb-bar/zephyr-chipyard-sw`.** Both HTTPS and SSH return
  `Permission ... denied to copparihollmann` (auth succeeds, authorization fails). The 3
  at-risk profile-DB commits are therefore secured **locally only**, as a verified bundle:
  `artifacts/repo_reconciliation/zephyr-chipyard-sw-*.bundle` (721 MB, "records a complete
  history"). Needs write access, or a fork, before they are durably safe.

### Next
- M1: fetch, rescue-verify, branch `feat/k1-modelblaster-closed-loop`, rebase onto fresh
  `origin/dev`, resolve the ModelBlaster three-way.

---

## M1 — Repo reconciliation — **DONE** (2026-08-26)

Full detail: `artifacts/repo_reconciliation.md`

### Passed
- Safety net first: verified 721 MB zephyr bundle + 177 MB XPU-RT bundle + annotated tag
  `premerge-k1-20260826` on the pre-rebase HEAD. No reset, no clean, no `--remote`.
- `git fetch --all --prune` — `origin/dev` moved `fe6feca..587f96f` (6 commits), and brought
  `docs/mlp_dronet_yolo_spike_reproduction.md` (59 KB) with it. **Open question #4 resolved:**
  the doc exists; it simply post-dated our last fetch.
- Branch `feat/k1-modelblaster-closed-loop` created and **rebased onto current `origin/dev`**.
  Now 42 ahead / 0 behind. All 42 original commit subjects verified present.
- Rebase picked up upstream's `06c187f Fix XPU-RT scheduler bugs; ...` — the main payoff.
- **363 tests pass, 1 skipped** after the rebase.
- Submodule worktree verified still at `418136f` on its rescue branch afterwards.

### Failed / surprises
- Untracked `xpu-rt/xpurt.egg-info/*` blocked the rebase at commit 2/43 (pip editable-install
  output). Backed up and removed.
- **One genuine conflict**, `xpu-rt/plot.py`: upstream and we had independently fixed the same
  FreeType raster-overflow bug. Theirs closed the figure; ours scaled dpi by op count and
  re-raised. Resolved by **combining both**, not choosing — ours leaked a matplotlib figure
  per sweep cell.
- The `zephyr-chipyard-sw` gitlink **cannot be stashed** — it reflects the submodule's actual
  HEAD. Rebased with `-c diff.ignoreSubmodules=all` instead.

### Blocker (unchanged, needs the user)
- Still no push access to `ucb-bar/zephyr-chipyard-sw`, and no fork exists at
  `copparihollmann/zephyr-chipyard-sw`. `gh` is not installed on this host.
  The 3 recalibration commits remain bundle-only.
- The parent gitlink bump was deliberately **not committed** — recording a SHA that exists on
  no remote would make the fragility permanent. Do it after the fork push.

### ModelBlaster three-way — RESOLVED
Investigated rather than assumed, and the answer inverted the brief's premise:

| Candidate | Relationship |
|---|---|
| `main` @ `dbbdcf0a` (what the submodule recorded) | 143 commits **behind** the live tree |
| nested `79328776` | strict **ancestor** of `dbbdcf0a`, 85 further behind — never a divergence |
| live `/scratch2/agustin/ModelBlaster` @ `3da8192` | 143 ahead of main, 2 behind |

Decisive detail: **`pipeline/apply_split_hint.py` exists only on the live branch, not on
`main`** — that is the split-feedback machinery M7 is built on, and which `profile_loader`
already understands via the `<orig>.tile_<i>` convention. `main` also lacks the v26 sharding
proof-of-concept (`l0.conv` split, -48.6% wall on yolov8) and the FireSim-validated kernel
restorations.

Action taken: branched `feat/k1-xpurt` off `3da8192`, merged `origin/main` into it
(**zero conflicts**; brings the KernelBlaster submodule + XPU-RT wiring), pushed to
`ucb-bar/ModelBlaster` as `a0310c6`, initialized `XPU-RT/ModelBlaster` there, and committed
the gitlink bump (`52fc103`). The SHA is on the remote, so this pointer is safe to record —
unlike the zephyr one, which is why that one was left uncommitted.

### Next
- M2: init `merlin/third_party/iree_bar`, `setup.sh` (toolchain already fetched), compile the
  three models for `spacemit_x60`, profile on the K1.

---

## M4 recipe correction (found early, while M2 builds)

The plan took the IME compile flags from
`merlin/benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh`. That script is
**not** the working recipe, and two of its assumptions are wrong:

1. It expects `third_party/iree_bar/tests/e2e/SpacemiT/matmul_i8_2048.mlir` and
   `check_hotloop_asm.py`. **Neither exists** at the pinned IREE SHA (`2b7dd40a`). Only
   `compile_{opu,rvv,vmadot}.sh`, `matmul_{i8,q_i8,fp8}.mlir`, `conv_{i8,q_i8,fp8}.mlir` do.
2. Its flags disagree with the recipe that actually works.

The authoritative recipe is `third_party/iree_bar/tests/e2e/SpacemiT/compile_vmadot.sh`:

| Flag | benchmark script (wrong) | `compile_vmadot.sh` (use this) |
|---|---|---|
| `--iree-llvmcpu-enable-ukernels` | `all` | **`none`** |
| `--iree-llvmcpu-enable-vector-contract-custom-kernels` | `false` | **`true`** |
| `--iree-codegen-mmt4d-use-intrinsics` | absent | **present** |
| preprocessing | absent | **img2col + quantized-conv-to-conv** |
| verification | `check_hotloop_asm.py` (missing) | **`llvm-objdump -d --mattr=+...+xsmtvdot \| grep vmadot`** |

So IME goes through **vector.contract custom kernels + mmt4d intrinsics**, not ukernel
bitcode. The M4 acceptance gate becomes the objdump grep, which needs
`<build>/llvm-project/bin/llvm-objdump` from the merlin build.

Note `compile_vmadot.sh` hardcodes paths into the *sibling* checkout
`/scratch2/agustin/merlin` (build dir `build/vanilla/host/debug/iree-spacemit-3.10.0.dev`),
which is a different merlin version (`33a57a0c`) from the XPU-RT submodule (`4582be0`) and a
different build-dir layout. Do not assume the path; resolve it from whichever build we make.

---

## M3a — K1 resource model — **partly done** (2026-08-26, while M2 builds)

Commit `0e55e93`. New `xpu-rt/capabilities.py` + `xpu-rt/tests/test_k1_capabilities.py`
(22 tests). Full suite now **385 passed, 1 skipped**.

### What landed
- `K1_CAPABILITIES` — the measured table: `CPU_P` (cluster 0) = scalar/rvv/**ime**,
  `CPU_E` (cluster 1) = scalar/rvv. Sourced from the SIGILL probe, not from a datasheet.
- `check_implementation_legality` — rejects `ime` on `CPU_E` **before** scheduling, reporting
  every violation at once. Necessary because an IME kernel on cluster 1 traps rather than
  degrading.
- `build_machine_combinations_with_impls` — implementation alternatives as combinations that
  **share a machine set**, so `combinations_overlap` (set intersection) serialises them with
  no solver change. A single-implementation config reproduces the existing
  `build_machine_combinations` byte for byte.
- `core_ids_for_combination` — `CPU_E#1` → physical core **5**, for affinity masks. Cluster 1
  does not start at 0, and the runtime needs physical ids.
- A test that pins the *wrong* design (`{"ime": 4}` as separate machines) and shows it lets
  the scheduler double-book one physical core.

### Finding: a cluster is ONE resource, not four
`build_machine_combinations` emits **cumulative prefixes** — `['CPU_P#0']`,
`['CPU_P#0','CPU_P#1']`, … and deliberately never `['CPU_P#1']` alone. So every cluster-0
combination contains `CPU_P#0`, every pair intersects, and **at most one dispatch occupies
cluster 0 at any instant**, however many cores are declared.

`machines: {cpu_p: 4}` therefore means *"one dispatch may use up to 4 cores"*, **not**
*"four dispatches may run side by side"*. Concurrency comes from machine **kinds**, so on the
K1 the genuinely-parallel dispatch count is **2** (one per cluster).

This is self-consistent — the profile tree stores `topo_0_1_2_3` as the 4-hart time for a
single dispatch — and it matches how the existing FireSim configs are written
(`{"cpu_p": 1, "cpu_e": 1}` plus `topo_tag_override`). But it is a real ceiling for a periodic
MLP + DroNet + YOLO workload, where the natural reading of "8 cores" is 8-way concurrency.
Captured as `test_combinations_are_cumulative_prefixes_so_a_cluster_is_ONE_resource` so it is
a deliberate choice rather than a surprise mid-experiment.

**Decision needed before M6** (not blocking M2): accept 2-way cluster-level concurrency, or
extend the combination builder to emit independent per-core combinations. The latter is a
contained change — the overlap machinery already handles it — but it multiplies the
combination count and needs per-core (not per-cluster) profiles.

---

## M2 — Toolchain + runtime — **runtime half DONE** (2026-08-26)

### Passed
- SpaceMiT cross toolchain: `riscv64-unknown-linux-gnu-gcc 14.3.0`.
- `merlin/third_party/iree_bar` initialized recursively (5.0 GB incl. llvm-project),
  referenced off the sibling clone to save a full re-download.
- **Host `iree-compile` built** (`merlin/build/host-vanilla-release/install/bin/iree-compile`).
- **Both SpaceMiT runners cross-compile and run on the K1**:
  `merlin-dispatch-scheduler` and `merlin-baseline-async`, deployed to
  `/root/mb_k1/bin/`, verified executing on the board.

### Three build breakages, each at a different stage (merlin `d07b116`)
1. **Configure** — `iree_runtime_plugin.cmake` pointed `add_subdirectory` at
   `projects/xpu-rt`, which does not exist and is untracked. History: `bf8483d` moved xpu-rt
   into `projects/`, then **`4582be0` — the commit the submodule pins** — moved it to
   `samples/common/xpu-rt` and left the reference behind. `samples/CMakeLists.txt`
   deliberately does not add `common/xpu-rt`, so this file is its only entry point.
2. **Compile** — both sample `main.c` include `iree/base/tooling/flags.h`; the pinned IREE has
   no `base/tooling` directory at all. `iree/base/internal/flags.h` provides exactly the two
   symbols used.
3. **Link** — not a source bug. Activating the `merlin-dev` conda env exports `LDFLAGS` with
   `-L$CONDA_PREFIX/lib`; CMake captured it into `CMAKE_{EXE,MODULE,SHARED}_LINKER_FLAGS` at
   configure time, so every RISC-V link resolved `-lstdc++` to the **x86** one
   (`libstdc++.so is incompatible with elf64-littleriscv`). Fixed by reconfiguring with the
   conda paths stripped. **Anyone configuring a target build from an activated conda env will
   hit this.**

### Resource model updated per decision
`capabilities.py` gained `granularity="prefix" | "per_core"`; the K1 will use **per_core**
(true 8-way concurrency), which requires **single-core (`topo_0`) profiles**, not the 4-hart
`topo_0_1_2_3` ones. A test pins that pairing. Suite: **392 passed, 1 skipped**.

### Next
- Compile MLP / DroNet / YOLOv8n int8 for `spacemit_x60` × {scalar, RVV} and profile
  single-core on the board.

---

## M2/M4 — First real K1 measurements — **DONE for MLP + DroNet** (2026-08-26)

### A measurement bug caught before it poisoned anything
The first numbers came out of a runtime built with `IREE_ENABLE_ASSERTIONS=ON` (the build
type *was* Release, but IREE re-enables assertions on top, which is what triggers
google-benchmark's "Library was built as DEBUG" warning). Same dispatch, same core:

| | median |
|---|---|
| assertions ON | 80.0 us |
| assertions OFF | **63.2 us** |

**~26% inflation on every dispatch.** Rebuilt with `-DIREE_ENABLE_ASSERTIONS=OFF`; all
recorded numbers are from that build.

### A second one: the wrong compiler
`compile_all_models.sh` resolves `MERLIN_DIR` to a *sibling* checkout before the submodule
(`REPO_ROOT/../merlin`), so the first compile silently used
`/scratch2/agustin/merlin/.venv/bin/iree-compile` — a pip-installed IREE emitting bytecode
16.0, which the runtime we built (17.0) refuses to load. Fixed by setting `MERLIN_DIR`
explicitly. **Always pin `MERLIN_DIR` when a sibling merlin exists.**

### MLP (5 dispatches, cluster 0 core 0, RVV vs scalar)
Total 335.7 us vs 357.4 us — RVV buys only **1.06x**. The reason is visible in the data:
every dispatch costs 63-78 us regardless of what it does (an `elementwise_10` costs the same
as a matmul), so MLP is **entirely dispatch-overhead-bound**, with a floor around 63 us.
That is a direct argument for *fusing* MLP dispatches, and it is exactly the kind of
evidence the granularity advisor should be turning into a recommendation.

### DroNet (19 dispatches, cluster 0 core 0, RVV)
**113.7 ms total on one core.** Top-5 dispatches are 75.3% of it:

| dispatch | median | op |
|---|---|---|
| 1 | 22.87 ms | conv_32x56x56x3x3x3 |
| 13 | 18.33 ms | conv_128x4x4x128x3x3 |
| 5 | 15.80 ms | conv_32x14x14x32x3x3 |
| 4 | 14.68 ms | conv_32x14x14x32x3x3 |
| 9 | 14.00 ms | conv_64x7x7x64x3x3 |

CV is 0.2-1.4%, so these are stable. **113.7 ms single-core cannot meet a 33.3 ms period**
— that is the real schedulability question this workload poses, and it is measured rather
than assumed.

### IME: works, and correctly does not apply to MLP
`verify_ime_build.sh` disassembles and counts `vmadot`, because `+xsmtvdot` does **not**
guarantee the instruction is emitted — IREE falls back to RVV silently when shapes do not
match the micro-tile, producing a binary labelled IME with no IME in it.

| build | vmadot | why |
|---|---|---|
| IREE's reference `matmul_i8.mlir` | 1 | toolchain proven working |
| `dronet.q.int8` | 1 | conv → img2col → real GEMM |
| `mlp.q.int8` | **0** | every matmul is `1xNxK` — M=1, GEMV |

MLP's zero is **correct**: its dispatches are literally `matmul_1x32x10`,
`matmul_1x32x32`, `matmul_1x2x32`. A 4x4x8 matrix micro-tile has nothing to bite on. So IME
is architecturally inapplicable to this MLP, which is a finding, not a failure.

---

## M6/M7 — Schedule executed on hardware, and one closed loop measured (2026-08-27)

### Predicted vs actual, 360/360 dispatches joined by stable key

| | |
|---|---|
| service-time error, all dispatches | **+14.1%** median |
| service-time error, dispatches >=1 ms | **+4.8%** median |
| queueing share of elapsed time | **87.6%** |
| periodic instances missing their window | **42 / 42**, worst 685 ms late |

The systematic **+17-25%** on the large convolutions is not profile error. Solo profiles
were taken one dispatch at a time; the run has both clusters busy. That gap **is** the
contention term, measured rather than modelled.

Two runner limitations had to be fixed to get here, and one remains: the runner dispatches
to a per-*cluster* worker pool, so `CPU_P#2` is parsed and recorded but **not enforced** —
IREE's local-task picks the core. Intra-cluster placement must be treated as unpinned until
that is wired through. The first run made this obvious: with a 4-core pool against
single-core profiles, dispatches ran **3.75x faster than planned** (18.29 ms planned vs
4.88 ms actual). All calibration numbers above are from a run pinned to one core per
cluster, matching the profiles.

### The advisor's output on real data
`compile_advice.json`: 30 items, 10 actionable. It gets the shape right:
- 5 x `split` on DroNet convs exceeding the 10 ms inter-release slot (22.9 ms worst)
- `fuse_with_successor` on MLP — **~94% of its 336 us is per-dispatch launch overhead**
- `choose_implementation` where measurement supports it: IME for `dronet.14` (-25.8%),
  **scalar** for `dronet.16` (-23.2%) and `dronet.18` (-19.1%)
- 20 x `unchanged`, recording negative results so a later round does not re-propose them

### IME: a negative result, kept
IME is **7.9% slower than RVV overall** for DroNet (122.7 ms vs 113.7 ms). Only one `vmadot`
was emitted in the whole binary, so the build pays different data-tiling costs without the
benefit. Per dispatch it wins exactly twice. An advisor that could only ever say "use the
accelerator" would be useless; this one says where it helps and where it does not.

### The loop, closed and measured on hardware

| | before | after | delta |
|---|---|---|---|
| **retargeted dispatches** (50) | 21 991 us | 16 172 us | **-26.5%** |
| untouched dispatches | 1 207 491 us | 1 215 659 us | +0.7% |
| **all dispatches** | 1 229 482 us | 1 231 831 us | **+0.2%** |
| makespan | 1 014 813 us | 1 020 057 us | +0.5% |

Predicted -21.2% on the retargeted set; measured **-26.5%**. The advice was right, and the
hardware beat the prediction.

**And it does not matter at system level.** Those 50 dispatches are 1.8% of total service
time, so a 26% local win is ~0.5% globally — inside run-to-run noise. By the accept/reject
criteria this change is **accepted per-dispatch** (correctness holds, local gain confirmed
on hardware) and **rejected as a system-level improvement**. That distinction is the whole
point of executing the schedule instead of trusting the kernel benchmark.

The advisor already named the actions that would matter — `split` the 22.9 ms conv, `fuse`
MLP's 94% overhead — and both change the dispatch graph, which is compiler-front-end work
rather than a post-pass over a finished schedule.

### Testing the advisor's other recommendation: MLP fusion
The advisor said ~94% of MLP's 336 us is per-dispatch launch overhead. The obvious
compiler-side levers were tried and **none of them work**, which is itself the finding:

| build | dispatches | total median |
|---|---|---|
| RVV baseline (data-tiling on) | 5 | 335.7 us |
| RVV + data-tiling off + detensoring | 5 | **352.3 us** (+4.9%) |

Dropping data-tiling did remove the five `_encoding_*` modules, but the dispatch **count** is
unchanged — the three matmuls are sequentially dependent with different shapes, so IREE will
not merge them, and `enable-detensoring` / `fuse-horizontal-contractions` do not either.

That is exactly what "94% launch overhead" predicts: **no codegen flag can touch a cost paid
per dispatch launch.** The only lever is emitting fewer dispatches, which is a graph-level
transformation — ModelBlaster's `apply_fusion_hint`, not an IREE flag. This is the concrete
reason the ModelBlaster path (M5) matters rather than being an alternative front end.

---

## M8 — Baselines, and a calibration result that reframes everything (2026-08-27)

`scripts/k1_baselines.py` runs the ladder on the board and reports predicted and
measured side by side, never merged.

### The ladder as first run (8 scheduler machines, 2 physical cores)

| rung | pred service | meas service | meas makespan | queue% | misses |
|---|---|---|---|---|---|
| B0 static placement | 1 143 476 | 1 146 316 | 1 141 405 | 70.3 | 10 |
| B1 XPU-RT greedy | 1 143 476 | 1 228 329 | **1 013 698** | 87.6 | **41** |
| B2 + impl selection | 1 139 372 | 1 229 807 | 1 018 049 | 87.7 | 42 |
| P1 greedy_periodic | 1 143 476 | 1 229 536 | 1 014 015 | 87.6 | 41 |

Broken down by model, this looked damning for the scheduler:

```
B0: dronet 10/10 miss   mlp  0/32 miss
B1: dronet 10/10 miss   mlp 31/32 miss
P1: dronet 10/10 miss   mlp 31/32 miss
```

XPU-RT buys **11% makespan and loses 31 of 32 MLP deadlines**, and the
periodic-aware solver does not rescue it.

### Except that was measuring a configuration error, not a scheduler

The config declared `machines: {cpu_p: 4, cpu_e: 4}`, but
`merlin-dispatch-scheduler` executes on exactly **two** worker pools, pinned to
whatever `--cpu_p_cpu_ids` / `--cpu_e_cpu_ids` name. Running it with `0` and `4`
collapses **4 scheduler machines onto 1 physical core**. The 87.6% queueing was
that over-subscription, not the hardware.

Re-run with a self-consistent model — one machine per pool, one core per pool,
single-core profiles (`data/toplevel/networks_k1_2core.json`):

| | predicted | measured | error |
|---|---|---|---|
| service | 3 553 628 us | 3 547 101 us | **0.18%** |
| makespan | 2 838 340 us | 2 859 786 us | **0.75%** |
| queueing share | — | **1.8%** | (was 87.6%) |
| service error, dispatches >=1 ms | — | **-1.6%** median | (was +4.8%) |

**When the resource model matches what the runtime can honour, prediction is
accurate to well under 1%.** The earlier +14% service error and 87.6% queueing
were artifacts of the mismatch. This is why the plan insisted on fixing
prediction before drawing scheduler conclusions — every conclusion from the
8-machine rungs above is about a configuration that cannot exist on this runner.

**Open, and the single most valuable next fix:** teach the runner to honour
per-dispatch core placement (`CPU_P#2`), which it currently parses and discards.
Until then the K1 can be scheduled as 2 resources faithfully, or as 8 resources
only in simulation. Everything needed on the XPU-RT side is already in place —
`capabilities.py` emits per-core combinations and `machine_combination_mode`
selects them.

---

## Codex kernel generation — correct, and rejected on merit (2026-08-27)

`LLM_PROVIDER=codex`, model `gpt-5.6-sol`, prompt sha256[:16] `a10e56e72849432c`,
19 818 in / 9 803 out tokens, logged with `provider: codex`. Full detail and the
source in `artifacts/k1_run/codex/`.

The kernel is good work: `vwmacc_vv_i32m4` over the contiguous reduction plus
`vredsum`, exactly the right shape for an M=1 GEMV. It compiles clean and runs
**bit-exact** on the board (`max_abs_err=0`).

| `linear_s8` total | scalar ref | curated RVV | Codex RVV |
|---|---|---|---|
| rdtime ticks | 8095 | **1280** | 1807 |

**Against the scalar reference it is 4.48x faster.** Against
`kernels/rvv/rvv_linear_s8_direct.c`, which already exists in the tree, it is
**41% slower**. Accept requires correctness *and* an improvement in the selected
metric; only the first holds, so it is archived rather than promoted, and the
curated kernel stays in place.

Worth recording that the first framing of this result compared Codex against the
*scalar* baseline and read as a 3.4x win. The comparison that decides the
question is against the best kernel already available, and it inverts the
conclusion.

It loses only on the two large-K calls and wins slightly at K=16, which points
at reduction blocking as the specific thing a second round should address.

---

## Round 1 through ModelBlaster: advice → fusion → Codex → measured → rejected

The full loop, end to end, on `mlp_control`:

1. Measured profile on the board says elementwise `elu_s8` is **39.7%** of runtime
   — a share that only appeared *after* the curated RVV linear kernel landed.
2. `compile_advice_mlp_control.json` recommends `fuse_with_successor`, with that
   evidence attached.
3. `scripts/advice_to_fusion_hint.py` translates it into
   `modelblaster.fusion_hints/v1`.
4. `apply_fusion_hint` rewrites the IR: **7 ops → 4**.
5. The fused op `linear_s8_elu_s8` has no RVV kernel, so **Codex** generates one
   (`gpt-5.6-sol`, prompt `bd46d6231ff1e79c`, 20 108 in / 7 897 out).
6. Built, deployed, run on the K1: **PASS**, correct.

| configuration | ticks | vs best | vs round 0 |
|---|---|---|---|
| reference scalar (round 0) | 8965 | 4.22x | 1.00x |
| **curated RVV linear, unfused** | **2122** | **1.00x** | **4.22x** |
| fused, reference kernel | 2986 | 1.41x | 3.00x |
| fused, Codex RVV kernel | 2892 | 1.36x | 3.10x |

### Verdict: REJECT the fusion round

Correct on hardware and still **36% slower** than the unfused incumbent. Two
reasons, both visible in the artifacts: the elu tail needs a per-element `expf`
that does not vectorise, so fusing buys one avoided tensor write and pays for a
scalar tail; and the Codex fused kernel never emits `vwmacc` (its intrinsic set
is `vle8`/`vmv`/`vsetvl` only), so it vectorises the reduction less effectively
than the curated linear kernel it replaced.

The advice was **well-founded** — 39.7% in elementwise ops is exactly the kind of
thing that should be investigated — and the outcome is still a rejection. That is
the loop working: the recommendation is a hypothesis, the board is the referee,
and both rounds so far have been rejected on measurement rather than accepted on
plausibility.

**Best configuration on the board remains the unfused curated RVV build: 2122
ticks = 88.4 us at 24 MHz, 4.22x faster than the round-0 reference.**

---

## Three-model set complete on the physical K1 via ModelBlaster (2026-08-27)

All bit-exact (`max_abs_err=0`), pinned to core 0, scalar reference kernels,
profiles filed under `gen/profile_mb/scalar/spacemit_x60/<model>/`:

| model | rdtime ticks | wall @24 MHz | verify |
|---|---|---|---|
| mlp_control | 9 028 | 0.38 ms | PASS |
| dronet | 3 777 286 | 157.4 ms | PASS |
| yolov8_nano | 96 503 982 | 4021 ms | PASS (75 600 outputs) |

YOLOv8n was the model missing from the merlin/IREE path — merlin has no YOLO
under `models/`, while ModelBlaster carries it as a PyTorch definition. So the
three-model set the brief asks for exists on the board through the ModelBlaster
path, and MLP + DroNet additionally through merlin/IREE.

These are scalar-kernel numbers and should be read as a correctness and
plumbing result, not a performance one — the curated RVV kernels are a 4.2x
improvement on mlp_control alone.

### A bug this surfaced
`report_run` derived the model name by walking three directories up from
`io.npz`, which mislabels any layout other than
`examples/<model>/<quant>/generated/`. The K1 flow roots its build at
`build/k1/<model>/<quant>/`, so every profile was filed under a model called
**`k1`** — wrong tree, no error, and unfindable by the model it describes. Fixed
by letting the runner state the name (`--model-name`), with the walk kept as the
default for existing callers.

---

## Per-core execution: placement now honoured, concurrency still 2

`--pin_per_core` (merlin `46fe201`) creates one pinned local-task device per
physical core and routes each dispatch to the device matching the `CPU_P#N`
index the schedule assigned. Previously that index was parsed and discarded.

| | before | after |
|---|---|---|
| service-time error, all dispatches | +14.1% | **+11.3%** median |
| queueing share | 87.6% | **87.7%** |

**Placement is now faithful; concurrency is not.** The runner still uses two
long-lived worker threads, one per hardware target, so at most two dispatches
are in flight regardless of how many devices exist. That, not the device
pinning, is what the 87.7% queueing measures.

So the K1 can currently be scheduled faithfully in two configurations:
- **2 resources** (one machine per worker pool) — self-consistent end to end,
  and predicted-vs-actual lands within **0.18%**.
- **8 resources with per-core placement** — placement is executed correctly,
  but the runtime serialises to 2-way, so the schedule's concurrency assumption
  is not met.

Closing the gap means N worker threads in `scheduler_runner.cc`, which is a
refactor of the dispatch loop rather than a config change. Everything upstream
of it is ready: `capabilities.py` emits per-core combinations,
`machine_combination_mode` selects them, and the runner now executes placement.

---

## Contention, measured under control — and it inverts the obvious assumption

`runtime/scripts/k1_contention.py` pins the dispatch under test to core 0 and runs
a co-runner on a chosen other core, comparing against the same dispatch with
nothing else running. The co-runner is a **different** benchmark module on
purpose: an identical one shares its weights, and a same-L2 co-placement then
looks *helpful* rather than contended (measured: 1.034x with the same module vs
1.088x with a different one on the same dispatch).

| dispatch | solo | co-runner same cluster | co-runner other cluster |
|---|---|---|---|
| dronet.0 | 0.581 ms | 1.088x | **1.298x** |
| dronet.10 | 0.416 ms | 1.103x | **1.312x** |
| dronet.11 | 0.222 ms | 0.995x | 0.937x |
| dronet.12 | 9.581 ms | 1.053x | **1.233x** |
| dronet.13 | 18.321 ms | 1.034x | **1.137x** |
| dronet.14 | 1.317 ms | 1.014x | 1.031x |
| **median** | | **1.043x** | **1.185x** |

**Co-running on the *other* cluster costs ~18%; on the *same* cluster ~4%.**
That is the opposite of what the resource model would suggest — cores 0-3 and
4-7 have separate 512K L2s, so spreading across clusters looks like the way to
avoid interference, and on this SoC it is roughly four times worse.

The magnitude also matches the unexplained gap in the first predicted-vs-actual
join: solo profiles ran 17-25% optimistic on the large convolutions during a run
with both clusters busy, and cross-cluster co-running measures 13-31% here.
So that gap was contention, and specifically *cross-cluster* contention.

**Confidence: moderate, mechanism unexplained.** Six dispatches, four
repetitions, one co-runner. Two of the six show no effect. A plausible story is
shared memory-controller or interconnect pressure dominating L2 isolation, but
this experiment does not establish it. Before this changes placement policy it
wants: more co-runner kinds, more repetitions, and a memory-bandwidth control to
separate interconnect pressure from cache effects.

If it holds, the scheduling consequence is concrete and contrarian: **prefer
packing concurrent work onto one cluster** rather than spreading it, which is
the opposite of the default intuition.

---

## Phases 14–15 (VitFly, SmolVLA): two of three VitFly ops landed

Both are explicitly post-first-milestone in the plan. Scoping them precisely so
the next person starts from facts rather than a guess.

### VitFly `LSTMNet` — 6 of 9 ops already exist

| op | ModelBlaster kernel | source line |
|---|---|---|
| conv2d | `conv2d_s8` | `conv1`, `conv2` |
| batchnorm2d | `batchnorm2d_s8` | `bn1`, `bn2` |
| relu | `relu_s8` | `F.relu` |
| maxpool2d | `maxpool2d_s8` | `maxpool` |
| linear | `linear_s8` | `fc1/fc2/fc3` |
| cat | `cat3_c1_s8` | `torch.cat` |
| avgpool2d | **MISSING** | `avgpool` |
| leaky_relu | **MISSING** (`leaky_relu` exists in fp32, no `_s8`) | `F.leaky_relu` |
| **LSTM** | **MISSING** | `LSTM(input=665, hidden=395, layers=2)` |

Two distinct pieces of work, and they are not the same size:

1. **`avgpool2d_s8` and `leaky_relu_s8`** — ordinary elementwise/pooling kernels.
   Each needs a `KernelSpec` plus a branch in `extract_graph.py`'s int8 module
   dispatch, which currently handles only BatchNorm2d, Conv2d, Dropout, ELU,
   MaxPool2d, Sigmoid, SiLU and Upsample. Small and well-understood.

2. **LSTM** — the actual project. `nn.LSTM` is a fused multi-gate module; the
   int8 extractor has no path for it, so it must either be decomposed into
   matmul + sigmoid + tanh + elementwise (all of which exist) or added as a
   fused `lstm_s8` op. Beyond codegen, it needs **hidden state carried across
   periodic invocations**, which the current harness has no concept of: every
   dispatch is stateless and `run_model` starts from a fixed input. That is a
   change to the harness and to the scheduler's notion of an instance, not just
   a new kernel — a recurrent model's instance *k* is not an interchangeable
   replica of instance *k-1*, which is exactly why the plan calls it out.

### SmolVLA — as the plan predicted
`models/smolvla.py` exists and is reachable only through
`extract_graph_export.py` (torch.export), **not** the FX int8 path. There is no
`examples/smolvla/`. So it loads and exports; int8 lowering is a separate
milestone, and nothing here blocks on it.


### VitFly update: `avgpool2d_s8` and `leaky_relu_s8` landed and verified

Both ops now exist end to end. Each needed four pieces, because a kernel is not
usable until all of them are present: a `KernelSpec` with argtypes, a branch in
the int8 extractor's module dispatch, a case in the int8 golden simulator, and
call emission in `generate_skeleton`.

Verified on hardware rather than added untested: `models/vitfly_frontend.py` is
LSTMNet's convolutional front end with layer parameters taken verbatim from
upstream. It extracts, builds and runs on the K1 **bit-exact**
(`max_abs_err=0`), 42 549 ticks (1.77 ms), with both new ops in the profile:

| op | kernel | ticks |
|---|---|---|
| avgpool | `avgpool2d_s8` | 1051 |
| lrelu | `leaky_relu_s8` | 88 |

Two details that are silent when wrong, both now pinned in the semantics:
`avgpool2d_s8` does **not** requantize (the mean of values on a scale is on that
scale, so the output scale is overwritten to the input's, as `maxpool2d_s8`
does); and `count_include_pad` selects **only the divisor**, because padded taps
contribute 0 to the sum either way under symmetric quantization.

**Still outstanding for full VitFly: the LSTM**, and it remains a different kind
of work — no path in the int8 extractor, and hidden state must survive across
periodic invocations, which the harness has no concept of.

---

## CORRECTION: the baseline ladder was measuring the runtime, not the scheduler

An earlier section of this log recorded that "XPU-RT buys 11% makespan and loses
31 of 32 MLP deadlines", and that `greedy_periodic` did not rescue it. **That
conclusion was wrong**, and it was wrong for the same reason the calibration
section flagged: the runtime could not execute the schedule it was given.

The runner had two worker threads, one per hardware target, so an 8-way schedule
was serialised onto two threads no matter how many cores or devices existed. The
deadline misses measured that serialisation. Per-core *devices* (`--pin_per_core`)
fixed where a dispatch ran but not how many ran at once; one worker **per core**
was the missing half.

Same schedules, same profiles, same board, with the runner fixed:

| rung | makespan | queue% | dronet | mlp |
|---|---|---|---|---|
| B0 static placement | 1149.8 ms | 70.5% | 10/10 miss | **0/32 miss** |
| **B1 XPU-RT greedy** | **441.8 ms** | 5.4% | 10/10 miss | **1/32 miss** |
| B2 + impl selection | 448.1 ms | 6.2% | 10/10 miss | 2/32 miss |
| P1 greedy_periodic | 449.5 ms | 6.0% | 10/10 miss | 2/32 miss |

**XPU-RT scheduling is 2.6x better on makespan than static placement and costs
one MLP deadline out of 32.** Predicted makespan 412.8 ms against 441.8 ms
measured — **7.0% error**.

DroNet misses 10/10 in *every* configuration including static placement, and that
is a property of the workload rather than of any scheduler: one instance is
113 ms of measured work and the window is 33.3 ms. No placement policy fixes
that; it needs the dispatch sharded across cores, which is the B4 rung.

The methodological point is the same one the calibration section made, and it
has now bitten twice: **before concluding anything about a scheduler, check that
the runtime can execute the schedule.** Both times the model and the profiles
were fine and the execution environment was not, and both times the wrong
conclusion was the flattering-to-nobody kind that looks like a real finding.

---

## Phases 14–15 complete

### VitFly (Phase 14) — full LSTMNet on the board, bit-exact

All three missing ops landed: `avgpool2d_s8`, `leaky_relu_s8`, and a fused
`lstm_s8` cell. Each needed four pieces — KernelSpec, extractor branch, golden
simulator case, skeleton emission — plus, for the LSTM, `nn.LSTM` handling
(tuple output resolved through its `getitem(0)`, discarded `(h_n, c_n)` skipped)
and `unsqueeze`/`squeeze` as views.

**Statefulness came from storage, not new machinery.** `h_state`/`c_state` are
ordinary intermediates, so `buffers.c` gives them file-scope arrays — `.bss`,
zero-initialised once and retained across `run_model()` calls. Nothing resets
them, deliberately: invocation *k* reads what *k-1* wrote.

`models/vitfly_lstm.py` on the K1: **`max_abs_err=0`**, 712 470 ticks (29.7 ms).

| op | ticks | share |
|---|---|---|
| lstm.l0 | 383 347 | 53.8% |
| lstm.l1 | 282 992 | 39.7% |
| everything else | 46 131 | 6.5% |

**Three real bugs, all caught by insisting on bit-exactness** and none of which a
tolerance would have surfaced:
1. The kernel wrote `h_state[t]` inside the loop that also reduces over the whole
   previous hidden vector, so unit *t+1* read the new `h[t]` — the recurrence
   consuming its own output mid-step. Off by up to **12 LSB**.
2. float32 accumulation was insufficient; both sides are now double/float64.
   2 LSB → 1.
3. numpy's `np.round` is half-to-**even**, C's `round()` is half-away-from-zero.
   The last LSB.

### SmolVLA (Phase 15) — exports, and the work list is quantified

Three genuine incompatibilities fixed to make it export at all: non-persistent
rotary `inv_freq` buffers absent from `state_dict`, `abs()` over a bool attention
mask during calibration, and the IO writer demanding a scale for that bool input.

`scripts/smolvla_op_inventory.py` reads the torch.export graph directly rather
than ModelBlaster's IR — the walker emits **2 op records for a 450M model**,
which measures its own coverage, not the model.

**4379 nodes, 63 distinct aten ops. Excluding 2480 shape/bookkeeping nodes:
1422 compute nodes covered, 477 gaps — 74.9%.**

The gaps are three families, not scatter:

| family | ops | count |
|---|---|---|
| RMSNorm | `pow`, `rsqrt` | 173 |
| attention | `scaled_dot_product_attention`, `softmax` | 60 |
| RoPE | `sin`, `cos` | 82 |
| plus | `layer_norm`, `embedding` | 79 |

That is a work list rather than "SmolVLA needs attention support".

**Two of those families then landed**, each with the full four pieces and
verified on hardware via `models/norm_block.py` (`max_abs_err=0`):

| added | covers |
|---|---|
| `layernorm_s8` | `layer_norm` 75 |
| `rmsnorm_s8` | `pow` 107 + `rsqrt` 66 — SmolVLA writes RMSNorm out rather than calling a module |
| GELU extractor branch | `gelu_s8` had a kernel and **no way to emit it** |

**SmolVLA compute coverage 74.9% → 89.2%.** The 205 remaining are RoPE
(`sin` 41, `cos` 41), attention (`scaled_dot_product_attention` 36) and a long
tail of small ops.

The GELU case is worth keeping in mind when reading any coverage number here:
kernel presence is not coverage. `gelu_s8` existed and counted as covered while
nothing in the extractor could produce it, so the inventory now says explicitly
that its figure is an upper bound.

### zephyr-chipyard-sw — resolved

The 3 orphaned commits now live in XPU-RT (which **is** pushable) as a 28 KB
bundle, verified by fetching into a clean clone: yields `418136fe` and tree
`a62bdd21`, byte-identical. A `format-patch` series was tried first and **does
not apply** — the profile CSVs defeat textual patching — which only surfaced
because the restore was tested rather than assumed.

---

## SmolVLA int8 complete: 98.1% coverage, and the soundness question answered

The last two families landed. `sdpa` is **decomposed**, not fused — `matmul_s8`
already carries `transpose_b` and `scale_div`, which is exactly QK^T/√d, and
`softmax_s8` exists; three existing kernels beat one new one duplicating them.
`sin_s8`/`cos_s8` cover RoPE. Extractor branches also added for `softmax` and
elementwise `mul` (another kernel nothing could emit).

**SmolVLA compute coverage 74.9% → 89.2% → 98.1%.** The 36 remaining nodes are
comparisons, `bucketize`, `cumsum`, `embedding`, `linspace`.

### The scores scale was wrong, and it mattered
`sq*sk*√D` is the obvious choice and it **clips**: on `attn_block` it covered
0.022 against an actual score range of 0.095, and cosine at that step was
**0.914**. In int8 units each operand has σ≈127/3, so the dot over D terms has
σ≈√D·(127/3)² and the 1/√D cancels — score magnitude is ~independent of D, near
1800·sq·sk at 3σ. `32·sq·sk` gives ~2× headroom; the step recovers to ~0.999.

### Is int8 attention/RoPE numerically sound? Yes — measured

| model | int8 vs fp32, cosine |
|---|---|
| attn_block (attention + RoPE) | **0.999899** |
| norm_block (layernorm/gelu/rmsnorm) | **0.999921** |
| lstm_tiny (stateful LSTM) | **0.999987** |
| mlp_control (trained reference) | **0.999975** |

`int8_quality.py` keeps this separate from kernel correctness on purpose. The
harness asks whether the device computes what the int8 reference says — exact or
a bug. This asks whether the int8 *model* still agrees with the fp32 model it
came from — never exact. Conflating them means shipping a broken kernel because
"int8 is lossy anyway", or rejecting a correct one because PTQ moved the answer.

### A bug in my own measurement, worth recording
That tool first reported **cosine 0.03** and looked like proof int8 attention was
destroyed. It was the tool. The synthetic models had no seed, so `get_model()`
built *different random weights* than extraction had used — the comparison was
between two different networks. `mlp_control` scored well only because it loads a
trained checkpoint, which masked the bug. A control (plain random-weight MLPs
quantize to 0.9997 even at 6 layers) is what showed the number had to be wrong
rather than a real finding. Models are now seeded.

**All seven models run bit-exact on the board**: mlp_control, dronet,
vitfly_frontend, lstm_tiny, vitfly_lstm, norm_block, attn_block.

---

## Phase 1 — measurement integrity

### Attempted

Make the instruments trustworthy before adding any rung, on the principle that
a conclusion drawn on a known-broken instrument is worth less than no
conclusion. Six changes.

### Commands

```bash
# merlin runner rebuild + deploy
cmake --build merlin/build/spacemit-merlin-perf --target merlin-dispatch-scheduler -j16
scp .../merlin-dispatch-scheduler k1:/root/mb_k1/bin/
ssh k1 'cd /root/mb_k1 && ulimit -n 8192 && ./bin/merlin-dispatch-scheduler \
  schedule_b4.json local-task 1 1 0 --vmfb_dir=/root/mb_k1 \
  --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5,6,7 --visible_cores=8 \
  --variant_p=RVV --variant_e=RVV --pin_per_core=1 --trace_csv=.../trace_B4.csv'
python3 scripts/join_k1_trace.py --schedule schedules/scheduled_networks_k1_B4_shard_greedy_profiled.json \
  --trace artifacts/k1_run/baselines/trace_B4.csv
python3 scripts/plot_k1_trace_gantt.py --composite "B0=...:..." "B1=...:..." "B4=...:..." \
  --out paper/figures/k1_measured_gantt_ladder.png
```

### Passed

* **Trace now carries placement.** `cores`, `observed_cpu_start`,
  `observed_cpu_end` appended in `dispatch_output.h`, fed the runner's own
  `held_mask`. Verified on hardware: 430 rows, `cores` populated
  (`0`, `0+1`, `2+3`, `0+1+2+3`), 218 sharded rows.
* **Per-core utilization is computable for the first time.** B4: CPU_P#0-3 at
  50.4-54.4%, CPU_E#0-3 at 47.7-59.7%; clusters CPU_P 51.6%, CPU_E 54.2%.
* **Affinity verified, not assumed.** `migrated=0` across four runs. The pin
  was previously best-effort with the return value discarded.
* **One metrics implementation.** `xpu-rt/trace_metrics.py` replaces three
  divergent copies. Reports what a periodic workload is actually specified in:
  miss *rate*, response p50/p90/p99 measured from `k*T`, achieved frequency,
  per-core and per-cluster utilization, and idle *intervals*.
* **The two miss definitions are now named apart.** `metrics.py` counts per
  dispatch (`op_deadline_miss_count`, B1: 160 of 360); `trace_metrics.py`
  counts per instance (`instance_deadline_misses`, B1: 10 of 10). Both were
  previously called `deadline_miss_count`.
* **Units.** Nine `*_us` fields in `_metrics.json` carried millisecond values —
  a 412 ms schedule reported as `makespan_us: 412.83`. Keys retained for
  compatibility, `*_ms` aliases added, `units_note` states it.
* **Prediction error is now decomposed.** Implementation comes from the trace's
  existing `vmfb_path` (third path segment) — it was in every trace ever taken
  and never read. Op type comes from the module-name tail.
* **Profiler.** `op`/`shape` parsed from the module name and written to the two
  CSV columns that were literal `""` for the whole campaign; explicit
  `--benchmark_min_warmup_time`; `cycles_est` renamed
  `cycles_est_from_assumed_clock` because it is median wall time x an assumed
  1.6 GHz, not a counter — `rdcycle` SIGILLs here.
* **Reusable measured Gantt.** `scripts/plot_k1_trace_gantt.py`, physical-core
  lanes from the trace, shard as one spanning bar, `--composite` for the
  ladder. Neither existing renderer could do this: `xpu-rt/plot_gantt.py`
  expects ModelBlaster's schema (zero shared column names), and merlin's own
  `plot_dispatch_trace.py` packs synthetic lanes.

### Measurements

Prediction error, B4, decomposed — the aggregate hides a 14x spread:

| op type | n | median rel err |
|---|---|---|
| matmul | 138 | +32.5% |
| reduction | 36 | +23.5% |
| elementwise | 160 | +20.4% |
| **conv** | 96 | **+2.3%** |

Aggregate median +19.5%, but **+1.7% restricted to dispatches >= 1 ms**. So the
cost model is accurate on the work that matters and wrong on small dispatches,
where a fixed per-dispatch overhead it does not model dominates.

Worst mispredictions are all sharded DroNet convs: `dronet0_dispatch_1`
predicted 6.096 ms from the 4-hart solo profile, measured **10.212 ms (+67.5%)**.
The profile is not wrong — the shard is contending with concurrent work. This is
the first direct evidence that a solo multi-core profile is not a valid cost for
a shard running alongside other dispatches, and it is what Phase 6 has to
quantify.

### Failed / corrected

**A retraction.** The B4 figure of "1 MLP deadline miss" was n=1 and is not
reproducible. Seven runs of the identical schedule give **7-9 misses of 38
(~24%)**; the single run showing 1 was a lucky outlier. Cause: MLP completes at
~7 ms against a 10 ms deadline, so its miss count is a knife-edge and is not a
stable discriminator between rungs. Makespan, by contrast, is stable to 1.0%
across runs.

I initially suspected my own instrumentation, since it added ~4% to MLP dispatch
time. Removing the allocation from the trace path left the miss count unchanged
at 9, so the instrumentation was not the cause. The allocation-free path stays
anyway.

**Also corrected:** DroNet's achieved frequency is the more honest framing of
the ladder than its miss count, which is 100% everywhere:

| rung | DroNet achieved | required | worst lateness |
|---|---|---|---|
| B0 static | 8.70 Hz | 30 Hz | 816.78 ms |
| B1 XPU-RT | 22.64 Hz | 30 Hz | 108.75 ms |
| B4 + shard | **29.02 Hz** | 30 Hz | **27.15 ms** |

B4 reaches 96.7% of the required rate.

### Blocker

None.

### Next

Phase 2: wire `capabilities.check_implementation_legality` into the production
path (it has zero callers today), reject `--variant_e=IME` in the runner, wire
the objdump gate, correct the IME record, and enable the vendored RVV int8
mmt4d ukernels that `enable-ukernels=none` currently makes dead code.

**Open question this phase raised:** all eight cores sit at ~50-60% utilization
while DroNet misses 100% of its deadlines. The board is half idle, so the
binding constraint is the dependency chain, not capacity — which means B3
(granularity) has a clearer route to improvement than more cores do.

### Repo SHAs

XPU-RT `feat/k1-modelblaster-closed-loop`, merlin `c2f7c35`, ModelBlaster
`34dc890`.

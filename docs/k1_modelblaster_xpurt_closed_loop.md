# SpaceMiT K1: profile → schedule → build → run → advise → rewrite

A runbook you can follow from a fresh checkout to a measured result on the
physical K1. Every number quoted here was measured on the board; where something
did not work, that is recorded too, because the failures cost more time than the
successes did.

**Read this first.** The execution path changed. Measurements used to go through
merlin's `merlin-dispatch-scheduler` running IREE VMFBs; they now go through
**ModelBlaster's own generated C kernels**, executed by a Linux port of the
schedule-driven multi-model harness (`ModelBlaster/harness_xpurt_linux`). The
reason is not aesthetic: `apply_fusion_hint` / `apply_split_hint` rewrite
ModelBlaster's IR, and a rewritten IR has no VMFB, so a runner that resolves
each dispatch to a `.vmfb` on disk *cannot execute a granularity change*. With
generated C, the thing the scheduler places is the thing that runs, and a hint
is directly runnable.

Everything the IREE path measured is still true of the IREE path, and the
numbers are kept — in [§11](#11-the-retired-ireemerlin-path), clearly labelled,
because several of them are the reason the current design looks the way it does.

### Where this currently stands, measured

Per-model service time on one core, ModelBlaster generated C, `rvv_x60`, int8,
every model bit-exact against its golden (`max_abs_err=0`):

| model | scalar | rvv_x60 | speedup |
|---|---|---|---|
| mlp_control | 0.37 | 0.083 | 4.5x |
| lstm_tiny | 0.08 | 0.059 | 1.4x |
| vitfly_frontend | 1.77 | 0.382 | 4.6x |
| vitfly_lstm | 29.69 | 9.48 | 3.1x (7.2x warm — see below) |
| dronet | 157.38 | 9.79 | 16.1x |
| yolov8_nano | 4020.93 | 226.87 | 17.7x |

Three models scheduled together on the board, measured, `max_abs_err=0`:

| yolov8_nano | cores it needs | makespan | predicted | deadline misses |
|---|---|---|---|---|
| 3 Hz | 0.68 | 821.2 ms | 819.3 | 3/110 |
| 4 Hz | 0.91 | 908.1 ms | 908.5 | **0/123** |
| 5 Hz | 1.13 | 960.1 ms | 957.3 | **0/130** |
| 6 Hz | 1.36 | 988.7 ms | 986.0 | **0/135** |

alongside `mlp_control` at 100 Hz and `dronet` at 30 Hz throughout. Prediction is
within 0.3% of measurement at every point.

Two caveats that belong next to those tables, not in a footnote:

* **`vitfly_lstm`'s 3.1x is dominated by a fixed cost, not by the kernel.**
  `run_model_k1.sh` scp's the ELF and runs it once, so the profile includes
  first-touch faulting of a 1.7 MB `const` weight array. Same binaries with
  pages already resident: **3.26 ms vs 23.58 ms scalar, 7.2x**. Both the before
  and after figures are cold, so the ratio is a fair comparison, but the
  absolute numbers carry a constant that dominates whichever build is faster.
* **3 Hz — the lightest load — is the only point that misses.** `mlp_control`
  instances 13/14/15 start 11-14 ms late, then execute in 0.07-1.14 ms; their
  harts were busy until 1-12 us before they started. They were ready and queued
  behind non-preemptible in-flight YOLO/DroNet dispatches (~2.6 ms each, against
  MLP's 10 ms period). The solver predicted no misses, so this is a real gap in
  its model rather than the board being slow.

---

## 0. Prerequisites

Nothing below assumes a particular home directory. Set these once; every command
in this document uses them.

```bash
export XPURT_ROOT="${XPURT_ROOT:-$(git rev-parse --show-toplevel)}"   # this checkout
export MB_ROOT="$XPURT_ROOT/ModelBlaster"

export MODELBLASTER_K1_HOST=k1          # ssh host alias for the board
export REMOTE_ROOT=/root/mb_k1          # staging dir on the board
export MODELBLASTER_K1_REMOTE_ROOT="$REMOTE_ROOT"   # name the MB runners read

# riscv64 *Linux glibc* cross toolchain prefix. The board runs Bianbu/Linux, so
# this is NOT the newlib/elf toolchain. Any of these works; pick what you have.
export CROSS=riscv64-unknown-linux-gnu-

# A Python with torch + numpy. Model extraction imports torch; the XPU-RT
# scheduler side does not.
export PY=python3
```

| you need | why | how to check |
|---|---|---|
| Python ≥3.11 with **torch**, **numpy** | `extract_graph` builds the model in PyTorch and does the int8 PTQ | `$PY -c 'import torch, numpy'` |
| `matplotlib` | the scheduler writes a Gantt PNG on every run; `plot_xpurt_trace` needs it too | `$PY -c 'import matplotlib'` |
| a riscv64-linux-gnu **gcc** | every board binary is cross-compiled and statically linked | `${CROSS}gcc --version` |
| **ultralytics**, only for yolov8 | `get_model` streams pretrained weights out of a `.pt` through the ultralytics loader | `$PY -c 'import ultralytics'` |
| `cmake` ≥3.16, `ninja` or `make` | `harness_xpurt_linux` is a CMake project | `cmake --version` |
| ssh access to the board | see the stanza below | `ssh $MODELBLASTER_K1_HOST true` |

On the machine this was developed on the environments are split, and the split
is worth knowing because no single one of them can run the whole yolov8 path:

| env | has | lacks |
|---|---|---|
| `merlin-dev` | torch 2.9.1 | **ultralytics** |
| `xpurt` | torch, **ultralytics 8.4.39** | — use this one for yolov8 |
| XPU-RT's own `.venv` | the scheduler side, matplotlib, mosek | torch |

An earlier version of this document claimed `merlin-dev` carried ultralytics
8.4. It does not, and the failure is a `ModuleNotFoundError` several minutes
into an extraction rather than at the start.

The cross toolchain is GCC 14.3.0 at

    merlin/build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin

Export it on `PATH` explicitly, and if you launch a build with `nohup` or in the
background, export it INSIDE that script. A background run here once failed
every single build because the PATH was set in the interactive shell and not in
the `nohup`ed one; the fix is a `command -v ${CROSS}gcc || exit 1` guard at the
top of the script, which turns twenty minutes of confusing failures into one
line.

None of these paths is baked into anything you have to edit:

```bash
# example only -- substitute your own
source "$CONDA_ROOT/etc/profile.d/conda.sh" && conda activate merlin-dev
export PY="$CONDA_PREFIX/bin/python"
export CROSS="$CHIPYARD/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-"
```

`ModelBlaster/scripts/run_xpurt_k1.sh` still carries a developer default for
`CROSS`. **Export `CROSS` yourself** rather than relying on it; it points at one
person's chipyard checkout.

`pydot` is *not* needed any more. It was a dependency of the IREE DOT-graph
parser, which belongs to the retired path.

### Which `modelblaster` are you importing?

There is more than one checkout of ModelBlaster on a typical dev box, and pip
editable installs make `import modelblaster` resolve to whichever one was
installed — same module names, different behaviour, no warning.
`run_xpurt_k1.sh` refuses to run when `modelblaster` does not resolve inside the
checkout it lives in; if you see

```
refusing to run: 'modelblaster' resolves to /some/other/checkout/..., not this checkout
```

that guard is working. Fix `PYTHONPATH` (`$MB_ROOT/src:$MB_ROOT`) or the
editable install, do not work around it.

### The SSH stanza matters

```
Host k1 spacemit
    HostName <board-ip>
    User root
    IdentityFile ~/.ssh/<your-key>
    IdentitiesOnly yes
    IdentityAgent none          # <- this line
```

**`IdentityAgent none` is not optional.** With a stale `ssh-agent` in the
environment, `ssh` hangs at `Next authentication method: publickey` and the
board's own log shows `Connection closed by authenticating user root [preauth]`
— i.e. the *client* gave up. `ssh-add -l` hanging is the tell. The same fault
breaks `git` over SSH; use `GIT_SSH_COMMAND` with the same option.

No key material, IP or credential belongs in this repository. The board's
address and key path live in your `~/.ssh/config` and nowhere else.

---

## 1. The board

| | |
|---|---|
| Console | `/dev/ttyUSB4` — CH340 (`1a86:7523`), 115200 8N1, group `dialout` |
| Host | `ssh $MODELBLASTER_K1_HOST` over wlan0; both Ethernet ports are down |
| OS | Bianbu 3.0, Linux 6.6.63, glibc 2.41 |
| SoC | Spacemit X60, 8 × riscv64, VLEN=256 (`zvl256b`) |

### Topology, measured not assumed

| Cluster | Cores | L2 | scalar | RVV | IME (`smt.vmadot`) |
|---|---|---|---|---|---|
| 0 | 0,1,2,3 | 512K shared 0-3 | yes | yes | **executes** |
| 1 | 4,5,6,7 | 512K shared 4-7 | yes | yes | **SIGILL** |

`/proc/cpuinfo` reports an identical ISA on all 8 harts and never mentions IME —
it is a vendor extension the kernel does not enumerate. The table above comes
from `artifacts/k1_bringup/*/ime_probe.c`, which pins each core and executes the
raw encoding under a `SIGILL` handler. `runtime/scripts/deploy_k1.sh` builds and
installs it; re-run it with
`ssh $MODELBLASTER_K1_HOST $REMOTE_ROOT/bin/ime_probe`.

The same facts are the machine-readable registry `ModelBlaster/cores/spacemit_k1.json`,
which the schedule ingester reads. Cluster 0 cores carry `ime_*` capabilities and
cluster 1 cores deliberately do not — an IME kernel on cluster 1 does not
degrade, it dies, so that list is a correctness constraint and not a performance
hint.

All 8 cores share **one** cpufreq policy (`related_cpus 0-7`), already
`performance` at a fixed 1.6 GHz. The clusters cannot be clocked independently.
No governor change is needed, so none was made.

### Five facts that will bite you

**`rdcycle` raises SIGILL from userspace.** `rdtime` works, and is a fixed
**24.000 MHz** (41.7 ns tick; `/proc/device-tree/cpus/timebase-frequency`
agrees). Any generated code that times itself with `rdcycle` does not run slowly
here — it dies. This is why every generation step below passes
`--platform linux`, and why `ModelBlaster/runtime/mb_posix_compat.h` hard-codes
`MB_POSIX_TICKS_PER_SEC = 24000000`. **Convert tick counts with 24 MHz, never
the 1.6 GHz core clock**; getting it wrong scales every profile by 67×, silently.

**Load average sits at 2.00 on an idle board.** Two vendor kernel threads
(`vq0`, `vq1`) are wedged in uninterruptible sleep at 0% CPU. Benign. Do not
chase it.

**Co-running across clusters is worse than within one.** See [§10](#10-contention-measured-under-control-and-it-inverts-the-obvious-assumption).
It contradicts the shared-L2 intuition, and it is measured.

**A vector build can execute scalar code and still report success.** Curated
kernels are looked up by EXACT op name
(`kernels/<backend>/<backend>_<op>_<algorithm>.c`), so a fused op --
`conv2d_batchnorm2d_silu_s8` -- matches nothing even when `conv2d_s8`,
`batchnorm2d_s8` and `silu_s8` all have kernels. Selection falls back to the
scalar reference, writes `"source": "reference"` into `kernel_picks.json`, and
the build succeeds. This cost this project its headline numbers: yolov8_nano
measured **0.81x against the scalar build** -- slower -- because 99.8% of it was
scalar code paying a vector build's overhead, and DroNet's "2.51x" was really
86.7% scalar. `scripts/check_kernel_coverage.py` now fails the build on this and
is wired into `run_xpurt_k1.sh`; override with `MB_KERNEL_COVERAGE=warn` only
while iterating. Weight by measured time, not dispatch count -- DroNet's
fallback was 13% by count and 86.7% by time.

**`core_kind` is a placement pool, and the walker used to collapse it.**
`generate_xpurt_main.py` spawns one scheduler worker per **(core_kind, hart)**
pair the dispatch table uses. It previously spawned one per *kind* and claimed
entries by `core_kind` alone, ignoring the `hart` each entry already carried --
so a schedule using four `rvv` cores executed on one thread, and the number of
dispatch pairs on different harts overlapping in time was exactly **zero**. A
2-model workload still met every deadline that way (29% of one thread was
enough); a 3-model one missed **119 of 123** while the solver predicted none.
If you are reading an old trace, check `worker_hart` against `hart` before
believing any statement about parallelism.

---

## 2. Deploy the board-side binaries

**This is step one on a fresh board, and it used to be missing.** Both runners
scp *their own* harness as a side effect of running it, so nothing ever put the
capability probe — or a previously built harness you want to re-run — on the
board. That is the first hard stop for a new reader.

```bash
runtime/scripts/deploy_k1.sh --list       # resolved manifest, no ssh
runtime/scripts/deploy_k1.sh --dry-run    # + local validation, still no ssh
runtime/scripts/deploy_k1.sh              # copy, verify, report
```

What it does:

* builds `ime_probe` from `artifacts/k1_bringup/*/ime_probe.c` (the only
  board-side binary with no other producer) into `build/k1/bin/`;
* copies an **explicit manifest** — `bin/ime_probe`, every locally built
  single-model harness (`ModelBlaster/build/k1/*_harness` → `$REMOTE_ROOT/bin/`),
  and every locally built schedule harness
  (`ModelBlaster/build/k1_xpurt/_build/*/xpurt_harness` → `$REMOTE_ROOT/xpurt/<schedule>`);
* **verifies each file landed** by comparing sha256 on both sides (falling back
  to size, and saying so, if the board has no `sha256sum`);
* is **idempotent** — a file whose remote hash already matches is skipped, so
  re-running costs one ssh round trip and copies nothing.

Extra files, and overrides:

```bash
runtime/scripts/deploy_k1.sh some/binary:bin/other_name    # explicit entry
runtime/scripts/deploy_k1.sh --only 'bin/ime_probe'        # glob over remote paths
runtime/scripts/deploy_k1.sh --host other-k1 --remote-root /srv/mb
runtime/scripts/deploy_k1.sh --force                       # re-copy even if current
```

Two guards worth knowing about, because both correspond to real failures:

* Every entry is checked to be a **riscv64 ELF** before it is copied
  (`e_machine == EM_RISCV`). Conda environments export their own `CC`/`LDFLAGS`
  and a cross build that quietly produced an x86 binary is a real failure mode
  here; catching it locally is much cheaper than a "cannot execute binary file"
  on the board.
* `ime_probe` is compiled `-march=rv64gcv`. The probe *assembles* a `vsetvli` in
  order to discover at **runtime** whether the core traps on it; the toolchain's
  default `rv64gc` rejects the mnemonic at assembly time
  (``unrecognized opcode ... extension `v' ... required``). The `-march` flag
  decides what can be encoded, not what the hardware will accept — which is the
  entire point of the probe.

Nothing else needs deploying: every binary in this flow is linked `-static`
(`-O2 -static` in both `ModelBlaster/harness_linux/Makefile` and the `run_xpurt_k1.sh` CMake
invocation), so the board needs no libraries, no runtime and no `.vmfb` tree.

---

## 3. One model, one core: generate, build, run, verify, profile

```bash
cd "$MB_ROOT"
PROFILE_OUT_ROOT="$XPURT_ROOT/gen_mb/profile" \
  bash scripts/run_model_k1.sh mlp_control int8 rvv_x60 0
#                              <model>      <quant> <target> <cpu>
```

extract → skeleton (`--platform linux`) → kernels → cross-build → deploy → run →
verify → profile, in one command. Positional arguments only; everything else is
environment (`MODELBLASTER_K1_HOST`, `MODELBLASTER_K1_REMOTE_ROOT`, `CROSS`,
`OUT_ROOT`, `PROFILE_OUT_ROOT`, `BACKEND`).

Artefacts land at, and these are the real paths — there is no `<gen>` placeholder:

```
ModelBlaster/build/k1/<model>/<quant>/graph.json      the IR (fusion/split hints edit THIS)
ModelBlaster/build/k1/<model>/<quant>/{weights,io}.npz
ModelBlaster/build/k1/<model>/<quant>/generated/      the generated C
ModelBlaster/build/k1/<model>_<quant>_<target>_harness   the riscv64 static ELF
ModelBlaster/build/k1/<model>/<quant>/profile_k1.csv  per-dispatch ticks
$XPURT_ROOT/gen_mb/profile/<backend>/spacemit_x60/<model>/<model>.<quant>/<spec>/topo_<cpu>/results.csv
```

`--platform linux` is what swaps `rdcycle` for `rdtime`; without it the binary
SIGILLs on its first timed dispatch. The runner defaults
`--profile-clock-mhz` to **24**, not 1600, for the same reason.

Valid `<target>` values are ModelBlaster backend tags:
`scalar`, `rvv`, `rvv_x60`, `rvv_f16`, `rvv_hetero`, `rvv_opu`, `scalar_f16`,
`gemmini`, `gemmini_q31`. On the K1 use `scalar` (reference, portable) or
`rvv_x60` (`-march=rv64gcv_zvl256b -mabi=lp64d`, plus the RVV intrinsics compat
header). `rvv_x60` is a K1-specific *build* of the `rvv` *kind* — see the
core-kind trap in [§6](#6-build-and-run-the-schedule-on-the-board).

Reference results, all bit-exact (`max_abs_err=0`), core 0, scalar reference
kernels:

| model | rdtime ticks | wall @24 MHz | verify |
|---|---|---|---|
| mlp_control | 9 028 | 0.38 ms | PASS |
| dronet | 3 777 286 | 157.4 ms | PASS |
| yolov8_nano | 96 503 982 | 4 021 ms | PASS (75 600 outputs) |

Read those as a correctness-and-plumbing result, not a performance one: the
curated RVV kernels are a 4.2× improvement on `mlp_control` alone. Note also
that the smallest dispatch costs **62 ticks ≈ 2.6 µs**, against the retired IREE
path's ~63 µs floor — the per-dispatch overhead that dominated the IREE MLP was
a property of that runtime, not of this hardware.

### `PROFILE_OUT_ROOT` must end in `profile`

XPU-RT's profile loader looks for
`<gen_root>/profile/<hw>/<target>/<model>/<basename>/…/<topo_tag>/results.csv`.
So a profile tree is only findable if its parent directory is literally named
`profile`. The existing ModelBlaster profiles in this repo were written to
`gen/profile_mb/…`, which **no `gen_root` can address** — verified. Either write
new profiles to `$XPURT_ROOT/gen_mb/profile` as above, or expose the old tree
under that name:

```bash
mkdir -p "$XPURT_ROOT/gen_mb" && ln -s ../gen/profile_mb "$XPURT_ROOT/gen_mb/profile"
```

`gen_mb` rather than `gen`: `gen/profile` is the retired IREE tree, and mixing
timings from two different runtimes in one profile database is exactly the class
of error [§11](#11-the-retired-ireemerlin-path) exists to prevent.

### Kernel generation is Codex-only

```bash
export LLM_PROVIDER=codex
export CODEX_CALLS_LOG=artifacts/k1_run/codex_calls.jsonl
BACKEND=llm bash scripts/run_model_k1.sh mlp_control int8 rvv_x60 0
```

`BACKEND` selects `generate_kernels --backend {reference,llm}`; the default,
`reference`, uses curated kernels and calls no model at all. There is no
fallback from Codex to Bedrock, by design and by test
(`ModelBlaster/tests/test_codex_provider.py`). If Codex is unavailable the
kernel step fails loudly and the caller falls back to reference/curated kernels —
deterministic artifacts already in the tree, not another model. The call log
records `provider: codex`; note it deliberately does **not** reuse
`bedrock_client._append_call_log`, which hardcodes `"provider": "bedrock"`.

#### Compare against the best kernel you already have, not the reference

The first Codex kernel (`artifacts/k1_run/codex/`) was bit-exact on the board and
**4.48× faster than the scalar reference** — a number it would be easy to report
as a win. Against `ModelBlaster/kernels/rvv/rvv_linear_s8_direct.c`, which was already in the
tree, it is **41% slower**:

| `linear_s8` total, rdtime ticks | scalar ref | curated RVV | Codex RVV |
|---|---|---|---|
| | 8095 | **1280** | 1807 |

Accept requires correctness **and** an improvement in the selected metric. It is
archived, not promoted. Always run the incumbent on the same board in the same
conditions before claiming a generated kernel is an improvement.

---

## 3b. Bringing a RETRAINED detector on board

A model retrained elsewhere — new classes, new input geometry — needs three
things to agree, and nothing checks two of them for you.

```bash
export MODELBLASTER_YOLOV8N_WEIGHTS=/path/to/best.pt   # the fine-tune
export MODELBLASTER_YOLOV8N_NC=2                       # its class count
export MODELBLASTER_YOLOV8N_INPUT=64x96                # H x W; `96` = square
export MODELBLASTER_YOLOV8N_CALIB_DIR=/path/to/dataset/images/val
export MODELBLASTER_YOLOV8N_CALIB_IMAGE=/path/to/dataset/images/val/one.png
bash scripts/run_model_k1.sh yolov8_nano int8 rvv_x60 0
```

### The three that must agree

1. the geometry the model was TRAINED at,
2. `MODELBLASTER_YOLOV8N_INPUT` here,
3. the board's preprocess.

Only the third is yours to get right at deploy time; the first two are a
config you can silently mismatch. Note what does NOT need to agree: the
ultralytics EXPORT geometry. Conv and BN weight shapes depend on channel
counts, not on input resolution, so a checkpoint trained at 64x96 loads into a
model configured at any size — this path reads the `state_dict`, not an
exported graph.

### Rectangular input, and why it is not a micro-optimisation

`MODELBLASTER_YOLOV8N_INPUT` takes `64x96` (H x W) as well as a bare square
size. Every dimension must be a multiple of 32 — YOLOv8's deepest level
downsamples by 32, so `96x48` is refused even though half of it looks legal.

A square input forces a non-square camera frame to be letterboxed, and the
padding is then convolved at full cost through the whole backbone. For the
90x60 FPV frame here, a 96x96 letterbox is a predicted 88.5 ms of which ~26 ms
convolves grey bars; the matching 64x96 is 62.6 ms with zero padding. That is
not a speed/accuracy trade — the content box inside the letterbox IS 64x96, so
the rect keeps every real pixel.

Exact-aspect sizes are scarcer than they look. For 3:2 the ladder is 64x96,
then 128x192 — a 3.5x cost jump, because 96x144 would be the natural half-step
and 144 is not a multiple of 32.

### PTQ calibration is not optional here, and its default is absent

`models/yolov8_nano.py` defaults `MODELBLASTER_YOLOV8N_CALIB_DIR` to
`datasets/idsia/samples/sc`, which **does not exist in this checkout**. The
documented fallback is `torch.randn`, and for this model that is not a mild
degradation:

> With `torch.randn` activation scales saturated the cls-head logits and
> produced 50 false-positive detections at confidence=1.0 ... the int8 ceiling
> tracks fp32 within 1-2 ulp instead of L-inf ~ 59.

So the default path quantises a detector against noise. It builds, verifies,
and profiles perfectly cleanly, and hands you a latency number for a model that
detects nonsense. Point `CALIB_DIR` at the deployment distribution — the actual
validation set of the actual dataset — and confirm from the log that it says
`calibrated across N samples`.

`CALIB_IMAGE` is a SEPARATE knob for the single io-pinned golden used by the
host-vs-board bit-exactness check. Setting only `CALIB_DIR` leaves the golden
as noise. `max_abs_err` is still a valid test of "the generated C matches
PyTorch" — any fixed input tests that — but it exercises no real-data
behaviour. Set both.

### A custom `nc` needs a custom checkpoint

`MODELBLASTER_YOLOV8N_NC != 80` against the STOCK `yolov8n.pt` is refused: its
cv3 head is COCO-80 and loading it into a 2-class model is a real shape error.
With `MODELBLASTER_YOLOV8N_WEIGHTS` pointing at a fine-tune, any `nc` is
allowed — permitted, not trusted, since the loader still raises per tensor on
any shape mismatch and names it.

The trap to know: `MODELBLASTER_YOLOV8N_PRETRAINED=0` is the obvious way past
a refusal and it silently discards the training, shipping random weights that
build, run, profile and detect nothing.

---

## 4. Make the models visible to the scheduler

XPU-RT needs two things per network: a **dispatch dependency graph** and a
**profile**. §3 produced the profile. The graph comes straight from the IR:

```bash
cd "$MB_ROOT"
export PYTHONPATH="$MB_ROOT/src:$MB_ROOT"
for m in mlp_control dronet; do
  $PY -m modelblaster.pipeline.emit_dispatch_graph \
      --ir "build/k1/$m/int8/graph.json" \
      --out-root "$XPURT_ROOT/gen_mb/vmfb" \
      --target spacemit_x60 --hw rvv_x60
done
# -> gen_mb/vmfb/<model>/spacemit_x60/rvv_x60/<model>.int8/<model>.int8_dispatch_graph.json
```

The `vmfb` directory name is a fossil of the IREE layout that XPU-RT's
`dispatch_deps_path` reader expects; there are no VMFBs in it. Each entry's `id`
is the same `dispatch_id` the profiler writes, so the graph joins to
`results.csv` directly.

Then a workload config. `data/toplevel/networks_k1_mb.json` is the worked
example — 4 MLP-class + DroNet on 8 cores, single-core profiles:

```jsonc
"hardware": {
  "machines":   { "cpu_p": 8 },                 // ONE pool -- see the trap below
  "profile_hw": { "cpu_p": "rvv_x60" },
  "profile": { "target": "spacemit_x60", "topo_tag": "topo_0",
               "topo_tag_override": true, "gen_root": "gen_mb" }
},
"scheduler": { "machine_combination_mode": "singletons", "use_profiled": true, ... },
"networks": {
  "mlp_control": { "id": 0, "identifier": "mlp_control", "period": 10,
                   "window_duration": 10,
                   "dispatch_deps_path": "gen_mb/vmfb/mlp_control/spacemit_x60/rvv_x60/mlp_control.int8/mlp_control.int8_dispatch_graph.json" },
  "dronet":      { "id": 1, "identifier": "dronet", "period": 33.3, ... }
}
```

Three settings decide what the model means:

* **`machine_combination_mode: "singletons"`** — every core independently
  schedulable, so 8 dispatches genuinely run at once. The alternative,
  `"prefix"`, makes a *cluster* one resource that may be given up to 4 cores,
  and then at most 2 dispatches run concurrently on the whole board.
  `"shard"` additionally offers aligned power-of-two core blocks, so the solver
  can choose to spread one dispatch across several cores.
* **`topo_tag: "topo_0"` with `topo_tag_override: true`** — singletons **must**
  be paired with single-core profiles. A 4-hart timing here would credit each
  core with the whole cluster's throughput. (With `machine_combination_mode:
  "shard"` you must set `topo_tag_override: false` instead, so each block is
  costed with its own N-hart profile; otherwise a 4-core shard is charged the
  single-core time while blocking four cores, and the solver "correctly" never
  picks one, for a purely clerical reason.)
* **`"machines": {"cpu_p": 8}`, not `{"cpu_p": 4, "cpu_e": 4}`.** This is a
  trap, verified: `ingest_xpurt_schedule` resolves `CPU_P#n` and `CPU_E#n` to
  the *n*-th registry core **of the named kind**, and every core in
  `cores/spacemit_k1.json` is kind `rvv`. With `--cpu-p-kind rvv --cpu-e-kind rvv`
  (which is what the K1 wants — both clusters are RVV-capable and measured
  equivalent at a 0.996 ratio), `CPU_P#0` and `CPU_E#0` both resolve to
  `cluster0_core0`, hart 0. An 8-machine schedule then double-books harts 0-3
  and never touches cluster 1. A single pool of 8 maps 1:1 onto harts 0-7.

IME is expressed as an implementation *on cluster-0 cores*, never as a separate
machine — see `xpu-rt/capabilities.py`. Modelling it as `{"ime": 4}` extra
machines produces schedules that cannot run: the IME machine is busy while the
core it executes on is still marked idle. (No ModelBlaster kernel targets IME
today anyway; see [§9](#9-what-ime-actually-does).)

---

## 5. Schedule

```bash
cd "$XPURT_ROOT"
python3 scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_k1_mb.json \
    --solver greedy --profiled
```

This side needs no torch. Useful flags, all real:
`--solver {milp,greedy,greedy_periodic,decomposed}`,
`--scheduler {cpsat,heft,peft,edf,fifo,critical_path,…}`, `--time-limit`,
`--max-periodic-iters`, `--no-profiled`, `--prune-periods` /
`--no-prune-periods`, `--random-seed`.

Outputs, all derived from the config's basename:

```
schedules/scheduled_networks_k1_mb_greedy_profiled.json          the schedule
schedules/scheduled_networks_k1_mb_greedy_profiled_metrics.json  makespan, misses, …
schedules/scheduled_networks_k1_mb_greedy_profiled_report.json   SchedulerReport v2
plots/networks_k1_mb_greedy_profiled.png                         predicted Gantt
```

It also prints a `pdb_hash` over the profile CSVs it actually read — check it
changes when you re-profile, and does not when you do not.

> **Units.** `metrics.json` reports `makespan_us=727.97` for a schedule that is
> 728 **milliseconds** long. Nine `*_us` fields carry millisecond values; `*_ms`
> aliases were added and `units_note` states it, but the old keys were kept for
> compatibility and are still wrong. Do not read a `_us` number from this file
> without checking `units_note`.

---

## 6. Build and run the schedule on the board

One command does generate → ingest → walker → cross-build → deploy → run:

```bash
cd "$MB_ROOT"
CORE_KINDS=rvv bash scripts/run_xpurt_k1.sh \
    --schedule "$XPURT_ROOT/schedules/scheduled_networks_k1_mb_greedy_profiled.json" \
    --models mlp_control,dronet \
    --backends rvv_x60
```

Flags: `--schedule` (required), `--registry`, `--models`, `--backends`,
`--quant`, `--out-root`, `--cpu-ids`, `--no-trace`, `--jobs`.
Environment: `MODELBLASTER_K1_HOST`, `MODELBLASTER_K1_REMOTE_ROOT`, `CROSS`,
`PY`, `REGISTRY`, `CPU_P_KIND`, `CPU_E_KIND`, `CORE_KINDS`, `BACKEND`,
`LLM_PROVIDER`.

The five stages, and what each leaves behind:

1. per `(model, backend)`: `extract_graph` → `generate_skeleton --platform linux`
   → `generate_kernels`, staged as
   `build/k1_xpurt/<model>/<quant>/<backend>/` (one weights/buffers TU per model,
   shared across backends). Existing outputs are **reused**, so a second run is
   fast — and stale, if you edited the IR without clearing them.
2. `ingest_xpurt_schedule` → `<schedule>.{c,h}` dispatch table.
3. `generate_xpurt_main --platform linux` → the walker.
4. `cmake -S harness_xpurt_linux` cross-build → `xpurt_harness`
   (`-O2 -static`, riscv64).
5. `scp` to `$REMOTE_ROOT/xpurt/<schedule-name>`, run under
   `ulimit -n 8192` (optionally `taskset -c $CPU_IDS`), pull back stdout and
   split out the trace CSV.

Everything through stage 4 runs without the board, which is the fastest way to
check a config: point `MODELBLASTER_K1_HOST` at an unreachable name and the run
completes stages 1-4 and stops at the `ssh`.

### `core_kind` is not the backend tag

`CORE_KINDS` describes what the **schedule** says (`rvv`, matching
`cores/spacemit_k1.json`); `--backends` describes what the **binary** was built
as (`rvv_x60`). They are parallel lists — kind *k* is executed by backend *k* —
and they are not the same string. Conflating them makes every worker refuse
every entry (`strcmp("rvv_x60","rvv") != 0`) and the run completes having
executed nothing, with `entries_done=0` and an all-zero trace. That is why the
command above sets `CORE_KINDS=rvv` explicitly while building `rvv_x60`.

### Why the per-model / per-backend split is not simplified

`model.c`'s `buf_*` intermediates must have exactly **one** definition per model,
shared across backends. Giving each backend its own copy made a cross-backend
dispatch within one network read zeroed scratch — that is what corrupted
DroNet's output when `rvv` ran `maxpool1` and `scalar`'s `conv_modules.3` then
read its own backend's `buf_maxpool1`. Weights are linked once for the same
reason.

### Output

```
build/k1_xpurt/_gen/<schedule>/<schedule>_stdout.txt   full board stdout
build/k1_xpurt/_gen/<schedule>/<schedule>_trace.csv    the trace block, extracted
build/k1_xpurt/_build/<schedule>/{cmake.log,build.log}
```

The trace columns are

```
entry_id, network, instance, dispatch_id, op, name, core_kind, hart,
predicted_start_ms, predicted_duration_ms, worker_kind_idx,
actual_start_cycles, actual_end_cycles
```

and `actual_*_cycles` are **rdtime ticks at 24 MHz**, not core cycles. The
stdout also carries `MODELBLASTER_OUTPUT_*` (verify against golden),
`MODELBLASTER_PROFILE_*` (per-op ticks per backend) and
`MODELBLASTER_WALL_CYCLES[_INST]` blocks.

**Verify before you report.** A run that produced timings but failed
`max_abs_err` measured something, but not the function you meant. The runner
greps the stdout for `MODELBLASTER_VERIFY|max_abs_err|FAIL|PASS`; read it.

---

## 7. Reading the result

### The measured Gantt

```bash
$PY scripts/plot_xpurt_trace.py \
    build/k1_xpurt/_gen/<schedule>/<schedule>_stdout.txt \
    --clock-mhz 24 --source k1 \
    --out plots/<schedule>_measured.png --csv artifacts/k1_run/<schedule>_trace.csv
```

Positional `input` is the **stdout capture** (or `-` for stdin), not the CSV.
`--clock-mhz` defaults to 10 (Zephyr-on-spike); on this board it is **24**.

Known break, hit on the one stdout capture currently in the tree: if the
schedule's dispatches all carry `duration: 0` — as a fully fused FireSim
schedule does — `_summary()` divides by a zero predicted makespan and raises
`ZeroDivisionError` before writing the PNG. Schedules produced by §5 carry real
durations and do not hit it.

### Predicted vs actual

`ModelBlaster/scripts/emit_measured_report.py` is meant to overlay the trace on
the predicted `SchedulerReport` and hand the result back to the advisor:

```bash
$PY scripts/emit_measured_report.py \
    --predicted-report "$XPURT_ROOT/schedules/<schedule>_report.json" \
    --trace  build/k1_xpurt/_gen/<schedule>/<schedule>_trace.csv \
    --out    artifacts/k1_run/<schedule>_measured.json \
    --clock-mhz 24
```

(The flag is `--predicted-report`. It is **not** `--schedule`; a `--schedule`
flag for this script appears in older notes and has never existed.)

**It currently matches 0 rows, and you should not report numbers from it until
that is fixed.** Verified against a real K1 trace and its own predicted report:
`matched 0/7`. The join key is `(network, instance, dispatch_id)`; the trace has
all three, but XPU-RT's report entries carry neither `network` nor `instance` —
they encode both inside `name`, as `mlp_control0_dispatch_0`. So `_key_for()`
falls back to `("", 0, id)` and never matches `("mlp_control", 0, 0)`. The fix is
to parse `name` in `_key_for()`, the same `<network><instance>` split
`ingest_xpurt_schedule` and `generate_xpurt_main` already implement.

Two more traps in the same file, both real:

* `--clock-mhz` defaults to **1000** (FireSim). On the K1 that under-reports
  every measured time by 41.7×.
* On FireSim traces the `actual_*_cycles` columns are *mtime ticks of 1000 CPU
  cycles*, so the column name is wrong there too and the old default compounded
  the error. On the K1 the columns really are the harness's own timer ticks, and
  24 MHz is the right conversion.

Until that join is fixed, use `XPU-RT/scripts/join_k1_trace.py`, which does the
same comparison per dispatch and reads both trace schemas:

```bash
$PY scripts/join_k1_trace.py --schedule <sched.json> --trace <trace.csv> \
    --ir <graph.json for each model in the schedule>
```

**Pass `--ir` for every model, or read nothing below the summary.** The trace's
`dispatch_id` is a record slot rather than the IR dispatch id, and the two
diverge by the number of zero-cost ops before them — see
[§8](#the-traces-dispatch_id-is-a-record-slot-not-a-dispatch-id). The tool
audits this itself, from the schedule's and the trace's own op-kind columns, and
refuses rather than printing per-dispatch numbers that compare different ops.

On the 3-model 4 Hz run it reports median service-time error **−2.4%** over 1585
dispatches (−1.3% restricted to dispatches over 1 ms), and 48.7% of elapsed time
spent queueing rather than computing — which is the split that decides whether a
deadline miss calls for a faster kernel or an earlier placement.

The per-network sum remains the fallback that needs no IR: sum
`predicted_duration_ms` from the trace's own columns against
`(actual_end - actual_start) / 24e3` ms from the same rows. Both come from the
one file the binary emitted, which sidesteps the labelling hazard described in
`artifacts/agentic_branch_salvage.md`: *the trace's labels must be provably
generated from the same artifact the binary executed.*

---

## 8. Advice, and feeding it back

The advisor reads a `SchedulerReport` and says what to change:

```bash
python3 xpu-rt/advisor.py \
    --report schedules/scheduled_networks_k1_mb_greedy_profiled_report.json \
    --top-k 5 --json --emit artifacts/k1_run/advice.json
```

On the worked example it reports the bottleneck resource, `granularity:
too_fine`, and recommends coarsening — `749/749 dispatches below 1000 cycles`.
That recommendation is actionable, because ModelBlaster can execute a
granularity change:

```bash
cd "$MB_ROOT"
$PY -m modelblaster.pipeline.apply_fusion_hint \
    --hint  <hint.json> --model mlp_control \
    --ir    build/k1_xpurt/mlp_control/int8/graph.json \
    --out   build/k1_xpurt/mlp_control/int8/graph.fused.json
# the dual, for splitting one op across cores:
$PY -m modelblaster.pipeline.apply_split_hint \
    --hint <hint.json> --model dronet \
    --ir   build/k1_xpurt/dronet/int8/graph.json \
    --out  build/k1_xpurt/dronet/int8/graph.split.json
```

`apply_fusion_hint` also takes `--pairwise`. Both are pure JSON-in/JSON-out
rewrites: the scheduler never edits C. Rebuild by pointing the flow at the
rewritten IR — **never edit `graph.json` in place**; a crash then leaves the
source corrupted.

`mlp_control` goes from 7 ops to 4 (three fused `linear_s8+elu_s8` pairs plus the
tail linear). Note the evidence only appeared **after** the previous round: once
the curated RVV linear kernel landed, the `elu` ops went from noise to **39.7%**
of runtime. Fusion advice that would have been wrong at round 0 is right at
round 1, which is the argument for closing the loop rather than optimising once.

### The invariant that governs every rewrite

From `ModelBlaster/artifacts/agentic_fuse_split/WARNING.md` and the salvage audit, in the
order they cost time:

1. A rewrite may not reduce modelled work unless a kernel exists that performs
   the merged work. Otherwise the schedule counts the work as gone and the
   hardware still does it.
2. Costs must reach the scheduler through the live ingest path, never a
   side-file. A corrective patch that edits a `results.csv` the scheduler does
   not read produces an "honest" number within 0.05 ms of the fiction it was
   correcting.
3. A corrective patch must be independently checked to have taken effect. An
   implausibly small delta after a large intended change is evidence of a no-op.
4. The trace's labels must be provably generated from the same artifact the
   binary executed.

### Where each piece of the loop actually lives

The modules are not all named after the concepts, which has cost more than one
person an afternoon.

| the concept | where it lives |
|---|---|
| granularity analysis | `xpu-rt/granularity_advisor.py` (the verdict) |
| candidate generation + scoring | `xpu-rt/rewrite.py` (`Candidate`, `score_candidates`) |
| the driver | `scripts/granularity_loop.py` — verdict → scored candidates → Contract-2 hint |
| hint assembly | `xpu-rt/bundle.py` |
| the rewriters | three, not one: `ModelBlaster/pipeline/apply_{fusion,split,unfuse}_hint.py` |
| the decision | search: `ModelBlaster/scripts/decision_loop.py` · **verdict: `scripts/compare_candidates.py`** |
| advice → hint | `scripts/advice_to_{fusion,split,unfuse}_hint.py`, `advice_to_kernel_choice.py` |
| the acceptance rule | `xpu-rt/candidate_objective.py` |
| board bundles | `scripts/run_xpurt_bundle.py`, `run_bundle_firesim.sh` (FireSim-era) |

`rewrite.py`, `bundle.py` and `granularity_loop.py` were absent from this branch
for a while — they were added on `origin/xpurt-scheduler-advisor` and never
merged forward, so `decision_loop.py` shelled out to a file that was not there
and the automated loop could not run at all. Restored in
`feat/k1-granularity-bridge`.

**The predicted cost model has no per-dispatch launch overhead**, so fusing
tiny dispatches leaves the predicted makespan unchanged — every fuse candidate
scores Δmakespan 0.00. `granularity_loop.py` therefore judges MERGE by the
dispatches and cross-device transitions it removes, and SPLIT by makespan
delta, which is visible in prediction because parallelism is. The merge payoff
is real but only shows on hardware, so the predicted number ranks; it does not
decide.

**Searching and deciding are different jobs.** `decision_loop.py` accepts on
total cycles, which is `candidate_objective`'s ninth and last term. That is
fine as a cheap search filter and wrong as a verdict — the DroNet split was
rejected at term 5 while *improving* critical-task p99, which a cycles gate
cannot see in either direction. Search with `decision_loop`, decide with
`compare_candidates.py`.

### The four verbs, and how each one reaches the compiler

The scheduler never edits C. It states a recommendation with evidence, a
narrow adapter turns it into one of ModelBlaster's hint contracts, and a
JSON-in/JSON-out rewriter applies it. Every stage is a file, so any stage can be
debugged alone.

| verb | producer | advice → hint | consumer | the VERB on hardware | the BRIDGE end to end |
|---|---|---|---|---|---|
| `fuse_with_successor` | `overhead_advice` | `advice_to_fusion_hint.py` | `apply_fusion_hint.py` | **rejected**, 36% slower | ✅ drove that rung |
| `split` | `blocking_advice` | `advice_to_split_hint.py` | `apply_split_hint.py` | **rejected**, +13.7% | ⚠️ that rung used a hand-written hint |
| `unfuse` | `unfuse_advice` | `advice_to_unfuse_hint.py` | `apply_unfuse_hint.py` | never | host only |
| `choose_implementation` | `implementation_advice` | `advice_to_kernel_choice.py` | `generate_kernels --keep-reference-ops` | never | host only |
| `shard` | `shard_advice` | — | `MB_SHARD_FACTOR` (build-level) | never | **cannot fire**: see below |

**Read the last two columns separately.** "The verb reached a board verdict" and
"this chain produced that verdict" are different claims, and only
`fuse_with_successor` satisfies both. The DroNet split that measured +13.7% was
driven by a hint written by hand — `advice_to_split_hint.py` did not exist
then — so the split bridge is verified against the rewriter and against real
advice, but has not yet been the thing that produced a board number. Collapsing
the two columns is how a chain gets credited with a measurement it did not make.

Two verbs have reached a board verdict and **both were rejected on
measurement**. That is the loop working, not failing, and it is the reason the
rejections are on the record rather than the successes.

A third measured result is a refusal rather than a rung: on the `dense2`
workload `advice_to_split_hint` emits **0 splits**, because yolov8_nano's
largest dispatch is 2.238 ms against a 16.527 ms periodic slot (13.5%) and
`blocking_advice` fires only above 100%. Nothing blocks, so nothing needs
splitting, and B4 put 4-way OC sharding at +76% total work. The 64×96 graph is
already fine enough for that workload — which is what dropping from 160×160
bought.

Every bridge takes `--advice --ir --model --out`. Each derives what it can from
the measurement rather than taking it as a parameter — `advice_to_split_hint`
computes `n_splits` from `ceil(service_time / max_target_piece)`, rounded up to
a divisor of the tilable axis, where `decision_loop.py` hard-codes `2`.

#### Every bridge proves the advice came from this graph, first

`dispatch_id`s renumber on every rewrite, so an id that still resolves is not
evidence that it resolves to the *same op*. Worse, the obvious check does not
fire for the ops that matter: `profile_writer` writes the literal string
`noshape` wherever it could not read a shape, and it could not read one for any
fused op until `_conv_shape_of` existed — so every fused dispatch in every
profile on disk is `noshape`, and the op **kind** alone proves nothing because a
model's topology is identical at every input size.

This was found by hitting it. The split bridge cheerfully joined the 320×320
`yolov8_nano` profile (226.86 ms, 90 dispatches, every conv `noshape`) against
the deployed 64×96 IR (46.4 ms), passing every per-op check, and derived split
factors from service times 25× too large.

So identity is established from the **whole advice set** — the elementwise and
concat dispatches carry real signatures even when the convs do not
(`xpu-rt/advice_join.py`, shared by all four bridges so they cannot disagree
about what counts as the same graph):

```
advice(320×320) vs IR 64×96     33 of 33 checkable dispatches disagree → refused
advice(320×320) vs IR 320×320   33 of 33 agree → proceeds
```

#### The trace's `dispatch_id` is a record slot, not a dispatch id

`generate_skeleton` sizes the harness's profile record array by the ops that
emit a kernel call — `view` and the `chunk2_c1` family emit none — and the
harness stamps each record with its **slot** in that array. The column is called
`dispatch_id`. It drifts from the IR numbering that the schedule, the profile
CSV and the advice all use, by the number of zero-cost ops seen so far.

`dronet` and `mlp_control` have none, so their numbering is the identity. That
is why every earlier per-dispatch validation on this path was clean.
`yolov8_nano` has 8, and 44 of its 90 dispatches join to an op of a different
kind:

| joined on | median rel. err | mean | worst \|abs\| | worst case |
|---|---|---|---|---|
| the trace's raw id | −3.6% | +204.4% | 16 913 µs | d81: pred 17.465, "meas" 0.577 (−96.8%) |
| the IR dispatch id | **−2.4%** | **−2.6%** | 2 249 µs | d77: pred 9.567, meas 11.816 (+23.5%) |

Both sets of numbers are real. Only one compares an op against itself: trace
slot 81 is `detect.cv3_1_2`, a 0.577 ms `conv2d_s8`, while IR dispatch 81 is
`detect.cv3_0_1.conv`, an 18.7 ms fused conv.

Note the **median barely moved**. A summary statistic does not reveal this; only
the spread does. Pass `--ir <graph.json>` (repeatable) to `join_k1_trace.py` and
`plot_predicted_vs_measured.py` to translate. The audit itself is always on and
needs no IR — the schedule carries the op kind in `module_name`, the trace in
its own `op` column — and it refuses by default rather than printing numbers
that compare different ops.

The upstream fix is for the harness to stamp the IR `dispatch_id`; that needs a
board rebuild and re-run, and this makes every trace already taken readable.

### Running a REWRITTEN IR on the board

The loop's output is a graph.json that no `--model` can regenerate, so the
runner has to be handed it:

```bash
bash ModelBlaster/scripts/run_xpurt_k1.sh \
    --schedule <scheduled_*.json> \
    --staged-ir <network>:<dir containing graph.json, weights.npz, io.npz> \
    --models <network>,<...> --backends rvv_x60,rvv_x60
```

`--staged-ir` copies the graph, records its source path and sha256 in
`.staged_from` (deliberately *not* `.extract_config`, so a staged graph can
never satisfy the extraction reuse check), and **renames the IR's `name` field
to the network name** — every generated C symbol mangles from it
(`model_<name>_op_record_t`, `MODEL_<NAME>_OUTPUT_SIZE`), while `xpurt_main.c`
mangles from the schedule's network name. Leave them different and the harness
fails to compile against a type it just generated.

The same flag covers a network whose name is not a `models/` module —
`networks_dense2` calls its detector `yolov8_nano_64x96`, which is
`yolov8_nano` built at another input size.

### Two toolchain traps that cost a board slot each

**Use GCC 14.3, not 13.2.** `run_xpurt_k1.sh` defaults `CROSS` to chipyard's
`riscv64-unknown-linux-gnu-` 13.2, while the single-model profiling path builds
with the spacemit 14.3 one. GCC 13.2 **reorders the `__riscv_vsetvl_*`
intrinsics**, so a widening instruction executes under the narrow vtype:

```
vsetvli e32,m4     <- sets SEW=32
vsetvli e8,m1      <- clobbers it to SEW=8
vle8.v / vle8.v
vsext.vf4          <- ILLEGAL: widening 8->32 needs SEW=32
```

Measured in `kernel_add_s8`: SIGILL with no stdout, `epc 0x17020`, `badaddr`
equal to that instruction's own encoding. Same source and flags, GCC 14.3
emits `vsetvli e32,m4 / vle8 / vsext.vf4` and it runs. Set:

```bash
export CROSS=<...>/spacemit-toolchain-linux-glibc-x86_64-v1.1.2/bin/riscv64-unknown-linux-gnu-
```

This does not invalidate earlier scheduled runs: a wrong vtype is an *illegal
instruction*, so it crashes rather than quietly computing a wrong answer.

**A network name may end in a digit.** `<network><instance>` has no separator,
so `yolov8_nano_64x96` reads as `yolov8_nano_64x` + instance 96 under a
trailing-digit split. That emitted `#include "yolov8_nano_64x_model.h"` and
failed the build; it also greyed the model out of every figure. Pass
`--networks` to `generate_xpurt_main` (the runner does), and prefer the trace's
own `network` column in any renderer.

### `shard`: documented, deliberately not wired

`shard_advice` is per dispatch; the mechanism (`MB_SHARD_FACTOR`, with
per-shard re-packed weights) is **build-level and whole-model**. Before building
a bridge across that gap, two things had to be true, and neither is:

* **The advice cannot fire at all.** It needs `profiles_by_cores` with a 1-core
  baseline *and* multi-core measurements. Only `topo_0` exists — all 16 profile
  directories on disk. This is a data-collection gap, not a code gap.
* **The measured ceiling is low.** B4 measured 4-way OC sharding costing
  **+76% total work** before it buys any parallelism, so its ceiling is 2.27×,
  not 4×.

So the verb stays documented and unwired, which is an outcome rather than an
omission. The mechanism is real and tested (`_OC_SLICEABLE_CONV_OPS`, per-shard
weight re-packing) and reachable by setting `MB_SHARD_FACTOR` at codegen. What
does not exist is advice that selects it, and it should not be written against
data that does not exist.

### Two worked round-trips, both rejections

Both are complete and both ended in the rung being rejected on measurement,
which is what these are here to show.

**Fusion — `mlp_control`, 7 ops → 4** (`artifacts/k1_run/round1_mlp_control/`).
The evidence only appeared *after* the previous round: once the curated RVV
linear kernel landed, the `elu` ops went from noise to **39.7%** of runtime.
Advice that would have been wrong at round 0 is right at round 1 — the argument
for closing the loop rather than optimising once.

```bash
$PY scripts/advice_to_fusion_hint.py --advice artifacts/k1_run/compile_advice.json \
    --ir ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json \
    --model mlp_control --out round1/fusion_hint.json --pair-only
cd "$MB_ROOT" && $PY -m modelblaster.pipeline.apply_fusion_hint \
    --hint round1/fusion_hint.json --model mlp_control \
    --ir  build/k1_xpurt/mlp_control/int8/graph.json \
    --out build/k1_xpurt/mlp_control/int8/graph.fused.json
$PY scripts/diff_dispatch_graph.py --before …/graph.json --after …/graph.fused.json
```

Result: **36% slower on the board**. Rejected.

**Split — DroNet dispatch 0, OC 32 → 2×16**
(`artifacts/k1_run/round_B3_dronet_split/`). Host-verified bit-exact first
(100 352 elements, 0 differing, tiles distinct), then measured: **+13.7%**.
Rejected. It first read as −0.2% against a **stale baseline** that predated the
`_zfh_zvfh` march change and sat inside the same round directory — panel c of
`k1_granularity_b3` exists to show that trap.

The split path now reaches the ops that carry the runtime.
`apply_split_hint` accepts `conv2d_batchnorm2d_s8` and
`conv2d_batchnorm2d_silu_s8` — 97% of yolov8n and 29% of DroNet — verified
bit-exact on the deployed graphs with the unmodified
`verify_ir_rewrite_host.py` verdict:

```
yolov8_nano d0 ×2   24576 elems  golden 0/0  diff 0  bit_exact True
yolov8_nano d0 ×4   24576 elems  golden 0/0  diff 0  bit_exact True
dronet      d3 ×2    6272 elems  golden 0/0  diff 0  bit_exact True
dronet     d13 ×4    2048 elems  golden 0/0  diff 0  bit_exact True
```

**A split graph has no profile.** Do not schedule it by dividing the parent's
cost by `n`: B4 measured 4 OC tiles costing ~0.44 of the parent *each*. The
order is rewrite → rebuild → **reprofile on the board** → schedule on the
measured finer profile. A schedule built on derived costs measures the
derivation, not the solver.

### Figures

Committed scripts, measured data, output into the gitignored `out/figures/`.
`scripts/figstyle.py` holds the print rcParams and the palette — a model keeps
one colour across every figure, which it did not before.

```bash
$PY scripts/plot_loop_schematic.py --out-dir out/figures
$PY scripts/plot_granularity_evolution.py --out-dir out/figures
$PY scripts/plot_k1_trace_gantt.py --trace <trace.csv> --schedule <sched.json> \
    --out out/figures/k1_schedule_measured.png --window-ms 300
$PY scripts/plot_predicted_vs_measured.py --schedule <sched.json> --trace <trace.csv> \
    --ir <each model graph.json> --out-dir out/figures
```

| figure | what it shows |
|---|---|
| `k1_loop_schematic` | the loop with the contract on every arrow, plus each verb and where its round-trip actually reached |
| `k1_granularity_b3` | the DroNet split rung, including the stale-baseline trap |
| `k1_schedule_measured` | three models on eight real harts, measured |
| `k1_predicted_vs_measured` | the profile's predictive quality (median −2.4% over 1585 dispatches, four decades) beside the same data joined on the raw id |

### What does not work on this path yet

* **`runtime/scripts/verify_ime_build.sh`** needs merlin's `llvm-objdump` (or
  `OBJDUMP=`). It gates an IREE artifact; nothing on the current path emits IME
  instructions at all.
* **The harness stamps a record slot, not the IR `dispatch_id`** — see above.
  Every consumer works around it; the producer should be fixed.
* **`shard_advice` cannot fire** for want of multi-core profiles — see above.

Previously listed here and now resolved: `emit_compile_advice.py` reads
ModelBlaster profiles (`--profile-format csv`, now the default);
`join_k1_trace.py` reads both trace schemas via `xpu-rt/k1_trace.py`;
`advice_to_fusion_hint.py` is unblocked along with three sibling bridges; and
`scripts/apply_compile_advice.py` is deleted — it rewrote `.vmfb` paths that do
not exist on this path, and `choose_implementation` now has a live consumer.

---

## 9. What IME actually does

Full record, with disassembly, in `artifacts/k1_run/ime_gate/FINDINGS.md`. The
short version, because three earlier claims in this document were wrong:

* **The micro-tile is 4×4×8 and hardware-forced.** The spec's MAC-unit table is
  indexed by `vl*SEW`, not VLEN: at VLEN=256, SEW=8, `vl=32` → `M×N×K = 4×4×8`.
  K is pinned at 8; deep reductions are a loop of accumulating `vmadot`s.
  `vsetvli` must set `vl=32, e8, m1` immediately before the instruction or it
  SIGILLs.
* **The discriminator is M, not K.** Every `xsmtvdot` lowering path in the
  vendored IREE requires `M0=4 && N0=4`. MLP gets no `vmadot` because **M=1**
  (GEMV), not because its K is small.
* **The one `vmadot` is real, executed, and 15/16 wasted.** In
  `dronet$…_matmul_1x1x2048`, inside a loop with a live back edge — and four
  instructions later a *masked* `vse32.v` stores **one** of the sixteen int32
  results. It is 23% faster than the RVV form while discarding 15/16 lanes, and
  it is 0.075% of DroNet's runtime: worth 0.027 ms of 122.7.
* **The "IME wins" in `compile_advice.json` are not IME wins.** Dispatches 14
  and 7 contain no `vmadot` at all; the speedups are incidental codegen variation
  from the `+xsmtvdot` data-tiling path.
* **No convolution can reach IME here.** Every `xsmtvdot` hook in that IREE is
  matmul-only; no conv→img2col/mmt4d path is wired to it. DroNet is 111 of its
  122.7 ms in convolutions.
* **The ukernel route was tried and refuted.** `IME_ukernel` measures 122.73 ms
   — identical to `IME` — while containing **zero** `vmadot`. So the entire
  +7.9% "IME penalty" is the data-tiling path acting on code that is pure RVV
  either way. Kept in the yaml as a recorded negative result so nobody re-runs
  it.

The honest statement is that **IME is untested on this workload, because the
workload never reaches it.** Two routes would change that: pad narrow-M matmuls
to `M0=4` instead of shrinking the tile to `{1,4,8}`; or wire conv through
img2col/mmt4d so it can hit the matmul hooks — the only route with access to the
111 ms where DroNet's time actually is.

For a ModelBlaster kernel, note there are **no clang builtins and no intrinsics
header** for this extension: you need `.insn r 0x2b, 3, 0x71, …` or the LLVM IR
intrinsic. LLVM has no scheduling model for `smt.vmadot`, so latency and
throughput must be measured, not looked up. One `vmadot` = 128 int8 MACs against
32 for an RVV `vwmul`+`vwadd` pair, so the instruction-count ceiling is 8:1.

---

## 10. Contention, measured under control — and it inverts the obvious assumption

`runtime/scripts/k1_contention.py` pins the dispatch under test to core 0 and
runs a co-runner on a chosen other core, comparing against the same dispatch
with nothing else running. The co-runner must be a **different** benchmark: an
identical one shares its weights, and a same-L2 co-placement then looks
*helpful* rather than contended (1.034× with the same module vs 1.088× with a
different one, same dispatch). The script's own docstring still says "the same
benchmark binary" and is wrong; the measurements below used a different one.

| dispatch | solo | co-runner same cluster | co-runner other cluster |
|---|---|---|---|
| dronet.0 | 0.581 ms | 1.088× | **1.298×** |
| dronet.10 | 0.416 ms | 1.103× | **1.312×** |
| dronet.11 | 0.222 ms | 0.995× | 0.937× |
| dronet.12 | 9.581 ms | 1.053× | **1.233×** |
| dronet.13 | 18.321 ms | 1.034× | **1.137×** |
| dronet.14 | 1.317 ms | 1.014× | 1.031× |
| **median** | | **1.043×** | **1.185×** |

**Co-running on the *other* cluster costs ~18%; on the *same* cluster ~4%.** That
is the opposite of what the resource model suggests — cores 0-3 and 4-7 have
separate 512K L2s, so spreading across clusters looks like the way to avoid
interference, and on this SoC it is roughly four times worse.

The magnitude also matches the otherwise unexplained gap in the first
predicted-vs-actual join: solo profiles ran 17-25% optimistic on the large
convolutions during a run with both clusters busy, and cross-cluster co-running
measures 13-31% here. So that gap was contention, and specifically
*cross-cluster* contention.

**Confidence: moderate, mechanism unexplained.** Six dispatches, four
repetitions, one co-runner kind; two of the six show no effect. A plausible story
is shared memory-controller or interconnect pressure dominating L2 isolation, but
this experiment does not establish it. Before it changes placement policy it
wants more co-runner kinds, more repetitions, and a memory-bandwidth control to
separate interconnect pressure from cache effects.

If it holds, the scheduling consequence is concrete and contrarian: **prefer
packing concurrent work onto one cluster** rather than spreading it.

The measurements were taken through the IREE path, on IREE dispatches. Nothing
about the mechanism is IREE-specific, but they have **not** been reproduced
against the generated-C kernels, and doing so is the cheapest way to raise the
confidence level.

---

## 11. The retired IREE/merlin path

Kept because the numbers explain the current design, not because you should run
it. Nothing below is a step.

* **Assertions cost 26%.** IREE re-enables assertions on top of `Release`;
  same dispatch, same core, **80.0 µs with, 63.2 µs without**. The
  google-benchmark "Library was built as DEBUG" warning was the tell.
* **`unset LDFLAGS` was load-bearing.** Activating a conda env exports
  `LDFLAGS=-L$CONDA_PREFIX/lib`; CMake captured it and every RISC-V link then
  resolved `-lstdc++` to the x86 one
  (`ld.lld: … is incompatible with elf64-littleriscv`). The same class of bug is
  why `deploy_k1.sh` checks `e_machine` before copying.
* **Per-dispatch overhead dominated.** MLP: 5 dispatches, 335.7 µs total, every
  dispatch 63-78 µs *regardless of work* — ~94% launch overhead. DroNet:
  19 dispatches, 113.7 ms, top-5 dispatches 75% of it. The generated-C path's
  smallest dispatch is 2.6 µs, which is the whole argument for the move.
* **Cluster symmetry.** cluster-1/cluster-0 median ratio **0.996** on
  compute-bound dispatches: the clusters are equivalent for RVV, so placement
  across them is a scheduling choice, not a performance one.
* **Worker count, not device pinning, was the concurrency limit.** The runner
  created one device per cluster and ran two worker threads total, so an 8-way
  schedule was serialised onto two threads. `--pin_per_core=1` is what created
  one pinned device and one worker per physical core — **and it is not
  optional**, which is why the historical command is written here with it:

  ```bash
  # RETIRED -- for the record only
  ssh k1 'cd /root/mb_k1 && ./bin/merlin-dispatch-scheduler schedule.json local-task 1 1 0 \
      --vmfb_dir=/root/mb_k1 --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5,6,7 \
      --visible_cores=8 --variant_p=RVV --variant_e=RVV --pin_per_core=1 \
      --trace_csv=/root/mb_k1/trace.csv'
  ```

  | runner configuration | makespan | queueing | MLP deadline misses |
  |---|---|---|---|
  | 1 worker/target, cluster device | 1017.2 ms | 87.6% | 32/32 |
  | 1 worker/target, per-core device | 1022.1 ms | 87.7% | 31/32 |
  | **8 workers, per-core device** | **448.5 ms** | **6.1%** | **2/32** |

* **Pin to as many cores as the profiles used.** `--cpu_p_cpu_ids=0,1,2,3`
  against single-core profiles ran dispatches 3.75× faster than planned
  (18.29 ms planned, 4.88 ms actual) and made every calibration number
  meaningless.
* **Prediction error decomposes.** Aggregate median +19.5%, but **+1.7%**
  restricted to dispatches ≥1 ms: the cost model is accurate on the work that
  matters and wrong on small dispatches, where an unmodelled fixed per-dispatch
  overhead dominates. By op type: matmul +32.5% (n=138), reduction +23.5%,
  elementwise +20.4%, conv **+2.3%**.
* **Sharding is real on this hardware.** DroNet's per-instance service time
  falls **113.7 → 62.0 → 32.4 ms** on 1/2/4 cores, and a 4-core shard cuts
  worst-case lateness from 108.8 ms to 27.2 ms (22.6 → 29.0 Hz against 30 Hz
  required). Note the sibling branch's headline "-48.6% from sharding" does
  **not** survive its own artifacts — its "unsharded baseline" binary contains
  only the two sharded tiles; see `artifacts/agentic_branch_salvage.md`. Cite
  the K1 numbers, not that one.

  **Superseded, and by how much:** those are IREE-era, single-model,
  pinned-core measurements. Through ModelBlaster's generated C with the fused
  conv kernels present, DroNet is **9.79 ms on ONE core** — the 113.7 ms
  starting point was 86.7% scalar reference code. The scaling conclusion
  (sharding helps DroNet's convolutions, and does not help the MLP) still
  stands; the absolute numbers do not. Re-derive before citing.
* **A retraction.** "B4 has 1 MLP deadline miss" was n=1 and is not
  reproducible: seven runs of the identical schedule give 7-9 misses of 38.
  MLP completes at ~7 ms against a 10 ms deadline, so its miss count is a
  knife-edge and not a discriminator between configurations. Makespan is stable
  to 1.0% across runs; achieved frequency is the honest framing.
* **A closed loop, measured, and rejected at system level.** Retargeting 50
  dispatches on compile advice took them from 21 991 µs to 16 172 µs
  (**−26.5%** measured against −21.2% predicted) while total service time moved
  **+0.2%** — those dispatches are 1.8% of the workload. Accepted per-dispatch,
  rejected as a system-level win. Executing the schedule is what tells you the
  difference; a kernel benchmark cannot.

---

## 12. Failure modes, in one place

| symptom | cause | fix |
|---|---|---|
| `ssh` hangs at `Next authentication method: publickey` | stale `ssh-agent` | `IdentityAgent none` in the ssh stanza |
| binary SIGILLs on its first timed dispatch | `rdcycle` from userspace | `--platform linux` (already passed by both runners) |
| every timing 67× off | converted with 1.6 GHz instead of 24 MHz rdtime | `--profile-clock-mhz 24`, `--clock-mhz 24` |
| `entries_done=0`, all-zero trace | `core_kind` ≠ backend tag (`rvv` vs `rvv_x60`) | set `CORE_KINDS` to the schedule's kind |
| schedule uses only harts 0-3; cluster 1 idle | `CPU_P#n` and `CPU_E#n` alias to the same registry core | one pool: `"machines": {"cpu_p": 8}` |
| solver reports "no profile found" | profile tree not under a directory named `profile` | `PROFILE_OUT_ROOT=$XPURT_ROOT/gen_mb/profile`, `gen_root: "gen_mb"` |
| a 4-core shard is never selected | `topo_tag_override: true` charges it the single-core cost | `topo_tag_override: false` with `machine_combination_mode: "shard"` |
| `No module named 'modelblaster'` from CMake | `backend_rename` was invoked with `PYTHONPATH=<repo>/..`, which only works if the checkout dir is *named* `modelblaster` | fixed in `harness_xpurt_linux/CMakeLists.txt` (now `<repo>/src:<repo>`) |
| `refusing to run: 'modelblaster' resolves to …` | an editable install points at a different checkout | fix `PYTHONPATH`/the install; do not bypass |
| `emit_measured_report` matches 0 rows | report entries have no `network`/`instance`; they are inside `name` | unfixed — see §7 |
| `plot_xpurt_trace` raises `ZeroDivisionError` | schedule has all-zero predicted durations | use a schedule from §5 |
| `makespan_us` looks 1000× too small | nine `*_us` fields carry ms | read `*_ms` / `units_note` |
| load average 2.00 on an idle board | wedged vendor kernel threads `vq0`/`vq1` | ignore |

---

## Artifacts

```
artifacts/k1_bringup/<ts>/        board inventory, topology, IME probe + source
artifacts/k1_progress.md          running log: what was tried, what failed, why
artifacts/k1_run/ime_gate/FINDINGS.md   what IME really does, with disassembly
artifacts/agentic_branch_salvage.md     audit of the sibling branch; the label-provenance rule
ModelBlaster/artifacts/agentic_fuse_split/WARNING.md   the bookkeeping-fiction speedup, kept verbatim
ModelBlaster/cores/spacemit_k1.json     the measured board topology, machine-readable
gen_mb/profile/<backend>/spacemit_x60/<model>/<model>.<quant>/<spec>/topo_<cpu>/results.csv
gen_mb/vmfb/<model>/spacemit_x60/<backend>/<basename>/<basename>_dispatch_graph.json
schedules/scheduled_<config>_<solver>_profiled{,_metrics,_report}.json
ModelBlaster/build/k1/<model>/<quant>/          single-model IR, weights, generated C
ModelBlaster/build/k1_xpurt/{<model>,_gen,_build}/   schedule-driven build tree
```

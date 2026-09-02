# firesim1 runbook — continuing the freshness evaluation on real FPGAs

Everything in Phases 0–12 of the freshness evaluation is profile-driven and runs
on `garden` with no FPGA. This document covers the handoff for the parts that
need hardware: Phase 13 (validate candidates through ModelBlaster → FireSim) and
Phase 14 (extend execution traces).

Verified 2026-08-03. Every fact below was read off the machine, not assumed.

---

## 1. Why firesim1 rather than the local rig

The Alveo board attached to `garden` is a single shared resource behind dima's
`firesim_queue.py daemon`, and its selected `default_hw_config` is
`alveo_u250_firesim_dual_saturn_v256d128`, which has neither Gemmini nor OPU.

`firesim1.millennium.berkeley.edu` (169.229.49.76) has **six Alveo U250 boards**,
no queue daemon, 64 cores and 503 GB RAM (472 GB free as measured). We have
passwordless sudo over the entire local-FPGA flow.

`firesim2.millennium.berkeley.edu` (169.229.49.75) has **4× NVIDIA L40S** (46 GB
each, driver 580.126.18 / CUDA 13.0), typically 0 % utilised, but **no FPGA and
no XRT**. It is a GPU host, not an alternative simulation target. Its CPU is
heavily contended (load ≈ 23, 18 users, 20 GB RAM free), so the GPUs are free
but the host feeding them is not.

---

## 2. Access

```bash
# firesim1 has NO Host block in ~/.ssh/config (only firesim2 does)
ssh -o User=copparihollmann firesim1.millennium.berkeley.edu
```

**ssh needs a live forwarded agent.** Every on-disk key (`id_ed25519`,
`DIMA_SLICE`, `id_rsa`) is rejected by millennium with
`Permission denied (publickey)`, and the socket recorded in `~/.ssh/AGENT_VARS`
is stale. What works is a live agent holding the laptop key
(`coppa@Agustin-X1`):

```bash
export SSH_AUTH_SOCK=/tmp/ssh-<...>/agent.<pid>   # find with: ssh-add -l against each
```

That socket dies with its session. Worth refreshing `~/.ssh/AGENT_VARS` and
adding a `Host firesim1` block so this is not rediscovered every time.

Permissions, as measured:

```
groups: u-copparihollmann, firesim(1268), tstech16c, tstech16i, bwrcgroup
sudo:   (ALL) NOPASSWD: /usr/local/bin/firesim-*
```

`firesim-*` covers `firesim-bind-xdma`, `firesim-unbind-xdma`,
`firesim-chmod-xdma-perm`, `firesim-change-pcie-perms`,
`firesim-load-xdma-module`, `firesim-fpga-util.py`,
`firesim-xvsecctl-flash-fpga`, `firesim-generate-fpga-db.py`. So the whole flow
is available without asking anyone.

Note `/dev/xdma*` are `root:root 0600` at rest — group membership alone does not
grant access. `firesim-chmod-xdma-perm` is what opens them, which is also a
usable signal that nobody has claimed a board (see §4).

---

## 3. Which board is "FPGA 0" — READ THIS BEFORE FLASHING

dima has agreed to leave **FPGA 0** to us. That phrase is ambiguous on this
machine and the two readings are **different physical boards**:

| xdma index | BDF | serial | firesim-db `device` |
|---|---|---|---|
| `xdma0` | `1d:00.0` | 2133072BM053A | `xcu250_0_1` |
| `xdma1` | `1e:00.0` | 2133072BM057A | `xcu250_0_3` |
| **`xdma2`** | **`3d:00.0`** | **21330497N029A** | **`xcu250_0`** |
| `xdma3` | `3f:00.0` | 21330497N01TA | `xcu250_0_5` |
| `xdma4` | `40:00.0` | 21330497P056A | `xcu250_0_4` |
| `xdma5` | `41:00.0` | 2133072BM059A | `xcu250_0_2` |

The `xdma` index comes from PCI probe order (confirmed from
`dmesg | grep probe_one`), the `device` name from `/opt/firesim-db.json`. So:

- **FireSim slot 0** = `xcu250_0` = BDF `3d:00.0` = serial `21330497N029A` = `xdma2`
- **`xdma0`** = `xcu250_0_1` = BDF `1d:00.0` = serial `2133072BM053A`

dima works through FireSim, so "FPGA 0" most likely means `xcu250_0` /
`3d:00.0`. **Confirm by serial before flashing** — `21330497N029A` — because
picking the wrong reading means flashing a board someone else is using. Always
address boards by `--serial` or `--bdf`, never by a bare index.

---

## 4. Checking a board is actually free

There is **no lock file, reservation table, or queue daemon** on firesim1
(unlike garden). Coordination is social — ping dima and chrisdong.

The available evidence, in decreasing strength:

```bash
ls -la /dev/shm                      # FireSim allocates sim slots here; empty => no sim
pgrep -af "FireSim-|vivado|hw_server|xsim"
ls -l /dev/xdma2_user                # still root:root 0600 => nobody ran firesim-chmod-xdma-perm
who
```

`fuser /dev/xdma*` is **not** informative as a normal user: the nodes are
root-only, so it cannot see holders either way and prints nothing whether the
board is busy or idle. Do not read its silence as "free".

Be aware `chrisdong` runs jobs on firesim1 that use it only as a *launcher* for
AWS F2 instances (`--strategy firesim` with ssh tunnels to `ubuntu@…` and
`firesim.pem`). Those consume CPU but touch no local board — do not mistake them
for local FPGA use, and do not assume the CPU is idle either.

---

## 5. Fresh clone — do NOT reuse garden's chipyard trees

The chipyard working trees on garden carry uncommitted state and hand-placed
submodule pointers (dima owns the Q31 overlay). Build from a **fresh clone on
firesim1**:

```bash
git clone git@github.com:ucb-bar/XPU-RT.git
cd XPU-RT
git checkout feat/freshness-validity-eval
git submodule update --init --recursive hw/chipyard
```

That last command resolves the **whole Q31 hardware stack** with no manual
remote surgery, because XPU-RT pins `hw/chipyard` at `f69af12a` — the `xpurt`
branch tip — whose nested pointers are:

| submodule | fork / branch | carries |
|---|---|---|
| `generators/gemmini` | CobbledSteel/gemmini `xpurt` | Q31 acc_scale, **`Q31Configs.scala`**, GemminiConfigs Rocket wrappers |
| `generators/rocket-chip` | CobbledSteel/rocket-chip `xpurt` | `WithRocketFPU{32,16}`, FP16-only FPU |
| `generators/saturn` | CobbledSteel/saturn-vectors `xpurt` | FP-stripping VectorParams, Saturn OPU dual-core |
| `toolchains/…/riscv-isa-sim` | CobbledSteel/riscv-isa-sim `saturn-opu-extension` | OPU Spike support |
| `gemmini/software/libgemmini` | ucb-bar/libgemmini `xpurt` | `gemmini_params.h` for Q31Ws 32×32 |

Commit `239e1bb` ("bump hw/chipyard: vendor xpurt tip") exists specifically to
make this one command work on a fresh clone.

**Do NOT run a bare `git submodule update --init --recursive`.**
`zephyr-chipyard-sw` holds two unpushed commits (`03ee2e2`, `de22a1d`) with
bit-exact FireSim recalibration data plus five modified profile CSVs; a blind
update discards them. Restrict the update to `hw/chipyard`. `merlin` is out of
scope for this project (LLVM bulk), and `sims/IsaacLab` stays unpopulated.

---

## 5b. Local FireSim on the U250s works — confirmed from other users' trees

Recorded because an earlier draft of this file treated it as an open question on
the grounds that firesim1 has no `/opt/xilinx/xrt` and no `vivado` on PATH. That
caution was wrong on both counts:

- **XRT is not part of this flow.** The Alveo U250 path is XDMA plus either
  Vivado JTAG or `xvsecctl` MCAP programming (`/usr/local/sbin/xvsecctl` is
  installed and `firesim-xvsecctl-flash-fpga` is in the passwordless sudo list).
- **Vivado is installed**, just not on the default PATH:
  `/ecad/tools/xilinx/2025.1/Vivado`, with `/ecad/tools/fpga.bashrc` to set the
  environment up. `/ecad` is a 59 T NFS mount with 57 T free.

The direct evidence that local runs happen, from readable trees:

| tree | evidence |
|---|---|
| `/scratch/ferran/chipyard1/sims/firesim/deploy` | `default_platform: XilinxAlveoU250InstanceDeployManager`, `default_simulation_dir: /scratch/ferran/FIRESIM_RUNS_DIR`, `default_hw_config: rocket_u250_lowfreq` |
| `/scratch/chrisdong/chipyard-fsim/sims/firesim/deploy` | `default_hw_config: alveo_u250_saturn_refv256d128_singlecore_20mhz`; `results-workload/` holds `…-kb_local_slot_2` and `…-kb_local_slot_3` dated 2026-08-03 |

Those `kb_local_slot_N` names are local-FPGA slots, so at least two boards were
in active local use the same day. That is consistent with slot 0 being the one
dima set aside — and it also means §4's "looks free" checks must be re-run
immediately before claiming a board, not trusted from this document.

Note `/scratch` on firesim1 is **97 % full** (3.5 T, ~116 G free). A chipyard
clone plus conda env plus FireSim build is tight; check headroom first. `/` has
182 G free.

## 6. Bitstreams

The bitstream this evaluation's profiles were measured against is
`alveo_u250_firesim_shuttle_gemmini_opu`
(`FireSimGemminiAndOPUShuttleConfig`: tile 0 Shuttle+Gemmini RoCC, tile 1
Shuttle+Saturn OPU vLen=128). It is **already built** — a 57 MB
`firesim.tar.gz` under garden's `results-build/2026-05-06--22-08-53-…`. Copy it
rather than rebuilding, and point a firesim1 `config_hwdb.yaml` entry at the
local copy.

**There is no prebuilt Q31 bitstream anywhere.** `config_hwdb.yaml` on garden
has zero Q31 entries. The `xpurt` branch does carry the FireSim target config
`FireSimREFV256D128DualSaturnOPUGemmini32x32Q31WsConfig`, so a Q31 bitstream is
*reproducible* from the pinned pointer — but it is a fresh Vivado build (hours),
not a copy. An earlier draft of this project's limitations wrongly said the Q31
scala config was absent from the checkout; it is present, in the **gemmini
generator**, not in chipyard's own config directory.

This pass deliberately uses `gemmini` + `rvv_opu`, not Q31, so the Q31 build is
not on the critical path. It matters only if the `mlp_control` backend-
differentiation limitation (see §7) becomes worth removing.

---

## 7. What hardware can and cannot validate here

Phase 13 can validate **timing fidelity** — does the schedule execute as the
solver predicted, and are the FireSim-measured cycles reproduced.

Phase 13 **cannot validate freshness**. No producer→consumer dataflow exists
between DroNet and `mlp_control` at any layer: XPU-RT `edges` become MILP
precedence constraints, ModelBlaster's `deps` are `k_sem` ordering edges, and
buffers are named `buf_<model_id>_<tensor>` — mangled by model id only, with no
instance or version, and all K instances of a network sharing one output buffer.
So `producer_instance_consumed` stays
`inferred_from_schedule_timestamps` on hardware exactly as in simulation. Any
report must say so.

Two independent blockers to know about before planning hardware work:

- **Phase 15 (runtime epoch-boundary switching) does not compile.**
  `runtime/tools/xpurt_scheduler_runner.c` assigns six
  `scheduler_runner_config_t` fields (`schedule_next_path`, `telemetry_fd`,
  `cpu_cpu_ids`, `qnn_gpu_enabled`, `qnn_hta_enabled`, `telemetry_jsonl_path`)
  that do not exist in the pinned merlin `scheduler_runner.h`. merlin is out of
  scope for this project, so Phase 15 is blocked independently of FPGA access.
- **The 1 GHz clock assumption.** All ms figures are FireSim-measured cycles /
  1e6 at an assumed 1 GHz. The bitstreams close timing at 25–30 MHz, and the
  on-device trace counter is mtime (~1 MHz on FireSim, 10 MHz on Spike). At the
  real 25 MHz a 10 ms control period is impossible. Hardware runs must report
  raw cycles and restate the assumption; they do not resolve it.

---

## 8. Running through ModelBlaster

Use `ModelBlaster/scripts/run_bundle_firesim.sh`. On garden that means
`FIRESIM_QUEUE=1` to go through dima's shared queue. On firesim1 there is no
queue, so the board must be claimed manually per §3–§4.

**Never edit a shared `deploy/config_runtime.yaml` in place** on a machine
someone else is using — add an hwdb entry and select it for your own run.

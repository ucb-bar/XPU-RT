# SpaceMiT K1 bring-up — Milestone 0

Captured 2026-08-26 from host `garden` (136.152.139.10).
No credentials are recorded here or anywhere in this tree.

## Access

| | |
|---|---|
| Console | `/dev/ttyUSB4` — CH340 (`1a86:7523`), 115200 8N1, group `dialout` |
| Not the K1 | `/dev/ttyUSB0` — CP2102N (`10c4:ea60`) is a different board; silent |
| Hostname | `k1` |
| IP | `10.44.98.236` on **wlan0** (WiFi). `end0`/`end1` are DOWN. |
| SSH | `ssh k1` (stanza added to `~/.ssh/config`), key `/scratch2/agustin/DIMA_SLICE` |
| User | `root` (`PermitRootLogin yes`, key already in `/root/.ssh/authorized_keys`) |
| RTT | ~4 ms |

### SSH gotcha — cost ~20 min, do not rediscover

`ssh root@10.44.98.236` **hangs at `Next authentication method: publickey`** whenever a
stale `ssh-agent` socket is in the environment. On this host `SSH_AUTH_SOCK` pointed at a
dead agent (`ssh-add -l` itself hangs). The board is blameless: `journalctl -u ssh` shows
this same key succeeding from this same host minutes earlier, and our failed attempts land
as `Connection closed by authenticating user root ... [preauth]` — the *client* gave up.

Fix, baked into the `~/.ssh/config` stanza: **`IdentityAgent none`**.

## Hardware

- **Spacemit(R) X60**, 8 × riscv64, Bianbu 3.0 (Plucky Puffin), Linux 6.6.63, glibc 2.41.
- Uptime 37 days.

### Cluster topology — CONFIRMED, not assumed

L2 cache domains give hard evidence for the 4+4 split:

| Cluster | Cores | L2 | L1i/L1d |
|---|---|---|---|
| 0 | **0,1,2,3** | 512K unified, `shared_cpu_list=0-3` | 32K/32K private |
| 1 | **4,5,6,7** | 512K unified, `shared_cpu_list=4-7` | 32K/32K private |

This matches the brief's hypothesis. Note the in-tree Merlin runner defaults to
`--cpu_e_cpu_ids=4,5` — only 2 of cluster 1's 4 cores. We should use all four.

### Frequency — one domain, already pinned

- `cpufreq-dt`, a **single policy (`policy0`) with `related_cpus = 0-7`**.
  All 8 cores share one frequency domain; the clusters **cannot** be clocked independently.
- Governor is already **`performance`**, fixed at **1 600 000 kHz**.
- OPPs available: 614.4 / 819 / 1000 / 1228.8 / 1600 MHz.
- **No governor change is needed for profiling**, so none was made.
- **`PROFILE_CLOCK_MHZ = 1600`.**

### ISA — RVV yes, IME not enumerated

All 8 harts report an identical ISA string:

```
rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm
_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt
_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt
_sscofpmf_sstc_svinval_svnapot_svpbmt
```

- RVV 1.0 with `zve64d` and `zvfh` is present on **every** core, both clusters.
  So "RVV on cluster 0 vs cluster 1" is a fair comparison — same ISA, same L2 size.
- **`xsmtvdot` / IME / `smt.vmadot` appears nowhere in the ISA string** — but that is an
  enumeration gap, not an absence. See the probe below, which settles it.

## IME is real, and it is cluster-0 only — MEASURED

`/proc/cpuinfo` cannot answer this, so `ime_probe.c` (archived beside this file) pins each
core in turn, enables the vector unit, executes the raw IME encoding under a `SIGILL`
handler, and reports whether it trapped. Encoding taken from
`merlin/benchmarks/SpacemiTX60/compile_matmul_xsmt_i8_ukernel_all.sh:37`:

```
.insn r 0x2b, 0x3, 0x71     # smt.vmadot: opcode=custom-1, funct3=3, funct7=0x71
```

Result, identical across 3 runs (`ime_capability_probe.txt`):

| Cores | Cluster | scalar | RVV | IME (`smt.vmadot`) |
|---|---|---|---|---|
| **0,1,2,3** | 0 | yes | yes | **EXECUTED** |
| **4,5,6,7** | 1 | yes | yes | **SIGILL** |

The differential is the evidence: same static binary, same instruction, same run — the only
variable is which core it is pinned to. A merely-reserved encoding would have executed
everywhere.

**This confirms the brief's hypothesis with measurement, and it confirms the resource model
the plan calls for:** IME is a capability of a *subset of cores*, not an independent engine.
An IME dispatch on core 2 occupies core 2 — nothing else may run there concurrently.

```
K1_C0_CORE0..3   capabilities = [scalar, rvv, ime]
K1_C1_CORE4..7   capabilities = [scalar, rvv]
```

Attempting an IME kernel on cluster 1 must be **rejected before execution** — on real
hardware it does not degrade, it traps.
- `zicntr` present ⇒ `rdcycle`/`rdtime` exist; `perf_event_paranoid = 2`.

## Load average 2.00 on an idle board — benign

`uptime` reports a steady 2.00 with no process above 0.3% CPU. Cause:

```
D  153 vq0  process_theread
D  154 vq1  process_theread
```

Two vendor kernel threads wedged permanently in **uninterruptible (D) sleep**. They inflate
loadavg but consume no CPU, so they do **not** contaminate profiling. Do not chase this.

Also present: a leftover debug loop from an earlier session,
`sh -c 'while true; do echo K1HELLO | timeout 3 nc -l -p 9999; done'` (pid 5861). Idle;
left alone.

## Toolchain — nothing on the board

`gcc`, `g++`, `cc`, `clang`, `cmake`, `make`, `ninja`, `perf` are **all missing**. Only
`python3` 3.13.3. The Bianbu apt repo offers `gcc`/`g++` 14.2.0, `build-essential`,
`cmake` 3.31.6 if we choose to build natively.

Consequence: **cross-compilation is the default path**, and `setup.sh`'s SpaceMiT toolchain
fetch (glibc x86_64 v1.1.2) is load-bearing. Watch that its glibc target is ≤ 2.41.

## Exit criteria

- [x] `ssh k1 lscpu` works non-interactively
- [x] Cluster topology recorded (and proven via L2 sharing)
- [x] Clock/DVFS state recorded
- [x] IME core set established by runtime probe: cluster 0 (cores 0-3) only

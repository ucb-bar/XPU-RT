# Reproducing the arm experiments (fine-grain latency sweep, simulation)

How to go from a bare machine to the figures in `finegrain_four_tasks.png` and the
numbers in `g5fine/RESULTS.txt`: install, get the policy and the environments, run
the fine-grain latency ladder on one task, scale it across six AWS workers,
analyse it correctly, and plot it.

**What the study measures.** Six "arms". Each is a latency/cadence pair MEASURED on
a physical QRB5165 board (see `XPU-RT/qnn_models/octo/REPRODUCE_SCHEDULES.md` for
how each of those numbers was obtained), replayed in SIMPLER-env against the Octo
policy. Success is MEASURED in simulation under that MODELLED latency.

| arm | latency ms | cadence ms | what it is on the board |
|---|---:|---:|---|
| `lat0` | 0.0 | native | ideal reference, no compute latency |
| `pipe110` | 117.7 | 117.6 | pipelined, 110 ms cadence, 10 instances |
| `pipe200` | 231.8 | 203.0 | pipelined, 200 ms cadence, 5 instances |
| `serial283` | 283.4 | 283.4 | 3-way serial CPU+DSP+HTA, one in flight |
| `fp32_555` | 555.0 | 555.0 | fp32 numerically-valid CPU path |
| `cpu685` | 684.8 | 684.8 | CPU-only int8 monolith |

The table lives in `g5fine/job.sh` (the sweep's own copy) and in
`repro/common.sh` (`arm_lat` / `arm_per`). Keep them identical.

> **Known defect in the two pipelined arms, being corrected as of 2026-09-06.**
> The `latency` column for `pipe110` and `pipe200` is the board's *throughput*
> (wall / number of inference instances), not the *observation age*. For a serial
> arm the two coincide; for a pipelined arm they do not. MEASURED from the board
> traces, the end-to-end age of a result is **~385 ms** for the 110 ms schedule
> (not 117.7) and **~260 ms** for the 200 ms schedule (not 231.8) — a fresh result
> arrives every ~111 ms, but about four inferences are in flight, so each result
> is roughly 4 x 111 ms old. `g5fine/job_fix.sh` adds `pipe110fix`
> (385.1 / 111.4) and `pipe200fix` (260.5 / 219.2). Until those land, the
> sweep's finding that `pipe110` costs the controller nothing is measured against
> an optimistic latency for that arm. Derivation and the underlying traces:
> `XPU-RT/qnn_models/octo/REPRODUCE_SCHEDULES.md` section 1.1.

---

## 0. The scripts

Everything below is driven by `repro/`. Each script takes `-h`, has an explicit
usage line, uses absolute paths, and is safe to re-run.

| script | what it does |
|---|---|
| `repro/common.sh` | shared paths, `octo_env`, the arm table, `awsq`. **Sourced, not run.** |
| `repro/01_setup_local.sh` | build (or `--check-only` verify) the conda env from bare |
| `repro/02_smoke.sh` | post-install proof: imports, GPU, and every task's derived timebase |
| `repro/03_validation_gate.sh` | stock vs fine harness at zero latency — **run this before any sweep** |
| `repro/04_run_arm_local.sh` | one arm, or the whole ladder, on this box |
| `repro/05_aws_instances.sh` | `status` / `start` / `stop` / `hosts` for the six workers |
| `repro/06_aws_joblists.sh` | build `g5fine/joblist_w{0..5}.txt` for a (task, seed range) |
| `repro/07_aws_launch.sh` | push the harness, launch, and **verify the runner count** |
| `repro/08_aws_fetch.sh` | pull summaries (`--logs` for the per-job stdout) and check completeness |
| `repro/09_analyse.sh` | `analyze_fine.py` + `compare_tasks.py` + `arm_timing.py` |
| `repro/10_plot.sh` | the four sweep figures |
| `repro/11_traces.sh` | per-tick state traces, then the EE-path and motion-cost figures |

Pre-existing scripts they wrap, unchanged: `g5fine/{push,relaunch,launch,fetch}.sh`,
`g5fine/{analyze_fine,compare_tasks,arm_timing}.py`, `g5fine/make_raw_google.sh`,
`plot_*.py`, `finegrain_eval.py`, `trace_eval.py`.

---

## 1. Install, from a bare machine

```bash
ROOT=$HOME/simpler CONDA_ROOT=/scratch2/dima/miniforge3 \
  /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/repro/01_setup_local.sh
```

~40 minutes from bare. It ends with `SETUP OK`. Re-running when everything is
present takes about a minute; it is idempotent. To check an existing box without
touching it, `01_setup_local.sh --check-only`.

Pinned revisions (identical to the ones the AWS workers were built with, in
`../g5/setup_simpler_g5.sh`): SimplerEnv `06accaca`, ManiSkill2_real2sim
`ef7a4d4f`, octo `241fb351`, python 3.10, and `../g5/requirements_g5.txt` — a
`pip freeze` of the validated environment with the three editable installs
stripped out, so every transitive dependency lands at the reference version
before the editables go in with `--no-deps`.

### The traps. Every one of these has cost real time.

**`setuptools>=81` removes `pkg_resources`, and sapien 2.2.2 imports it at module
scope** (`sapien/core/renderer_config.py:4`). `requirements_g5.txt` does not pin
setuptools and `conda create python=3.10` now seeds 84.0.0, so on any *fresh*
machine `import simpler_env` dies with
`ModuleNotFoundError: No module named 'pkg_resources'`. The original worker
predates setuptools 84 and carries 80.10.2, which is why this only bites new
boxes.

```bash
pip install "setuptools==80.10.2"
```

`01_setup_local.sh` does this and `--check-only` fails loudly on any other version.

**The AWS Deep Learning AMI ships no conda at all.** `ami-012ba162b9cd2729c` has
the driver, CUDA and `/etc/vulkan/icd.d/nvidia_icd.json`, but no conda —
`../g5/setup_simpler_g5.sh` assumes one at `/home/ubuntu/miniforge3`. That is what
`../g5wide/bootstrap.sh` exists for: it installs **Miniforge3 24.11.3-2** and then
hands off. `01_setup_local.sh` folds the same step in (it only installs if
`$CONDA_ROOT/bin/conda` is missing).

**`libvulkan1` (the Vulkan *loader*) is not in the DLAMI** either. Install
`libvulkan1 vulkan-tools`. The AMI's apt mirror (`us-east-1.ec2.archive.ubuntu.com`)
has served HTTP 503 at ~5 kB/s; `../g5/setup_simpler_g5.sh` substitutes
`mirror.math.princeton.edu` with a `sed` on `/etc/apt/sources.list`.

**Required environment, every run** (`repro/common.sh:octo_env`, matching `../env.sh`):

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json   # else SAPIEN picks llvmpipe
export XLA_PYTHON_CLIENT_PREALLOCATE=false                  # else JAX takes ~75% of the GPU
export TOKENIZERS_PARALLELISM=false
export DISPLAY=""
```

On the DLAMI there is a fifth, and its failure mode is silent rather than loud:
the AMI exports `LD_LIBRARY_PATH=/usr/local/cuda/lib64:...` (CUDA 12.8), whose
`libcusolver.so.11` **shadows** the `nvidia-cusolver-cu12` wheel jaxlib
0.4.20+cuda12.cudnn89 needs. JAX prints *"CUDA backend failed to initialize"* and
**falls back to CPU** instead of erroring. Do not simply unset it — conda's
`libicui18n.so.78` then binds the system `libstdc++.so.6`, which lacks
`CXXABI_1.3.15`, and `import sqlite3` dies. Point it at the env and nothing else:
`export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"` (`../g5/g5_env.sh`).
`repro/02_smoke.sh` fails if JAX ended up on CPU.

**Pins that must not be "upgraded":** cuDNN 8.9.7.29 (jaxlib 0.4.20 is built
against cuDNN 8.x, not 9.x), scipy 1.11.4, tensorflow 2.15.0,
tensorflow-metadata 1.14.0, jax 0.4.20, and the `dlimp` git dependency. Short
list in `../../constraints.txt`.

**CHECKPOINT PINNING — pass `--ckpt` explicitly, always.** `run_eval.py` *defaults*
to `octo-small-1.5`; `finegrain_eval.py` defaults to `octo-small` (1.0); the
published SIMPLER "octo-small" row is 1.0. Mixing them invalidates any
comparison, and the gap is not small — closed-loop, 1.0 vs 1.5 is **52.8% vs
26.4%** on eggplant and **45.8% vs 4.2%** on spoon
(`XPU-RT/qnn_models/octo/OCTO_INT8_QRB5165.md` §8). Every `repro/` script uses
`$CKPT`, default `hf://rail-berkeley/octo-small`. Also never extrapolate
octo-small from octo-base: on spoon they are 47.2% and 12.5%.

**This box is shared.** `/scratch2/dima/miniforge3`, env `octo_sim`, one TITAN RTX
(24 GB). Other people use the GPU. Check before choosing concurrency:

```bash
nvidia-smi; free -g
```

3-4 concurrent 24-episode runs fit. `repro/04_run_arm_local.sh` prints both and
caps itself at `CONC` (default 3). Note the check is advisory — another user's
job can start between the check and your launch.

### Verify the install

```bash
repro/02_smoke.sh
```

~1 min, ends with `SMOKE OK`. It proves the imports work, that JAX is on the GPU,
and that every task's timebase is what the sweep assumed. **VERIFIED by running,
2026-09-06** — output was exactly:

```
jax 0.4.20  devices=[cuda(id=0)]

task                              cf  sim  tick  act dt ms  horizon ms
widowx_spoon_on_towel              5  500   25H     200.00     12000.0
widowx_put_eggplant_in_basket      5  500   25H     200.00     24000.0
google_robot_close_drawer          3  513   27H     333.33     37666.7
google_robot_pick_coke_can         3  513   27H     333.33     26666.7
```

---

## 2. Policy and environments

Both are fetched on first use, no separate step:

* **Policy** — `OctoModel.load_pretrained("hf://rail-berkeley/octo-small")` pulls
  from HuggingFace into `$HF_HOME` (the workers set
  `HF_HOME=/home/ubuntu/hf_cache`). ~1.4 GB. 136,670,604 parameters, of which
  109,628,544 is the frozen T5-base text encoder.
* **Environments** — `simpler_env.make(<task>)`. The scene/asset packs are
  downloaded by `01_setup_local.sh` step 5:
  `python -m mani_skill2_real2sim.utils.download_asset bridge_v2_real2sim -y`
  and `... ycb -y`.

The four tasks: `widowx_spoon_on_towel`, `widowx_put_eggplant_in_basket`,
`google_robot_close_drawer`, `google_robot_pick_coke_can` — short names `spoon`,
`egg`, `drawer`, `coke` throughout.

---

## 3. Methodology you must know before running anything

These are not background reading. Results are **wrong** without them.

### 3.1 A seed is not an episode draw

`--init-rng s` seeds **only the policy's JAX sampling key**. Object placement comes
from `obj_init_options.episode_id`, swept 0..23 in every run of every arm
(`finegrain_eval.py:563`, `run_eval.py`). The 24 official visual-matching episode
*configurations* are therefore identical across every seed and every arm. A seed
is a replicate over a fixed episode set, not a fresh draw of episodes.

The configs are a small grid, and they differ enormously in difficulty.
MEASURED here (`simpler_env.make(t).unwrapped._xy_configs` / `._quat_configs`):

| task | grid | full xy span of the config set |
|---|---|---|
| `widowx_put_eggplant_in_basket` | 8 positions x 3 orientations = 24 | 2.0 cm x 3.0 cm (source object; target fixed) |
| `widowx_spoon_on_towel` | 12 positions x 2 orientations = 24 | 15 cm x 15 cm (source and target) |

> A "~4 x 6 cm" figure for this spread circulates in the project notes. **It is not
> what this check returns** for either widowx task, and I could not find its
> source. Treat the two measured rows above as the numbers, and "~4 x 6 cm" as
> **UNVERIFIED**. The conclusion it was used to support is unaffected and is
> separately measured: over that small grid, per-config success rates run from
> **17.5% (ep22) to 93.7% (ep00)** — MEASURED on eggplant arm A over 63 seeds,
> `../g5wide/RESULTS.txt` line 33.

### 3.2 Therefore: cluster-corrected intervals, not binomial

Episodes cluster by config, so a binomial on *n* episodes overstates precision by
the design effect `DEFF = 1 + (m-1)*ICC`, with the episode config as the cluster.
`analyze_fine.py` reports naive *and* design-corrected intervals side by side;
quote the design-corrected one. MEASURED ICCs in the fine sweep run **0.13**
(egg `lat0`) to **0.58** (drawer `serial283`), DEFF 2.2 to 6.3.

There is a hard ceiling here: with K = 24 configs fixed, `n_eff -> K/ICC` as seeds
go to infinity. For the coarse eggplant baseline that limit is `n_eff = 119` — a
million seeds would still leave the marginal rate known only to about +/-9 points.
Only *more configs* lift it. More seeds help the **paired** contrast, not the
marginal rate.

### 3.3 Paired contrasts use t quantiles, not normal

The per-seed penalty against `lat0` is a paired difference over seeds, so its
interval uses `t(df = n_seeds - 1)`. With a handful of seeds the normal quantile
understates it badly: at 2 seeds **t(1) = 12.706 against z = 1.96, a 6.5x error**.
`analyze_fine.py` carries the t table and flags `df < 3` as unusable.
`compare_tasks.py` does the same for the cross-task difference-in-differences,
with a pooled SE and Welch's df (seeds are disjoint across tasks, so the two
penalties are independent).

### 3.4 The harness is NOT run-to-run deterministic

MEASURED: three byte-identical invocations of the same configuration gave episode
0 = **False / True / False** on the same machine, same GPU, same seed, same flags
(`NONDETERMINISM.md`). Almost certainly non-deterministic GPU reductions in
XLA/JAX. The untested remedy is `XLA_FLAGS=--xla_gpu_deterministic_ops=true` plus
`TF_DETERMINISTIC_OPS=1`; **nobody has verified it works here**, so treat it as
UNVERIFIED.

What that does and does not break:

* **Aggregate rates reproduce; no individual episode does.** Never cite,
  replay, or present a specific `(seed, episode_id)` outcome.
* The pairing is on the episode **config** set, which *is* deterministic, so the
  paired penalties and the difference-in-differences stand.
* "Same seed" means "same replicate index", not "same policy-noise realisation" —
  the framing in `../g5wide/SEED_ALLOCATION.md` is looser than it reads.
* Videos cannot be labelled as replays of a swept episode. `render_ladder_videos.sh`
  retries until a run produces the target outcome and keeps that run; each clip is
  internally consistent, and the success rate on it is the sweep's aggregate.

Separately: **same box + same seed is bit-reproducible**, **different GPU is not**.
`coke lat0 rng80` scores 9/24 on the local TITAN RTX and 14/24 on an A10G (measured
twice, two different A10G boxes). So a google_robot *marginal* rate is only
comparable to one measured on the same GPU. Paired contrasts are safe because
every arm of a seed runs on the same worker.

### 3.5 Horizon, tick and action-dt are DERIVED — never hardcode them

`finegrain_eval.py` derives all three from the env:

* **tick** — the divisor of `sim_freq` that is also a multiple of the native
  `control_freq`, nearest `--tick-ms-target` (default 40 ms)
* **action dt** — the env's native control period
* **horizon** — the env's registered `max_episode_steps` x native control period,
  i.e. exactly the wall-clock the stock baseline gets

widowx is 5 Hz control / 500 Hz sim -> **25 Hz tick, 200.00 ms action dt**.
google_robot is 3 Hz / **513 Hz** sim -> **27 Hz tick, 333.33 ms**; 513 = 3^3 x 19,
so **25 Hz does not divide it and is illegal there**. The horizon trap is the
expensive one: eggplant is 120 steps = 24000 ms but spoon/carrot/cube are 60
steps = 12000 ms, so the old hardcoded 24000 ms handed spoon **double** the
baseline's wall-clock and would have inflated its success. Do not pass
`--horizon-ms`, `--action-dt-ms` or `--tick-hz` unless you are deliberately
ablating.

### 3.6 google_robot needs `--actuation native`

`--actuation auto` (the default) selects `native` for google_robot and `fine` for
widowx, which is what you want. Do not override it. Two bugs had to be fixed to
get there (`GOOGLE_ROBOT_PORT.md`):

1. **Fine-tick actuation under a planner-interpolated controller.** widowx's
   `arm_pd_ee_target_delta_pose_align2` is a pure integrator of the deltas, so
   ticking 5x faster with 1/5-scaled deltas is a good approximation. google_robot
   uses `arm_pd_ee_delta_pose_align_interpolate_by_planner`, which *plans* a
   trajectory subject to velocity and acceleration limits — shrink the control
   period to 37 ms and the arm barely moves. MEASURED on `close_drawer` seed 80:
   stock 10/24, `--tick-hz 3` 9/24, fine harness at 27 Hz with 1/9 deltas
   **0/24**. Under `native` the env runs at its own `control_freq` with unscaled
   deltas, and the fine grid only decides *which* result is current at each native
   step. A guard now refuses to run at all if the control mode contains
   `interpolate_by_planner` while `--actuation fine` is in force with a delta
   scale != 1.
2. **The gripper is a DELTA on google_robot, with a sticky state machine.** The
   harness had been binarising it as `2*(g>0.5)-1`, the widowx *absolute* joint
   convention. google_robot's `gripper_pd_joint_target_delta_pos_interpolate_by_planner`
   takes a delta, and SimplerEnv's octo wrapper sends the open<->close *transition*
   and repeats it for `sticky_gripper_num_repeat = 15` control steps
   (`../../octo15_inference.py`, the `policy_setup == "google_robot"` branch).
   Sending +1 to a delta controller means "open by one unit, forever". The harness
   now runs the same state machine, advancing it once per **actuation** (not per
   result — per result would make the sticky window `15 x cadence`, penalising
   fast-cadence arms on any grasping task, which is exactly the contrast being
   measured). Transcription verified against `octo15_inference.py` over 200 random
   sequences x 120 steps: 0 mismatches.

**Resolution limit of `native`, and you must state it in any result.** Arrivals and
snapshots stay on the 37.037 ms tick grid but commands are sampled at the native
control boundary, so changing the modelled latency alters the actuated command
sequence only when an arrival crosses a boundary. For an arm whose cadence equals
the native period that is a **333.3 ms** step on google_robot (200 ms on widowx).
None of the six arms collapse onto each other — they differ in cadence too — but
this damps google_robot's apparent latency sensitivity, so the grasp-vs-push
reading is better supported than a raw cross-embodiment comparison.

### 3.7 Runs where the fine model is the pessimistic bound

The harness models pure **zero-order hold**: the landed action is repeated for the
whole gap. A deployed chunked policy has a cheaper option — Octo predicts a
4-entry chunk = 800 ms of motion, so during a 283 ms gap it could walk forward
through the chunk instead of repeating entry 0. That is **not** modelled, so these
numbers are the pessimistic bound for a chunk-capable policy and the correct bound
for a non-chunked one.

---

## 4. The validation gate — run it before any sweep

```bash
repro/03_validation_gate.sh drawer 80        # task seed
repro/03_validation_gate.sh spoon  90
```

~15 min per task. Runs the same 24 configs twice at zero modelled latency — stock
`run_eval.py` and `finegrain_eval.py --latency-ms 0`, both pinned to
`hf://rail-berkeley/octo-small` — and prints the difference in episodes.
**Pass criterion: every arm within two episodes of stock.** Anything larger and
the harness is modelling a bug, not latency.

The reference gate, MEASURED on the local TITAN RTX at harness md5 `b9be655f74a6`
(`GOOGLE_ROBOT_PORT.md`):

| task | seed | stock | fixed fine harness | delta |
|---|---|---|---|---|
| `google_robot_close_drawer` | 80 | 10/24 = 41.7% | 10/24 = 41.7% | 0 |
| `google_robot_close_drawer` | 81 | 10/24 = 41.7% | 9/24 = 37.5% | -1 |
| `google_robot_pick_coke_can` | 80 | 10/24 = 41.7% | 9/24 = 37.5% | -1 |
| `google_robot_pick_coke_can` | 81 | 7/24 = 29.2% | 9/24 = 37.5% | +2 |

Remember §3.4: on google_robot these are only comparable **on the same GPU**. The
table above is TITAN RTX vs TITAN RTX; the sweep itself is all A10G.

---

## 5. The fine-grain sweep on one task, on this box

```bash
repro/04_run_arm_local.sh spoon serial283 90     # one arm
CONC=3 repro/04_run_arm_local.sh spoon all 90    # the whole 6-arm ladder
```

~7 min per 24-episode widowx run uncontended; google_robot is longer (113 native
steps vs 60). The ladder at `CONC=3` is 20-30 min. Output lands in
`runs/<task>_<arm>_rng<seed>/summary.json` and `logs/<task>_<arm>_rng<seed>.log`.

Under the hood this is one `finegrain_eval.py` invocation per arm, with latency and
cadence kept as two separate numbers:

```bash
python finegrain_eval.py --task widowx_spoon_on_towel \
    --ckpt hf://rail-berkeley/octo-small \
    --latency-ms 283.4 --issue-period-ms 283.4 \
    --init-rng 90 --n 24 --out runs/spoon_serial283_rng90
```

`--latency-ms` is the age of the observation behind an applied result;
`--issue-period-ms` is how often a fresh result arrives. Serial arms have
cadence == latency; the pipelined ones do not, and that difference is the point —
`pipe110` issues a fresh result every 117.6 ms against the policy's native 200 ms
action period, so the controller replans ~1.7x more often.

The pre-existing `run_arm.sh` and `sweep.sh` are the older, eggplant-only 3-seed
drivers for the same thing; `04_run_arm_local.sh` is the task-parameterised version
that uses the same arm table as the AWS `job.sh`.

---

## 6. Scaling across the AWS workers

Six `g5.xlarge` (A10G 24 GB, 4 vCPU, 16 GB), region **us-east-1**, key
`~/.ssh/firesim.pem`, SG `sg-0cb78aaa92a9912fd`, AMI `ami-012ba162b9cd2729c`,
tagged `Owner=dima`, `Purpose=octo-latency-sweep`.

| worker | instance | AZ |
|---|---|---|
| w0 | `i-043560448065532ff` | us-east-1a (the original box) |
| w1 | `i-05289cfa068e00c74` | us-east-1b |
| w2 | `i-09db19b86bffe6acb` | us-east-1c |
| w3 | `i-078a22ff12e6c885d` | us-east-1d |
| w4 | `i-039d68dfbf56a4f61` | us-east-1f |
| w5 | `i-0f7e65981bc8d4d7c` | us-east-1b |

**Credentials live on the manager, `ubuntu@3.88.218.39` (`~/.aws`), not on this
box.** `repro/common.sh:awsq` runs every `aws` call there over ssh. VERIFIED
working 2026-09-06: `05_aws_instances.sh status` returned all six as `stopped`
in the AZs above.

```bash
repro/05_aws_instances.sh status
repro/05_aws_instances.sh start          # ONE AT A TIME
repro/05_aws_instances.sh hosts --write  # refresh the IP map
```

### The four AWS rules

**Start them ONE AT A TIME.** A single six-id `start-instances` call fails whole
with `InsufficientInstanceCapacity` — g5.xlarge capacity in these AZs is thin and
the API is all-or-nothing per call. `05_aws_instances.sh start` never batches and
reports each outcome separately. (The workers are spread over five AZs for the same
reason: the originally specified subnet `subnet-061e6fb07d6ea44da` in us-east-1a
had no g5.xlarge capacity.)

**Public IPs CHANGE on every stop/start.** Every IP in `g5fine/hosts.sh`,
`../g5wide/WORKERS.md` and `../g5wide/SEED_ALLOCATION.md` is stale the moment the
boxes stop. Re-read them and rewrite `hosts.sh` *before* pushing or fetching, or
you time out — or, worse, reach somebody else's instance.
`05_aws_instances.sh hosts` prints the new map; `--write` installs it and keeps
the previous copy as `hosts.sh.bak-<timestamp>`.

**NEVER trust the exit code of the ssh that detaches the remote `xargs`.** It can
hang past its own timeout while the launch succeeded, and it can return 0 while
nothing started. The only evidence is the process count:

```bash
ssh -n ubuntu@$IP "pgrep -f 'finegrain_ev[a]l' | wc -l"    # must be exactly 3
```

Trusting the exit code once produced **two workers silently running six runners
each**, writing into the same run directories. `relaunch.sh` refuses to launch onto
a box that already reports a non-zero count, polls for the expected count
afterwards, and flags any box that does not reach it; `07_aws_launch.sh` re-checks
after that. Note the `xargs` parent and the per-job `bash .../job.sh` wrappers do
*not* contain the string `finegrain_eval`, so the count is exactly the number of
python runners.

**Bracket every `pgrep`/`pkill` pattern.** `'finegrain_ev[a]l'` cannot match the
command line that is running it; `'finegrain_eval'` can, and will, and then you
are reading your own grep back as a runner or killing your own shell.

### The full AWS run

```bash
repro/05_aws_instances.sh start
repro/05_aws_instances.sh hosts --write

FRESH=1 repro/06_aws_joblists.sh drawer 110 119     # 6 arms x 10 seeds
        repro/06_aws_joblists.sh coke   120 129
repro/07_aws_launch.sh all                          # push + launch + verify
repro/07_aws_launch.sh verify                       # poll progress any time
repro/08_aws_fetch.sh --logs
repro/05_aws_instances.sh stop
```

`06_aws_joblists.sh` encodes the two design rules: **seed s -> worker (s mod 6)**,
with every arm of a seed on the same box (so a GPU effect adds variance to the
paired difference but cannot confound it), and the per-worker list **shuffled with
a fixed seed** — the tail of an `xargs -P3` queue runs at lower concurrency than
the body, and with the arms in a fixed order the same arm lands in that tail on
every box, correlating an arm with its machine load.

> VERIFIED by running: `06_aws_joblists.sh drawer 110 119` + `... coke 120 129`
> reproduces the shipped `g5fine/joblist_w{0..5}.txt` **exactly as sets** —
> 18/18/24/24/18/18 jobs, 120 total.

**The joblists are shared state.** They live in `g5fine/` and another sweep may be
staged in them right now (as of 2026-09-06 they hold the `pipe110fix` /
`pipe200fix` arms from `g5fine/job_fix.sh`). Without `FRESH=1` the script
*appends*, so you can stage several tasks into one launch; with `FRESH=1` it
truncates — and therefore copies the previous contents to
`joblist_w*.txt.bak-<timestamp>` and prints what they held before doing so. Look
at that line before launching.

Seeds must be globally disjoint per `(task, arm)`; record whatever you issue in
`../g5wide/SEED_ALLOCATION.md` *before* launching or the pooled analyses stop
being valid. Consumed: 0-79 coarse (a different timing model, not poolable with
these), 80-84 the broken google probe (quarantined in `g5fine/runs_prefix_broken/`
and rejected by both analysis scripts), 90-99 spoon, 100-109 egg, 110-119 drawer,
120-129 coke. **20 is reserved and unissued.**

Wall time for a 6-arm x 10-seed x 2-task block (120 runs, 2,880 episodes) is a
few hours at 3 concurrent per box. The coarse sweep's 4,752 episodes cost 11.01
instance-hours = **$11.08**.

### STOP them when done. Never terminate.

```bash
repro/05_aws_instances.sh stop
```

Terminating deletes the root volumes and with them the *working environment*: the
Miniforge3 install, `setuptools==80.10.2`, the pinned repos, the checkpoint cache,
and a **passed per-worker validation gate** — a rebuilt box has to re-earn that
before its numbers may be pooled. Rebuild is ~40 min of
`bootstrap.sh` + `setup_simpler_g5.sh` plus a gate run; restart from `stopped` is
~2 min.

Stopped is not free: **6 x 200 GB gp3 = 1.2 TB at ~$0.08/GB-month = ~$96/month**
(~$16/box). That overtakes the entire coarse sweep's compute in under four days of
idling. If the follow-ups are not going to run soon, prune deliberately per
`../g5wide/WORKERS.md` — and **pull the per-job logs first**
(`08_aws_fetch.sh --logs`), because `fetch.sh` copies only `summary.json`.

---

## 7. Analyse

```bash
repro/09_analyse.sh                    # -> g5fine/RESULTS_raw_repro.txt
```

~20 s. Runs `analyze_fine.py`, `compare_tasks.py` and `arm_timing.py` over
`g5fine/runs/`. (`g5fine/make_raw_google.sh` is the original of this and writes
`RESULTS_raw_google.txt`; `09_analyse.sh` writes a separate file so it cannot
clobber a published one.)

Read it in this order:

1. **`analyze_fine.py`** — per-arm marginal rate with naive *and* design-corrected
   intervals, ICC/DEFF/n_eff, then the paired per-seed penalty against `lat0` with
   t intervals and a "seeds worse" count. Quote the design-corrected interval for
   marginals and the paired penalty for contrasts.
2. **`compare_tasks.py`** — difference-in-differences between tasks. Within a task
   the penalty is paired so config difficulty cancels; across tasks the seeds are
   disjoint, so the two penalties are independent and get a pooled SE with Welch's
   df.
3. **`arm_timing.py`** — what each arm did to the *command stream*: observation age
   at actuation, dispatches per episode, and new-results-per-actuation. This is
   what separates "it hurt because the observation was stale" from "it hurt
   because the command stream changed shape". `upd/act` is capped at 1.0 on
   google_robot by the 3 Hz control grid — which is exactly why a faster cadence
   cannot buy more replanning there the way it can in the widowx fine model.

The three quarantine/validity rules the scripts enforce for you: a google_robot run
without `--actuation native` is rejected; runs are keyed
`<task>_<arm>_rng<seed>`; `df < 3` paired intervals are flagged unusable.

## 8. Plot

```bash
repro/10_plot.sh all
```

~20 s each; every script recomputes from `g5fine/runs/` at plot time, so a figure
is only as current as the last fetch.

| figure | what it shows |
|---|---|
| `finegrain_four_tasks.png` | the headline: 4 envs, 2 embodiments, marginal success (design-corrected) and the paired penalty |
| `finegrain_tasks.png` | the two widowx envs, same encoding |
| `finegrain_tasks_comparison.png` | funnel / sensor->actuation age / duty cycle — *where* latency breaks the task |
| `completion_time.png` | completion time over **successful episodes only** (a selection effect the script states on the figure) |

Encoding, kept across all of them: colour = category (ideal / HW accelerated /
CPU only), marker = schedule (star none, circle pipelined, square serial),
linestyle or fill = environment.

## 9. Traces: end-effector paths and motion cost

```bash
repro/11_traces.sh ee        # ~10 min   -> traces/egg_<arm>/       (1 episode each)
repro/11_traces.sh energy    # ~3-4 h    -> traces_energy2/{drw,egg}_<arm>/ (24 each)
repro/11_traces.sh plot      #           -> eetrace_*.png, energy_by_schedule.png
```

`trace_eval.py` is `finegrain_eval.py` plus per-tick state dumps —
`ep*_ee_xyz.npy`, `ep*_ee_quat.npy`, `ep*_qvel.npy`, `ep*_link_com.npy`,
`ep*_bodies.json`, `ep*_action_age_ms.npy`, `ep*_bg.png`. Those files, not the
summaries, are what `plot_ee_traces.py` and `energy_analysis.py` read. There was
no driver script for this step — the traces were made by hand; `11_traces.sh` is
that step written down, with the parameters read back out of the existing traces'
own `summary.json` (seed 100, octo-small 1.0, n=1 for the EE traces, n=24 for the
energy traces).

**One source per figure.** `energy_analysis.py` reads `traces_energy2/` if
`traces_energy2/_COMPLETE` exists and falls back to `traces_energy/` otherwise.
Never half-populate one directory from two sessions: mixing arms across independent
runs of a non-deterministic harness puts run-to-run noise into the arm differences.
`11_traces.sh energy` removes `_COMPLETE` first and only re-touches it when all 12
arm directories are present.

What `energy_analysis.py` reports and, as importantly, does not: **joint motion
effort** (`INTEGRAL sum_i omega_i^2 dt`, in its own units, not watts — ratios
between schedules are invariant to the viscous coefficient) and **physical work**
(lift and kinetic, real joules from the real link masses over every link's COM).
Not modelled: static holding torque (high-latency arms hold still *more*, so every
metric here understates them), drivetrain and electrical losses, and work done on
the object during contact. It scores successful episodes only, so each arm is a
different subpopulation — n is printed everywhere, and eggplant CPU-only has n=0
successes and cannot be scored.

---

## 10. Known-good outputs

Check your run against these. All MEASURED, all on the A10G workers unless noted.

**Single runs** (`g5fine/runs/<worker>/<task>_<arm>_rng<seed>/summary.json`):

| task | arm | seed | worker | result |
|---|---|---|---|---|
| spoon | `lat0` | 90 | w0 | **11/24 = 45.8%** |
| spoon | `pipe110` | 90 | w0 | 12/24 = 50.0% |
| spoon | `serial283` | 90 | w0 | 4/24 = 16.7% |
| spoon | `cpu685` | 90 | w0 | 0/24 = 0.0% |
| egg | `lat0` | 100 | w4 | **9/24 = 37.5%** |
| egg | `pipe110` | 100 | w4 | 17/24 = 70.8% |
| egg | `serial283` | 100 | w4 | 5/24 = 20.8% |
| egg | `cpu685` | 100 | w4 | 0/24 = 0.0% |
| drawer | `lat0` | 110 | w2 | 9/24 = 37.5% |
| coke | `lat0` | 120 | w0 | 11/24 = 45.8% |

Stock-harness reference, local TITAN RTX: `google_robot_close_drawer` **41.7%
(10/24) on both seeds 80 and 81**; `google_robot_pick_coke_can` 41.7% on seed 80
and 29.2% on seed 81.

Because of §3.4 these individual runs are **not** expected to reproduce
episode-for-episode. Aggregates are.

**Pooled ladder** — 240 episodes per arm (10 seeds x 24 configs).
VERIFIED by re-running `analyze_fine.py` on 2026-09-06; this is its current output:

| task | `lat0` | `pipe110` | `pipe200` | `serial283` | `fp32_555` | `cpu685` |
|---|---:|---:|---:|---:|---:|---:|
| spoon | 45.4% | 49.6% | 27.9% | 13.8% | 4.2% | 0.8% |
| egg | 50.0% | 63.3% | 41.7% | 17.1% | 2.9% | 1.7% |
| drawer | 40.0% | 38.3% | 41.7% | 43.8% | 37.9% | 31.7% |
| coke | 44.2% | 46.2% | 40.4% | 33.8% | 15.4% | 8.8% |

Paired penalty vs `lat0`, percentage points, t(9) 95% CI:

| task | `pipe110` | `pipe200` | `serial283` | `fp32_555` | `cpu685` |
|---|---|---|---|---|---|
| spoon | +4.2 [-3.8, +12.1] | -17.5 [-26.9, -8.1] | -31.7 [-40.5, -22.9] | -41.2 [-51.1, -31.4] | -44.6 [-53.4, -35.8] |
| egg | +13.3 [+0.6, +26.0] | -8.3 [-19.8, +3.2] | -32.9 [-43.8, -22.0] | -47.1 [-56.2, -38.0] | -48.3 [-58.4, -38.3] |
| drawer | -1.7 [-7.8, +4.5] | +1.7 [-5.8, +9.1] | +3.8 [-2.4, +9.9] | -2.1 [-11.3, +7.2] | -8.3 [-15.4, -1.3] |
| coke | +2.1 [-8.0, +12.1] | -3.8 [-12.0, +4.5] | -10.4 [-17.3, -3.5] | -28.8 [-36.1, -21.4] | -35.4 [-42.9, -27.9] |

Sanity anchors: `lat0` spoon 45.4% validates against the published SIMPLER
octo-small spoon **47.2%**. `pipe110` is not distinguishable from zero latency on
spoon and is *better* than it on eggplant (+13.3, CI excludes zero) — the mechanism
is cadence, not latency. Everything from `serial283` down is significant, monotone
and unanimous across seeds on both widowx tasks.

Observation age at actuation, from `arm_timing.py` (ms, spoon / drawer):

| arm | spoon | drawer |
|---|---:|---:|
| `lat0` | 80.0 | 0.0 |
| `pipe110` | 157.6 | 154.7 |
| `pipe200` | 312.3 | 316.4 |
| `serial283` | 404.8 | 409.6 |
| `fp32_555` | 810.9 | 777.3 |
| `cpu685` | 1003.4 | 1012.6 |

---

## 11. When it does not work

| symptom | cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'pkg_resources'` on `import simpler_env` | setuptools >= 81. `pip install setuptools==80.10.2`. |
| `ErrorExtensionNotPresent` from SAPIEN, or it picks llvmpipe | `VK_ICD_FILENAMES` unset, or `libvulkan1` missing. |
| "CUDA backend failed to initialize: Unable to load cuSOLVER", then everything is slow | `LD_LIBRARY_PATH` shadowing. Set it to `$CONDA_PREFIX/lib` and nothing else. `02_smoke.sh` catches this. |
| OOM, or another user's job dies | JAX preallocating. `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Lower `CONC`. Check `nvidia-smi` first — the box is shared. |
| Fine harness scores 0/24 at zero latency on google_robot | `--actuation fine` on a planner-interpolated controller. Use `auto`/`native`. The harness now guards against it. |
| Success rate is implausibly high on spoon/carrot/cube | a hardcoded `--horizon-ms 24000` (eggplant's). Do not pass it; let the env derive it. |
| Stock and fine disagree by a lot at zero latency | check `--ckpt` on both. `run_eval.py` defaults to 1.5, the fine harness to 1.0. |
| An episode outcome will not reproduce | expected. §3.4. Only aggregates reproduce. |
| `rsync`/`ssh` to a worker times out | stale IPs. `05_aws_instances.sh hosts --write`. |
| `start-instances` fails `InsufficientInstanceCapacity` | you batched. Start one id at a time; retry that one id in a few minutes. |
| A worker shows 6 runners, or run dirs look interleaved | a double launch. `pkill -f 'finegrain_ev[a]l'` on that box (bracketed!), delete the affected run dirs, relaunch that box alone. |
| `analyze_fine.py` prints "no runs fetched yet" | nothing under `g5fine/runs/*/*/summary.json`. Re-run `08_aws_fetch.sh`. |
| A paired CI is flagged "95% CI unusable" | fewer than 4 seeds. Add seeds; do not switch to a normal quantile. |
| `energy_analysis.py` uses the wrong trace set | `traces_energy2/_COMPLETE` missing. Finish the 12 arms, or accept the `traces_energy/` fallback. |

---

## 12. Provenance

Reproduced and re-verified **2026-09-06**. `octo_work/` is not a git repository, so
the harness is identified by md5 (first 12 hex).

**Harness and analysis** (`sim_eval/roselite/finegrain/`):

```
b9be655f74a6  finegrain_eval.py          <- the harness the google sweep ran
95d207238267  trace_eval.py
cc9a5d24d501  g5fine/analyze_fine.py
244088c7e472  g5fine/compare_tasks.py
23b567e81fbe  g5fine/arm_timing.py
40d4bd8e7361  g5fine/job.sh              <- the authoritative arm table
377bb5d6559d  g5fine/push.sh
91692563edb6  g5fine/relaunch.sh
965d6f47316f  g5fine/fetch.sh
4e5a4200fbfa  plot_four_tasks.py
709066af6d6b  plot_tasks_finegrain.py
d1813dc557d3  plot_tasks_comparison.py
664854753e9c  plot_completion_time.py
885d2b7ee68d  plot_ee_traces.py
4b147fd99e8d  energy_analysis.py
29a61979a880  ../../octo15_inference.py
42d1f560b1af  ../../run_eval.py
ae20dea710b2  ../env.sh
```

`g5fine/RESULTS.txt` §5 records the *spoon* sweep at harness md5 `cb302b43f647`;
`b9be655f74a6` is the later, google-robot-fixed harness, and `GOOGLE_ROBOT_PORT.md`
shows widowx is bit-identical across that change (spoon lat0 rng90 re-ran to the
same 11/24 with the identical success set `{0,4,8,9,10,12,15,18,20,21,22}`).

**Checkpoint.** `hf://rail-berkeley/octo-small` — version **1.0**, pinned
explicitly on every invocation. Not 1.5, which is what `run_eval.py` defaults to.

**Upstream revisions.** SimplerEnv `06accaca93535902d408da4855f21cece12bceb7`,
ManiSkill2_real2sim `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682`,
octo `241fb3514b7c40957a86d869fecb7c7fc353f540`, python 3.10,
jax 0.4.20 / jaxlib 0.4.20+cuda12.cudnn89, setuptools 80.10.2.

**Seeds.** 90-99 spoon, 100-109 egg, 110-119 drawer, 120-129 coke — 6 arms x 10
seeds x 24 configs = 240 episodes per arm per task, 5,760 episodes total. Seed
s -> worker (s mod 6). Not poolable with 0-79 (coarse timing model) or 80-84 (the
pre-fix google probe, quarantined in `g5fine/runs_prefix_broken/`). 20 is reserved
and unissued. The ledger is `../g5wide/SEED_ALLOCATION.md`.

**Hardware.** Sweep: 6 x g5.xlarge (A10G), us-east-1, `stopped` as of
2026-09-06 (verified). Local box and the validation gates: one NVIDIA TITAN RTX
(24 GB), shared.

**What in this document was verified by running, on 2026-09-06:** `02_smoke.sh`
(all four timebases); `06_aws_joblists.sh` (reproduces the shipped joblists, and
its `FRESH=1` backup path); `05_aws_instances.sh status` via the manager (all six
stopped); `09_analyse.sh` (the pooled tables above are its live output);
`10_plot.sh four`; the config-grid measurement in §3.1; and every md5 above.
Also the exact `finegrain_eval.py` invocation `04_run_arm_local.sh` issues, run
short (`spoon serial283 rng90 --n 2`): it derived the correct 12000 ms spoon
horizon, a 0.2 delta scale, 43 dispatches/episode and a mean observation age of
405.2 ms — matching the 405.3 ms the `serial283` arm is documented to produce.
**Documented from reading, not re-run:** the install path on a genuinely bare
machine, the AWS start/launch/fetch cycle end to end (the instances are stopped),
the full 24-episode `03_validation_gate.sh` and `04_run_arm_local.sh` runs, and
`11_traces.sh`.

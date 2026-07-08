# Replicating the forest-trail navigation demo from a fresh clone

This traces every dependency of one specific command — the "3-model" DroNet +
MLP + YOLO forest-trail pilot — back to its source, and marks each link in
the chain as **committed** or **not committed**, and **documented** or
**not documented**. Use it as the checklist for a dry-run in a scratch clone.

Reference command:

```bash
conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \
    --dronet_weights logs/dronet/2026-05-10_20-22-08_finetune/best.pt \
    --checkpoint logs/rsl_rl/crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt \
    --schedule_json schedules/scheduled_networks_mlp10_dronet20_yolov8_firesim_static_decomposed_profiled.json \
    --yolo --trail curved --curvature_seed 12 \
    --num_periods 200 \
    --save_video full_firesim_sched_seed_12_finetuned_4model.mp4
```

## Dependency graph

```
git clone + submodule init
        │
        ▼
xpurt conda env (Isaac Sim 5.1, IsaacLab, PyTorch, rsl_rl, ultralytics)  ⚠ not documented
        │
        ▼
sims/scripts/pilot/, sims/isaaclab_tasks/forest_trail/, sims/training/,   ⚠ NOT COMMITTED
sims/scripts/utils/, sims/scripts/train/  (this entire reorg + the forest
task + the fine-tuning tools only exist in this working tree)
        │
        ├──► DroNet weights ───────────────────────────────────┐
        │    IDSIA download (manual) → extract_idsia.py         │
        │    → train_dronet.py → best.pt (base)                 │  ⚠ base flow documented,
        │    → collect_sim_data.py (needs Isaac Sim running)    │    finetune flow is NOT
        │    → finetune_dronet.py → best.pt (finetune, the one  │
        │      actually used by the reference command)          │
        │                                                       │
        ├──► RL inner-loop checkpoint ──────────────────────────┤  ✅ documented
        │    sims/scripts/train/train_full.sh                   │
        │    → logs/rsl_rl/crazyflie_steering_tracking/*/model_*.pt
        │                                                       │
        ├──► schedule JSON ─────────────────────────────────────┤  ⚠ tool documented,
        │    data/toplevel/networks_mlp10_dronet20_yolov8_       │    this specific
        │    firesim_static.json  (NOT COMMITTED)                │    input/output pair
        │    → scripts/run_xpurt_schedule.py                     │    is not committed
        │    → schedules/scheduled_networks_..._profiled.json    │    or named as an example
        │      (NOT COMMITTED)                                   │
        │                                                       │
        └──► runtime downloads (automatic, need internet) ──────┘  ✅ works with zero setup
             - yolov8n.pt (ultralytics, auto-downloaded)
             - human character USD (S3, direct HTTPS, no Nucleus needed —
               falls back to a bundled local USD if unreachable)
        │
        ▼
pilot_forest_with_dronet_scheduled.py runs
```

---

## Step-by-step

### 0. Hardware / driver assumptions (undocumented anywhere, observed working)

- NVIDIA GPU with ≥ 10 GB free VRAM (observed run used ~4.6 GB on a TITAN RTX 24 GB)
- Driver 560.35.05, CUDA 12.6 (host driver; the conda env's `torch` bundles its own CUDA 12.8 runtime)
- No `$DISPLAY` needed — Isaac Sim's `carb.windowing-glfw` fails to init and Kit
  falls back to headless-capable rendering regardless; matplotlib resolves to
  the `Agg` backend and renders fine without a display.

### 1. Clone + submodules

```bash
git clone <repo> && cd FreshScheduler
git submodule update --init sims/IsaacLab
```

`sims/IsaacLab` is a proper git submodule (`.gitmodules`), so this part is
sound. (`hw/chipyard`, `merlin`, `zephyr-chipyard-sw` are unrelated to this
demo.)

### 2. The `xpurt` conda environment — ⚠ **not documented**

No `environment.yml`, `requirements.txt`, or setup script exists anywhere in
the repo for this environment (only `env.yml` exists, and that's a *different*,
unrelated Python-3.9 env for the `xpu-rt` MILP scheduler — `mosek`/`cvxpy`,
nothing to do with Isaac Sim). `sims/scripts/utils/setup_env.sh` only exports
`PYTHONPATH`, and does so against a **stale path** (`/scratch2/dima/IsaacLab/...`
instead of the vendored `sims/IsaacLab/...` the current scripts actually use).

Current working manifest (observed via `pip show` / `pip list` in the live
`xpurt` env — captured here for reference, **not a recipe**; the original
install commands/index URLs used to build this env were not recorded
anywhere, so this would need to be reverse-engineered from NVIDIA's Isaac Sim
5.1 install docs + a matching PyTorch cu128 wheel):

| Package | Version |
|---|---|
| Python | 3.11.15 |
| `isaacsim` (+ all `isaacsim-*` subpackages) | 5.1.0.0 |
| `isaaclab_tasks` (upstream, pip-installed — distinct from this repo's `sims/isaaclab_tasks`) | 0.11.14 |
| `torch` | 2.7.0+cu128 |
| `rsl-rl-lib` (import name `rsl_rl`) | 5.0.1 |
| `ultralytics` | 8.4.39 |
| `gymnasium` | 1.2.1 |
| `imageio` / `imageio-ffmpeg` | 2.37.0 / 0.6.0 |
| `matplotlib` | 3.10.3 |

**Action item:** run `conda list -n xpurt --explicit` (or `pip freeze`) and
commit it — right now nobody could rebuild this environment from the repo
alone.

**Also undocumented: the IsaacLab install step itself.** `isaaclab`,
`isaaclab_rl`, `isaaclab_assets`, and `isaaclab_contrib` are not just on
`sys.path` — `pip show` confirms they're **editable installs**
(`Editable project location: .../sims/IsaacLab/source/<pkg>`) into the
`xpurt` env. That only happens after running IsaacLab's own installer
against this submodule checkout, almost certainly:

```bash
cd sims/IsaacLab
./isaaclab.sh --install   # or -i; installs system deps (sudo apt-get:
                           # cmake etc.), pins setuptools<82, ensures a
                           # CUDA-matched torch, then `pip install -e`
                           # every extension under source/*
```

This presupposes the `isaacsim` pip package (5.1.0.0 here) is already
installed in the env — `isaaclab.sh` does not install Isaac Sim itself, and
there's no `_isaac_sim` symlink in this checkout, so it's using the newer
pip-based Isaac Sim install rather than a separate binary install. Nothing
in this repo documents either half of this (installing `isaacsim` via pip,
or running `isaaclab.sh --install` against it) — it's a real gap, distinct
from and in addition to the "no environment.yml" gap above.

### 3. Uncommitted code — ⚠ **the critical blocker**

None of the following exist in git history — a real fresh clone gets none of
it:

| Path | git status |
|---|---|
| `sims/scripts/pilot/*.py` (all 4 pilot scripts, including the one in the reference command) | untracked |
| `sims/scripts/train/*`, `sims/scripts/play/*`, `sims/scripts/debug/*` | untracked |
| `sims/scripts/utils/schedule_dispatch.py` and everything else in `utils/` | untracked |
| `sims/scripts/README.md` | untracked |
| `sims/isaaclab_tasks/forest_trail/` (32 files, 46 MB — the entire forest-trail task: scene, curved-trail generator, tree/human placement, and the **local fallback USD assets** trees/human rely on) | untracked |
| `sims/training/*` (train_dronet.py, finetune_dronet.py, collect_sim_data.py, dataset_*.py, README.md) | **now committed** (see §4) |
| `sims/README.md`, `sims/STEERING_POLICY_USAGE.md`, and the other loose `sims/*.md` docs | untracked |
| The two shutdown-hang / asset-race fixes applied this session to all 4 pilot scripts (`_disable_app_control_on_stop_handle`, `wait_for_textures = False`) | **uncommitted working-tree edits** — exist only in this checkout, not even staged |

Only `qnn_models/dronet.py` (the `DronetTorch` model class) and the
`track_steering_vision` half of `sims/isaaclab_tasks` are actually committed.
Committed HEAD still has the *old* flat layout (`sims/scripts/train_steering_tracking.py`
etc., currently showing as `D` deleted in the working tree) — meaning **the
reference command's script path doesn't exist in git at all today.**

This is the dominant finding: everything downstream (env setup, model
training, schedule generation) is moot for a genuinely fresh clone until this
gets committed.

### 4. DroNet weights

`--dronet_weights logs/dronet/2026-05-10_20-22-08_finetune/best.pt`

`logs/` is untracked (1.2 GB on disk; not even mentioned in `.gitignore`, just
never added) but the *generation* flow is split across two paths:

- **Base training — ✅ documented** (`sims/training/README.md` §3): download
  IDSIA archive manually → `extract_idsia.py` → `train_dronet.py` → `best.pt`.
- **Fine-tuning — ✅ now documented** (`sims/training/README.md` §3b, added
  and committed in this pass, along with the underlying scripts themselves —
  see §3 above). The checkpoint the reference command actually uses is a
  *fine-tuned* run (`..._finetune`), not a base run: on disk there are 4 base
  runs (2026-04-27) and 4 fine-tune runs (2026-05-10); the tooling was added
  6 days after the original documentation pass and had never been folded
  into the README until now.
  1. `sims/training/collect_sim_data.py --headless --num_samples 5000` — needs
     Isaac Sim running, teleports the drone along the trail, saves labeled
     frames.
  2. `sims/training/finetune_dronet.py --checkpoint <base best.pt> --data_root datasets/sim_forest/extracted/000 --sim_data --epochs 20 --lr 1e-4`

### 5. RL inner-loop checkpoint — ✅ documented

`--checkpoint logs/rsl_rl/crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt`

```bash
bash sims/scripts/train/train_full.sh   # 4096 envs, 2000 iters, headless, ~20 min
```

Also untracked (part of `logs/`), but the training path is documented
consistently in three places (`sims/scripts/README.md`, `sims/STEERING_POLICY_USAGE.md`,
`sims/training/README.md` §4). One caveat unrelated to this specific
checkpoint but worth knowing: `train_harsh.sh` still silently trains the
*non*-harsh config — `train_steering_tracking.py` hardcodes
`TrackSteeringEnvCfg()` and passes it as `cfg=` to `gym.make()`, which
overrides the `--task ...Harsh-v0` name. `TRAINING_CONFIG_BUG.md` /
`ENVIRONMENT_MISMATCH_FIX.md` document this as "RESOLVED" — but the
resolution was "use the regular config for playback," not a code fix. The
bug is still live in the current script.

### 6. Schedule JSON — ⚠ tool documented, this file is not

`--schedule_json schedules/scheduled_networks_mlp10_dronet20_yolov8_firesim_static_decomposed_profiled.json`

The general pipeline is well documented in `docs/end_to_end_xpurt_firesim.md`
(tracked, 267 lines):

```
data/toplevel/networks_<name>.json  →  scripts/run_xpurt_schedule.py  →  schedules/scheduled_networks_<name>_<solver>_profiled.json
```

But for this specific demo:

- `data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static.json` (the
  topology input) is **untracked**, and is not one of the examples named in
  `docs/end_to_end_xpurt_firesim.md` (which only lists the `_q31profile`
  sibling variant).
- The output schedule JSON itself is **untracked** (of 54 schedule files on
  disk, only 10 are committed).
- Reproducing it also depends on profiled hardware timing data
  (`gen/profile/...`) whose provenance for the `firesim` hardware profile
  used here isn't traced in this doc — treat it as a separate, deeper
  rabbit hole if the dry run needs to regenerate this file from scratch
  rather than just committing the existing one.

### 7. Runtime auto-downloads — ✅ work with zero setup

- **`yolov8n.pt`**: `ultralytics.YOLO("yolov8n.pt")` downloads it
  automatically on first use. Needs outbound internet; no repo-side action
  needed.
- **Human character USD**: `sims/isaaclab_tasks/forest_trail/forest_scene.py`
  fetches from `https://omniverse-content-production.s3-us-west-2.amazonaws.com/...`
  directly over HTTPS — **no Nucleus server required** (confirmed in code
  comments and by a live run). Falls back to a bundled local USD
  (`forest_trail/assets/human.usda`) if the S3 asset is unreachable — but
  that fallback asset only exists because `forest_trail/` is untracked-but-
  present in *this* checkout (see §3); a fresh clone has neither the S3 path
  guaranteed nor the local fallback.
- Trees use bundled local assets (`forest_trail/assets/pine_tree.usda` /
  `pine_sapling_small/`) — same caveat, only present because `forest_trail/`
  hasn't been committed yet.

### 8. Run

Once 1–7 are satisfied, the reference command should run as-is.

---

## Summary: not committed

- Entire `sims/scripts/{pilot,play,debug,utils}/` reorg + README (note: `train/` is
  still uncommitted too — only `sims/training/` was committed in this pass)
- Entire `sims/isaaclab_tasks/forest_trail/` package (code + 46 MB local assets)
- All loose `sims/*.md` docs (`README.md`, `STEERING_POLICY_USAGE.md`, etc.)
- The specific topology + schedule JSON pair used by the reference command
  (left as-is for now — orthogonal to the sim-runner flow)
- This session's two shutdown/asset-race fixes to the 4 pilot scripts (still only in the working tree — the pilot scripts themselves aren't committed yet either)
- `logs/`, `datasets/` (expected/appropriate to exclude — just noting they're not even named in `.gitignore`)

Committed in this pass: `sims/training/` (base + fine-tune DroNet tooling,
now including the fine-tune docs).

## Summary: not documented

- How to build the `xpurt` conda env from nothing (no manifest, no install script, stale `setup_env.sh` path)
- The IsaacLab install step itself (`isaaclab.sh --install` against the submodule, plus the prerequisite `isaacsim` pip install) — see §2
- How to regenerate this specific schedule JSON (topology file untracked, hardware profile provenance untraced) — left out of scope for now
- `sims/README.md` is stale — describes DroNet integration and the forest trail as future "Next Steps" when both already exist and are the primary demo per `sims/scripts/README.md`

Resolved in this pass: the DroNet fine-tuning flow (`collect_sim_data.py` →
`finetune_dronet.py`) is now documented in `sims/training/README.md` §3b.

## Suggested next step (not run yet)

Dry-run in a fresh worktree/temp clone once the above is committed, in this
order: (1) submodule init, (2) recreate `xpurt` env from a captured
`pip freeze`/`conda list --explicit`, (3) commit and pull in the
`sims/scripts/`, `sims/isaaclab_tasks/forest_trail/`, `sims/training/` trees
plus this session's two fixes, (4) either commit the existing DroNet/RL
checkpoints and schedule JSON directly (simplest for a verification dry run)
or re-run the training/scheduling flows from scratch, (5) run the reference
command headless with a small `--num_periods` and confirm a non-empty video
chunk, matching the smoke test already done in this session.

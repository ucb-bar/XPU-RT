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
xpurt conda env (Isaac Sim 5.1, IsaacLab, PyTorch, rsl_rl, ultralytics)  ✅ documented (docs/xpurt_env_setup.md), not yet run from scratch
        │
        ▼
sims/scripts/pilot/, sims/isaaclab_tasks/forest_trail/,                  ✅ COMMITTED
sims/scripts/utils/schedule_dispatch.py, sims/training/
        │
        ├──► DroNet weights ───────────────────────────────────┐
        │    IDSIA download (manual) → extract_idsia.py         │
        │    → train_dronet.py → best.pt (base)                 │  ✅ documented
        │    → collect_sim_data.py (needs Isaac Sim running)    │    (sims/training/README.md §3, §3b)
        │    → finetune_dronet.py → best.pt (finetune, the one  │
        │      actually used by the reference command)          │
        │                                                       │
        ├──► RL inner-loop checkpoint ──────────────────────────┤  ✅ documented
        │    sims/scripts/train/train_full.sh (still uncommitted,│
        │    but only needed to *retrain*, not to *run*, the demo)
        │    → logs/rsl_rl/crazyflie_steering_tracking/*/model_*.pt
        │                                                       │
        ├──► schedule JSON ─────────────────────────────────────┤  ⚠ this specific file
        │    data/toplevel/networks_mlp10_dronet20_yolov8_       │    left out of scope
        │    firesim_static.json  (NOT COMMITTED, by choice)     │    (orthogonal); a tracked
        │    → scripts/run_xpurt_schedule.py                     │    substitute exists —
        │    → schedules/scheduled_networks_..._profiled.json    │    see §6
        │      (NOT COMMITTED, by choice)                        │
        │                                                       │
        └──► runtime downloads (automatic, need internet) ──────┘  ✅ works with zero setup
             - yolov8n.pt (ultralytics, auto-downloaded)
             - human character USD (S3, direct HTTPS, no Nucleus needed)
             - pine_sapling_small tree asset (Poly Haven, CC0) — regenerated
               via download_trees.py, not committed (see §7)
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

### 2. The `xpurt` conda environment — ✅ documented

Written up end-to-end in **`docs/xpurt_env_setup.md`**, including the "is
there already a recipe?" question (no — `conda-meta/history` shows only
`conda create -n xpurt python=3.11`, everything else was ad-hoc `pip`), with
a frozen fallback manifest at `docs/xpurt_pip_freeze_2026-07-08.txt`. Covers
the IsaacLab install step too (`isaaclab`/`isaaclab_rl`/`isaaclab_assets`/
`isaaclab_contrib`/`isaaclab_mimic`/`isaaclab_tasks` are editable pip
installs from `sims/IsaacLab/source/*`, wired up by `./isaaclab.sh --install`
— not just a `sys.path` trick).

Not yet done: actually running that flow on a clean machine. It was
reverse-engineered from the live env's installed packages and from
`isaaclab.sh`'s own version pins (which match to the patch version), not
executed from scratch — that's exactly what the dry run below is for.

### 3. Code — ✅ now committed

Everything the reference command needs to *run* (as opposed to *retrain* a
checkpoint) is committed:

| Path | Commit |
|---|---|
| `sims/scripts/pilot/*.py` (all 4 pilot scripts, incl. this session's shutdown-hang + asset-race fixes) | `3ab117d` |
| `sims/scripts/utils/schedule_dispatch.py` | `3ab117d` |
| `sims/isaaclab_tasks/forest_trail/` (code + hand-authored fallback assets; the large Poly Haven binary is deliberately excluded — see §7) | `04bf9b9` |
| `sims/training/*` (base + fine-tune DroNet tooling) | `42bd9d1` |

Still uncommitted, but **not required to run the demo** (only to retrain a
checkpoint from scratch): `sims/scripts/{train,play,debug}/`,
`sims/scripts/README.md`, the loose `sims/*.md` docs (`README.md`,
`STEERING_POLICY_USAGE.md`, etc. — `sims/README.md` is also stale, see the
bottom summary).

### 4. DroNet weights

`--dronet_weights logs/dronet/2026-05-10_20-22-08_finetune/best.pt`

`logs/` is untracked (1.2 GB on disk; not even mentioned in `.gitignore`,
just never added) — this checkpoint has to be regenerated, not cloned. Both
stages are documented in `sims/training/README.md`: §3 (base, IDSIA) and §3b
(fine-tune on sim-collected data, the path that actually produced this
specific checkpoint).

### 5. RL inner-loop checkpoint — ✅ documented, script uncommitted

`--checkpoint logs/rsl_rl/crazyflie_steering_tracking/2026-04-13_12-23-08/model_6998.pt`

```bash
bash sims/scripts/train/train_full.sh   # 4096 envs, 2000 iters, headless, ~20 min
```

`train_full.sh`/`train_steering_tracking.py` are still uncommitted (§3) —
only matters if the dry run needs to retrain rather than reuse an existing
checkpoint. Documented consistently in `sims/scripts/README.md`,
`sims/STEERING_POLICY_USAGE.md`, `sims/training/README.md` §4. Unrelated
caveat worth knowing: `train_harsh.sh` still silently trains the *non*-harsh
config (`train_steering_tracking.py` hardcodes `TrackSteeringEnvCfg()` and
passes it as `cfg=` to `gym.make()`, overriding the `--task ...Harsh-v0`
name) — `TRAINING_CONFIG_BUG.md`/`ENVIRONMENT_MISMATCH_FIX.md` call this
"RESOLVED" but only documented a workaround, not a code fix.

### 6. Schedule JSON — left out of scope, but a tracked substitute exists

`--schedule_json schedules/scheduled_networks_mlp10_dronet20_yolov8_firesim_static_decomposed_profiled.json`

Per an earlier decision this is orthogonal to the sim-runner flow and was
left alone: `data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static.json`
(topology input) and the schedule JSON itself are both still untracked, and
regenerating them depends on `gen/profile/...` hardware timing data whose
provenance isn't traced here.

For the dry run this doesn't have to block anything:
`schedules/scheduled_networks_periodic_dronet50ms_yolov8_firesim_greedy_profiled.json`
**is tracked** and contains real `dronet0`–`dronet3` + `yolov8_nano` dispatch
keys (no `mlp` key — the pilot script handles that gracefully, falling back
to running the MLP ungated). It exercises the same DroNet+YOLO scheduling
code paths as the reference file; swap it in via `--schedule_json` for a
dry run that doesn't depend on the untracked topology/schedule pair.

### 7. Runtime downloads / third-party assets

- **`yolov8n.pt`**: `ultralytics.YOLO("yolov8n.pt")` downloads it
  automatically on first use. Needs outbound internet; no repo-side action
  needed.
- **Human character USD**: `forest_scene.py` fetches from
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/...`
  directly over HTTPS — no Nucleus server required. Falls back to a
  committed local procedural USD (`forest_trail/assets/human.usda`, hand-
  authored primitives, no license concern) if the S3 asset is unreachable.
- **Pine tree asset (`pine_sapling_small`)** — this is a real third-party
  asset, and was specifically audited:
  - **License**: [Poly Haven](https://polyhaven.com), **CC0** (verified
    against `polyhaven.com/license`) — free for any use including
    commercial, redistribution explicitly allowed, no attribution required.
  - **How obtained**: `forest_trail/assets/download_trees.py` hits Poly
    Haven's public files API (`api.polyhaven.com/files/pine_sapling_small`)
    and downloads the 1k-resolution USDC geometry + 9 PBR texture maps from
    `dl.polyhaven.org`.
  - **Is the fetch replicable?** Yes, now — but it was **broken**: Poly
    Haven's API/CDN returns 403 for urllib's default `Python-urllib/x.y`
    User-Agent (curl and browser UAs pass fine). Fixed in this pass
    (`sims/isaaclab_tasks/forest_trail/assets/download_trees.py`) and
    verified with a real from-scratch redownload of all 10 files, diffed
    byte-for-byte (`md5sum`-identical) against the working copy.
  - **What's committed vs. fetched**: the USDC + textures (~46 MB) are
    **not** committed — excluded via a scoped `.gitignore` inside
    `forest_trail/assets/` and regenerated by running `download_trees.py`.
    The hand-authored override layer that fixes a broken MDL material
    remap in the raw USDC (`pine_sapling_small_patched.usda`, ~3 KB, not
    part of the download) **is** committed as source.
  - If `download_trees.py` is never run, `forest_scene.py` falls back to
    the committed procedural `pine_tree.usda` proxy (simple cylinder +
    cone geometry) — the scene still runs, just with plainer trees.

### 8. Run

Once 1–7 are satisfied, the reference command should run — or, for a dry
run that doesn't depend on the two orthogonal untracked pieces (§6),
substitute the tracked schedule file and reuse/regenerate checkpoints per
§4–§5.

---

## Summary: not committed

- `sims/scripts/{train,play,debug}/`, `sims/scripts/README.md` — not required to *run* the demo, only to *retrain* a checkpoint
- Loose `sims/*.md` docs (`README.md`, `STEERING_POLICY_USAGE.md`, etc.) — `sims/README.md` is also stale (see below)
- The specific topology + schedule JSON pair used by the reference command — left out of scope by choice; a tracked substitute exists (§6)
- The Poly Haven tree binary payload (~46 MB) — left out by choice, regenerable via the now-fixed `download_trees.py`
- `logs/`, `datasets/` (expected/appropriate to exclude — just noting they're not even named in `.gitignore`)

Committed across this session: `sims/training/` (`42bd9d1`),
`docs/xpurt_env_setup.md` + frozen manifest (`43401f2`),
`sims/isaaclab_tasks/forest_trail/` (`04bf9b9`),
`sims/scripts/pilot/*` + `schedule_dispatch.py` (`3ab117d`).

## Summary: not documented

- How to regenerate the specific `mlp10_dronet20_yolov8_firesim_static` schedule JSON (topology file untracked, hardware profile provenance untraced) — left out of scope for now
- `sims/README.md` is stale — describes DroNet integration and the forest trail as future "Next Steps" when both already exist and are the primary demo per `sims/scripts/README.md`

Resolved across this session: the DroNet fine-tuning flow, the `xpurt` env
build flow (including the IsaacLab install step), and the Poly Haven asset
license/provenance/fetch-script status.

## Suggested next step

The code-completeness half of the dry run is now unblocked. Order:
(1) submodule init in a scratch clone, (2) build the `xpurt` env per
`docs/xpurt_env_setup.md` (untested from scratch — this is the main open
risk), (3) either copy over the existing DroNet/RL checkpoints or retrain
per §4–§5, (4) run `download_trees.py` or accept the procedural tree
fallback, (5) run the reference command with the tracked substitute
schedule (§6) and a small `--num_periods`, confirm a non-empty video chunk
— matching the smoke test already done earlier in this session, now from a
clean clone instead of the working tree.

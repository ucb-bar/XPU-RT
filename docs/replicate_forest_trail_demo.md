# Replicating the forest-trail navigation demo from a fresh clone

How to get the DroNet + MLP + YOLO forest-trail pilot running, starting from
nothing but a clone of this repo.

Reference command (what you're working toward):

```bash
conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \
    --dronet_weights logs/dronet/<timestamp>_finetune/best.pt \
    --checkpoint logs/rsl_rl/crazyflie_steering_tracking/<timestamp>/model_<N>.pt \
    --schedule_json schedules/<some_schedule>.json \
    --yolo --trail curved --curvature_seed 12 \
    --num_periods 200 \
    --save_video out.mp4
```

## Dependency graph

```
git clone + submodule init
        │
        ▼
xpurt conda env (Isaac Sim 5.1, IsaacLab, PyTorch, rsl_rl, ultralytics)
        │
        ▼
sims/scripts/pilot/, sims/isaaclab_tasks/forest_trail/,
sims/scripts/utils/schedule_dispatch.py, sims/training/
        │
        ├──► DroNet weights
        │    IDSIA download (manual) → extract_idsia.py
        │    → train_dronet.py → best.pt (base)
        │    → collect_sim_data.py (needs Isaac Sim running)
        │    → finetune_dronet.py → best.pt (finetuned, recommended)
        │
        ├──► RL inner-loop checkpoint
        │    sims/scripts/train/train_full.sh
        │    → logs/rsl_rl/crazyflie_steering_tracking/*/model_*.pt
        │
        ├──► schedule JSON
        │    data/toplevel/networks_<name>.json
        │    → scripts/run_xpurt_schedule.py
        │    → schedules/scheduled_networks_<name>_<solver>_profiled.json
        │
        └──► runtime downloads (automatic, need internet)
             - yolov8n.pt (ultralytics, auto-downloaded)
             - human character USD (S3, direct HTTPS, no Nucleus needed)
             - pine_sapling_small tree asset (Poly Haven, CC0)
        │
        ▼
pilot_forest_with_dronet_scheduled.py runs
```

---

## 0. Prerequisites

- NVIDIA GPU with ≥ 10 GB free VRAM.
- A recent NVIDIA driver with CUDA 12.x support (the conda env's `torch`
  bundles its own CUDA 12.8 runtime; the host driver just needs to support
  that).
- No `$DISPLAY` required — Isaac Sim and matplotlib both work fine headless.

## 1. Clone + submodules

```bash
git clone <repo> && cd FreshScheduler
git submodule update --init sims/IsaacLab
```

## 2. Build the `xpurt` conda environment

Follow **[`docs/xpurt_env_setup.md`](xpurt_env_setup.md)** end to end. It
covers installing Isaac Sim, running IsaacLab's installer against the
vendored submodule, and the project-specific extras (ultralytics, imageio,
matplotlib) — including version pins and an EULA-acceptance step that are
easy to trip over on a first attempt.

## 3. Get a DroNet checkpoint

```bash
# Base training (needs the IDSIA dataset -- see sims/training/README.md §2)
python sims/training/train_dronet.py

# Recommended: fine-tune on sim-collected data for better in-sim performance
python sims/training/collect_sim_data.py --headless --num_samples 5000
python sims/training/finetune_dronet.py \
    --checkpoint logs/dronet/<base_run>/best.pt \
    --data_root datasets/sim_forest/extracted/000 \
    --sim_data --epochs 20 --lr 1e-4
```

Full details, all CLI flags, and expected metrics: `sims/training/README.md`
§3 (base) and §3b (fine-tune).

## 4. Get the RL inner-loop checkpoint

```bash
bash sims/scripts/train/train_full.sh   # 4096 envs, 2000 iters, headless, ~20 min
```

Checkpoints land in `logs/rsl_rl/crazyflie_steering_tracking/<timestamp>/`.
Details: `sims/scripts/README.md`, `sims/STEERING_POLICY_USAGE.md`,
`sims/training/README.md` §4.

## 5. Get a schedule JSON

The pilot script's `--schedule_json` needs an XPU-RT scheduler output. To
generate your own from a workload spec:

```bash
python scripts/run_xpurt_schedule.py --profiled data/toplevel/networks_<name>.json
```

See `docs/end_to_end_xpurt_firesim.md` for the full workload-spec → schedule
pipeline. If you just want something that runs, several pre-generated
schedules are already checked into `schedules/` — any file containing
`dronet*`/`yolov8_nano` dispatch keys works with this pilot script (it falls
back to running the MLP ungated if there's no `mlp` key), e.g.
`schedules/scheduled_networks_periodic_dronet50ms_yolov8_firesim_greedy_profiled.json`.

## 6. Third-party assets (no action needed)

These resolve automatically the first time you run the pilot script:

- **`yolov8n.pt`** — downloaded by `ultralytics` on first use.
- **Human character** — streamed from an NVIDIA-hosted S3 bucket over HTTPS;
  falls back to a bundled procedural USD if unreachable.
- **Pine tree model** — [Poly Haven](https://polyhaven.com) asset, **CC0**
  licensed (free for any use, no attribution required). Fetch it explicitly
  for better-looking trees:
  ```bash
  python sims/isaaclab_tasks/forest_trail/assets/download_trees.py
  ```
  If you skip this, the scene falls back to a simpler procedural tree proxy
  and still runs fine.

## 7. Run

```bash
conda run -n xpurt python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \
    --dronet_weights logs/dronet/<timestamp>_finetune/best.pt \
    --checkpoint logs/rsl_rl/crazyflie_steering_tracking/<timestamp>/model_<N>.pt \
    --schedule_json schedules/<some_schedule>.json \
    --yolo --trail curved --curvature_seed 12 \
    --num_periods 200 \
    --save_video out.mp4
```

Video chunks get written every `--video_flush_periods` periods (default 20)
as `<out>_p<N>.mp4`. For a quick smoke test rather than a full 200-period
run, pass `--num_periods 3 --video_flush_periods 1` to see output almost
immediately.

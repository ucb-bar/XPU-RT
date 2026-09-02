# DroNet Training

Two independent models power the forest-trail navigation stack. This document
covers both and explains how to reproduce our training runs from scratch.

```
IDSIA trail images
    │
    └─► train_dronet.py ──► best.pt ──► collect_sim_data.py + finetune_dronet.py ──► best.pt (finetuned)
                                                                                            │
                                                                                            ▼
                                                                    --dronet_weights (pilot_forest_*.py)
                                                                                            │
Isaac Lab sim (4096 envs)                                                                  │ steering command
    │                                                                                      ▼
    └─► train_steering_tracking.py ──► model.pt ──► inner-loop thrust controller
```

The fine-tuning step (§3b) is optional but recommended — see there for why.

---

## 1. Prerequisites

All training runs under the `xpurt` conda environment (Isaac Sim + PyTorch):

```bash
conda activate xpurt
```

DroNet supervised training only needs PyTorch + torchvision (no Isaac Sim
required, so it runs on any machine with a GPU or even CPU).

---

## 2. IDSIA Forest Trails Dataset

DroNet is trained on the **IDSIA Forest Trails** dataset (Giusti et al., 2016):
*"A Machine Learning Approach to Visual Perception of Forest Trails for Mobile
Robots"*, IEEE Robotics and Automation Letters.

The dataset consists of 15 trail-segment recordings, each with three
synchronized cameras mounted at −30° (left, `lc`), 0° (centre, `sc`), and
+30° (right, `rc`). Camera identity is the steering label:

| Sub-directory | Meaning                | Target yaw rate  |
|---------------|------------------------|------------------|
| `lc`          | heading too far left   | `−omega_max`     |
| `sc`          | on the trail centre    | `0.0`            |
| `rc`          | heading too far right  | `+omega_max`     |

### Download

Download the `files-archive` zip from the IDSIA project page and place it at:

```
datasets/idsia/files-archive
```

The archive is a zip-of-zips (~14 GB on disk after extraction). Segment `014`
is held out as test-only per the dataset authors' note in its `info.txt`.

### Extract (one-time)

```bash
python sims/training/extract_idsia.py
```

This unpacks all 15 inner zips into `datasets/idsia/extracted/`, skipping
files that already exist. Re-running is safe and fast.

Optional: extract only specific segments (e.g. while debugging):

```bash
python sims/training/extract_idsia.py --only 000 001 002
```

Smoke-test the dataset after extraction:

```bash
python sims/training/dataset_idsia.py
# Expected output: ~12000 train, ~4000 val, ~4000 test frames
```

---

## 3. DroNet Supervised Training

### Quick start

```bash
python sims/training/train_dronet.py
```

Defaults match our best run:
- model: `small` (112 × 112 input, ~0.37 M parameters)
- 30 epochs, batch 64, lr 1e-3, AdamW + cosine decay
- val segments: 011, 012 (withheld from training; 014 always held out)
- output: `logs/dronet/<YYYY-MM-DD_HH-MM-SS>/`

On an NVIDIA GPU, one full run takes ~2 minutes (≈50 s/epoch).

### Shell wrapper

```bash
bash sims/training/train_dronet.sh          # defaults
bash sims/training/train_dronet.sh --epochs 50 --lr 5e-4   # overrides
```

### All CLI options

```
--data_root PATH         Extracted IDSIA directory  (default: datasets/idsia/extracted)
--out_dir PATH           Checkpoint parent dir       (default: logs/dronet)
--model_size {small,large}   small=112px, large=224px   (default: small)
--img_size INT           Override input edge length
--epochs INT             (default: 30)
--batch_size INT         (default: 64)
--lr FLOAT               (default: 1e-3)
--weight_decay FLOAT     (default: 1e-4)
--num_workers INT        DataLoader workers          (default: 4)
--omega_max FLOAT        |yaw_rate| for lc/rc labels (default: 1.0 rad/s)
--val_segments ID ...    Segments to hold out        (default: 011 012)
--device {cuda,cpu}      (auto-detected)
--seed INT               (default: 0)
--log_every INT          Steps between loss prints   (default: 20)
```

### Outputs

```
logs/dronet/<timestamp>/
├── config.json      hyperparameters used for this run
├── best.pt          state_dict at lowest val MSE   ← use this for inference
├── last.pt          state_dict at final epoch
└── history.json     per-epoch {train_mse, val_mse, means_by_class, lr, elapsed_s}
```

### Expected results (small model, 30 epochs)

| Metric              | Typical value |
|---------------------|---------------|
| Final train MSE     | ~0.08         |
| Best val MSE        | ~0.45–0.55    |
| means_by_class `lc` | −0.2 to −0.3  |
| means_by_class `sc` | ±0.2          |
| means_by_class `rc` | +0.6 to +0.8  |

The three `means_by_class` values should have the correct sign ordering
(`lc` < `sc` < `rc`). If they don't, something is wrong with the label sign
convention or data extraction.

---

## 3b. Fine-tuning DroNet on Simulation Data

IDSIA is real-world trail footage; the forest-trail Isaac Lab scene looks
different enough (synthetic trees, lighting, camera FOV) that a DroNet
trained purely on IDSIA underperforms in sim. The fix is to collect labeled
frames *from the sim itself* and fine-tune the IDSIA-trained checkpoint on
them. This is a second, independent training pass — it does not replace
step 3, it starts from its output.

```
best.pt (from step 3)
    │
    └─► collect_sim_data.py ──► datasets/sim_forest/extracted/<seg>/
                                 (frames + labels.csv)
                                       │
                                       └─► finetune_dronet.py ──► best.pt (fine-tuned)
                                                                       │
                                                                       ▼
                                                        --dronet_weights (pilot_forest_*.py)
```

### Step 1 — Collect labeled frames from the sim

`sims/training/collect_sim_data.py` teleports the drone to randomized
positions along the trail (lateral offset + heading error), settles a few
physics steps, then captures an FPV frame. The label is the corrective yaw
rate implied by the known offset/heading error — larger error → stronger
correction — discretized into `lc`/`sc`/`rc` directories (IDSIA-compatible)
with continuous labels also written to a companion `labels.csv`.

Requires Isaac Sim (same `xpurt` env as the pilot scripts, not just
plain PyTorch like step 3):

```bash
conda run -n xpurt python sims/training/collect_sim_data.py \
    --headless --num_samples 5000
```

Key options:

```
--num_samples INT           Total frames to collect        (default: 5000)
--out_dir PATH               IDSIA-style segment output dir  (default: datasets/sim_forest/extracted/000)
--img_size INT                Camera height in px (width = ×4/3) (default: 480)
--trail_length FLOAT           Sampled trail length (m)       (default: 30.0)
--with_humans                  Include procedural humans in the scene
--max_lateral_offset FLOAT      Max sampled offset from centre (m) (default: 1.2)
--max_heading_error_deg FLOAT    Max sampled heading error (deg)   (default: 35.0)
--height_range FLOAT FLOAT        Drone height sampling range (m)   (default: 0.8 1.2)
--omega_max FLOAT                 |yaw_rate| label scale (rad/s)    (default: 1.0)
--seed INT                        (default: 0)
--settle_steps INT                 Sim steps before capture per sample (default: 5)
```

Output layout matches IDSIA (`videos/{lc,sc,rc}/*.jpg` + `labels.csv` with
columns `filename, class, steering_label, x, y_off, heading_err_deg, height`)
— see `sims/training/dataset_sim.py` for the reader.

### Step 2 — Fine-tune from the IDSIA checkpoint

`sims/training/finetune_dronet.py` loads an existing checkpoint and
continues training at a lower LR on the sim-collected data. It never
overwrites the input checkpoint — output goes to a new
`<timestamp>_finetune/` run dir, same layout as step 3's output.

```bash
python sims/training/finetune_dronet.py \
    --checkpoint logs/dronet/<base_run>/best.pt \
    --data_root datasets/sim_forest/extracted/000 \
    --sim_data --epochs 20 --lr 1e-4
```

Note: this step is plain PyTorch, same as step 3 — no Isaac Sim needed once
the data is collected.

Key options:

```
--checkpoint PATH        Pre-trained DroNet state_dict.pt   (required)
--data_root PATH          Extracted IDSIA tree, or sim segment dir (default: datasets/idsia/extracted)
--sim_data                 Treat --data_root as sim-collected (labels.csv) instead of IDSIA
--out_dir PATH               Parent dir for the finetuned run    (default: logs/dronet)
--model_size {small,large}    Must match --checkpoint            (default: small)
--epochs INT                   (default: 20)
--batch_size INT                (default: 64)
--lr FLOAT                       10x lower than from-scratch      (default: 1e-4)
--weight_decay FLOAT               (default: 1e-4)
--val_segments ID ...               IDSIA-mode only               (default: 011 012)
--seed INT                           (default: 0)
```

Output: `logs/dronet/<timestamp>_finetune/{config.json,best.pt,last.pt,history.json}`
— identical shape to step 3's output, and equally valid for `--dronet_weights`.

---

## 4. Inner-loop RL Policy Training (MLP)

The inner-loop MLP converts a steering command (from DroNet) into per-motor
thrust commands. It is trained with PPO in Isaac Lab, independently of DroNet.

### Quick start

```bash
bash sims/scripts/train/train_full.sh                  # 4096 envs, 2000 iters, headless
```

Or launch directly:

```bash
conda run --no-capture-output -n xpurt \
    python -u sims/scripts/train/train_steering_tracking.py \
    --headless --num_envs 4096 --max_iterations 2000
```

Small-scale smoke test (~20 min on 1 GPU):

```bash
conda run --no-capture-output -n xpurt \
    python -u sims/scripts/train/train_steering_tracking.py \
    --num_envs 512 --max_iterations 500
```

Checkpoints are saved to `logs/rsl_rl/crazyflie_steering_tracking/<timestamp>/`.

For full details on the RL environment, reward terms, and observation space
see the task configuration itself — there is no separate write-up:
[`sims/isaaclab_tasks/track_steering_vision/config/crazyflie/track_steering_env_cfg.py`](../isaaclab_tasks/track_steering_vision/config/crazyflie/track_steering_env_cfg.py)
defines the observation and reward terms, and
[`.../agents/rsl_rl_ppo_cfg.py`](../isaaclab_tasks/track_steering_vision/config/crazyflie/agents/rsl_rl_ppo_cfg.py)
the PPO hyperparameters. A `_harsh` variant of the env config sits beside
the first.

---

## 5. Running Inference

With both checkpoints in hand:

```bash
# Forest trail demo with DroNet + scheduled inner-loop policy
python sims/scripts/pilot/pilot_forest_with_dronet_scheduled.py \
    --dronet_weights logs/dronet/<timestamp>/best.pt \
    --camera_update_period 0.1 \
    --num_periods 40 \
    --forward_velocity 1.0

# Or without the XPU-RT schedule dispatcher (always-on inference):
python sims/scripts/pilot/pilot_forest_with_dronet.py \
    --dronet_weights logs/dronet/<timestamp>/best.pt
```

The `--dronet_weights` flag accepts any `best.pt` or `last.pt` produced by
`train_dronet.py` or `finetune_dronet.py` (§3b) — both are plain `state_dict`s
compatible with `DronetTorch(img_dims=(112,112), img_channels=3, output_dim=1, small=True)`.

---

## 6. File Map

```
sims/training/
├── README.md               this file
├── train_dronet.py         supervised training script (§3)
├── train_dronet.sh         shell wrapper (mirrors train_full.sh style)
├── dataset_idsia.py        PyTorch Dataset + smoke-test __main__ (IDSIA)
├── extract_idsia.py        one-time IDSIA archive extractor
├── collect_sim_data.py     sim-data collector for fine-tuning (§3b step 1)
├── dataset_sim.py          PyTorch Dataset for sim-collected data (§3b)
└── finetune_dronet.py      fine-tune an existing checkpoint (§3b step 2)

datasets/
├── idsia/
│   ├── files-archive       outer zip download (place here manually, ~14 GB)
│   └── extracted/          output of extract_idsia.py
│       ├── 000/videos/{lc,sc,rc}/*.jpg
│       ├── ...
│       └── 014/            held-out test segment (never used for train/val)
└── sim_forest/
    └── extracted/000/      output of collect_sim_data.py (IDSIA-style layout + labels.csv)

logs/
├── dronet/<timestamp>/              DroNet supervised training outputs (§3)
├── dronet/<timestamp>_finetune/     DroNet fine-tuning outputs (§3b)
└── rsl_rl/crazyflie_steering_tracking/<timestamp>/   RL policy outputs
```

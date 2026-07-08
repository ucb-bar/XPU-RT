# Setting up the `xpurt` conda environment

Answers two questions: **is there an existing recipe to replicate `xpurt`?**
(no — see below) and **what's the actual flow to build one that runs the
forest-trail demo?** (this doc).

## Is there a global conda setup for this already?

No. `conda-meta/history` inside the live `xpurt` env records exactly one
conda-level operation, ever:

```
==> 2026-04-09 16:44:39 <==
# cmd: conda create -n xpurt python=3.11
```

Everything else — Isaac Sim, PyTorch, IsaacLab, rsl_rl, ultralytics, YOLO,
matplotlib, and ~350 other pip packages accumulated over time — was
installed with `pip` afterward, and pip installs aren't recorded in
`conda-meta/history`. There's no `environment.yml`, no `requirements.txt`,
no install script anywhere in the repo. `sims/scripts/utils/setup_env.sh`
only exports `PYTHONPATH`, and does so against a stale, pre-vendoring path.

The flow below was reverse-engineered from what's actually installed
(`pip show <pkg>`, `pip index versions <pkg>` against plain PyPI with no
custom index configured, and reading `sims/IsaacLab/isaaclab.sh`'s own
`ensure_cuda_torch()` — its exact version pins for `torch`/`torchvision`
match what's installed to the patch version, so this is very likely the
actual flow that was originally run, not a guess from scratch) — but it has
**not been executed end-to-end on a clean machine**. That's the point of the
planned dry run in a temp directory.

## The flow

```bash
# 1. Bare env — this part *is* recorded (conda-meta/history)
conda create -n xpurt python=3.11 -y
conda activate xpurt

# 2. Isaac Sim, from plain PyPI — no NVIDIA-specific index needed for 5.1.0.0
#    (confirmed: `pip index versions isaacsim` resolves with zero extra-index
#    configured in this env's pip config)
pip install "isaacsim[all,extscache]==5.1.0.0"

# 3. IsaacLab itself — run against the vendored submodule, NOT a separate
#    IsaacLab checkout. This is the step that's easy to miss: it's not
#    enough to have sims/IsaacLab on PYTHONPATH, isaaclab/isaaclab_rl/
#    isaaclab_assets/isaaclab_contrib/isaaclab_mimic/isaaclab_tasks are all
#    `pip install -e` targets, confirmed via `pip show isaaclab` showing
#    `Editable project location: .../sims/IsaacLab/source/isaaclab`.
cd sims/IsaacLab
./isaaclab.sh --install
# ↑ this one command also, internally:
#   - installs system deps via `sudo apt-get` (cmake, build-essential, ...)
#     — expect a sudo prompt on a fresh machine
#   - pins setuptools<82
#   - runs its own ensure_cuda_torch(): installs torch==2.7.0 +
#     torchvision==0.22.0 from https://download.pytorch.org/whl/cu128
#     (exact match to what's installed today — this pin lives in
#     isaaclab.sh itself, not something we need to specify separately)
#   - pip installs all 4 supported RL frameworks (rsl_rl, rl_games, skrl,
#     stable_baselines3) — confirmed all 4 are present in the live env
cd ../..

# 4. Project-specific extras — not part of Isaac Sim or IsaacLab, needed by
#    the pilot/training scripts themselves (DroNet/YOLO inference, FPV plot,
#    video export):
pip install ultralytics "imageio[ffmpeg]" matplotlib
```

That's the whole flow. `gymnasium` comes in transitively via `isaaclab`'s
own dependencies — no separate install line needed.

### Sanity check

```bash
python -c "
import torch, isaacsim, isaaclab, rsl_rl, ultralytics, imageio, matplotlib
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('isaaclab', isaaclab.__file__)
"
```

Expect `torch 2.7.0+cu128 cuda True` and `isaaclab.__file__` pointing at
`sims/IsaacLab/source/isaaclab/...` (editable install, not site-packages).

## Exact-reproduction fallback

The curated flow above is what's *necessary* for the demo. The live `xpurt`
env also has ~350 other pip packages from unrelated work (this conda env is
shared across projects, not demo-dedicated) — freezing all of them isn't a
clean recipe, but it is available as a fallback if the curated flow above
drifts or someone wants a byte-for-byte match: `docs/xpurt_pip_freeze_2026-07-08.txt`
(`pip freeze --exclude-editable`, captured 2026-07-08). This is a point-in-time
snapshot, not maintained — prefer the curated flow above and only reach for
this if something in it fails to resolve.

## Open question for the dry run

`install_system_deps` in `isaaclab.sh` calls `sudo apt-get install`. The
exact package list wasn't re-derived here (it's in `isaaclab.sh` itself, not
duplicated in this doc to avoid drift) — on a fresh machine without those
packages already present, the dry run should surface exactly what's missing.

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
actual flow that was originally run, not a guess from scratch) — and it has
now been **executed end-to-end on a clean machine** (fresh git clone in
`/tmp`, brand-new conda env, following this flow verbatim). It ran the
reference pilot command successfully, but only after fixing three real bugs
the dry run surfaced (§ below) that weren't visible from just reading the
original env's installed-package state.

## The flow

```bash
# 1. Bare env — this part *is* recorded (conda-meta/history)
conda create -n xpurt python=3.11 -y
conda activate xpurt

# 2. Isaac Sim, from plain PyPI — no NVIDIA-specific index needed for 5.1.0.0
#    (confirmed: `pip index versions isaacsim` resolves with zero extra-index
#    configured in this env's pip config)
pip install "isaacsim[all,extscache]==5.1.0.0"

# 3. Every fresh Isaac Sim invocation prompts interactively for an EULA
#    (isaacsim/kit/kit_app.py:check_eula()) unless this is set. It does NOT
#    persist across separate shell/process invocations on its own -- the
#    EULA_ACCEPTED marker file only gets written by the *interactive*
#    input() path, never by this env-var bypass -- so export it for the
#    whole session (every command below, and every future pilot-script
#    run) rather than one-off per command.
export OMNI_KIT_ACCEPT_EULA=Y

# 4. IsaacLab itself — run against the vendored submodule, NOT a separate
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

# 5. IMPORTANT: verify cuDNN survived step 4 intact before going further.
#    On the dry run, resolving the RL frameworks' dependencies (stage 4)
#    pulled in an orphaned `nvidia-cudnn-cu13` package alongside the correct
#    `nvidia-cudnn-cu12` -- nothing ends up actually requiring cu13
#    afterward (`pip show nvidia-cudnn-cu13` shows empty Required-by), but
#    both packages install to the *same* file path
#    (nvidia/cudnn/lib/libcudnn.so.9), so whichever installs *last* wins on
#    disk while both dist-infos claim ownership. cu13 (built for a newer
#    CUDA than this host's driver supports) silently clobbered cu12,
#    breaking every cuDNN op in the process with
#    `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` --
#    reproducible, not transient, and easy to misdiagnose as an Isaac-Sim
#    problem since it only surfaces the first time something (e.g. DroNet)
#    actually calls into cuDNN.
python -c "import torch; torch.nn.Conv2d(3,4,3).cuda()(torch.randn(1,3,8,8,device='cuda')); print('cuDNN OK')"
# If that fails, check for the orphan and fix forward:
pip show nvidia-cudnn-cu13 >/dev/null 2>&1 && pip uninstall -y nvidia-cudnn-cu13
pip install --force-reinstall --no-deps "nvidia-cudnn-cu12==9.7.1.26"

# 6. Project-specific extras — not part of Isaac Sim or IsaacLab, needed by
#    the pilot/training scripts themselves (DroNet/YOLO inference, FPV plot,
#    video export). Pin ultralytics -- unpinned `pip install ultralytics`
#    resolved to a newer release during the dry run whose opencv-python
#    dependency forces numpy>=2, silently breaking numpy for
#    isaaclab/isaaclab-rl/isaaclab-tasks/isaacsim-kernel (all pin numpy<2;
#    isaacsim-kernel pins numpy==1.26.0 exactly, an ABI dependency, not just
#    a loose bound). Versions below are what's confirmed working today:
pip install "ultralytics==8.4.39" "opencv-python==4.11.0.86" "numpy==1.26.0" "imageio[ffmpeg]" matplotlib
```

That's the whole flow. `gymnasium` comes in transitively via `isaaclab`'s
own dependencies — no separate install line needed.

### Sanity check

```bash
OMNI_KIT_ACCEPT_EULA=Y python -c "
import torch, isaacsim, isaaclab, rsl_rl, ultralytics, imageio, matplotlib
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('isaaclab', isaaclab.__file__)
"
```

Expect `torch 2.7.0+cu128 cuda True` and `isaaclab.__file__` pointing at
`sims/IsaacLab/source/isaaclab/...` (editable install, not site-packages).
And remember `OMNI_KIT_ACCEPT_EULA=Y` on every future invocation that
imports `isaacsim` (including the pilot scripts themselves) — it's a
per-process bypass, not a one-time acceptance.

## Exact-reproduction fallback

The curated flow above is what's *necessary* for the demo. The live `xpurt`
env also has ~350 other pip packages from unrelated work (this conda env is
shared across projects, not demo-dedicated) — freezing all of them isn't a
clean recipe, but it is available as a fallback if the curated flow above
drifts or someone wants a byte-for-byte match: `docs/xpurt_pip_freeze_2026-07-08.txt`
(`pip freeze --exclude-editable`, captured 2026-07-08). This is a point-in-time
snapshot, not maintained — prefer the curated flow above and only reach for
this if something in it fails to resolve.

## Dry run result

Executed 2026-07-08: fresh `git clone` into `/tmp`, brand-new `xpurt_dryrun`
conda env, this flow run verbatim (including hitting and fixing the three
issues folded into steps 3/5/6 above), then the reference pilot command run
with a small `--num_periods` and a tracked substitute schedule file (see
`docs/replicate_forest_trail_demo.md` §5, "Get a schedule JSON"). Result: DroNet/MLP/YOLO all
executed, three video chunks produced and verified with `ffprobe`
(valid H.264, real frames), and shutdown exited cleanly with no hang.

One thing not exercised: `install_system_deps` in `isaaclab.sh` calls
`sudo apt-get install`, and this host already had those packages present
from building the original `xpurt` env, so the dry run couldn't confirm
what apt-get actually pulls in on a machine that's never had them. The
exact package list wasn't re-derived here (it's in `isaaclab.sh` itself,
not duplicated in this doc to avoid drift).

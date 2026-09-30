# What the flights were flown in

Every flight number in this repository comes from Isaac Lab, not from the board. The board supplies
the measured cadence and latency each flight replays; the flight itself is a simulation. This page
records the simulator that produced them, because nothing else in the repository did.

## The environment

| | |
|---|---|
| interpreter | `/scratch2/agustin/miniforge3/envs/env_isaaclab/bin/python`, CPython **3.11.15** |
| Isaac Sim | **5.1.0.0** |
| Isaac Lab | commit **`4df6560e187f2cc66685b41b21b259f4485d0c22`** (branch `training-checkpoints-develop`, 2026-04-08) |
| PyTorch | **2.7.0+cu128** |
| NumPy | **1.26.0** |
| warp-lang | **1.12.1** |

The Isaac Lab commit is exactly what the `sims/IsaacLab` submodule is pinned to, so the pin in this
repository is the version that produced the results. It is not initialised in every working copy;
`git submodule update --init sims/IsaacLab` fetches it.

## How a flight finds Isaac Lab

`sims/scripts/sweep_rate_demo.py` -- the script every campaign, display search, scene census and
energy run ultimately calls -- puts Isaac Lab's four source packages on `sys.path` before importing
`isaaclab.app`. It resolves the directory in this order:

1. `$ISAACLAB_SOURCE`, if set;
2. `sims/IsaacLab/source`, the pinned submodule, if it is checked out;
3. the checkout the results were produced against.

So a clean clone needs only `git submodule update --init sims/IsaacLab`, and nothing in the file has
to be edited. Setting `ISAACLAB_SOURCE` overrides both.

Other scripts under `sims/scripts/` still name the original checkout directly. They are training,
evaluation and probe utilities, not on the path any figure in `results/codesign_feedback/refined/`
depends on; `grep -rl /scratch2/dima/IsaacLab sims/scripts` lists them.

## What this does and does not buy

The environment is recorded, and the Isaac Lab version is pinned and fetchable. **Flights are still
not bit-reproducible**: the same seed re-flown can differ, which is why every displayed pair is
re-verified on the scene it flies rather than carried over from a census row, and why the claims in
the figures rest on censuses of 48-60 flights per arm rather than on a single flight.

The board side is pinned separately and more tightly: each Gantt row's trace, per-core sampler and
manifest carry a sha256 in the figure's sidecar, the executed schedule table is compared against the
solved one, and both arms are checked to have run the same staged YOLO IR. See
[`artifact_checklist.md`](../Artifact/artifact_checklist.md).

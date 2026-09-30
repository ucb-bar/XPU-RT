# 01 — build and deploy: kernels, the ROS 2 node, the IME kernel

**Needs the K1 board and the SpaceMiT cross toolchain. Nothing here runs on the host alone.**

Both arms of every comparison run the *same* generated C. The kernels are generated from one IR by
ModelBlaster; XPU-RT's runtime and the ROS 2 node link the same objects and differ only in the
orchestrator. That is what makes the comparison two runtimes over one network rather than two
networks, and it is why this step comes first.

Authoritative recipes, in the order they are used:

| what | where the command is | note |
|---|---|---|
| per-network kernels + skeleton (the deployed build) | `docs/Baselines/ros_baseline_reproduction.md` §1 | `yolov8_nano_64x96` → `yolo`, `fused_full` → `nav`, `mlp_control` → `ctrl`; `nav` needs `-march=rv64gcv_zvfh` for its fp16 linear |
| the 4-hart sharded YOLO | `docs/Baselines/ros_baseline_reproduction.md` §1 "The 4-hart YOLO" | `MB_SHARD_FACTOR=4`; **`CROSS` must be in the environment** or the curated kernels fail verify and the generator falls back to the scalar reference silently |
| the traced ROS 2 node binaries | `docs/Baselines/ros_baseline_reproduction.md` §2 | the `_pool` and plain variants; the libraries that are easy to miss are listed there |
| the IME fused kernel and its build | `docs/K1/ime_kernel_reproduction.md` §2–§3 | the build log must say `[conv2d_batchnorm2d_silu_s8/ime_vmadot_4x4x8] ... curated[ime] verify PASS` |

## The scripts that do it

```bash
eval "$(bash scripts/setup_spacemit_toolchain.sh)"   # puts CROSS in the environment
scripts/board_deploy_ros_node.sh                     # env MODELBLASTER_K1_HOST=k1
scripts/board_campaign.sh                            # steps 1-5: stage, build, measure
```

* `scripts/board_deploy_ros_node.sh` pushes `board/k1_ros_mb/` (the node, the samplers, `cpu_hog`)
  and `ModelBlaster/runtime/modelblaster_pool/`, then compiles the two objects. It does **not** push
  the per-network kernels: those are generated from the IR and staged by `board_campaign.sh`, which
  records their sha256 per run. It prints `DEPLOY_DONE`.
* `scripts/board_campaign.sh` is the whole measurement campaign in one sequence, so that nothing
  ever compiles natively on the board while something else is being measured there (a native build
  shows up in the sweep). Its step 1 stages the sharded YOLO and times it standalone on 1 and 4
  harts against the golden output into
  `results/codesign_feedback/ros_traced/yolo_standalone/`; step 2 links the pool-enabled traced ROS
  node; steps 3–5 are measurement and belong to [`../03_measure/`](../03_measure/).

`ModelBlaster/scripts/check_kernel_coverage.py <generated dir>` must print `OK` after a generate —
that is the check that catches the silent scalar fallback.

## What is already in the repo, and why

`board/k1_ros_mb/` holds the ROS 2 baseline node (`ros_mb_chain_traced.cpp`), the per-core sampler,
the shared trace origin (`rdtime_now.c`), the load generator and the worker pool, with a tracked
`MANIFEST.sha256`. Every ROS arm is that one program with different flags, and it lived only on the
board: without it in the repo a reimaged board would take every baseline with it.

```bash
scripts/board_source_snapshot.sh            # pull the board's sources into board/k1_ros_mb/
scripts/board_source_snapshot.sh --verify   # needs the board: fail if the board differs from the repo
```

The `--verify` form is the board line of `docs/Artifact/artifact_checklist.md` §3, and the one check in that
list that `artifact/verify_no_hardware.sh` does not run.

## First: get the submodule, because a clone cannot

Every command on this page runs inside `ModelBlaster/`, and a plain clone leaves that directory
**empty**. The superproject pins `2b11d038`, which does contain the fused IME kernel, both benches,
both measured tables and the pool's per-slice tracing — `docs/K1/ime_kernel_reproduction.md` §2–§3 and
`ros_baseline_reproduction.md` §2b are followable from the pinned state. It is simply not pushed:
it is nine commits ahead of `origin/feat/split-linear-along-m` and `git ls-remote` does not have it,
so `git submodule update --init ModelBlaster` fails instead of checking out something older.
Advancing the pointer would not help — no pushed commit carries this work.

[`../history/`](../history/) carries the commit as a bundle; `docs/K1/ime_kernel_reproduction.md` §0 is
the three-command recipe, and `scripts/verify_modelblaster_inputs.py` checks that the bundle's tip is
still the commit the superproject pins. This is the second bullet of `docs/Artifact/artifact_checklist.md` §5.

## Not verified here

Every command on this page needs the board or the cross toolchain, so none of them was run while
this directory was written. Each is quoted from, or is the script named by, the document in the
table above.

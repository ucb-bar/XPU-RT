# zephyr-chipyard-sw: 3 commits that exist on no remote

`zephyr-3commits.bundle` (28 KB) carries
`feat/firesim-bitexact-profile-recalibration` (`418136f`) and its two parents,
on top of `ae5d736`.

## Why they live here

They exist on **no remote and in no other clone**. This account can push
`ucb-bar/ModelBlaster` and `ucb-bar/merlin` but **not**
`ucb-bar/zephyr-chipyard-sw`: SSH authenticates (`Hi copparihollmann!`) and the
push is refused, so it is an authorization gap, not a credential problem. A fork
is the normal answer and needs the GitHub API or the web UI.

XPU-RT *is* pushable, so the commits live here until that access exists. 28 KB
in a repo that reaches GitHub beats 721 MB on one disk.

## Restoring

```bash
cd zephyr-chipyard-sw
git fetch /path/to/zephyr-3commits.bundle HEAD:feat/firesim-bitexact-profile-recalibration
```

**Verified**: fetched into a clean clone this yields commit
`418136fe7fe52f1469d0955d0ba79d79ac06ea00` and tree
`a62bdd218ceebdc3eea196d8245c08fff0b493c1` — byte-identical to the original.

A `format-patch` series was tried first and **does not apply** (the profile CSVs
defeat textual patching), which is why this is a bundle. The full-history bundle
at `../zephyr-chipyard-sw-*.bundle` remains the richer restore path while the
disk is intact; this is the copy small enough to live in git.

## Contents

Profile-DB data: bit-exact FireSim recalibration for dronet, yolov8_nano and rvv
mlp_control/yolov8_fused. 12 files, all under `gen/profile/**`, +962/-326. The
SHA the superproject records (`1ff7f6f`) touches only `samples/rose*/**`, so a
merge has **zero file overlap**.

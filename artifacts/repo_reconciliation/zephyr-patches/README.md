# zephyr-chipyard-sw: 3 commits with no remote

These reconstruct `feat/firesim-bitexact-profile-recalibration` (`418136f`) on
top of `ae5d736`, the merge-base with the SHA the XPU-RT superproject records
(`1ff7f6f`).

## Why they live here

They exist on **no remote and in no other clone**. The account has push access to
`ucb-bar/ModelBlaster` and `ucb-bar/merlin`, but **not** to
`ucb-bar/zephyr-chipyard-sw` — SSH authenticates (`Hi copparihollmann!`) and the
push is refused, so this is an authorization gap, not a credential problem. A
fork would be the normal answer and requires the GitHub API or the web UI.

XPU-RT *is* pushable, so keeping the patches here puts the work somewhere
durable and off-disk rather than leaving it to a single verified bundle on
`/scratch2`. The full bundle (721 MB, complete history) is at
`../zephyr-chipyard-sw-*.bundle` and is the better restore path when the disk is
intact; these patches are the copy that survives the disk.

## Restoring

```bash
cd zephyr-chipyard-sw
git checkout -b feat/firesim-bitexact-profile-recalibration ae5d736
git am /path/to/zephyr-patches/*.patch
```

Content is profile-DB data: bit-exact FireSim recalibration for dronet,
yolov8_nano and rvv mlp_control/yolov8_fused — 12 files, all under
`gen/profile/**`, +962/-326. The recorded `1ff7f6f` touches only
`samples/rose*/**`, so a merge has **zero file overlap**.

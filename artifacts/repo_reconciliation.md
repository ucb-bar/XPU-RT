# Repository reconciliation — K1 bring-up

Performed 2026-08-26. Nothing was reset, cleaned, or force-updated.

## Safety net taken BEFORE any history operation

| Artifact | Contents |
|---|---|
| `artifacts/repo_reconciliation/zephyr-chipyard-sw-20260826-203857.bundle` | 721 MB, `--all`, verified "records a complete history" |
| `artifacts/repo_reconciliation/xpurt-20260826-212844.bundle` | 177 MB, `--all` |
| tag `premerge-k1-20260826` | annotated, on the pre-rebase HEAD `e70a4b4` |
| `~/.ssh/config.bak.*` | before the `Host k1` stanza was appended |

## Per repository

### XPU-RT — reconciled

| | Before | After |
|---|---|---|
| Branch | `feat/freshness-validity-eval` | `feat/k1-modelblaster-closed-loop` |
| HEAD | `e70a4b4` | `336c2ca` |
| Base | `origin/dev` @ `fe6feca` (23 days stale) | `origin/dev` @ `587f96f` |
| Position | 42 ahead / 0 behind (of stale dev) | **42 ahead / 0 behind** (of current dev) |

The fetch moved `origin/dev` `fe6feca..587f96f`, six commits, including
`06c187f Fix XPU-RT scheduler bugs; add mlp+dronet+yolo spike reproduction` — so the
rebase **gained upstream scheduler fixes**, which is the main reason it was worth doing.

Rebase outcome: all 42 original commit subjects verified present afterwards (set
comparison, zero missing). **363 tests pass, 1 skipped.**

Two obstacles, both handled:
- Untracked `xpu-rt/xpurt.egg-info/*` blocked commit 2/43. These are `pip install -e`
  output; backed up to the scratchpad and removed.
- **One real conflict**, in `xpu-rt/plot.py` (`c3a8f0b`). Upstream and we had
  *independently* fixed the same FreeType "raster overflow" bug that was destroying solved
  schedules. Upstream used a fixed dpi retry ladder with `finally: plt.close()`; ours scaled
  dpi by op count and re-raised so the caller decides, but **never closed the figure**.
  Resolved by **combining both** — adaptive dpi + ladder + re-raise + `finally: plt.close()`
  — rather than picking a side. A leaked figure per cell is a real cost across a 45-cell sweep.

The old branch `feat/freshness-validity-eval` is untouched at `e70a4b4` and still matches
its remote.

### zephyr-chipyard-sw — RESCUED LOCALLY, still not durably safe

Branch `feat/firesim-bitexact-profile-recalibration` @ `418136f`, **no upstream configured**.
Three commits (12 files, all `gen/profile/**` CSVs, +962/-326) that exist on **no remote and
in no other clone on this machine**:

```
418136fe7fe  profile DB: bit-exact FireSim recalibration data + rvv mlp_control/yolov8_fused
03ee2e2604d  profile DB: yolov8_nano hetero cycles from FireSim measurement
de22a1d4f36  profile DB: dronet hetero cycles from bit-exact FireSim measurement
```

**Push is not possible:** both HTTPS and SSH return
`Permission to ucb-bar/zephyr-chipyard-sw.git denied to copparihollmann`. SSH auth itself
succeeds (`Hi copparihollmann!`), so this is an authorization gap, not a credential problem.
No fork exists at `copparihollmann/zephyr-chipyard-sw`, and `gh` is not installed here.

⇒ Secured by bundle only. **Open action: create a fork and push.**

The submodule worktree was verified still at `418136f` on its own branch after the rebase.
The parent gitlink remains modified (`1ff7f6f` recorded vs `418136f` checked out); it was
deliberately **not** committed, because recording a SHA that exists on no remote would make
the fragility permanent. Commit that bump only after the fork push lands.

Merge outlook: merge-base with the recorded `1ff7f6f` is `ae5d736`; our side touches only
`gen/profile/**` and the other side only `samples/rose*/**` — **zero file overlap**.

### ModelBlaster — three states, unresolved

| Location | SHA | Status |
|---|---|---|
| `XPU-RT/ModelBlaster` (submodule) | `dbbdcf0a` | **empty**, uninitialized. Is `ucb-bar/ModelBlaster` `main` tip. |
| `XPU-RT/zephyr-chipyard-sw/modelblaster` (nested) | `79328776` | **empty**, uninitialized. **Strict ancestor** of `dbbdcf0a`, 85 commits behind. |
| `/scratch2/agustin/ModelBlaster` (standalone) | `3da8192` | **the live working tree**, branch `feat/triple-fused-conv-bn-silu`, pushed. |

The brief's premise — that the nested pointer carries fixes the root lacks — is **backwards**.
There is no divergence to reconcile between the two pointers; it is a fast-forward. The real
question is how the live `3da8192` relates to `main`.

Init traps for later: `hw/chipyard` and `zephyr-chipyard-sw/modelblaster` use **SSH** URLs
while everything else is HTTPS; no submodule declares a `branch =` key, so
`--remote` is not a usable shortcut anywhere.

### merlin — untouched, now provisioned

`4582be0` (`v0.1.0-rc1-46-g4582be0`), clean. Added (untracked, not a history change):
`build_tools/riscv-tools-spacemit/spacemit-toolchain-linux-glibc-x86_64-v1.1.2`,
4.7 GB, `riscv64-unknown-linux-gnu-gcc 14.3.0`.

## Open actions

1. **Create a fork of `zephyr-chipyard-sw` and push the rescue branch.** Until then those
   3 commits live only on this disk.
2. Decide the canonical ModelBlaster checkout, then init both submodule paths at it.
3. Verify the Zephyr build tolerates whichever SHA is chosen (`dbbdcf0a` removed ~171k lines
   and moved KernelBlaster to a submodule).

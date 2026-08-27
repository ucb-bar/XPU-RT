# Repository reconciliation

Regenerated 2026-08-27 from live state. The previous version of this file was
written mid-session and every "after" SHA in it had gone stale; it also did not
mention the sibling ModelBlaster clone, which turned out to hold the largest
body of at-risk work. Regenerate rather than edit in place — the whole value of
this file is that its SHAs are true.

## Live state

| repo | working dir | branch | HEAD | upstream | off-remote | dirty (tracked) |
|---|---|---|---|---|---|---|
| XPU-RT | `/scratch2/agustin/XPU-RT` | `feat/k1-modelblaster-closed-loop` | `f042202` | `origin/feat/k1-modelblaster-closed-loop` | **0** | 3 |
| ModelBlaster (submodule, canonical) | `XPU-RT/ModelBlaster` | `feat/k1-xpurt` | `163fd02` | `origin/feat/k1-xpurt` | **0** | 3 |
| merlin | `XPU-RT/merlin` | `feat/k1-xpurt` | `e32529a` | `origin/feat/k1-xpurt` | **0** | 0 |
| zephyr-chipyard-sw | `XPU-RT/zephyr-chipyard-sw` | `feat/firesim-bitexact-profile-recalibration` | `b6e359c` | `fork/…` (see below) | **0** | 0 |
| ModelBlaster (sibling clone) | `/scratch2/agustin/ModelBlaster` | `feat/k1-xpurt` | `a0310c6` | `origin/feat/k1-xpurt` | **0** | 0 |

XPU-RT is **0 behind / 74 ahead** of `origin/dev` (`587f96f`), relative to the
last fetch on 2026-08-26 21:28. Whether upstream has moved since is unknown
without fetching.

## Submodule pointers recorded in XPU-RT HEAD

```
ModelBlaster         163fd0202b0d2ad40d38fb494bf237da107655d6
merlin               e32529a88ca39414c31bb3636e7108f24ab12420
zephyr-chipyard-sw   e9a969412390ae0dc8ba3864696b8958124af522
```
and inside zephyr, the nested pointer `modelblaster → 650558ae` (uninitialized).

## Canonical ModelBlaster checkout

**`XPU-RT/ModelBlaster` at `feat/k1-xpurt` is canonical.** It is a strict
descendant of every other candidate, so there is no divergence to reconcile:

| other pointer | behind `163fd02` by |
|---|---|
| nested `zephyr/modelblaster` `79328776` | 237 |
| `origin/main` `dbbdcf0a` | 152 |
| sibling clone `a0310c6` | 8 |
| `3da8192` | 11 |

The runbook's commands are repo-relative (`ModelBlaster/scripts/run_model_k1.sh`),
i.e. they use the submodule. Two scripts still hard-code the *sibling* path and
should be repointed: `benchmarks/freshness_eval/run.py:457` and
`scripts/export_profile_db_to_results_csv.py:307`.

## Work rescued this session

Three bodies of work existed on no remote and were recoverable only from this
disk. All three are now durable.

**1. ModelBlaster pre-squash history — 133 commits.**
`feat/benchmark-harness-pre-squash` @ `5fed203` in the sibling clone, no
upstream. Its merge-base with the pushed `feat/benchmark-harness` is `79328776`,
and `git diff 5fed203 a7636e7` is 7232 files / 1,048,868 deletions — so "just a
pre-squash backup" was not a safe assumption. Pushed to
`ucb-bar/ModelBlaster` as `feat/benchmark-harness-pre-squash`. Off-remote count
is now 0.

**2. zephyr-chipyard-sw recalibration data — 3 commits.**
`418136f`, `03ee2e2`, `de22a1d` (bit-exact FireSim profile-DB recalibration),
on a branch with no upstream. First merged `origin/dev` into the branch — 21
behind, and **zero file-path overlap** (ours touches only
`gen/profile/sweep_v8/**`, theirs only `samples/executorch|tacit/**` and the
`modelblaster` gitlink), so the merge was mechanical → `b6e359c`. A direct push
to `ucb-bar/zephyr-chipyard-sw` is **denied** (403, no write access), so the
branch was pushed to a fork: `copparihollmann/zephyr-chipyard-sw`, remote
`fork`.

**The parent gitlink was deliberately NOT bumped to `b6e359c`.** A fresh XPU-RT
clone resolves submodules against the URL in `.gitmodules`, which is
`ucb-bar/zephyr-chipyard-sw` — and that remote does not have `b6e359c`, so
`git submodule update` would fail. The recorded `e9a9694` **is** on ucb-bar's
`dev`, so leaving it there keeps fresh checkouts working. The rescued data lives
in two places instead: the fork branch above, and the tracked 28 KB bundle at
`artifacts/repo_reconciliation/zephyr-patches/zephyr-3commits.bundle` (verified
to restore SHA `418136fe` and tree `a62bdd21`; its prerequisite `ae5d736` is on
ucb-bar's `dev`, so it applies to any fresh clone). Bumping the gitlink is
correct only once the branch reaches ucb-bar.

**3. merlin shard-reservation work — ~305 uncommitted lines.**
`scheduler_runner.cc`, `dispatch_types.h`, `dispatch_graph.h` carried the entire
runtime half of baseline B4 in the working tree only. Committed as `e32529a`
and pushed; the XPU-RT gitlink now records it.

## Safety artifacts

Taken before any history operation, and still on disk:

| bundle | bytes |
|---|---|
| `artifacts/repo_reconciliation/zephyr-chipyard-sw-20260826-203857.bundle` | 755,053,715 |
| `artifacts/repo_reconciliation/xpurt-20260826-212844.bundle` | 184,644,388 |
| `artifacts/repo_reconciliation/modelblaster-20260826-214542.bundle` | 18,661,235 |

These three are gitignored (`.gitignore:29`) because GitHub rejects files that
size; they are local safety copies, not durable storage. The 28 KB
`zephyr-patches/` bundle is deliberately **not** ignored — that exemption is
documented at `.gitignore:23-28` — because it is the only committed copy of
commits that are not on ucb-bar.

Also present: annotated tag `premerge-k1-20260826` → `e70a4b4a`, with branch
`feat/freshness-validity-eval` still at `e70a4b4` matching its remote. No
stashes exist in any repo.

## Remaining open items

1. **Get zephyr's branch onto `ucb-bar`** (needs write access or an accepted PR
   from the fork), then bump the XPU-RT gitlink from `e9a9694` to the merged SHA.
2. **Repoint the two hard-coded sibling paths** named above at the submodule.
3. **Initialize and pin the nested `zephyr/modelblaster`** pointer, currently
   `650558ae` and uninitialized. Note its URL is SSH
   (`git@github.com:ucb-bar/ModelBlaster.git`) while everything else is HTTPS,
   so `git submodule update --init` there needs an SSH agent; and no submodule
   declares `branch =`, so `--remote` would default to `master`/`HEAD` and must
   not be used.
4. **Verify the Zephyr build against the canonical ModelBlaster SHA.** Never
   done. `dbbdcf0a` removed ~171k lines and moved KernelBlaster to a submodule,
   so this is the real risk in the pointer story, not the SHA arithmetic.

## Credential scan

`git diff 587f96f..f042202` (added lines only) scanned for `password|passwd|
api_key|secret|token|BEGIN … PRIVATE KEY|ssh-rsa|AKIA|xox|ghp_|github_pat_|
Bearer|sshpass|scheme://user:pass@`. **No credential material found.** Hits are
LLM token *counts* and prose about FireSim's `NOPASSWD` sudo. No tracked `.env`,
`.netrc`, `.pem`, `.key`, `id_rsa`, or `id_ed25519` in any repo.

One item worth naming, not a leak:
`docs/k1_modelblaster_xpurt_closed_loop.md:26` publishes the on-disk *path* of a
private key (`/scratch2/agustin/DIMA_SLICE`). The key itself is not tracked
anywhere. Replacing the path with `~/.ssh/<your-key>` costs nothing and removes
a reader's map to it.

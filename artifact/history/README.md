# history — the exact source, offline

**Needs no hardware.**

Neither branch has been pushed: this repository's `feat/codesign-loop-hardening` and the
ModelBlaster submodule's `feat/split-linear-along-m` both carry commits their GitHub remotes do not. A plain `git clone` of either GitHub remote therefore cannot reach
the code the results were produced with. Two git bundles carry it.

```
xpurt.bundle          12.3 GiB   XPU-RT         feat/codesign-loop-hardening @ 08c6ebd8
modelblaster.bundle     20 MiB   ModelBlaster   feat/split-linear-along-m    @ 2b11d038
```

The XPU-RT bundle is much larger than the tree it checks out (16.3 GB of tracked files, which the
repository's own pack stores in 3.6 GiB). `git bundle` writes its pack in one pass rather than with
the full delta search a repacked repository has had, so the bundle trades size for the time it takes
to cut. Re-cutting it with `-c pack.window=50 -c pack.depth=250` gets closer to the repository's own
ratio and takes far longer; the size is not a defect in what the bundle contains, and
`git bundle verify` reports a complete history for it.

**Re-cut both after any commit that results depend on.** A bundle is a snapshot, and the branch has
been rewritten once already (the 369-to-8 squash), so an older bundle's tip can stop being an
ancestor of the branch entirely — at which point a reviewer cloning it gets a tree without the
results the page describes. `MANIFEST.sha256` is what says which tip is in there.

`MANIFEST.sha256` holds each bundle's sha256, its byte size and the full commit ids, and is the
record to read rather than this paragraph — the branch moves on, the bundle does not. That file is
**tracked**; the bundles themselves are not (they are large and regenerable, the same convention
`results/codesign_feedback/archive_v3/` uses for its tars).

## Clone from them

```bash
git clone -b feat/codesign-loop-hardening artifact/history/xpurt.bundle        xpurt
git clone -b feat/split-linear-along-m    artifact/history/modelblaster.bundle modelblaster
```

**Pass `-b`.** `git clone -b <branch>` checks out the named branch whatever refs the bundle
carries; `git bundle list-heads <bundle>` lists them. A bundle without a `HEAD` ref makes a bare
`git clone` fail with *"remote HEAD refers to nonexistent ref, unable to checkout"* and leave an
empty working tree. Verified 2026-09-28: with `-b`, the
ModelBlaster clone lands on `2b11d038`, the commit the superproject pins, with 1739 files.

The clone's `origin` is the bundle file; point it at the real remote afterwards if you want to
fetch newer work:

```bash
git -C xpurt        remote set-url origin https://github.com/ucb-bar/XPU-RT.git
git -C modelblaster remote set-url origin https://github.com/ucb-bar/ModelBlaster.git
```

To put the submodule inside the superproject clone instead of beside it:

```bash
git -C xpurt submodule init
git -C xpurt config submodule.ModelBlaster.url "$PWD/artifact/history/modelblaster.bundle"
git -C xpurt submodule update
```

`submodule update` checks out the pinned commit by sha, so it does not need `-b` and is the more
reliable of the two routes. This is what fills `ModelBlaster/` for a reviewer: the pin is correct
but the branch is unpushed, so a clone from GitHub alone leaves that directory empty and one test
under `tests/` cannot collect.

## Check them

```bash
cd artifact/history
awk '$1 ~ /^[0-9a-f]{64}$/ {print $1"  "$3}' MANIFEST.sha256 | sha256sum -c -
git bundle verify xpurt.bundle          # "The bundle records a complete history."
git bundle verify modelblaster.bundle
```

## Regenerate them

The bundles are cut from the working repository, so they can be rebuilt at any HEAD:

```bash
git -C <repo> bundle create artifact/history/xpurt.bundle HEAD feat/codesign-loop-hardening
git -C <repo>/ModelBlaster bundle create ../artifact/history/modelblaster.bundle HEAD feat/split-linear-along-m
sha256sum artifact/history/*.bundle      # then update MANIFEST.sha256
```

`HEAD` is named explicitly on purpose: without it the bundle has no `HEAD` ref and `git clone`
refuses with `ambiguous argument 'HEAD'`.

`git bundle list-heads <bundle>` prints what a bundle actually contains. The ids in
`MANIFEST.sha256` are the branch tips the bundles were cut at, which are ancestors of the commit
that added the manifest: a bundle cannot contain the commit that records its own hash.

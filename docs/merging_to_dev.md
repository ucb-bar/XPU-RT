# Landing this branch on `dev`

All of `dev` is merged. `HEAD..dev = 0` and `HEAD..origin/dev = 0`, and both
are ancestors of HEAD, so the push is a **fast-forward** — no merge commit:

```bash
git push origin HEAD:dev                      # XPU-RT
git -C ModelBlaster push origin HEAD:main     # ModelBlaster's integration branch
```

Do the ModelBlaster push first. XPU-RT records a gitlink; pushing a parent that
points at a commit no one can fetch is the one ordering that breaks a clone.

## One thing is blocked, and it is not ours to unblock

`zephyr-chipyard-sw` needs a commit that only exists on a personal fork.

The submodule pointer and the checkout had genuinely diverged — 7 commits
against 4, neither an ancestor. They merge cleanly (disjoint paths: ours
touched the `modelblaster` gitlink, theirs added FireSim profile CSVs), and the
merge is on
`copparihollmann/zephyr-chipyard-sw @ feat/firesim-bitexact-profile-recalibration`.
It carries the bit-exact FireSim profile-DB recalibration for dronet and
yolov8_nano hetero cycles — scheduler input data, not incidental — plus the
`modelblaster` bump described below.

Pushing it to `ucb-bar/zephyr-chipyard-sw` is refused with a 403. So XPU-RT's
zephyr pointer **stays at `b76ad31`** (ucb-bar `dev`'s tip). Recording a gitlink
only a fork can resolve would break `git submodule update --init` for everyone,
which is precisely the class of breakage this branch spent its time removing.

**Someone with write access needs to land that zephyr branch.** Until then the
two ModelBlaster checkouts cannot converge — see below.

## Why there are two ModelBlaster checkouts, and why they must match

ModelBlaster is reachable twice: XPU-RT's own top-level submodule (Flow B, the
K1 board) and `zephyr-chipyard-sw/modelblaster` (Flow A, spike/firesim). Dima's
`b66dd8c` removed the top-level one as redundant; this branch restores it
deliberately, and that decision stands.

What the duplication costs is worth stating plainly: they had drifted **214
commits against 77**, which means Flow A and the K1 flow were compiling
different kernels from the same op names, with nothing anywhere to say so. An
uninitialised submodule is an empty directory rather than an error, so it goes
unnoticed.

They are now merged onto one commit. Both pointers should name it:

* XPU-RT `ModelBlaster` → done, on this branch.
* `zephyr-chipyard-sw/modelblaster` → done on the fork, **blocked by the 403.**

When you bump one, bump the other.

## The license

`mosek.lic` is reachable from HEAD through the second parent of the merge that
took local `dev`'s CompGen snapshot. It is already on four `origin` branches
including this one, so this push adds no new exposure — but it makes later
removal a rewrite of five refs rather than four.

Three things in the tree already said it should not be committed: `.gitignore`
carries `*.lic` with "machine-local, never commit",
`benchmarks/freshness_eval/README.md` says it "must be present at the repo root
(gitignored)", and nothing reads it by path.

Treat it as compromised and re-issue it. That is its owner's call, not
something to fix by rewriting published branches.

## What the push carries

* merlin retired — the submodule, the IREE/VMFB runtime, and nine QRB5165
  scripts that could not run without it. Flow B is the K1 board now.
  `qnn_scheduler/` and `qrb5165_costs.json` stay: measured data outlives the
  compiler that fed it.
* the cross toolchain moved out of `merlin/build_tools/` into
  `tools/riscv-tools-spacemit/`, so nothing depends on merlin being on disk.
* all five compiler↔scheduler verbs have complete chains, `shard` included.
* both feedback channels reach ModelBlaster — batch through
  `run_xpurt_schedule.py --emit-feedback`, streaming through the walker's
  per-dispatch JSON lines.
* per-dispatch implementation choice is honoured by the binary rather than
  silently ignored.

## Before pushing

```bash
.venv/bin/python -m pytest xpu-rt/tests tests -q
cd ModelBlaster && python -m pytest tests pipeline/tests -q   # needs CROSS
eval "$(scripts/setup_spacemit_toolchain.sh)"
```

And run Flow A at least once from the merged ModelBlaster commit —
`scripts/repro_workload.sh` plus the four `networks_*_spike.json` specs. A green
K1 suite is not evidence that Flow A survived.

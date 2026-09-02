# Board tools

Four scripts that talk to a SpaceMiT K1 (BananaPi) over ssh. Everything they
deploy is ModelBlaster-generated C — there is no compiler, runtime library or
`.vmfb` in this directory any more.

## What used to be here

This tree was the merlin/IREE flow: `build_runtime.sh` linking against
`libxpurt_standalone.a`, `xpurt_scheduler_runner.c` driving `.vmfb` modules,
`compile_all_models.sh` invoking merlin, `profile_k1.py` benchmarking through
`iree-benchmark-module`. All of it is retired along with the merlin submodule,
because every kernel that runs on this board today comes out of ModelBlaster's
curated tree — a number measured against IREE-compiled kernels is a number for
code nobody runs.

Two things survived the move rather than being deleted:

* the **profile schema**. `results.csv` is still IREE-shaped, because that is
  what `xpu-rt/profile_loader.py` and `compile_advice.load_profiles_csv` read.
  ModelBlaster's `pipeline/profile_writer.py` writes it now. Old
  `profile.jsonl` artifacts still parse through `load_profiles`.
* the **cross toolchain**, which only ever lived inside merlin. It is now
  `scripts/setup_spacemit_toolchain.sh`, fetching to
  `tools/riscv-tools-spacemit/`.

## The scripts

| script | what it does |
|---|---|
| `scripts/deploy_k1.sh` | build, stage and run one model on the board |
| `scripts/verify_ime_build.sh` | check an IME build assembles and stays on cluster 0 |
| `scripts/k1_contention_mb.py` | how much co-runners slow a dispatch down |
| `scripts/k1_cost_by_pred.py` | what it costs to read what the previous dispatch wrote, from elsewhere |

The last two are measurements with their own write-ups —
`docs/K1/k1_contention.md` and `docs/K1/k1_cost_by_pred.md` — and both are worth
reading before quoting either number, because one of them is a **null result**
and the other is a model fitted to three measured classes rather than 64
independent measurements.

Operationally, [`docs/K1/k1_board.md`](../docs/K1/k1_board.md) is the runbook --
commands, timings, and the failure table. This page is what lives here.

## Before any of them

```bash
eval "$(scripts/setup_spacemit_toolchain.sh)"     # exports CROSS
```

Not optional. GCC 13.2 — what `CROSS` defaults to via chipyard's riscv-tools —
reorders the RVV `vsetvl` intrinsics so a widening instruction runs under the
narrow vtype, and the binary SIGILLs on the board with no stdout at all. The
script refuses anything below 14.

GCC 14.3 has its own trap in the opposite direction: it substitutes a wrong AVL
on a *chained* `vsetvl`, which is silent and wrong rather than loud. Pass the
element count to every width, and run
`ModelBlaster/scripts/check_rvv_avl.py` — that is what it is for.

## The rest of the loop

Building, profiling and scheduling live outside this directory:
`ModelBlaster/scripts/run_model_k1.sh` (with `PROFILE_OUT_ROOT` to emit a
profile), `scripts/run_xpurt_schedule.py`, and `docs/Feature/the_loop.md` for how they
fit together.

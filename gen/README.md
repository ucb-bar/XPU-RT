# gen/ — generated per-dispatch profiles

Per-dispatch cost tables the scheduler reads (`results.csv`, one row per dispatch per topology tag).

* `profile/<target>/…` — FireSim and spike targets (`gemmini_q31`, `V256D128_rvv`, `HTA`, `scalar`, `RVV`).
  A few entries are symlinks into the `zephyr-chipyard-sw` submodule and resolve once it is initialised.
* `profile_mb/` — the SpacemiT K1 profiles built with ModelBlaster (`rvv_x60`, `ime_x60`).


Variants of the K1 tables, each with its own README:

* `mb/` — ModelBlaster build outputs for the K1: dispatch graphs (`vmfb/`), IR, hybrid profiles;
  `mb/profile` links to `profile_mb/`.
* `mb_cal/` — YOLO's rows calibrated to the board, every other model linked to `mb/`.
* `mb_shard/`, `mb_shard_nav/` — per-width shard tables for YOLO, and for YOLO and nav.
* `mb_force/` — the matrix engine forced on, costed from the forced build's own dispatches.
* `clk25/` — the same cycles at a 25 MHz clock, the clock-invariance control.

Specs select a tree with `hardware.profile.gen_root`. Recorded specs and schedules name these trees
by their earlier top-level names (`gen_mb`, `gen_mb_shard`, `gen25`, ...); `xpu-rt/profile_roots.py`
maps each of those names to its directory here.

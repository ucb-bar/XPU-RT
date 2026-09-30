# 02 — the executed schedules, the codegen contract, the feasibility check

**Needs no hardware.** Everything here is host-side and costs no board time.

An XPU-RT arm is a *schedule table*: a JSON file naming, per dispatch, a start time, a duration and
a hardware target. The board's walker executes it literally. Three things therefore have to hold
before a board run means anything, and all three are checkable from a clean checkout.

## The tables the current figures draw

Each figure's panel I records its Gantt rows, and each Gantt sidecar names the schedule the XPU-RT
row executed (`xpu_schedule`):

| figure | XPU-RT row's table | Gantt sidecar |
|---|---|---|
| `warehouse_showdown_paper_r36` (the 36 Hz rate-sweep form) | `schedules/fig_w2pg36_greedy_clamped.json` | `results/codesign_feedback/refined/rate36/measured_gantt_r36_xpu_metrics.json` |
| `showdown_45hz_pinned_vs_rosdefault_s1003{,_ladder,_allcores,_allcores_merged,_c14}` | `schedules/fig_a_cpsat_hard_clamped.json` | `schedules/measured_gantt_v3_xpu_metrics.json` |

The ROS 2 rows have no table: a ROS 2 deployment is a set of launcher flags, not a schedule, which
is the whole point of the comparison.

`schedules/` is covered by a blanket ignore; the tables a documented check actually opens are
force-added so they are in the repository. 88 schedule files are tracked today.

## 1. Can the walker execute it? — `scripts/check_schedule_feasibility.py`

The solvers enforce no-overlap as a constraint, so a schedule straight out of one satisfies it by
construction. The schedules that do not come straight out of one are the reason this check exists:
MOSEK's per-network decomposition stitches solutions in ordinary Python, `combine_solved_windows`
concatenates per-window solutions and pushes back afterwards, and any hand-edited or hot-swapped
table is outside every constraint set. A double-booked core does not fail loudly on the board — the
second dispatch simply waits, and the run comes out slower than predicted, which in a results table
is indistinguishable from an optimistic profile or from contention.

```bash
.venv/bin/python scripts/check_schedule_feasibility.py --schedule schedules/fig_w2pg36_greedy_clamped.json
```

Exit 0 feasible, 1 infeasible, 2 could not be established.

## 2. Can the compiler build it? — `xpu-rt/codegen_contract.py`

`ModelBlaster/cores/spacemit_k1.json` says what the hardware can *execute*;
`ModelBlaster/cores/codegen_contract.json` says what ModelBlaster can *generate code for*. The
scheduler's option space was wider than the compiler's: `shard` mode lets every periodic instance of
a dispatch pick its own core block, and for a convolution that is unbuildable, because the packed
weight array is materialised per shard while generating the skeleton. The contract is data both
sides read.

```bash
.venv/bin/python xpu-rt/codegen_contract.py schedules/fig_a_cpsat_hard_clamped.json
```

`uniform_width_dispatch_ids()` asks the same question *before* solving, so the search cannot produce
a violating schedule; `violations()` asks after. Exit 1 on a refusing violation.

## 3. Is the file on disk the table the board ran? — `scripts/executed_tables.py`

A schedule JSON carries the dispatch table and a metadata block, and the metadata can be annotated
after a run — which changes the file's hash but not the table that executed. The ledger
(`results/codesign_feedback/executed_tables.json`) therefore records, per board manifest, the
schedule path, the manifest's `schedule_sha256` prefix, the sha256 of the canonical dispatch block,
and the annotating commit. 81 entries today.

```bash
.venv/bin/python scripts/executed_tables.py            # check every entry against the files on disk
.venv/bin/python scripts/executed_tables.py --write    # rebuild from the manifests and git history
```

`scripts/verify_showdown_figure.py` calls `check()` for every Gantt row it verifies, so a figure
cannot draw a table the board did not run.

## Both gates over the drawn tables, in one command

```bash
artifact/02_schedules/check_schedules.sh                       # the two tables above
artifact/02_schedules/check_schedules.sh <schedule.json> ...   # any others
```

A thin driver: it calls the two scripts of §1 and §2 and nothing else. Exit non-zero if either
refuses. Run here, both tables come back `FEASIBLE` and `no contract violation the schedule can
show`. The ledger of §3 is checked by `artifact/verify_no_hardware.sh`.

## Solving a table in the first place

`docs/Artifact/REPRODUCE.md` §1 step 2 is the one command
(`scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json --solver … --profiled`,
with `export XPURT_CPSAT_WORKERS=0`); `docs/Evaluation/figure_runbook.md` §3b item 2 names the specs and the
certificate and stage-2 drivers (`scripts/solve_certificates.sh`, `scripts/solve_stage2_hard.sh`);
`docs/Feature/solvers.md` and `docs/Feature/scheduler_solver_study.md` are the solver study itself. The tables a
figure draws are committed, so none of that is on the path to reproducing a figure.

## One table that does not pass, on purpose

`schedules/fig_w2pg36ime_greedy_clamped.json` is the IME arm: 1492 of its 4768 dispatches are placed
on the matrix engine. Run through §1 with the default `--gen-root gen/mb` it reports

```
INFEASIBLE: 1492 dispatch(es) request an implementation that has no kernel for their op;
            the walker stops at the first one with a FATAL and the run produces nothing.
  yolov8_nano_64x960_dispatch_12: ... has no ime kernel at any profiled width;
                                  its ime_x60 row reports 'curated[rvv]/rvv_oc_blocked_bn_silu_epilogue'
```

— the check reads the per-width profile tables under `--gen-root`, and in `gen/mb` those dispatches'
`ime_x60` rows name the RVV kernel. `docs/Artifact/artifact_checklist.md` §5 records where that arm stands
(measured on the board, control cadence 100.0 Hz, no flights flown against it, drawn in no figure)
and the state of the ModelBlaster submodule pointer; `docs/K1/ime_kernel_reproduction.md` is the kernel
itself, and [`../history/`](../history/) carries the submodule branch.

# K3 timing loop

One command that makes the planner's timings match the SpacemiT K3 board:

    plan -> build -> run on the board -> check -> measure -> compare -> fix the timings -> plan again

It stops when every model's frame takes as long on the board as the plan says (within 3% by default) and no
frame is late that the plan puts on time, or when the round limit is reached.

- [What it does](#what-it-does) · [The command](#the-command) · [What it needs](#what-it-needs) · [Inputs and outputs](#inputs-and-outputs)
- [The two gap rules](#the-two-gap-rules) · [The kept table and its drift check](#the-kept-table-and-its-drift-check) · [Checks](#checks)
- [What was measured with it](#what-was-measured-with-it) · [Limits](#limits) · [Files](#files) · [Tests](#tests)

## What it does

The planner (this repository) plans from a profile tree: one measured time per layer and option (core kind,
kernel type, number of cores), each measured with the layer running alone. On the board a frame takes longer
than the plan says, because a layer in a real schedule does not run exactly as long as it did alone, and
because starting a layer costs time the tree does not hold: waking the thread that runs it, most of all when
that thread is on another core.

The loop measures both on the board and feeds them back. Each round it

1. builds a corrected copy of the raw profile tree from every round measured so far (`fit.py`, `apply.py`,
   `tree_audit.py`); round 0 uses the raw tree as it is;
2. plans with that tree (`solve.sh`, `plan_summary.py`);
3. builds the plan with ModelBlaster's walker, runs it N times on the board at real-time priority 80 and
   copies the output back, md5-checked (`job.sh`);
4. checks every run and measures (`golden_c.sh`, `analyze.py`): a run counts only if its output is exact,
   every layer ran on the planned cores, and the trace matches the plan;
5. compares plan and board per model and decides: stop, or another round.

The raw tree is never modified. Every corrected tree is rebuilt from the raw tree and all rounds so far, so
a measurement is never forgotten when the planner moves a layer to another option.

## The command

Run from any folder; relative paths are relative to that folder.

```bash
python3 scripts/k3_timing_loop/timing_loop.py \
    --out <loop folder> --config <workload.json> --raw-tree <raw profile tree> \
    --modelblaster <ModelBlaster checkout with the stamps walker> \
    [--runs 20] [--max-rounds 5] [--delta 0.03] [--gap-rule layer|table] [--models yolov8_nano] [--host k3w]
```

Running the same command again continues after the last finished step. A finished board round is never
repeated; a round whose runs are on the board but not yet copied back is only fetched. `--start-from
<raw folder>[,<tree>] ...` adopts rounds that were run before. A higher `--max-rounds` on a later call
continues a loop that stopped at its limit. `timing_loop.py --help` lists every option.

## What it needs

- **A K3 board on a PREEMPT_RT kernel**, reachable as an ssh alias (default `k3w`) with key login, where the
  login may use real-time priority 80 (`ulimit -r` of 80 or more). The loop refuses to run otherwise.
- **A ModelBlaster checkout whose walker writes the two extra stamps** (`MB_XPURT_STAMPS=1`: `got_cores` and
  `kernel_end` in the trace). `--modelblaster`, default `zephyr-chipyard-sw/modelblaster` of this repository.
  Its `cores/codegen_contract.json` and `cores/spacemit_k3.json` are read as well.
- **The K3 builder image** (default name `k3-builder`, `K3TL_BUILDER_IMAGE`): a container image with the
  SpacemiT cross-compiler and ModelBlaster's Python packages whose entrypoint takes `K3_IP` and `K3_USER`
  and sets up ssh to the board with the key mounted at `/run/host-ssh/id_ed25519` (`builder.sh`). The loop
  reads the board's address and user from `ssh -G <alias>` and the key from `K3_SSH_KEY` (default
  `~/.ssh/id_ed25519`).
- **A raw profile tree** for the workload's models (`<tree>/profile/...`) and a **workload config** (the
  planner's networks json) whose `dispatch_deps_path` entries exist.
- Python with the planner's packages and numpy on the machine that runs the loop. Tried on macOS only.

Sharing the board: before every job the loop checks that nobody is logged in, the load is low, no other
board job runs, the notice `/tmp/K3_IN_USE` and the flock `/tmp/k3_board.lock` are free, and takes its own
lock file (`~/mb_k3rd/BOARD_LOCK`, owner = the login name unless `--owner` says otherwise). While its runs
are on the board it holds the flock and writes the notice. A busy board is waited for, never forced.
Nothing on the board is changed outside the loop's own folder (`~/mb_k3_timing_loop/`), the lock file and
the notice.

## Inputs and outputs

Everything is written under `--out` unless `--results-root`, `--raw-root`, `--tree-root` or `--solve-dir`
say otherwise.

| Path | What |
|---|---|
| `loop_state.json` | every round's tree, plan, raw folder, summary and verdict; rewritten after each step |
| `loop_report.md` | one table: per round and model planned and board frame time, late frames, verdict |
| `board_table.json` | gap rule `table`: the kept hand-off table (see below) |
| `solve/` | per round the workload config, the plan (`schedules/`) and the solve log |
| `trees/corr_<round>/` | the corrected profile tree of each round after round 0 |
| `raw/<round>_stamps/` | the board runs: per run the output, trace and placement; the build's table and IRs |
| `results/<round>/` | `summary.json`, `runs.csv`, `layers.csv`, `frames.csv.gz`, `handoff.json` |
| `results/corr_<round>/` | corrections, plan summary, the table used for that round |
| `logs/` | the log of every step |

Builds go to `<ModelBlaster>/build/k3_timing_loop_<name>_<round>/`.

## The two gap rules

A layer's **gap** is the time between "everything it waits for is done" and its start.

- **`layer`** (default). Each row that ran gets its own measured run time and its own measured gap, both
  written into the tree. Rows that never ran get the median of similar rows. Simple, but the gap is stored
  with a row although it depends on where the layer before it ran, so it is wrong as soon as the planner
  moves a neighbour.
- **`table`**. The tree holds the run-time corrections plus one flat start cost. The wake-ups are charged by
  the planner itself from a small table the loop measures and writes into the workload config:
  `scheduler.handoff_ms` (from core kind, to core kind: the cost when a layer's last input comes from
  another worker thread) and `scheduler.sender_ms` (the worker that has just woken another thread starts its
  own next layer late; `--no-sender` leaves it out). The planner then also avoids hand-offs that cost more
  than they save.

Run-time correction of a row, both rules: per round the 75th percentile of run / raw (clamped to 0.5 .. 4;
for rows under 20 us the difference instead of the ratio), then the median over every round the row ran in.

## The kept table and its drift check

With the `table` rule the loop keeps the table in `<out>/board_table.json` (`--save-table` writes a second
copy with a note of what it is valid for). After every round it compares the gaps the board showed with the
values the plan was charged. A value with 200 or more samples that is more than 2 us off is measured again
from all rounds for the next round (`--table-drift-us`, `--table-min-samples`; `--table-frozen` only
reports). `--handoff-table <file>` starts from a given table. A workload whose rounds lack one of the four
hand-off kinds needs a given table: the loop does not guess a value (yolov8_nano alone never hands off from
one A100 thread to another).

## Checks

The loop stops with the reason in `loop_state.json`, and fits nothing on that round, when a run's output is
not exact, a layer ran on other cores than planned, a run exits non-zero, the stamps are out of order, the
trace does not match the dispatch table or the plan, an md5 does not match, the corrected tree fails its
audit, or the plan's durations are not the tree's.

The output check uses a golden output made with the C kernels' rounding (`golden_c.sh`): ModelBlaster's
default golden rounds one operation differently and can be off by one step for yolov8_nano.

## What was measured with it

On a K3 with kernel 6.18.3-rt (PREEMPT_RT), SCHED_FIFO 80, 20 runs per round unless said. Measured with
these scripts before they moved here; the last row was run from this folder. Frame time planned / board
median (board / planned).

**yolov8_nano alone at 100 frames a second** (2026-11-19 and 2026-11-20)

| round | planned / board |
|---|---|
| raw tree | 3.72 / 5.48 ms (1.47) |
| `layer` rule, 3rd and 4th corrected round | 4.55 / 4.58 ms (1.006), 4.43 / 4.51 ms (1.020) |
| `table` rule | 4.48 / 4.53 ms (1.010) |
| `table` rule with the sender's delay | 4.57 / 4.61 ms (1.008) |
| from this folder, 2026-11-20, 10 runs each: raw tree, then one `table` round with a given table | 3.72 / 5.49 ms (1.47), then 4.63 / 4.49 ms (0.97) |

**mlp_control (100 Hz), dronet (30 Hz) and yolov8_nano at 170 Hz together** (2026-11-20)

| round | yolov8_nano | dronet | mlp_control | runs with a late frame |
|---|---|---|---|---|
| raw tree | 3.72 / 5.48 ms (1.47) | 0.39 / 1.13 ms | 0.015 / 1.66 ms | 20 of 20 |
| `layer` rule, 1st corrected round | 4.76 / 4.73 ms (0.99) | 0.65 / 0.67 ms | 0.048 / 0.090 ms | 0 of 20 |
| `table` rule | 4.59 / 4.63 ms (1.01) | 0.57 / 0.68 ms | 0.040 / 0.102 ms | 0 of 20 |
| `table` rule with the sender's delay, one more round | 4.65 / 4.88 ms (1.05) | 0.62 / 0.78 ms | 0.039 / 0.086 ms | 0 of 20 |

The table measured on that board: flat start cost 0.56 us; X100 to X100 11.1 us, X100 to A100 17.4 us,
A100 to X100 13.7 us, A100 to A100 17.4 us; sender's delay 5.4 us (X100) and 9.7 us (A100).

The raw runs, per-round results and notes of these experiments are kept outside this repository (the
group's K3 work notes: walker stamps, timing fix on the 3-model mix, hand-off table, 2026-11-19 and
2026-11-20).

## Limits

- **Small models.** dronet and mlp_control are not within 3% in any round. What they lose is waiting behind
  other models' layers (a core still in use, a layer that finished late), which no timing table covers.
- **The walker's release wait spins.** The walker holds a frame's first layer back until its planned start
  in a loop of `sched_yield()` calls at priority 80. While a worker thread spins like that on X100 core 1,
  2 or 3, layers that run on several cores start their helper threads more slowly: about 7 us more on X100
  and 20 to 30 us more on A100 cores, about 0.3 ms on a yolov8_nano frame. How long a worker spins depends
  on where the plan puts another model's first layer, so the corrections cannot learn it: the same row then
  looks slower in one plan than in the next. That is the 5% in the last mix round above (the worker spun for
  805 ms of the 1 s run, against 100 ms in the round before).
- **Tried with yolov8_nano, dronet and mlp_control only**, int8, on the backends `ime_k3_x100` and
  `ime_k3_a100` with the kernel choices fixed in `job.sh`.
- **One sample of the output is checked for yolov8_nano** (16 values and 4 sums, as the walker prints them);
  the two small models are checked on every value.
- The tree audit can re-run the profiling campaign's own audit on a corrected tree, but only when its script
  is given (`--profile-audit`); by default it checks structure and values.
- The bookkeeping files in a corrected tree (`t023_tree_entries.csv`, `t023_tree_report.json`) and the
  format ids inside the json files keep the names of the experiment they came from, so that recorded trees
  stay readable.

## Files

| File | Why the loop needs it |
|---|---|
| `timing_loop.py` | the loop: state, steps, stop rule, table drift check, report |
| `solve.sh`, `run_schedule_noplot.py` | plan one round with this checkout, without drawing the schedule picture (the plan is the same; drawing takes most of the time for a big workload) |
| `plan_summary.py` | what a plan predicts per model and what moved against the round before |
| `job.sh` | one board job: lock, build, smoke run, N runs in one login, md5-checked copy |
| `board.sh` | board free check and our lock, one login each |
| `builder.sh` | runs a command in the builder image |
| `fetch_resume.sh` | copies the run archive over a link that drops, resuming where it stopped |
| `golden_c.sh` | golden outputs with the C kernels' rounding, for the output check |
| `walker_run.py`, `loop_lib.py` | read a run: trace, placement, dispatch table, IRs; cut each layer into gap, wait for cores and run time; output and placement checks |
| `analyze.py` | checks every run of a round; frame times, late frames, per-layer table |
| `fit.py`, `apply.py`, `tree_audit.py` | corrections from all rounds; corrected tree; checks on that tree |
| `handoff_table.py` | measures the hand-off table and the sender's delay; plan against board per round |
| `test_timing_loop.py` | tests |

## Tests

```bash
python3 -m pytest scripts/k3_timing_loop
```

No board and no solver are needed: the stop rule, the drift checks, the state machine with stub steps
(resume, fail closed, adopted rounds, a raised round limit) and what the table rule writes into a round's
config. One more test replays recorded board rounds through the real scripts and must rebuild the recorded
corrected tree byte for byte; it is skipped unless the `K3TL_REPLAY_*` variables point at such data (see the
top of `test_timing_loop.py`).

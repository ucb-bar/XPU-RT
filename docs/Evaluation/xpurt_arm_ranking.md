# XPU-RT arm ranking — which implementation of the warehouse chain is best, and at what

Thirty solved-and-executed XPU-RT tables and thirteen further board points have been measured on
the SpaceMiT K1 for the warehouse camera→navigation→control chain. They are catalogued in
[`run_index.md`](../Artifact/run_index.md), which says what each one *is*. This page says which one is
**best**, on which axis, and whether an arm that no figure draws beats one that a figure does.

Every number here is a recorded constant in `scripts/measured_timing.py`, and
`scripts/measured_timing.py --verify` re-derives each one from the board trace it came from.
The one exception is §6, which is labelled as such.

## 0. How the ranking works

Three axes, in the order the paper cares about them.

1. **Does it meet the perception window.** A frame is late when its last YOLO dispatch ends more
   than the spec's `window_duration` after the frame was released. `--verify` counts this against a
   fixed 66.67 ms; for the three specs whose window is tighter than that (`wh_chain45_w2p` 44.44 ms,
   `wh_chain36_w2p` 55.56 ms and `wh_chain40_w2p` 50.00 ms) the count against the spec's own window
   is the same for all three, so no arm's standing depends on which of the two is used. An arm
   that misses the window ranks below every arm that meets it, whatever its median. The count is
   quoted as the registry records it; `--verify` checks the numerator against the frames it finds
   in the traces on disk, and for `p45freer` and `p45detr` that frame count is 51 rather than the
   66 and 63 the registry string carries.
2. **Camera→control chain**, the pooled median over three replicates (`per_frame_chain` in
   `make_measured_gantt_pair.py`), with the p95 where one is recorded.
3. **Control cadence**: the mean gap between control outputs, and the largest gap in the run. Every
   arm that meets its window commands at 99–102 Hz, so the tail is what separates them — a 29.88 ms
   maximum gap is three missed control ticks even when the mean is 10.00.

**Ranking across camera rates is meaningless and is not done here.** The camera period sets how
much perception work is in flight, so a 90 Hz arm and a 30 Hz arm are solving different problems:
`a90cpsat_hardr` (55.90 ms) is not "worse" than `p30freer` (26.75 ms). The tables below are
therefore grouped by the camera rate the *spec* was solved for, taken from the spec's
`networks.yolov8_nano_64x96.period`, and ranked only within a group. Comparing across groups is
only defensible for the two arms of a pair that differ in nothing else, which is what §2 does.

Bold marks the four arms the showdown figures draw: `p30freer`, `p36freer`, `p45freer`,
`acpsat_hardr`. "flights" is the number of un-quarantined episodes replaying that arm's cadence
trace (`flight_quarantine.flight_rows`), "figures" the number of figure sidecars that draw it.

## 1. The ranked table

### 30 Hz camera — 33.33 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `p30efullr` | yes (0/36) | 66.67 ms | 25.30 | 30.2 | 10.00 → 100 | 29.88 | given | 4 | yes | CP-SAT | — | — |
| 2 | `p30w4imer` | yes (0/36) | 66.67 ms | 26.75 | — | 10.00 → 100 | 10.02 | given | 4 | yes | greedy | — | — |
| 3 | `p30w4imecpr` | yes (0/36) | 66.67 ms | 26.75 | — | 10.00 → 100 | 10.34 | given | 4 | yes | CP-SAT | 48 | — |
| 4 | **`p30freer`** | yes (0/36) | 66.67 ms | 26.75 | 30.1 | 9.99 → 100 | 18.06 | solved | solver's choice | yes | CP-SAT | 192 | 11 |
| 5 | `p30w4forcer` | yes (0/36) | 66.67 ms | 30.08 | — | 10.00 → 100 | 10.03 | given | 4 | yes | CP-SAT | — | — |
| 6 | `fb30r1r` | yes (0/81) | 66.67 ms | 30.10 | — | 9.99 → 100 | 19.7 | solved | solver's choice | no | CP-SAT | 60 | — |
| 7 | `p30w4cpr` | yes (0/36) | 66.67 ms | 36.75 | — | 10.00 → 100 | 10.02 | given | 4 | no | CP-SAT | — | — |
| 8 | `p30w4r` | yes (0/36) | 66.67 ms | 36.75 | — | 10.00 → 100 | 10.03 | given | 4 | no | greedy | — | — |
| 9 | `p30imer` | yes (0/36) | 66.67 ms | 43.42 | — | 10.00 → 100 | 10.05 | given | 1 | yes | greedy | — | — |
| 10 | `a30cpsat_hardr` | yes (0/36) | 66.67 ms | 55.20 | — | 9.98 → 100 | 15.44 | solved | solver's choice | no | CP-SAT | 120 | 1 |
| 11 | `p30cpsat_hardr` | yes (0/36) | 66.67 ms | 60.08 | — | 10.00 → 100 | 10.08 | given | 1 | no | CP-SAT | — | — |
| 12 | `p30greedyr` | yes (0/36) | 66.67 ms | 60.09 | — | 10.00 → 100 | 10.03 | given | 1 | no | greedy | — | — |
| 13 | `a30greedyr` | yes (0/36) | 66.67 ms | 70.10 | — | 10.00 → 100 | 15.52 | solved | solver's choice | no | greedy | — | — |

### 36 Hz camera — 27.78 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **`p36freer`** | yes (0/42) | 66.67 ms | 25.82 | 34.3 | 9.84 → 102 | 19.69 | solved | solver's choice | yes | CP-SAT | 12 | 11 |
| 2 | `w2pg36r` | yes (0/96) | 55.56 ms | 30.10 | — | 9.98 → 100 | 17.23 | solved | solver's choice | no | greedy | 72 | 1 |

### 40 Hz camera — 25.00 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `w2pg40r` | yes (0/108) | 50.00 ms | 39.80 | — | 10.06 → 99 | 16.68 | solved | solver's choice | no | greedy | 60 | — |

### 45 Hz camera — 22.22 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **`p45freer`** | yes (0/66) | 66.67 ms | 27.46 | 37.2 | 9.92 → 101 | 20.19 | solved | solver's choice | yes | CP-SAT | 84 | 6 |
| 2 | `p45detr` | yes (0/63) | 66.67 ms | 30.07 | 43.6 | 9.83 → 102 | 24.27 | solved | solver's choice | yes | CP-SAT | — | — |
| 3 | `w2pgOCr` | yes (0/120) | 44.44 ms | 36.00 | 44.2 | 9.99 → 100 | 17.02 | solved | solver's choice | no | greedy | 84 | — |
| 4 | `sonlycpr` | yes (0/120) | 66.67 ms | 53.20 | 62.6 | 10.03 → 100 | 20.52 | solved | solver's choice | no | CP-SAT | — | — |
| 5 | **`acpsat_hardr`** | yes (0/120) | 66.67 ms | 56.80 | 72.3 | 10.00 → 100 | 19.36 | solved | solver's choice | no | CP-SAT | 4236 | 8 |
| 6 | `agreedyr` | **no** (120/120) | 66.67 ms | 748.00 | 1064.7 | 13.25 → 75 | 1036.96 | solved | solver's choice | no | greedy | 552 | — |

### 60 Hz camera — 16.67 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a60cpsat_hardr` | yes (0/72) | 66.67 ms | 58.50 | — | 10.06 → 99 | 15.3 | solved | solver's choice | no | CP-SAT | — | — |
| 2 | `a60greedyr` | **no** (72/72) | 66.67 ms | 583.60 | — | 17.48 → 57 | 667.9 | solved | solver's choice | no | greedy | — | — |

### 90 Hz camera — 11.11 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a90cpsat_hardr` | yes (0/108) | 66.67 ms | 55.90 | — | 10.00 → 100 | 15.04 | solved | solver's choice | no | CP-SAT | 1020 | — |
| 2 | `b5cpsat_hardr` | yes (0/108) | 66.67 ms | 58.40 | 68.8 | 9.95 → 101 | 16.35 | solved | solver's choice | no | CP-SAT | 240 | — |
| 3 | `b5greedyr` | **no** (108/108) | 66.67 ms | 150.30 | 172.1 | 11.56 → 87 | 47.88 | solved | solver's choice | no | greedy | 120 | — |
| 4 | `a90greedyr` | **no** (111/111) | 66.67 ms | 947.50 | — | 25.67 → 39 | 991.65 | solved | solver's choice | no | greedy | 840 | — |

### 120 Hz camera — 8.33 ms period

| # | arm | window met | perception window | chain ms | p95 | gap mean → Hz | gap max | placement | YOLO width | IME | solver | flights | figures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `a120hcpsat_hardr` | yes (0/36) | 66.67 ms | 59.90 | — | 10.00 → 100 | 15.37 | solved | solver's choice | no | CP-SAT | — | — |
| 2 | `a120hgreedyr` | **no** (36/36) | 66.67 ms | 566.30 | — | 35.77 → 28 | 513.81 | solved | solver's choice | no | greedy | — | — |

### 1b. The further board points

`XPURT_POINTS` holds thirteen arms measured the same way whose placement was written by hand
(`scripts/build_best_schedule.py`) rather than solved. None carries a `frames_late` count; each
carries an on-time flag derived from the release-to-start lag, and all but `best75alt2` are on
time. They are ranked here in the same groups so the solved arms can be read against them.

| camera Hz | point | chain ms | gap mean → Hz | gap max | on time | note |
|---|---|---|---|---|---|---|
| 25 | `rich25p4` | 29.6 | 10.00 → 100 | — | yes | + ffn_block 10 Hz, dronet 30 Hz |
| 36 | `w2pg36ime` | **22.3** | 9.94 → 101 | 21.45 | yes | conv on the IME; the fastest chain measured at any rate |
| 36 | `w2pg36base` | 30.1 | 9.98 → 100 | 19.58 | yes | the same source tree, every conv on RVV |
| 45 | `best45alt2c200` | 37.9 | 5.00 → 200 | — | yes | control asked for 200 Hz |
| 45 | `best45alt2long` | 40.1 | 10.00 → 100 | 10.03 | yes | five seconds, 996 gaps |
| 45 | `rich45alt2` | 40.8 | 9.98 → 100 | 12.12 | yes | heavier stack |
| 45 | `rich45alt2ime` | 40.9 | 9.97 → 100 | 12.15 | yes | heavier stack, ffn_block's linears on the IME |
| 45 | `best45alt2` | 45.6 | 9.97 → 100 | 15.14 | yes | two background hogs (`HOGS=2`) |
| 60 | `best60alt2` | 39.6 | 10.00 → 100 | — | yes | width 2, two frames in flight |
| 75 | `best75alt2` | 119.6 | 10.00 → 100 | — | **no** | the width-2 layout's ceiling is ~64 Hz |
| 90 | `best90alt1` | 57.4 | 10.00 → 100 | — | yes | six frames in flight, width 1 |
| 90 | `cam2alt1` | 71.8 | 10.00 → 100 | 10.03 | yes | two 45 Hz cameras |
| 90 | `cam2rich` | 74.3 | 9.96 → 100 | 12.23 | yes | two cameras and the heavier stack |

At 45 Hz every hand-placed point is slower than the solved `p45freer` (27.46 ms). At 36 Hz the
hand-placed pair is the exception and §3 returns to it.

## 2. The levers, quantified

Each row is a pair of executed arms that differ in one entry of the spec. Where a second entry
also differs it is named, because the per-width cost table (`gen/mb_shard`) and the per-width
navigation table (`gen/mb_shard_nav`) are what make a wide placement costable at all: a solve that
may choose width 4 cannot be priced from the single-width `gen/mb` tree.

### 2a. Solver — CP-SAT against greedy, same spec, same tag

| spec | camera Hz | greedy | CP-SAT | Δ chain | greedy window | CP-SAT window |
|---|---|---|---|---|---|---|
| `wh_chain90_solve_500` | 90 | 947.50 | 55.90 | **−891.60 ms (−94.1 %)** | 111/111 late | 0/108 |
| `wh_chain45_solve` | 45 | 748.00 | 56.80 | −691.20 ms (−92.4 %) | 120/120 late | 0/120 |
| `wh_chain60_solve_500` | 60 | 583.60 | 58.50 | −525.10 ms (−90.0 %) | 72/72 late | 0/72 |
| `wh_chain120_solve_h200` | 120 | 566.30 | 59.90 | −506.40 ms (−89.4 %) | 36/36 late | 0/36 |
| `wh_chain90_rich_solve_500` | 90 | 150.30 | 58.40 | −91.90 ms (−61.1 %) | 108/108 late | 0/108 |
| `wh_chain30_solve_500` | 30 | 70.10 | 55.20 | −14.90 ms (−21.3 %) | 0/36 | 0/36 |
| `wh_chain30_part` | 30 | 60.09 | 60.08 | −0.01 ms | 0/36 | 0/36 |
| `wh_chain30_part4` | 30 | 36.75 | 36.75 | 0.00 ms | 0/36 | 0/36 |
| `wh_chain30_part4ime` | 30 | 26.75 | 26.75 | 0.00 ms | 0/36 | 0/36 |

**The solver is the largest lever by a wide margin, and it is bimodal.** Where the camera period is
short enough that perception has to be interleaved across harts, greedy list scheduling does not
find a feasible interleaving at all: every frame misses its window and the chain runs 2.6–17× longer.
Where the spec already names a cluster per network, there is nothing left to decide and the two
solvers land within 0.01 ms of each other. The 30 Hz unpartitioned row (−14.90 ms, both feasible)
is the only case where the solver is a tuning rather than a feasibility switch.

### 2b. Placement — given in the spec against left to the solver

| pair | camera Hz | placement given | placement solved | Δ chain | note |
|---|---|---|---|---|---|
| `p30efullr` vs `p30freer` | 30 | 25.30 | 26.75 | **−1.45 ms for the given placement** | both `gen/mb_shard_nav`, both `enable_impls`, same horizon |
| `p30w4imecpr` vs `p30freer` | 30 | 26.75 | 26.75 | **0.00 ms** | profile tree differs (`gen/mb_shard` / `gen/mb_shard_nav`) |

Naming a cluster per network buys nothing the solver does not find for itself: the unconstrained
solve reaches the same 26.75 ms, placing 808 of YOLO's dispatches four-wide on the P cluster and
taking 679 IME dispatches (`docs/K1/partitioned_schedule.md` §6). `p30efullr`'s 1.45 ms advantage is a
phase effect recorded as one — its navigation stage is 0.78 ms *longer*, and what moves is the wait
from navigation ending to the next 100 Hz control tick — and it is paid for in the control tail:
29.88 ms maximum gap against `p30freer`'s 18.06.

### 2c. Convolution on the IME against RVV

| pair | camera Hz | RVV | IME | Δ chain | what else differs |
|---|---|---|---|---|---|
| `p30imer` vs `p30greedyr` | 30 | 60.09 | 43.42 | **−16.67 ms (−27.7 %)** | profile tree `gen/mb` → `gen/mb_shard` |
| `p30w4imer` vs `p30w4r` | 30 | 36.75 | 26.75 | **−10.00 ms (−27.2 %)** | nothing — same spec but `enable_impls` |
| `p30w4imecpr` vs `p30w4cpr` | 30 | 36.75 | 26.75 | **−10.00 ms (−27.2 %)** | nothing — same spec but `enable_impls` |
| `w2pg36ime` vs `w2pg36base` | 36 | 30.1 | 22.3 | **−7.80 ms (−25.9 %)** | nothing — one source tree, built twice |
| `p30w4forcer` vs `p30w4imecpr` | 30 | — | 30.08 vs 26.75 | **+3.33 ms for blanket IME** | `MB_IME_FORCE=1`, costed from `gen/mb_force` |

The engine is worth 25–28 % of the chain wherever it is offered, and the two clean pairs
(`p30w4*`, `w2pg36*`) agree with the two that also move the profile tree. The last row is the
ablation that makes the cost model earn its place: forcing all 63 IME-capable dispatches per frame
onto the engine instead of the 46 the measured table calls wins costs 3.33 ms.

### 2d. Sharding width — one hart per YOLO frame against the whole P cluster

| pair | camera Hz | width 1 | width 4 | Δ chain | what else differs |
|---|---|---|---|---|---|
| `p30w4imer` vs `p30imer` | 30 | 43.42 | 26.75 | **−16.67 ms (−38.4 %)** | nothing — same tree, same `enable_impls` |
| `p30w4r` vs `p30greedyr` | 30 | 60.09 | 36.75 | −23.34 ms (−38.8 %) | profile tree `gen/mb` → `gen/mb_shard` |
| `sonlycpr` vs `acpsat_hardr` | 45 | 56.80 | 53.20 | −3.60 ms (−6.3 %) | the solver is shown per-width costs and shards; same window, same horizon |

Width is the largest lever among arms that already meet their window: **−16.67 ms, 38 % of the
chain**, from a single spec field. The 45 Hz row is the same lever seen through the solver's cost
model rather than through a pin — showing CP-SAT the measured per-width table lets it shard, and it
is worth 3.60 ms there because at a 22.22 ms period the perception window is the binding constraint
rather than the hart count.

### 2e. CP-SAT search workers — 8 against 1

| pair | camera Hz | workers | chain ms | p95 | gap max | window |
|---|---|---|---|---|---|---|
| `p45freer` | 45 | 8 | 27.46 | 37.2 | 20.19 | 0 late |
| `p45detr` | 45 | 1 | 30.07 | 43.6 | 24.27 | 0 late |

One worker costs **+2.61 ms (+9.5 %)** on the chain, 6.4 ms on the p95 and 4.1 ms on the control
tail. The specs are byte-equal apart from a comment (`wh_chain45_free` /
`wh_chain45_free_det1`), so this is the price of a table that is reproducible by recipe: with
several search workers the answer depends on thread interleaving, and only the single-worker solve
repeats. Both tables meet every window.

### 2f. Which lever buys the most

- **By magnitude: the solver**, −891.60 ms at a 90 Hz camera, and it is the only lever that turns
  "every frame late" into "no frame late". It is also worth 0.00 ms once the placement is pinned.
- **Among arms that already meet the window: the shard width**, −16.67 ms (−38 %), ahead of the IME
  (−10.00 ms, −27 %), the solver's worker count (−2.61 ms) and the placement (−1.45 ms, and in the
  other direction from the one a reader might expect).
- Stacked at 30 Hz, width and the engine together take an unconstrained CP-SAT chain from 55.20 ms
  (`a30cpsat_hardr`, single-width `gen/mb`, no implementations offered) to 26.75 ms (`p30freer`,
  per-width tables and the IME offered) — **−28.45 ms, −51.5 %, from the same solver on the same
  hardware**. Nothing was pinned in either.

## 3. Is anything better than what we draw?

| drawn arm | camera Hz | its chain | best alternative at the same rate | that arm's chain | verdict |
|---|---|---|---|---|---|
| `p30freer` | 30 | 26.75 | `p30efullr` | 25.30 | **beaten by 1.45 ms** on the chain; loses on the control tail (29.88 vs 18.06 ms max gap); never flown |
| `p36freer` | 36 | 25.82 | `w2pg36ime` | **22.3** | **beaten by 3.52 ms (13.6 %)**; comparable tail (21.45 vs 19.69); never flown |
| `p45freer` | 45 | 27.46 | `p45detr` | 30.07 | not beaten — `p45freer` is the fastest 45 Hz arm measured |
| `acpsat_hardr` | 45 | 56.80 | `p45freer` | 27.46 | **beaten by 29.34 ms (51.7 %)** — and by `p45detr`, `w2pgOCr` and `sonlycpr` as well |

Plainly:

- **`w2pg36ime` is the fastest camera→control chain measured on this workload at any camera rate,
  22.3 ms, and no figure draws it.** It is the IME half of the A/B pair in
  `docs/K1/ime_kernel_reproduction.md` §5, built to isolate the engine rather than to be a showdown
  arm, and it has no flight campaign: its cadence trace `xpu_w2pg36ime.csv` exists and is replayed
  by nothing. It beats the drawn 36 Hz arm `p36freer` on latency while matching it on cadence
  (9.94 ms mean gap against 9.84). It is an arm that is better on latency and was never flown.
- **`p30efullr` beats the drawn 30 Hz arm on latency and loses on the tail.** 25.30 ms against
  26.75, but a 29.88 ms largest control gap — three control ticks — against 18.06. Its cadence
  trace `xpu_p30efull.csv` exists and has never been flown either. On the axis the showdown
  figures actually exercise, which is command cadence under a fixed 100 Hz tick, it is the weaker
  arm; on camera→control it is the stronger.
- **`acpsat_hardr` is the slowest 45 Hz arm that meets its window.** Four measured arms beat it at
  the same camera rate, one of which (`p45freer`) is drawn alongside it. That is by construction:
  it is the configuration the Tier A warehouse showdown uses, and it is drawn because it is
  that configuration, not because it is the best 45 Hz table. `p45freer`'s spec carries the
  per-width tables and the IME implementations that `wh_chain45_solve` does not; neither spec pins
  a placement.
- **At 60, 90 and 120 Hz only one arm per rate meets its window**, so there is no alternative to
  compare and nothing to draw against. `b5cpsat_hardr` (58.40 ms, the heavier stack at 90 Hz) is
  within 2.5 ms of the plain 90 Hz arm while carrying two more networks.

Nothing outside the registry beats a drawn arm either; §6 gives the screen.

## 4. Reproduction

[`run_index.md`](../Artifact/run_index.md) §1c already gives, per arm, the table the board executed, whether
it is tracked, and which of the two board recipes produced it. What it leaves as `<spec>` is
filled in here, together with the four spec fields the levers in §2 turn and the trace glob a
check opens. The generic recipes those two forms expand to:

```bash
# solve, both forms
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/<spec>.json \
    --solver {cpsat|greedy_periodic} --max-periodic-iters 1 --use-profiled --board-calibration $CAL

# board, the stage-2 form (label <tag><solver>r<k>)
bash scripts/solve_stage2_hard.sh <spec> <tag> 3000 && bash scripts/board_stage2.sh <tag> <spec>

# board, the direct form (label <label>r<k>)
REPS=3 bash scripts/board_partitioned30.sh <table> <label> <gen-root>
```

`XPURT_CPSAT_WORKERS` is 8 in `solve_stage2_hard.sh` and 1 for `p45detr`
(`scripts/solve45_determinism.sh`). `board_partitioned30.sh` switches to the three-backend walker
(`BACKENDS=rvv_x60,rvv_x60,ime_x60`) by itself whenever the table carries an `impl: ime`
placement; `p30w4forcer` additionally runs under `MB_IME_FORCE=1`, which changes the binary, so a
table costed against the table-guided tree must not be run under it.

Several families have a driver of their own that wraps both halves, and four of them generate
their spec rather than reading a committed one:

| arm(s) | driver | note |
|---|---|---|
| `acpsat_hardr`, `agreedyr`, `a30*`, `a60*`, `a90*`, `b5*` | `scripts/chain_rates2.sh` (`run_pair <spec500> <tag>`) | solves the certificate at `_h200` first, then the half-second table |
| `a120hcpsat_hardr`, `a120hgreedyr` | `scripts/chain_rates_high.sh`, `scripts/chain_rates_high_h200.sh` | 200 ms table |
| `p30greedyr` … `p30w4forcer` | `scripts/board_partitioned30.sh <table> <label> gen/mb_shard`, per `docs/K1/partitioned_schedule.md` §4 | the verbatim invocation on that page is `p30w4imecp`'s |
| `p30freer` | `docs/Evaluation/showdown_cam30_solver_placed_reproduction.md` §3 | verbatim, `gen/mb_shard_nav` |
| `p36freer` | `docs/Evaluation/showdown_cam36_allcores_reproduction.md` | spec generated by `scripts/make_free_spec.py 36` |
| `p45freer` | `scripts/board_free45.sh` | waits on `solve45_free.log`, then `board_partitioned30.sh` |
| `p45detr` | `scripts/solve45_determinism.sh` then `scripts/board_det45.sh` | `XPURT_CPSAT_WORKERS=1` |
| `w2pg36r`, `w2pg40r` | `scripts/xpu_greedy_shard_at_rate.sh <hz>` | writes `wh_chain<hz>_w2p.json` from `wh_chain45_w2p.json`, YOLO window `max(2×period, 40 ms)`; solves with no `--board-calibration` |
| `fb30r1r` | `scripts/rate_feedback_loop.sh 30` | writes `wh_chain30_fb1.json` from `wh_chain30_solve.json` |
| `w2pg36base`, `w2pg36ime` | `docs/K1/ime_kernel_reproduction.md` §5 | one source tree, built twice; the IME arm runs under `BACKENDS=rvv_x60,rvv_x60,ime_x60` |
| the 13 points of §1b | `scripts/build_best_schedule.py` then `scripts/run_xpurt_long.sh <table> <label> <reps>` | each table's `metadata` block records the generator's own arguments |

Two labels have no driver left in the tree and their spec is named by
`scripts/measured_timing.py` rather than by a recorded command: `sonlycpr`
(`wh_chain45_shardonly`, solve logs under `results/codesign_feedback/solver_v2/`) and `w2pgOCr`
(`wh_chain45_w2p`; the `OC` suffix is produced by nothing in `scripts/`, only the per-run board
logs survive). Both executed tables are tracked and both traces are archived, so the numbers
re-derive; the command that produced the run does not.

| arm | camera Hz | spec | placement in spec | `machine_width` | `enable_impls` | profile tree | horizon | traces |
|---|---|---|---|---|---|---|---|---|
| `p30efullr` | 30 | `data/toplevel/wh_chain30_part4ime_efull.json` | `allowed_machines` set | 4 | true | `gen/mb_shard_nav` | 500 ms | `trace_p30efullr[1-3]_other_run1.csv` |
| `p30w4imer` | 30 | `data/toplevel/wh_chain30_part4ime.json` | `allowed_machines` set | 4 | true | `gen/mb_shard` | 500 ms | `trace_p30w4imer[1-3]_other_run1.csv` |
| `p30w4imecpr` | 30 | `data/toplevel/wh_chain30_part4ime.json` | `allowed_machines` set | 4 | true | `gen/mb_shard` | 500 ms | `trace_p30w4imecpr[1-3]_other_run1.csv` |
| `p30freer` | 30 | `data/toplevel/wh_chain30_free.json` | none | — | true | `gen/mb_shard_nav` | 500 ms | `trace_p30freer[1-3]_other_run1.csv` |
| `p30w4forcer` | 30 | `data/toplevel/wh_chain30_part4force.json` | `allowed_machines` set | 4 | true | `gen/mb_force` | 500 ms | `trace_p30w4forcer[1-3]_other_run1.csv` |
| `fb30r1r` | 30 | `data/toplevel/wh_chain30_fb1.json` | none | — | false | `gen/mb_shard` | 1000 ms | `trace_fb30r1r[1-3]_other_run1.csv` |
| `p30w4r` | 30 | `data/toplevel/wh_chain30_part4.json` | `allowed_machines` set | 4 | false | `gen/mb_shard` | 500 ms | `trace_p30w4r[1-3]_other_run1.csv` |
| `p30w4cpr` | 30 | `data/toplevel/wh_chain30_part4.json` | `allowed_machines` set | 4 | false | `gen/mb_shard` | 500 ms | `trace_p30w4cpr[1-3]_other_run1.csv` |
| `p30imer` | 30 | `data/toplevel/wh_chain30_partime.json` | `allowed_machines` set | — | true | `gen/mb_shard` | 500 ms | `trace_p30imer[1-3]_other_run1.csv` |
| `a30cpsat_hardr` | 30 | `data/toplevel/wh_chain30_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a30cpsat_hardr[1-3]_other_run1.csv` |
| `p30cpsat_hardr` | 30 | `data/toplevel/wh_chain30_part.json` | `allowed_machines` set | — | false | `gen/mb` | 500 ms | `trace_p30cpsat_hardr[1-3]_other_run1.csv` |
| `p30greedyr` | 30 | `data/toplevel/wh_chain30_part.json` | `allowed_machines` set | — | false | `gen/mb` | 500 ms | `trace_p30greedyr[1-3]_other_run1.csv` |
| `a30greedyr` | 30 | `data/toplevel/wh_chain30_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a30greedyr[1-3]_other_run1.csv` |
| `p36freer` | 36 | `data/toplevel/wh_chain36_free.json` | none | — | true | `gen/mb_shard_nav` | 500 ms | `trace_p36freer[1-3]_other_run1.csv` |
| `w2pg36r` | 36 | `data/toplevel/wh_chain36_w2p.json` | none | — | false | `gen/mb_shard` | 1000 ms | `trace_w2pg36r[1-3]_other_run1.csv` |
| `w2pg40r` | 40 | `data/toplevel/wh_chain40_w2p.json` | none | — | false | `gen/mb_shard` | 1000 ms | `trace_w2pg40r[1-3]_other_run1.csv` |
| `p45freer` | 45 | `data/toplevel/wh_chain45_free.json` | none | — | true | `gen/mb_shard_nav` | 500 ms | `trace_p45freer[1-3]_other_run1.csv` |
| `p45detr` | 45 | `data/toplevel/wh_chain45_free_det1.json` | none | — | true | `gen/mb_shard_nav` | 500 ms | `trace_p45detr[1-3]_other_run1.csv` |
| `w2pgOCr` | 45 | `data/toplevel/wh_chain45_w2p.json` | none | — | false | `gen/mb_shard` | 1000 ms | `trace_w2pgOCr[1-3]_other_run1.csv` |
| `sonlycpr` | 45 | `data/toplevel/wh_chain45_shardonly.json` | none | — | false | `gen/mb_shard` | 1000 ms | `trace_sonlycpr[1-3]_other_run1.csv` |
| `acpsat_hardr` | 45 | `data/toplevel/wh_chain45_solve.json` | none | — | false | `gen/mb` | 1000 ms | `trace_acpsat_hardr[1-3]_other_run1.csv` |
| `agreedyr` | 45 | `data/toplevel/wh_chain45_solve.json` | none | — | false | `gen/mb` | 1000 ms | `trace_agreedyr[1-3]_other_run1.csv` |
| `a60cpsat_hardr` | 60 | `data/toplevel/wh_chain60_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a60cpsat_hardr[1-3]_other_run1.csv` |
| `a60greedyr` | 60 | `data/toplevel/wh_chain60_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a60greedyr[1-3]_other_run1.csv` |
| `a90cpsat_hardr` | 90 | `data/toplevel/wh_chain90_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a90cpsat_hardr[1-3]_other_run1.csv` |
| `b5cpsat_hardr` | 90 | `data/toplevel/wh_chain90_rich_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_b5cpsat_hardr[1-3]_other_run1.csv` |
| `b5greedyr` | 90 | `data/toplevel/wh_chain90_rich_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_b5greedyr[1-3]_other_run1.csv` |
| `a90greedyr` | 90 | `data/toplevel/wh_chain90_solve_500.json` | none | — | false | `gen/mb` | 500 ms | `trace_a90greedyr[1-3]_other_run1.csv` |
| `a120hcpsat_hardr` | 120 | `data/toplevel/wh_chain120_solve_h200.json` | none | — | false | `gen/mb` | 200 ms | `trace_a120hcpsat_hardr[1-3]_other_run1.csv` |
| `a120hgreedyr` | 120 | `data/toplevel/wh_chain120_solve_h200.json` | none | — | false | `gen/mb` | 200 ms | `trace_a120hgreedyr[1-3]_other_run1.csv` |

The thirteen hand-placed points of §1b have no spec: `scripts/build_best_schedule.py` writes the
placement directly, recording its own arguments in the table's `metadata` block, and the board form
is `scripts/run_xpurt_long.sh <table> <label> <reps>`.

## 5. Trace coverage

`scripts/verify_board_traces.py` reports **438 tracked or archived and matching, 0 failed**. That
set is every trace a documented check opens: the three replicates of each of the 30 `SOLVER_ARMS`,
every run of each of the 13 `XPURT_POINTS`, the ROS 2 runs, and the trace each drawn Gantt row
names. Every arm ranked in §1 and §1b is inside it.

`results/codesign_feedback/xpurt_long/` holds **473** `trace_*.csv` files, so **353 are outside
that guarantee** — they belong to run labels that no registry names, so no check opens them and
nothing asserts they are archived. By family:

| family | traces | what they are |
|---|---|---|
| `fb{a90,a120h,a120,b5}r{0,1,2}{cpsat_hard,cpsat_soft,greedy,…}` | 234 | the three rounds of `scripts/hil_feedback_study.sh`; of the per-rate loop only `fb30r1r` is registered |
| `best45alt{1,2b,4,4b,4s}`, `best45p4`, `best60alt1`, `long{25,45}`, `tiled{25,45}`, `coupled` | 41 | earlier hand-placed layouts |
| `*greedyOC*`, `ash{cpsat_hard,greedy,greedyOC}`, `bgreedy`, `a120greedy` | 33 | re-runs of the greedy and shard-solve tags |
| `w2p{cp,scp,greedy,ime,ime2}`, `w2pg{38,38u,38p1,50}` | 27 | the 45 Hz two-period family beyond `w2pgOCr` |
| `fb{30r2,36r1,36r2,45r1,45r2}` | 15 | the per-rate feedback rounds other than round 1 at 30 Hz |
| `p30espread` | 3 | control spread over the E cluster without widening navigation — the control for `p30efullr`'s phase explanation |

Two of these labels ran a table at the same path as a registered arm at a different time, and the
manifests distinguish them by `schedule_sha256`: `w2pgOCr1` ran
`fig_w2p_greedy_clamped.json` at `cb62c4ec…` and `w2pgreedyr1` ran the same path at `9d4a8972…`.
The path alone does not identify the table; the manifest hash does, which is what
`scripts/executed_tables.py` is built on.

Naming the gap precisely: a clean clone plus `results/codesign_feedback/archive_v3/` reproduces
every number on this page, and reproduces nothing about those 353 traces. Bringing a label inside
the guarantee means recording it in `measured_timing.SOLVER_ARMS` or `XPURT_POINTS`, at which point
`verify_board_traces.wanted()` globs for it and `--verify` re-derives it.

## 6. The unregistered labels, screened

The question in §3 cannot be answered from the registry alone, because a board-measured arm that no
registry names would not appear there. The 26 unregistered labels that are chain runs of this
workload were therefore screened with the same `per_frame_chain` and `_warm_gaps` the registry uses,
pooled over their three replicates:

```bash
.venv/bin/python - <<'EOF'
import sys, glob, statistics
sys.path.insert(0, "scripts")
import measured_timing as MT
from make_measured_gantt_pair import per_frame_chain, read_trace
for lab in ("p30espreadr", "fb30r2r", "w2pime2r", "sonlygr", "w2pcpr"):   # ... one per row below
    chain, gaps, late, n = [], [], 0, 0
    for t in sorted(glob.glob(f"{MT.RES}/xpurt_long/trace_{lab}[0-9]_other_run1.csv")):
        rows = read_trace(t)
        chain += [c for k, c, _ in per_frame_chain(rows) if k >= 1]
        gaps += MT._warm_gaps("mlp_control", t)[1:]
        for k in sorted({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}):
            fr = [r for r in rows if (r["net"], r["inst"]) == ("yolov8_nano_64x96", k)]
            rel = min((r["rel"] for r in fr if r.get("rel") is not None), default=None)
            if rel is not None and k >= 1 and rel >= 100:
                n += 1; late += (max(r["e"] for r in fr) - rel > 66.67)
    print(lab, statistics.median(chain), MT._pct(chain, 0.95),
          statistics.mean(gaps), max(gaps), f"{late}/{n}")
EOF
```

| label | chain ms | p95 | gap mean | gap max | frames late |
|---|---|---|---|---|---|
| `p30espreadr` | 26.75 | 30.09 | 10.00 | 26.14 | 0/36 |
| `fb30r2r` | 33.43 | 37.06 | 10.00 | 19.38 | 0/81 |
| `fb36r2r` | 43.16 | 59.30 | 10.02 | 18.39 | 0/96 |
| `fb36r1r` | 56.19 | 89.23 | 10.50 | 28.89 | 30/96 |
| `w2pime2r` | 56.46 | 77.47 | 10.40 | 17.44 | 10/120 |
| `sonlygr` | 57.42 | 76.26 | 10.40 | 17.27 | 7/120 |
| `w2pg38r` | 59.08 | 67.96 | 10.35 | 18.92 | 0/102 |
| `w2pscpr` | 59.72 | 67.87 | 10.18 | 17.85 | 0/30 |
| `w2pg38p1` | 60.59 | 70.50 | 10.36 | 19.44 | 0/102 |
| `w2pg38u` | 63.08 | 74.11 | 10.39 | 21.57 | 5/102 |
| `w2pcpr` | 63.45 | 73.42 | 10.16 | 20.32 | 0/120 |
| `a30greedyOCr` | 70.09 | 73.45 | 10.00 | 14.83 | 0/36 |
| `fb45r1r` | 74.97 | 100.55 | 10.67 | 18.82 | 64/120 |
| `fb45r2r` | 86.56 | 113.53 | 10.74 | 16.76 | 81/120 |
| `w2pg50r` | 87.16 | 119.62 | 10.86 | 20.18 | 107/135 |
| `ashgreedyOCr` | 91.81 | 119.06 | 10.84 | 16.74 | 94/120 |
| `bgreedyr` | 147.93 | 172.59 | 10.63 | 47.92 | 243/243 |
| `ashcpsat_hardr` | 290.99 | 442.70 | 14.17 | 22.61 | 120/120 |
| `w2pgreedyr` | 467.26 | 814.06 | 18.15 | 27.91 | 120/120 |
| `ashgreedyr` | 490.60 | 839.73 | 18.62 | 28.37 | 120/120 |
| `a60greedyOCr` | 580.49 | 709.65 | 17.47 | 667.58 | 72/72 |
| `agreedyOCr` | 747.96 | 1059.65 | 13.25 | 1038.88 | 120/120 |
| `a90greedyOCr` | 945.46 | 1045.42 | 25.71 | 991.70 | 111/111 |
| `a120greedyr` | 1370.72 | 1456.78 | 35.06 | 1385.53 | 144/144 |
| `a120greedyOCr` | 1370.71 | 1452.68 | 35.07 | 1385.63 | 144/144 |
| `w2pimer` | — | — | — | — | no chain frames in the trace |

The late column here is against the fixed 66.67 ms threshold `--verify` uses, not against each
label's own spec window, because several of these labels have no spec recorded anywhere.

**No unregistered arm beats a drawn one.** The best of them, `p30espreadr`, ties `p30freer` at
26.75 ms with a worse control tail (26.14 ms against 18.06), which is exactly what it was run to
show. The `*greedyOC*` re-runs reproduce their registered twins to within 3.2 ms
(`a30greedyOCr` 70.09 against `a30greedyr` 70.10, `agreedyOCr` 747.96 against `agreedyr` 748.00,
`a60greedyOCr` 580.49 against `a60greedyr` 583.60).

These numbers are derived here by the same functions the registry uses, but they are **not recorded
constants**, so `measured_timing.py --verify` does not cover them and `verify_board_traces.py` does
not guarantee their traces. They are a screen, not a measurement of record.

## 7. What was checked

```
scripts/measured_timing.py --verify        → 0 DRIFT, exit 0, every arm's traces present
scripts/verify_board_traces.py             → 438 tracked or archived and matching, 0 failed
artifact/verify_no_hardware.sh             → 48 passed, 0 failed
```

# Scheduling solvers for XPU-RT: survey, implementations, and measurements

The scheduling problem this repo solves is an **unrelated-parallel-machine
job-shop with precedence, sequence-independent transfer costs, and release
times / deadlines on the periodic jobs**. In the classical taxonomy that is
`R | prec, r_j, d_j | C_max` — NP-hard, and the deadline structure makes even
*feasibility* NP-hard, which is why every method here is judged on two axes:
the makespan it achieves and whether the schedule it returns is valid at all
(no missed periodic window, and enough periodic instances to cover its own
makespan).

This document surveys the mechanisms available, records which were
implemented against `xpu-rt`, and reports what they measured. The measurement
workloads are the Flow A / RISC-V ones (spike-profiled and FireSim), not the
Qualcomm ones.

## 1. The mechanism landscape

### 1.1 Constructive heuristics (what the repo had)

`greedy`, `greedy_periodic` and `decomposed` are **list scheduling** (Graham
1966): repeatedly pick a ready operation by some priority rule and place it.
They differ only in the priority rule, and all three place an operation on
whichever machine finishes it *soonest*.

Two known weaknesses of that rule show up directly in the measurements:

- **Earliest-completion priority is myopic about the critical path.** It
  schedules whichever ready op finishes first, which repeatedly defers the
  long chain that actually sets the makespan. The standard fix is a
  *lookahead* priority — see HEFT below.
- **Earliest-completion *placement* is wrong for a job with slack.** Finishing
  a 5 ms-period task in 0.9 ms instead of 2.7 ms buys nothing if both fit the
  window, but it costs the fast lane that the makespan-critical job needs.
  This is the observation `greedy_reserved` implements.

### 1.2 Exact methods

- **MILP, disjunctive/big-M formulation** (Manne 1960; Applegate & Cook 1991)
  — what `xpu-rt/scheduler.py` builds. One ordering boolean per operation
  *pair*, so the model is O(N²) in both variables and constraints. Solid at a
  few hundred operations, hopeless past that; see §3.
- **MILP, time-indexed formulation** (Pritsker et al. 1969) — one binary per
  (operation, machine, time slot). Tighter LP relaxation, but the variable
  count scales with the time horizon, which here is ~10⁵ microsecond ticks.
  Not viable and not implemented.
- **Constraint programming with interval variables and a global no-overlap
  propagator** (Baptiste, Le Pape & Nuijten 2001; OR-Tools CP-SAT) — states
  "these operations cannot overlap" as *one constraint per machine* instead
  of O(N²) rows, and CP-SAT's cumulative/disjunctive propagators plus
  no-good learning are built for exactly this problem class. **Implemented**;
  see §4.

### 1.3 Metaheuristics

All of these need a *decoder* that turns a search vector into a schedule. The
standard encoding for precedence-constrained scheduling is the **random key**
(Bean 1994): a real number per operation, decoded by a serial schedule
generation scheme that only lets the keys break ties among operations that are
*simultaneously eligible*, so every decoded point is precedence-feasible by
construction and the optimiser never needs a repair operator.

- **Particle swarm optimisation** (Kennedy & Eberhart 1995) over random keys.
  **Implemented.**
- **Simulated annealing** (Kirkpatrick et al. 1983) over the same encoding,
  as a control: it answers "is the swarm doing anything a temperature-guided
  random walk would not". **Implemented.**
- **Genetic algorithms / memetic algorithms** — the biggest literature for
  RCPSP (Hartmann & Kolisch's benchmark comparisons put GA+SGS at the top).
  Not implemented: PSO and SA share the decoder and already answer the
  question of whether *any* population search beats the constructive
  heuristics at these budgets.
- **Tabu search / large-neighbourhood search** — strong on job-shop
  specifically (Nowicki & Smutnicki 1996). Not implemented; the SA control
  suggests the payoff is in the initial solution, not the local search.

The decisive design point, learned the hard way here, is **seeding**. A search
seeded only from HEFT spends its whole budget climbing out of an infeasible
basin, because HEFT ignores deadlines entirely. Seeding the population from
*every* cheap heuristic — and holding each heuristic's own schedule as the
incumbent, not just its decoded key vector — is what makes the search
non-destructive. This is the memetic-algorithm pattern: metaheuristic on top
of constructive heuristics, never instead of them.

## 2. What was implemented

| module | what it adds |
| --- | --- |
| `xpu-rt/schedule_decoder.py` | The shared SGS. `(priority, machine choice) -> (t, alpha)` with insertion-based placement into idle gaps, precedence + transfer costs, machine conflicts, and periodic release times. Also `upward_rank` and a common `evaluate`. |
| `xpu-rt/metaheuristics.py` | `heft_schedule`, `pso_schedule`, `sa_schedule`, all driving the decoder; heuristic seeding and incumbent-holding shared between the two searches. |
| `xpu-rt/cpsat_scheduler.py`, `xpu-rt/_cpsat_solve.py` | CP-SAT backend. Runs out of process because OR-Tools needs a newer numpy/protobuf than the scheduler's environment pins; `XPURT_CPSAT_PYTHON` names the interpreter. |
| `xpu-rt/scheduler.py` | `cvxpy_solver` parameter + per-backend time-limit dispatch, replacing four hardcoded `cp.MOSEK` call sites. |
| `scripts/run_xpurt_schedule.py` | `--solver heft/pso/sa/cpsat`, `--cvxpy-solver`, `--search-budget`. |

`heft` also joins the `auto` candidate set.

## 3. Why the MILP does not scale here

See the README's solver section for the full constraint/variable accounting.
In brief: the pruning passes are effective on *constraints* (they remove 97%
of the non-overlap pairs on a 191-op workload), the dependency-chain pruning
is load-bearing (turning it off turns a 620 s solve into a timeout), but
`_periods_overlap` can only prune a pair when *both* operations carry
windows — so every pair involving a non-periodic operation survives by
construction. On the 1751-op workload that leaves 1.46M constraints, ~330 s to
build the model in Python, and a 41.9 GB peak before MOSEK ever sees it.

The MILP is also the only solver that does not reseed periodic networks to one
instance before its first pass, so it is handed several times more periodic
work than the heuristics end up scheduling. Matching the seeding is worth more
than any solver tuning: on FireSim dronet+yolov8 it turns a 332-op timeout
into a 242-op solve in 133 s.

## 4. Measurements

See the README for the per-workload heuristic comparison; this section
records the backend, search-method and corpus results.

### 4.1 One fixed instance: the spike 3-model workload

`networks_mlp_dronet_yolo_spike` at the instance counts its refinement loop
converges to (mlp_control x63, dronet x1) — 677 operations, 2 lanes. Every
method schedules the *same* instance, so this isolates solver quality from
the refinement loop's own behaviour.

| method | objective (ms) | all-ops (ms) | missed windows | wall |
| --- | --- | --- | --- | --- |
| `greedy` | 1242.41 | 1242.41 | 0 | 0.1 s |
| `greedy_periodic` | 665.47 | 693.89 | 0 | 0.1 s |
| **`greedy_reserved`** | **628.94** | 657.35 | 0 | **0.1 s** |
| `decomposed` | 1270.83 | 1270.83 | 0 | 0.1 s |
| `heft` | 630.91 | 630.91 | **284** | 0.04 s |
| `heft_edf` | 1241.61 | 1241.61 | 0 | 0.04 s |
| `pso` (30 s budget) | 628.94 | 657.35 | 0 | 15.9 s |
| `sa` (30 s budget) | 628.94 | 657.35 | 0 | 30.0 s |
| `cpsat` (180 s limit) | 643.09 | 733.99 | 0 | 180.6 s |
| `milp` (any backend) | no solution | | | model still building after 22 min |

The MILP row is not a timeout on the *solve*: at 677 operations cvxpy is
still compiling the model after 22 minutes, which matches grid E, where no
RISC-V workload above 212 operations finished inside 900 s. CP-SAT builds and
solves the same instance in under three minutes — the difference is the
formulation, not the solver: one `AddNoOverlap` per machine against
O(N^2) big-M ordering rows.

Three things to take from it:

- **Nothing beats `greedy_reserved` here.** PSO and SA match it exactly and
  stop; at a 30 s budget over a 1354-dimensional encoding, population search
  does not improve on a good constructive solution. CP-SAT gets within 2.3%
  in 180 s. The constructive heuristic costs 0.1 s.
- **HEFT is deadline-blind.** Its 630.91 ms looks competitive until you count
  the 284 missed windows. `heft_edf` fixes that but pays 2x makespan on this
  workload, because banding all periodic work above the non-periodic chain
  serialises them.
- **Seeding is what makes the metaheuristics usable at all.** Seeded only
  from HEFT they reached 3341 ms (PSO) and 822 ms (SA) — worse than every
  greedy picker — because they spent the budget climbing out of HEFT's
  infeasible basin. Seeding from all five heuristics *and* holding each
  heuristic's own schedule as the incumbent (the random-key round-trip is
  lossy) is what brings them to 628.94.

### 4.2 cvxpy backends

The MILP's backend is now selectable (`--cvxpy-solver`). Of the seven cvxpy
backends installed in this environment only three accept boolean variables —
verified by handing each a 3-boolean toy MIP; CLARABEL, SCS, OSQP and DAQP
raise `SolverError`. On `yolov8_only_spike` (212 ops) at a matched 120 s
limit:

| backend | makespan | licence |
| --- | --- | --- |
| **MOSEK** | **618.98 ms** | commercial (academic licence here) |
| HiGHS | 1707.66 ms | MIT, free |
| SCIPY (HiGHS under the hood) | 22096.92 ms | BSD, free |

MOSEK is 2.8x better than HiGHS and 36x better than SCIPY at equal budget, so
the free backends are not a drop-in substitute at these model sizes.

Swept across MILP-tractable instance sizes at a matched 120 s budget, with
CP-SAT and the best heuristics as reference (objective makespan, ms):

| method | yolov8_only (212 ops) | fsim dronet+yolov8 (242 ops) | spike 3-model (271 ops) |
| --- | --- | --- | --- |
| `greedy_reserved` (0.0 s) | 628.94 | 110.68 | 628.94 |
| `heft` (0.0 s) | 628.14 | 98.41 | 630.91 (35 missed) |
| `heft_edf` (0.0 s) | 628.14 | 104.58 | 666.22 |
| **`cpsat`** (120 s) | **614.16** | **96.33** | **615.39** |
| `milp:MOSEK` | 615.20 | 150.07 | 656.32 |
| `milp:HIGHS` | 1707.66 | 5000.00 | 8110.01 |
| `milp:SCIPY` | 22096.92 | 5000.00 | 9931.28 |

Three conclusions:

- **CP-SAT is the best exact method at every size measured**, and beats MOSEK
  at all three — narrowly at 212 operations (614.16 vs 615.20), decisively at
  242 (96.33 vs 150.07) and at 271 (615.39 vs 656.32, in half the wall time).
  It is also the only one that keeps working as the instance grows: at 677
  operations it still returns a valid schedule while the MILP model has not
  finished compiling. For a disjunctive no-overlap problem this is the
  expected ordering — CP-SAT states the constraint once per machine, the MILP
  spells it out per operation pair — and it is the strongest argument for
  moving the exact path off cvxpy entirely.
- **The free cvxpy backends are unusable past ~200 operations.** At 242
  operations *both* HiGHS and SCIPY return exactly 5000.00 ms, which is the
  big-M constant itself: a trivially feasible answer carrying no scheduling
  information at all, not a merely-weak incumbent.
- **The heuristics stay within 2.4%, 2.2% and 2.2% of the best exact answer,
  for zero seconds.** The exact methods buy a couple of percent for two to
  three orders of magnitude more wall time — worth it only where the
  heuristics produce no valid schedule at all.

### 4.2b Warm starting the exact solvers

Handing a solver an existing schedule as its initial integer solution works for
one of the two, and the difference is plumbing rather than principle.

**CP-SAT: worth more than 40x the compute.** On the 242-op RVV+Gemmini q31
instance, seeded from `heft` (46.91 ms):

| | 15 s budget |
| --- | --- |
| cold | 86.56 / 90.77 ms (two runs) |
| warm | **45.34 ms** |

Warm at 15 s beats *every* cold budget including 600 s (45.44 ms). CP-SAT
adopts the hint as its incumbent at 0.06 s and improves from there.

Getting there took three fixes, each of which produced a plausible-looking
number rather than an error, and only the solver's own log distinguished them:

1. **Partial hint.** Hinting `start` and the assignment booleans but not
   `duration`/`end` broke `end == start + duration`.
2. **Objective unhinted.** Leaving `cmax` out drew
   *"The solution hint is incomplete: 1210 out of 1211 non fixed variables
   hinted"* — and an incomplete hint is advice for the search, not a solution
   CP-SAT can adopt.
3. **Independent rounding.** Rounding each start and duration to integer
   microseconds separately makes abutting operations overlap by 1 us:
   *"The solution hint is complete, but it is infeasible!"*. Fixed by replaying
   the schedule on the solver's own integer grid (`_integerize`).

Only after all three does the log say *"The solution hint is complete and is
feasible. Its objective value is 46913"*.

**MOSEK: cvxpy does not pass a MIP start through.** Same instance, same seed,
120 s each:

| | objective | wall |
| --- | --- | --- |
| `heft` seed | 46.91 | 0.01 s |
| MOSEK cold | 112.264 | 185.8 s |
| MOSEK warm | **112.264** | 188.0 s |

Bit-identical, and the warm arm is 2.4x worse than the point it was handed. A
MIP start that reached the solver would be a feasible upper bound that
branch-and-bound could only match or beat, so returning the exact cold answer
means the values never arrived. This was measured with a *complete* start —
242 starts, 484 `alpha`, all 8,012 `beta` and `C_max = 46.91` — so it is not
the incomplete-hint failure above.

The cause is cvxpy's reduction chain: MOSEK receives transformed variables, not
`t`/`alpha`/`beta`, and cvxpy has no mapping to carry a user-set `.value`
through it. `warm_start=True` is the continuous path;
`MSK_IPAR_MIO_CONSTRUCT_SOL` arrives with nothing to construct from.

**`--solver milp_native` settles it.** `xpu-rt/mosek_native.py` builds the same
formulation straight against MOSEK's Optimizer API, where `putxxslice` +
`MSK_IPAR_MIO_CONSTRUCT_SOL` is the supported MIP-start path. Same instance,
same 120 s:

| | objective | note |
| --- | --- | --- |
| `heft` seed | 46.91 | 0.01 s |
| native, cold | 90.906 | 1.94x worse than the seed |
| **native, warm** | **46.912** | MOSEK log: *"Initial feasible solution objective: 4.6912e+01"* |
| cvxpy, cold | 112.264 | |
| cvxpy, warm | 112.264 | start discarded |

Three things follow:

- **The MIP start works natively.** MOSEK reports adopting the 46.91 ms
  incumbent, which it never did through cvxpy — confirming the blocker was
  cvxpy, not MOSEK.
- **It buys nothing beyond not losing.** Warm returns the seed to three
  decimals: 35,295 branches and 1.45M simplex iterations in 120 s without
  improving on a schedule HEFT produced in 10 ms. Warm beats cold (90.9 ->
  46.9) only because cold spends its budget rediscovering something worse.
- **The native model is better cold too** — 90.9 against cvxpy's 112.3 at the
  same budget, because cvxpy spends roughly a third of the wall clock compiling
  and hands MOSEK a transformed problem rather than these rows.

None of which makes the MILP competitive: its best result ties a 10 ms
heuristic after two minutes, while warm-started CP-SAT reaches 45.34 ms in 15 s.
The rewrite closes the question rather than opening a path.

### 4.3 Global trends: 30 generated spike workloads

Generated with `scripts/gen_random_workload.py` against the spike RVV +
scalar bank entry (3-5 networks, 2-4 periodic, real profile data on both
lanes). "Valid" means no missed periodic window *and* enough periodic
instances to cover the schedule's own makespan.

The entry was called `spike_rv` when these numbers were measured. Merging
`rose-2-dev` replaced `data/banks/` with that branch's fuller bank, where the
same machines/profile_hw/profile block is named `spike_het` (and
`spike_single_core`) — field-for-field identical, so the hardware these
workloads were generated against is unchanged. The *model* bank is not: it
now draws dronet's period from 20-100 ms rather than 40-400, and lets
yolov8_nano's instance count reach 0. Re-running the generator at the same
seeds therefore yields a different corpus. The table below is the corpus as
generated at the time; treat it as a measurement, not as something a fresh
`gen_random_workload.py` run reproduces.

| method | valid | missed windows | under-covers | both | median vs best |
| --- | --- | --- | --- | --- | --- |
| `greedy` | 19/30 | 0 | 11 | 0 | 1.008x |
| `greedy_periodic` | 19/30 | 0 | 11 | 0 | 1.008x |
| `greedy_reserved` | 19/30 | 0 | 11 | 0 | 1.008x |
| `decomposed` | 0/30 | 2 | 10 | 18 | — |
| `heft` | 0/30 | 26 | 0 | 4 | — |
| `heft_edf` | 23/30 | 1 | 5 | 1 | 1.000x |
| **`auto`** | **24/30** | 0 | 6 | 0 | **1.000x** |

- **`heft_edf` is the strongest single constructive method on this corpus**,
  and the opposite of its ranking on the hand-written 3-model workload. The
  ordering is workload-dependent every time it is measured, which is the
  standing argument for `auto`.
- **`decomposed` and plain `heft` never produce a valid schedule here.**
  HEFT's median worst-lateness is 728 ms; `decomposed`'s is 274 ms.
- **The dominant failure is not the picker, it is the refinement loop.** Six
  of the thirty defeat every method by under-coverage: the loop grows
  periodic instance counts from a makespan those same instances inflated, so
  it never reaches a self-consistent answer. Fixing the loop is worth more
  than any further solver work.

### 4.4 What did not pay off

- **Particle swarm and simulated annealing.** Both match the best heuristic
  and neither beats it, at 150x-300x the wall time. The encoding is not the
  problem — seeding fixed that — the budget is: 677 operations is a
  1354-dimensional search and 30 s buys a few hundred decodes.
- **CP-SAT**, on this workload. It is the only exact-ish method that can
  *build* a 677-op instance at all, and it returns a valid schedule, but at
  180 s it is still 2.3% behind a 0.1 s heuristic. It is worth revisiting on
  workloads where the heuristics produce no valid schedule — the six
  under-covering corpus cases are the obvious target.

### 4.5 The `wl_sweep` corpus: gating `heft_edf`'s deadline band

`scripts/solver_study/wl_sweep_bench.py` runs every solver over the 24
buildable `wl_sweep` specs — six workload families (`control_mix`,
`saturation`, `scale_ladder`, `tight_loop`, `vint_intro`, `vint_multi`) each in
four machine configurations: `hetero` (1 P + 1 E), `gempair` and `rvvpair` (two
identical lanes), and `quad` (2 P + 2 E, 6 combinations).
`scripts/solver_study/data/wl_sweep_baseline.json` freezes the run these
numbers are diffed against.

**The finding.** `heft_edf`'s periodic priority band was unconditional: every
periodic op outranked every non-periodic one. That was tuned on the 30 spike
workloads of §4.3, where plain `heft` scored 0/30 and the band was the
difference between a valid schedule and none. On `quad` there is enough machine
slack that the band buys nothing and costs real makespan — measured on the
three specs where plain `heft` was *also* valid, so the safety was pure loss:

| spec | `heft` | `heft_edf` | penalty |
| --- | --- | --- | --- |
| `control_mix_quad` | 33.63 | 36.12 | +7.4% |
| `saturation_quad` | 75.54 | 86.62 | +14.7% |
| `vint_multi_quad` | 3750.82 | 3939.28 | +5.0% |

Concentrated, not diffuse: the median penalty over the nine both-valid specs
was +0.00%.

**The laxity signal.** Each op gets its CPM *total float* — a forward pass for
the earliest finish it could reach and a backward pass for the latest finish
that still leaves the chain feasible, both using the fastest combination
available. The backward pass is what makes the signal usable. `op.max_end_t` is
the *instance* window, carried identically by every op of the instance, so the
raw `max_end - eft` of a chain head reads as though it had the whole window to
itself; the backward pass pushes the real deadline up the chain and gives every
op of an instance the same float, which is the right semantics — an instance is
tight or slack as a whole. Dividing by a span lower bound (critical path, or
total work per machine, whichever binds) makes it scale-free.

**No fixed threshold works.** Swept absolute (ms), normalised by the span lower
bound, and normalised by the window width, 20-35 values each:

| gate | valid | geomean vs `heft_edf` | worst vs `heft` |
| --- | --- | --- | --- |
| `float/L < 0.03` | 14/24 | 0.935 | 1.000 |
| `float/L < 0.08` | 19/24 | 0.960 | 1.050 |
| `float/L < 0.13` | 20/24 | 0.982 | 1.147 |
| `float < 14 ms` | 17/24 | 0.963 | 1.000 |
| unconditional | 20/24 | 1.000 | 1.147 |

Every constant that recovers `quad` drops windows somewhere else, and every
constant that is safe everywhere gives `quad`'s penalty straight back. The
reason is structural: laxity is very nearly configuration-invariant — a window
and a chain of work do not change when you add a core — while the crowding
those ops must survive changes a great deal. On `saturation`, the gate that
recovers `quad`'s makespan (lift 112 of 238) misses ten windows on `hetero`.

**What works is enumeration plus a guard.** A gate can only fall between two
adjacent laxity *levels*, and because laxity is close to a per-network property
the corpus admits only 7-8 levels per workload, endpoints included. So there is
nothing to tune: decode every gate, score each by (missed windows, makespan),
and keep the best. "Lift everything" is always in the list, so the result is
never worse than the unconditional band; "lift nothing" is too, so it is never
worse than plain `heft`. A gate that would drop a deadline is decoded, scored
and discarded before it can be returned — the guard is what makes a gate this
aggressive safe to ship.

One caveat found the hard way: the floats of ops that are *equally* tight come
back differing in the last ulp, because they are the same durations summed in a
different association order. `control_mix_hetero` has 140 periodic ops at 32
"distinct" laxities but only 6 levels, the other 26 being spread of order
1e-16. Gating inside that spread splits an instance chain at a boundary decided
by rounding; it did move the makespan — a further 2% on two specs — but not
reproducibly, so `laxity_levels` merges to a relative tolerance first and those
two percent are deliberately left on the table.

**Result** (objective / missed windows; `misses == 0` is valid):

| spec | config | `heft` | `heft_edf` before | after | delta |
| --- | --- | --- | --- | --- | --- |
| `control_mix` | gempair | 54.07 / 14 | 60.07 / 0 | 54.07 / 0 | **-9.99%** |
| `control_mix` | hetero | 38.17 / 14 | 44.43 / 0 | 42.14 / 0 | **-5.16%** |
| `control_mix` | quad | 33.63 / 0 | 36.12 / 0 | 33.63 / 0 | **-6.88%** |
| `control_mix` | rvvpair | 85.67 / 21 | 97.12 / 0 | 85.67 / 0 | **-11.79%** |
| `saturation` | gempair | 112.16 / 154 | 164.67 / 0 | 154.64 / 0 | **-6.09%** |
| `saturation` | hetero | 86.94 / 44 | 115.83 / 0 | 109.18 / 0 | **-5.75%** |
| `saturation` | quad | 75.54 / 0 | 86.62 / 0 | 75.54 / 0 | **-12.78%** |
| `saturation` | rvvpair | 170.96 / 47 | 217.56 / 0 | 182.98 / 0 | **-15.89%** |
| `scale_ladder` | gempair | 98.59 / 0 | 98.59 / 0 | 98.59 / 0 | — |
| `scale_ladder` | hetero | 97.16 / 0 | 97.16 / 0 | 97.16 / 0 | — |
| `scale_ladder` | quad | 97.16 / 0 | 97.16 / 0 | 97.16 / 0 | — |
| `scale_ladder` | rvvpair | 97.37 / 0 | 97.37 / 0 | 97.37 / 0 | — |
| `tight_loop` | gempair | 75.58 / 119 | 92.79 / 84 | 92.20 / 84 | -0.63% |
| `tight_loop` | hetero | 76.05 / 108 | 100.83 / 84 | 100.81 / 84 | -0.02% |
| `tight_loop` | quad | 45.92 / 84 | 73.44 / 84 | 45.92 / 84 | -37.48% |
| `tight_loop` | rvvpair | 75.60 / 84 | 91.21 / 84 | 75.60 / 84 | -17.11% |
| `vint_intro` | gempair | 3754.05 / 56 | 3754.05 / 0 | 3754.05 / 0 | — |
| `vint_intro` | hetero | 4160.69 / 0 | 4160.69 / 0 | 4160.69 / 0 | — |
| `vint_intro` | quad | 3750.82 / 0 | 3750.82 / 0 | 3750.82 / 0 | — |
| `vint_intro` | rvvpair | 11068.08 / 56 | 11068.08 / 0 | 11068.08 / 0 | — |
| `vint_multi` | gempair | 3754.05 / 140 | 3874.09 / 0 | 3757.46 / 0 | **-3.01%** |
| `vint_multi` | hetero | 4160.69 / 33 | 4239.12 / 0 | 4160.69 / 0 | **-1.85%** |
| `vint_multi` | quad | 3750.82 / 0 | 3939.28 / 0 | 3750.82 / 0 | **-4.78%** |
| `vint_multi` | rvvpair | 11068.08 / 140 | 11263.61 / 0 | 11069.59 / 0 | **-1.72%** |

- **Validity is unchanged at 20/24** — every spec that was valid still is, and
  no spec gained validity (the four `tight_loop` specs are statically
  infeasible: their periodic ops carry *negative* float, so no ordering saves
  them, and every method misses exactly 84 windows).
- **All three `quad` penalties are fully recovered**, to `heft`'s makespan to
  the last decimal.
- Sixteen of the 24 improve; the largest wins are on specs where `heft` was
  *invalid*, so they were never visible as a "penalty" at all:
  `saturation_rvvpair` -15.9%, `control_mix_rvvpair` -11.8%.
- Deadline margin is not spent to buy this. Worst-case lateness on all 20 valid
  specs is identical before and after — every window still finishes 1.2-13.1 ms
  early. Among the infeasible `tight_loop` four, total lateness falls on one,
  rises 1.1% on another, and is unchanged on two.
- Cost: 7-8 decodes instead of one, worst case 0.97 s on the 745-op
  `vint_multi` against 0.16 s — against 20 s for PSO and 60 s for CP-SAT.

**Out of corpus.** The same comparison on the 24 non-`wl_sweep` toplevel specs
that build from this data root (the `3net` family, `mlp10_dronet20_yolov8`, the
`smolvla`, `vint` and `yolov8` singles): 0 validity regressions, 0 objective
regressions, 15 improvements, up to -18.9% on `3net_rvvpair`. None of these
specs was used to design the gate.

### 4.6 Re-sweeping `greedy_reserved`'s `max_slowdown`

`_RESERVED_MAX_SLOWDOWN = 2.0` was swept over {1, 2, 4, 8, unbounded} on eleven
spike/FireSim workloads and never re-examined for the gemmini/RVV lanes. Swept
again over {1, 1.25, 1.5, 2, 3, 4, 6, 8, 12, 16, unbounded}:

- **On `wl_sweep` the knob is inert on 20 of 24 specs** — every value from 1.0
  to unbounded returns an identical schedule. It moves only on `hetero` and
  `quad`. On the same-kind pairs it is inert on 12/12, which is structural: the
  cap rejects a lane more than `max_slowdown` times slower than the op's own
  fastest, and with two identical lanes there is no such lane to reject. So
  `--reserved-max-slowdown` is a no-op on any single-lane-family machine set.
- Where it does move, `wl_sweep` weakly prefers *lower*: 1.0 and 1.25 give
  11/24 valid against 10/24 for everything ≥1.5, on the strength of one spec
  (`vint_multi_hetero`), with the geomean objective within 0.1% across the
  whole range.
- **That preference reverses on the other 24 specs.** There the knob moves on
  13/24, validity is 13/24 at every value, and values below 1.5 are much worse
  on makespan: `vint_only` 4231 ms at 1.0 against 2713 at ≥1.5 (+56%),
  `vint_stack` 4237 against 2858. So the `wl_sweep` signal is one spec deep and
  does not generalise.
- **3.0 weakly dominates 2.0 across all 48 specs measurable here**: identical on
  every `wl_sweep` spec, and better on two others — `3net_armA` 80 missed
  windows against 101, `3net_fused_v2` 74.6 ms against 76.6. 4.0 is *worse* than
  2.0 on `vint_multi_hetero` (4547.4 against 4546.6).

**Recommendation: leave `_RESERVED_MAX_SLOWDOWN` at 2.0, and keep
`greedy_reserved` in `auto`.** Not because either is earning its place here —
`greedy_reserved` is the unique best candidate on **0 of 48** specs, and
removing it from `auto` changes `auto`'s answer on **0 of 48** — but because
the evidence that put both there is on workloads this data root cannot build.
The eleven spike/FireSim specs the constant was tuned on all fail with
`profile_loader: required profile data is missing`, as does the QRB5165 3-way
workload where `greedy_reserved` reportedly matches the MILP optimum (33.57 ms
against 90.13 for every other picker). Raising the constant to 3.0 would trade
a documented 30 ms win on FireSim dronet@20ms+yolov8 — the measurement that
ruled out a 4x cap — for a 2 ms gain on one spec here, blind. A
`greedy_reserved` pass costs about a second, so dropping it buys a second and
risks a documented win.

**The measurement that would settle both:** restore the FireSim/spike profile
data to a data root and re-run those eleven specs at {2.0, 3.0}. If 3.0 holds
there it dominates on both corpora and the constant should move; if
`greedy_reserved`'s QRB5165 3-way win no longer reproduces, it should come out
of `auto` at the same time.


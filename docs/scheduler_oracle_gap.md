# How far from optimal are the cheap schedulers? An oracle gap study

`scripts/solver_study/data/wl_sweep_baseline.json` reports every solver on
every `wl_sweep` spec at the standard budgets, and the comparison it supports
is "which of these ten methods wins". That is not the question anyone actually
has. When `heft_edf` scores 1.000x relative-to-best it may be optimal, or the
whole field may be 30% off together and nobody would know, because *best* there
means *best among the methods we ran*.

This study puts an absolute scale under the relative one. For a chosen subset
it establishes (a) the best solution reachable at large effort and (b) a
**proven lower bound** — a number no schedule can beat — and reports the
interval between them. Where the two meet, the optimum is known and the
heuristic gap is exact. Where they do not, the honest answer is an interval,
and this document gives the interval rather than promoting a best-found
solution to an optimum.

Measured on `flow-c-qnn-qrb5165` at `5c66592e`, with the two CP-SAT model fixes
of `95db5778` in place. **No solver was modified.** Three scripts were added:
`scripts/solver_study/oracle_run.py` (upper bounds), `oracle_bounds.py`
(combinatorial lower bounds), and `oracle_cpsat_decide.py` +
`_oracle_cpsat_decide_child.py` (CP-SAT used as a decision procedure, which is
where most of the lower-bound strength comes from).

## 1. What is being bounded

`schedule_decoder.evaluate` minimises the makespan over the **non-periodic**
operations when any exist, and over all operations otherwise; call that set the
*targets*. Separately a schedule only counts as a solution when `misses == 0`,
i.e. every periodic operation finishes inside its own window. So the quantity
in question is

> `obj` = the latest completion among target operations, minimised over
> schedules that respect precedence, per-machine no-overlap, release times,
> **and every periodic window**.

Two consequences run through everything below.

**Validity is a gate, not a tiebreak.** A schedule that misses a window is not
a worse solution, it is not a solution. `heft` scores 86.94 ms on
`saturation_hetero` against the 111.50 ms of the best valid cheap method — and
misses 44 windows getting there. It is excluded, and so is every other row with
`misses > 0`. The best *valid* cheap heuristic is the number that goes in the
table.

**A bound must say what it bounds.** Every lower bound here is a bound on
`obj`, the target-only makespan, of a schedule that also meets all the windows.
Bounds are allowed to *drop* the window constraints (that only enlarges the
feasible set and so stays valid); they are not allowed to drop resource or
precedence constraints without saying so. None of these is a bound on
`all_ops`, the makespan over every operation, which is a different and larger
quantity.

## 2. Which specs, and why

Six specs, crossing three sizes with the two extreme lane structures, plus the
sibling-hart family that the `95db5778` fix was about:

| spec | ops | machines | combinations | lane structure |
|---|---|---|---|---|
| `scale_ladder_hetero` | 126 | 2 | 2 | 1 gemmini + 1 rvv |
| `scale_ladder_quad` | 126 | 4 | 6 | 2 gemmini + 2 rvv |
| `control_mix_gempair` | 295 | 2 | 3 | 2 gemmini harts |
| `control_mix_quad` | 295 | 4 | 6 | 2 gemmini + 2 rvv |
| `saturation_hetero` | 393 | 2 | 2 | 1 gemmini + 1 rvv |
| `saturation_quad` | 393 | 4 | 6 | 2 gemmini + 2 rvv |

`scale_ladder` is the smallest and has no periodic operations at all, so `obj`
is the plain makespan and the window constraint is absent — a clean control.
`control_mix` and `saturation` are 47% and 61% periodic, which is where the
window gate bites. `gempair` is the configuration whose sibling-hart
combinations (`[P#0]`, `[P#0,P#1]`, `[P#1]`) broke the CP-SAT model before
`95db5778`, so it is the one to check the fix against at real effort.

The cheap combinatorial bounds were then run across **all 24 buildable specs**
(§6), because they cost seconds and the cross-family pattern turns out to be
the most useful result in this document. Eight specs of the 32 do not build at
all (`bimodal`, `perception_heavy`: missing profile data) and are out of scope.

## 3. Upper bounds — best solution at maximum effort

`oracle_run.py` runs every cheap method first, keeps the best one that misses
no window, hands that to CP-SAT as a solution hint, and lets CP-SAT run at
`workers=4` for 120–1800 s at several random seeds. CP-SAT with more than one
worker is not deterministic, so seeds matter; where several seeds ran, the
spread is reported.

`CPSAT_LOG=1` confirms the hint is being used rather than silently dropped —
"The solution hint is complete and is feasible" appears in every run here,
which is what `95db5778` fixed and is worth re-checking whenever the model
changes.

## 4. Lower bounds

Three, and the max of them is the reported bound.

### 4.1 Critical path (`oracle_bounds.cp_bound`)

Give every operation its fastest feasible combination and every transfer zero
cost. Then `head[i] = max(min_start[i], max over predecessors p of head[p] +
mindur[p])` is a valid lower bound on `i`'s start whatever the assignment, and
`max over targets of head[i] + mindur[i]` bounds `obj`. This ignores every
resource conflict, so it is weak whenever the machines are the binding
constraint — and exact when they are not, which turns out to be most of the
time.

### 4.2 Energetic / area LP (`oracle_bounds.area_bound`)

The textbook per-machine total-work bound is **identically zero on these
workloads**: every operation can run on every combination, so the set of work
that "can only go on machine m" is empty. The generalisation that does work:
an operation running on combination `c` occupies *every* machine in `c` for its
whole duration — exactly what the per-machine `AddNoOverlap` says — so machine
`m` can absorb at most `b - a` machine-time inside any window `[a, b]`.

For a candidate objective `T`, an operation must lie entirely inside `[a, b]`
when `head[i] >= a` and `deadline[i] <= b`, where

    deadline[i] = min( max_end[i]      if i is periodic,
                       T - tail[i]     if a target is reachable from i )

and `tail[i]` is the longest min-duration path from `i`'s completion to a
target's completion. `T - tail[i]` is a deadline because a schedule with
objective `T` finishes every target by `T`, so everything on a path into a
target has to clear out ahead of it. The LP

    min 0   s.t.  sum_c x[i,c] = 1,  x >= 0,
                  for each machine m and window [a,b]:
                    sum_{i inside} sum_{c containing m} x[i,c]*dur[i,c] <= b-a

is a relaxation of "a schedule with objective `T` exists" — every real schedule
induces an integral `x` satisfying all of it. **LP infeasible ⇒ no schedule
achieves `T`.** Windows come from the observed `head`/`deadline` values on a
grid, refined by a cutting-plane pass that hunts the most-violated window
against the current LP solution. Bisecting on `T` gives the bound. Because each
refutation stands on its own, the reported value is valid whether or not the
bisection found the largest refutable `T`.

Certificates are printed, and the strongest ones are checkable by hand. On
`control_mix_gempair` the binding statement is: *in `[43.762, 51.745]`, 59
operations are pinned inside and need 15.972 ms of machine-time; two harts
supply 15.966 ms.* On `saturation_hetero` no single all-machine window is over
on its own and the refutation needs the LP across both machines — the script
says so rather than quoting a window that does not carry the argument.

Every INFEASIBLE verdict is re-derived a second time before it is believed. A
HiGHS status other than "proven primal infeasible" is not accepted as a proof
at all, and the ones that are get re-solved *elastically*: each window row is
allowed to overrun by `s >= 0` and the total overrun is minimised. That problem
is always feasible, so a positive optimum re-proves the refutation from a
different numerical path and an optimum of zero withdraws it. Across all 24
specs, no refutation was withdrawn and the bounds are bit-identical to a run
without the recheck — which is the point: the recheck is there so that a
numerical artefact could not have become a "proven" bound without anyone
noticing.

### 4.3 CP-SAT as a decision procedure (`oracle_cpsat_decide.py`)

CP-SAT reports an objective bound at the end of an optimisation, and on these
instances it is **almost worthless**: 4.86 ms against a 56.8 ms incumbent on
`control_mix_gempair`, 6.67 ms against 99.7 ms on `saturation_hetero`. The
search spends its budget improving the incumbent and its relaxation never
tightens.

Asked the strictly easier question *"is there any schedule with objective
≤ T?"*, the same solver often answers INFEASIBLE, and INFEASIBLE is a proof
that the optimum exceeds `T`. Bisecting on `T` turns CP-SAT into a lower-bound
engine.

The model must be the production model or the bound is about a different
problem, so nothing is rebuilt: `cpsat_scheduler.cpsat_schedule` constructs its
payload exactly as always and only the script path it hands the subprocess is
redirected (a module attribute set at runtime — no solver file is edited). The
child then runs `_cpsat_solve.main` verbatim with `CpModel.minimize` wrapped so
the objective variable also carries `objective <= T`. One extra row; everything
else identical.

## 5. Results

### 5.1 The table

| spec | ops / lanes | best valid cheap heuristic | best known at max effort | proven LB | heuristic -> best known | best known -> LB |
|---|---|---|---|---|---|---|
| `scale_ladder_hetero` | 126 / 2M 2C | `heft` 97.16 (0.0 s) | 97.16 (CP-SAT 60 s) **proven optimal** | 97.16 | +0.00% | +0.00% |
| `scale_ladder_quad` | 126 / 4M 6C | `heft` 97.16 (0.0 s) | 97.16 (CP-SAT 120 s) **proven optimal** | 97.16 | +0.00% | +0.00% |
| `control_mix_gempair` | 295 / 2M 3C | `heft_edf` 60.07 (0.0 s) | 51.75 (CP-SAT 1800 s, 3 seeds spanning 51.750-51.751) | 51.74 | +16.08% | +0.03% |
| `control_mix_quad` | 295 / 4M 6C | `pso` 33.62 (5.0 s) | 33.38 (CP-SAT 120 s) **proven optimal** | 33.38 | +0.71% | +0.00% |
| `saturation_hetero` | 393 / 2M 2C | `pso` 111.50 (8.7 s) | 86.24 (CP-SAT 1800 s) | 79.14 | +29.30% | +8.97% |
| `saturation_quad` | 393 / 4M 6C | `pso` 75.44 (8.8 s) | 73.52 (CP-SAT 120 s) **proven optimal** | 73.52 | +2.62% | +0.00% |

### 5.2 Where each lower bound came from

| spec | critical path | area LP | CP-SAT optimiser bound | CP-SAT decision bound | reported LB |
|---|---|---|---|---|---|
| `scale_ladder_hetero` | 97.16 | 97.16 | 97.16 | 97.16 | **97.16** |
| `scale_ladder_quad` | 97.16 | 97.16 | 97.16 | 97.16 | **97.16** |
| `control_mix_gempair` | 48.74 | 51.74 | 35.30 | UNKNOWN at 51.749 (600 s) | **51.74** |
| `control_mix_quad` | 32.60 | 32.60 | 33.38 | 33.38 | **33.38** |
| `saturation_hetero` | 72.20 | 79.14 | 19.76 | UNKNOWN at 83.0 (900 s) and 89.4 (300 s) | **79.14** |
| `saturation_quad` | 72.20 | 72.20 | 73.52 | 73.51 | **73.52** |

### 5.3 Independent cross-checks

An optimum asserted by one solver against one model is one bug away from
fiction, and the bug `95db5778` fixed was exactly that shape — CP-SAT answering
a wrong model correctly. Three checks, deliberately not sharing code.

**The bound, computed from the workload rather than from any solver.**
`oracle_bounds` reads only the `DecoderContext` — durations, precedence,
releases, windows — in exact float arithmetic. On the whole `scale_ladder`
family its critical-path value equals the objective the schedule actually
achieves, to four decimals, with no solver in the loop at all.

**Every stored schedule re-validated outside `evaluate`.** Precedence,
per-machine no-overlap, release times and windows were all rechecked from the
raw `(start, combination)` arrays. Every schedule quoted here passes: no window
is missed, and the worst window has 0.03 ms of slack (`saturation_hetero` at
120 s) up to 2.24 ms (`control_mix_quad`). The only violations found anywhere
are precedence and overlap slips of at most 5.0e-4 ms, which is the CP-SAT
microsecond grid and nothing else — see §7.

**MOSEK's MILP** (`xpu-rt/mosek_native.py`): a different formulation (big-M
disjunctions with an ordering bit per surviving pair) in a different solver,
pinned to four cores.

| spec | MOSEK bound | MOSEK best integer | wall | our LP bound | best known |
|---|---|---|---|---|---|
| `scale_ladder_hetero` | 97.163363 | 97.163562 | 0.8 s | 97.1634 | 97.163 |
| `scale_ladder_quad` | 97.163363 | 97.163363 | 75.9 s | 97.1634 | 97.163 |
| `control_mix_hetero` | 33.193 | 42.008 | 900 s | 36.375 | 37.246 |
| `control_mix_gempair` | 48.740 | 60.074 | 903 s | 51.745 | 51.750 |
| `saturation_hetero` | 72.310 | 115.832 | 901 s | 79.147 | 86.237 |

The two small rows corroborate: `scale_ladder_quad` closes exactly,
`scale_ladder_hetero` stops at MOSEK's default 1e-4 relative MIP gap and
brackets the optimum in [97.163363, 97.163562]. Both agree with CP-SAT and with
the critical-path bound. `scale_ladder_quad` also cross-checks the
*sibling-hart* handling — it carries the `[P#0]`/`[P#0,P#1]`/`[P#1]`
combinations whose no-overlap the CP-SAT model got wrong before `95db5778`, and
the MILP reaches the same answer through `combinations_overlap` pair tests
rather than per-machine interval lists.

The three large rows say something else. At 295 operations and up the pairwise
big-M model is the wrong tool on *both* sides. On the primal side MOSEK barely
moves off the `heft_edf` schedule it was handed: 60.074 and 115.832 are that
schedule exactly, unimproved after 900 s, and 42.008 on `control_mix_hetero` is
5% below its 44.433 start and still 13% worse than CP-SAT's 37.246 at 60 s. On
the dual side its bound lands at or barely above the critical path — 48.740 on
`control_mix_gempair` is the critical-path value to four decimals.
`oracle_bounds`, in 3–13 s on one core, beats MOSEK's 900 s bound on all three.

**Optimality re-proved as a refutation.** For each spec CP-SAT called OPTIMAL,
the decision procedure of §4.3 was asked separately whether anything reaches one
microsecond below the incumbent. All four came back INFEASIBLE:

| spec | incumbent | refuted at | wall |
|---|---|---|---|
| `scale_ladder_hetero` | 97.163 | 97.162 | 2.0 s |
| `scale_ladder_quad` | 97.163 | 97.162 | 1.2 s |
| `control_mix_quad` | 33.383 | 33.382 | 11.3 s |
| `saturation_quad` | 73.515 | 73.514 | 30.2 s |

This matters most for `control_mix_quad` and `saturation_quad`, where the
combinatorial bound is 2.4% and 1.8% short and CP-SAT's optimality claim is the
only thing between "best found" and "optimal". It now rests on two separate
runs answering two different questions.

**Where the decision procedure failed, it is reported as a failure.** Asked to
refute 51.749 and 51.744 on `control_mix_gempair` (600 s each) and 83.0 and
89.4 on `saturation_hetero` (900 s and 300 s), CP-SAT returned UNKNOWN every
time — neither a schedule nor a proof. The 51.744 attempt is worth singling
out: the area LP refutes that same target in 6 s, and CP-SAT could not confirm
it in 600 s. These runs contribute nothing to the bounds and are recorded so
the table cannot be read as if the solver was never asked.

## 6. The same bound across every spec

The two combinatorial bounds cost seconds, so they were run against every spec
that builds. That turns the whole sweep into an absolute measurement, and it is
the most useful thing in this document. The LB column is the strongest bound
obtained by any means; the "best known" column is the frozen baseline's best
valid row, replaced by the max-effort result on the six specs of §5.

The last column brackets how far the best *valid cheap* heuristic is from the
**true** optimum. The optimum lies in `[LB, best known]`, so the heuristic's
excess over it lies in `[cheap/best_known - 1, cheap/LB - 1]`. Both ends are
facts. Where the bracket collapses, the optimum is known and the excess is
exact; where it is wide, we know the heuristic is at least the low figure off
and cannot yet rule out the high one.

| spec | ops | lanes | best valid cheap heuristic | proven LB | best known | heuristic excess over the true optimum |
|---|---|---|---|---|---|---|
| `scale_ladder_hetero` | 126 | hetero | `heft` 97.16 (0.00 s) | 97.16 | 97.16 | **0% — provably optimal** |
| `scale_ladder_gempair` | 126 | gempair | `heft` 98.59 (0.00 s) | 98.59 | 98.59 | **0% — provably optimal** |
| `scale_ladder_rvvpair` | 126 | rvvpair | `heft` 97.37 (0.01 s) | 97.37 | 97.37 | **0% — provably optimal** |
| `scale_ladder_quad` | 126 | quad | `heft` 97.16 (0.00 s) | 97.16 | 97.16 | **0% — provably optimal** |
| `tight_loop_hetero` | 252 | hetero | — | — | **no valid schedule exists** | n/a |
| `tight_loop_gempair` | 252 | gempair | — | — | **no valid schedule exists** | n/a |
| `tight_loop_rvvpair` | 252 | rvvpair | — | — | **no valid schedule exists** | n/a |
| `tight_loop_quad` | 252 | quad | — | — | **no valid schedule exists** | n/a |
| `control_mix_hetero` | 295 | hetero | `sa` 44.07 (14.56 s) | 36.38 | 37.25 | 18.3% – 21.2% |
| `control_mix_gempair` | 295 | gempair | `heft_edf` 60.07 (0.01 s) | 51.74 | 51.75 | **16.1%** (optimum pinned to <0.05pp) |
| `control_mix_rvvpair` | 295 | rvvpair | `sa` 95.08 (20.00 s) | 81.55 | 91.46 | 4.0% – 16.6% |
| `control_mix_quad` | 295 | quad | `pso` 33.62 (5.04 s) | 33.38 | 33.38 | **0.7%** (optimum pinned to <0.05pp) |
| `saturation_hetero` | 393 | hetero | `pso` 111.50 (8.69 s) | 79.14 | 86.24 | 29.3% – 40.9% |
| `saturation_gempair` | 393 | gempair | `heft_edf` 164.67 (0.02 s) | 117.16 | 156.54 | 5.2% – 40.5% |
| `saturation_rvvpair` | 393 | rvvpair | `heft_edf` 217.56 (0.02 s) | 170.91 | 170.95 | **27.3%** (optimum pinned to <0.05pp) |
| `saturation_quad` | 393 | quad | `pso` 75.44 (8.78 s) | 73.52 | 73.52 | **2.6%** (optimum pinned to <0.05pp) |
| `vint_intro_hetero` | 661 | hetero | `sa` 4019.97 (20.01 s) | 3750.82 | 4019.97 | 0.0% – 7.2% |
| `vint_intro_gempair` | 661 | gempair | `heft_edf` 3754.05 (0.07 s) | 3754.05 | 3754.05 | **0% — provably optimal** |
| `vint_intro_rvvpair` | 661 | rvvpair | `heft_edf` 11068.08 (0.07 s) | 11068.08 | 11068.08 | **0% — provably optimal** |
| `vint_intro_quad` | 661 | quad | `heft` 3750.82 (0.08 s) | 3750.82 | 3750.82 | **0% — provably optimal** |
| `vint_multi_hetero` | 745 | hetero | `heft_edf` 4239.12 (0.05 s) | 3750.82 | 4208.46 | 0.7% – 13.0% |
| `vint_multi_gempair` | 745 | gempair | `heft_edf` 3874.09 (0.09 s) | 3754.05 | 3874.09 | 0.0% – 3.2% |
| `vint_multi_rvvpair` | 745 | rvvpair | `heft_edf` 11263.61 (0.08 s) | 11068.08 | 11263.61 | 0.0% – 1.8% |
| `vint_multi_quad` | 745 | quad | `heft` 3750.82 (0.08 s) | 3750.82 | 3750.82 | **0% — provably optimal** |

## 7. Threats to validity

**The microsecond grid.** CP-SAT works in integer microseconds: durations are
rounded to the nearest microsecond, and so are `min_start` and `max_end`.
Replaying a CP-SAT schedule with the exact float durations shows precedence and
no-overlap slips of up to 5.0e-4 ms per adjacency — measured on every run in
this study, and never larger. Two consequences. First, a CP-SAT objective is
exact only to about `chain depth x 0.5 us`, comfortably under 0.1 ms at these
depths and under 0.1% of every number here. Second, a CP-SAT "proven optimal"
is proven *on that grid*. The combinatorial bounds use exact float durations
and carry no such caveat, so the eight specs where the combinatorial bound
already meets the schedule exactly — all four `scale_ladder`, three of four
`vint_intro`, and `vint_multi_quad` — are optimal with no grid asterisk at all.
The four proved optimal by CP-SAT alone (`control_mix_quad`, `saturation_quad`
and the two `scale_ladder` rows it also closed) carry it.

**The horizon cap.** The payload gives every start variable the domain
`[min_start, horizon]`, with `horizon` the larger of "every operation at its
slowest, run back to back" and "the latest `max_end`". On windowed specs that
can be *smaller* than release-plus-total-work (99.76 ms against 158.36 ms on
`control_mix_gempair`), which makes the model formally a restriction rather
than a relaxation, and a bound from a restriction is not automatically a bound
on the original. It excludes nothing here: any schedule with objective `T` has
every target finishing by `T` and every periodic operation by its own
`max_end <= horizon`, and every `T` in play is far below the horizon.
`oracle_cpsat_decide.horizon_of` asserts `T <= horizon` before each decision so
this cannot silently stop being true.

**Zero transfer costs.** `transfer` is identically zero on every `wl_sweep`
spec, so the critical-path and area bounds lose nothing by ignoring it. On a
workload with real transfer costs the head/tail recursions would need the
*minimum* transfer added to stay valid, and the bounds as written would be
weaker but still correct.

**The area LP is a floor on a floor.** Its windows come from a capped grid plus
a cutting-plane pass. A richer window family, or an energetic-reasoning
argument that credits partial overlap rather than only fully-pinned operations,
would push it higher. Nothing here claims the reported lower bound is the best
obtainable one — only that it is valid.

**Nondeterminism.** CP-SAT ran at `workers=4`, which is not reproducible;
repeated runs of an identical configuration differ. Where several seeds ran the
spread is reported, and it is small next to the gaps being discussed.

**What was excluded, and why.** Every figure above is the best schedule with
`misses == 0`. Excluded by that rule: `heft` at 86.94 ms on
`saturation_hetero` (44 windows missed — it beats the best valid answer by 17%
and is not an answer), `heft` at 54.07 ms on `control_mix_gempair` (14 missed),
`greedy_periodic` and `greedy_reserved` on most windowed specs, and the frozen
baseline's `cpsat` row on `control_mix_quad` (33.38 ms, 2 missed). That last
one is worth a note: a 120 s run here reaches the same 33.38 ms with
`misses == 0`, and an independent replay of it — precedence, per-machine
overlap, releases and windows all rechecked outside `evaluate` — shows every
window met with at least 2.24 ms to spare. The frozen row's two misses are the
microsecond grid landing exactly on `evaluate`'s 1e-6 tolerance, not a real
overrun. It stays excluded either way; the rule is not negotiable, and the
schedule that replaced it is strictly better anyway.

## 8. Answer

**Are the cheap heuristics near-optimal, or is there real headroom?** Both, and
which one holds is now measurable rather than a matter of opinion. Of the 20
schedulable specs, `heft` or `heft_edf` is **provably optimal on 8** at a cost
of 0.00–0.08 s, and on two more a cheap method sits within 2.6% of an optimum
that is itself proven. On the remaining ten there is headroom, and on four of them the cheap
answer is **at least 16% above the optimum** — 27.3% on `saturation_rvvpair`,
against an optimum now pinned to 0.02%, and at least 29.3% on
`saturation_hetero`.

**Size is not the variable.** The 745-operation `vint_multi_quad` is solved to
its proven optimum by `heft` in 80 ms. The 393-operation `saturation_hetero` is
not solved by anything in this repo, and even after 1800 s of CP-SAT its
optimum is only known to lie in a 9%-wide band. Two of the three families where
`heft` is exactly optimal are the two largest in the sweep. Selecting a solver
by operation count is selecting on the wrong axis.

**The variable is which constraint binds.** Sort the 20 by which lower bound is
tight against the best known schedule and the picture is clean:

- **Precedence-dominated** — the critical path alone accounts for 97%+ of the
  best known objective: all four `scale_ladder`, three of four `vint_intro`,
  `vint_multi_quad`, `vint_multi_rvvpair`, `saturation_quad`. The DAG is the
  constraint, not the machines. *`heft` is optimal or within 2%, instantly*, and
  CP-SAT at 1800 s returns the same number. Nothing more expensive can help
  here, and now we can say that instead of suspecting it.
- **Area-saturated** — the machines are full enough that the energetic bound is
  tight: `control_mix_gempair` (bound 51.745 against a best known 51.750, a
  0.010% band, though CP-SAT could not close the last microsecond in 600 s so
  this is *pinned*, not *proven*), `saturation_rvvpair` (170.91 against 170.95),
  `control_mix_hetero` (2.3% apart). The optimum is essentially pinned. But
  *pinned is not the same as solved by the heuristics*: on `control_mix_gempair`
  the best valid cheap answer is 16.1% above that optimum, and on
  `saturation_rvvpair` 27.3% above. Saturation makes the problem easy to
  **bound** and hard to **solve**.
- **Contested** — neither bound is within 8%: `saturation_hetero` (LB/best
  known 0.92 after the long run, 0.79 against the frozen baseline),
  `saturation_gempair` (0.75), `control_mix_rvvpair` (0.89),
  `vint_multi_hetero` (0.89), `vint_intro_hetero` (0.93). Here we know the
  cheap heuristic is at least 4–29% off and cannot yet rule out 41%.

**So: yes, there is real headroom, and it is concentrated.** Four
configurations where a cheap heuristic is 16%, 18%, 27% and at least 29% above
the optimum — `control_mix_gempair`, `control_mix_hetero`,
`saturation_rvvpair`, `saturation_hetero` — are not marginal losses. A
scheduler that closed `saturation_rvvpair` would cut a 217.6 ms workload to
171.0 ms; one that closed `saturation_hetero` would cut 111.5 ms to at most
86.2 ms. And in every one of those cases warm-started CP-SAT already finds most
or all of the gain in 60–1800 s, so the gap is not an artifact of the problem
being intractable; it is the list-scheduling priority rule leaving value on the
table on exactly the instances where the machines, not the DAG, decide the
makespan.

**Lane structure predicts more than lane count, but only through that lens.**
Every `quad` (2+2) spec in the sweep is solved to proven optimality, always
within 2.6% of what a cheap method already had; four machines leave enough
slack that scheduling stops being the hard part. The two-machine configurations
spread across all three regimes depending on the workload, so "gempair is hard"
is false as stated: `scale_ladder_gempair` and `vint_intro_gempair` are exactly
optimal for free, while `saturation_gempair` carries the widest band in the
study.

**The cheap bound beat the expensive solver at its own job.** On
`control_mix_gempair` the area LP proved 51.745 ms in 6 s of one core. CP-SAT,
given 1800 s on four workers at three seeds, reported objective bounds of
32.30, 32.75 and 35.30 — its relaxation never came within 30% of what a
few hundred window constraints establish immediately. Same story on
`saturation_hetero`: 79.15 from a 13 s LP against 19.76 from 1800 s of CP-SAT
and 72.31 from 900 s of MOSEK. Asked instead to *refute* a target — the form of
the question CP-SAT is best at — it returned UNKNOWN at 51.749 and 51.744 on
gempair and at both 83.0 and 89.4 on saturation_hetero; the LP refutes 51.744
in 6 s. If a lower bound is wanted on these instances, `oracle_bounds.py` is the
tool, not `LAST_SOLVE["best_bound"]` and not the MILP relaxation.

**Where the frozen baseline is actively misleading.** Relative-to-best
compresses exactly the cases that matter. On `saturation_hetero`, `heft_edf`
reads 1.109x relative-to-best; against a proven lower bound it is at least
1.464x. On `control_mix_gempair` it reads 1.054x and is in fact 1.161x — the
denominator was itself. The *ranking* the frozen sweep produces is sound; the
*magnitudes* it implies understate the loss by a factor of three to four on
the instances where a better scheduler would pay for itself.

**One finding that is not about solvers at all.** The whole `tight_loop`
family — 4 of the 24 specs that build — admits **no valid schedule whatever**,
on any of the four hardware configurations. It is not that our methods miss
windows on it; the windows cannot be met. The proof is one line and needs no
solver: on `tight_loop_quad` the critical path, with every operation on its
fastest combination and zero resource contention, cannot start operation 230
before 42.05 ms, and that operation's periodic window closes at 10.85 ms. The
other three configurations fail the same way at operations 188, 230 and 251.
CP-SAT independently reports INFEASIBLE in presolve in 0.04 s. The generator's
claim that these periods are "tight but satisfiable" is false for this family,
and every `tight_loop` row in the frozen baseline — eight methods reporting
objectives between 45.9 and 106.9 ms — is a number about schedules that do not
exist. That is a workload-generation bug and it should be fixed before anyone
reads a `tight_loop` figure again.

### What we still do not know

| spec | proven lower bound | best schedule anyone has | the optimum is known to within |
|---|---|---|---|
| `control_mix_gempair` | 51.74 ms | 51.75 ms | **0.03%** |
| `saturation_hetero` | 79.14 ms | 86.24 ms | **8.97%** |
| `saturation_gempair` | 117.16 ms | 156.54 ms | **33.61%** |
| `control_mix_rvvpair` | 81.55 ms | 91.46 ms | **12.16%** |
| `control_mix_hetero` | 36.38 ms | 37.25 ms | **2.39%** |
| `vint_multi_hetero` | 3750.82 ms | 4208.46 ms | **12.20%** |

Closing the open ones needs either a stronger lower bound — energetic reasoning
that credits partial overlap rather than only fully-pinned operations, or a
machine-decomposition relaxation solved exactly — or a better upper bound than
warm-started CP-SAT reaches in half an hour. Both are open. What has changed is
that the question is answerable and quantified, instead of hidden behind a
ratio whose denominator was ourselves.

## 9. Reproducing this

Raw numbers are frozen in `scripts/solver_study/data/oracle_gap.json`
(lower bounds per spec with their certificates, every max-effort run with its
CP-SAT status and bound, the decision refutations, and the MOSEK cross-check).

    export XPURT_CODE_ROOT=<this checkout>
    export XPURT_DATA_ROOT=/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt
    export XPURT_CPSAT_PYTHON=<a venv with ortools>/bin/python
    export XPURT_ORACLE_OUT=<a scratch dir>

Lower bounds (seconds per spec, one core, no solver involved):

    python - <<'PY'
    import sys, os
    sys.path.insert(0, os.environ["XPURT_CODE_ROOT"] + "/scripts/solver_study")
    from wl_sweep_bench import build, DATA
    from schedule_decoder import DecoderContext
    import oracle_bounds as ob
    w, _ = build(f"{DATA}/data/toplevel/wl_sweep/networks_control_mix_gempair.json")
    ctx = DecoderContext(w)
    print("critical path", ob.cp_bound(ctx))
    print("schedulable at all?", ob.window_feasibility(ctx))
    print("area bound", ob.area_bound(ctx, hi=56.80, verbose=True))
    PY

Upper bound at max effort (one spec, one seed, one time limit):

    python scripts/solver_study/oracle_run.py networks_control_mix_gempair 0 1800

Is `T` reachable at all? (INFEASIBLE is a proof that the optimum exceeds it):

    python - <<'PY'
    import sys, os
    sys.path.insert(0, os.environ["XPURT_CODE_ROOT"] + "/scripts/solver_study")
    from wl_sweep_bench import build, DATA
    import oracle_cpsat_decide as ocd
    w, _ = build(f"{DATA}/data/toplevel/wl_sweep/networks_control_mix_quad.json")
    print(ocd.decide(w, 33.382, time_limit=300, workers=4))
    PY

`CPSAT_LOG=1` turns on CP-SAT's own log, which is the only reliable way to see
whether a warm start was adopted ("The solution hint is complete and is
feasible") and how its objective bound moved over the run.

# Decision Gate B: does switching among precomputed schedules retain more utility?

**Answer, in two parts.**

**With the specified control signal (observed input age), no** — and it is worse than
useless: the selector is inadmissible on 3 of 4 contention trajectories because it runs
a schedule that overruns the epoch by 2.7×. The cause is not tuning. The signal is
measured *downstream of the mitigation*, so a mitigation that works destroys the signal
that says whether it is still needed (§4).

**With a signal taken upstream of the mitigation, partly yes.** Offered work for the
coming epoch is monotone in contention and available at the epoch boundary. It restores
safety completely (20/21 threshold settings admissible on every trajectory) and retains
**+2 of 20** offered soft instances on the ramp trajectory — one per epoch spent at the
single burst where the cheap rung both suffices and fits. On trajectories that visit only
the extremes it retains nothing (§4b).

So the value demonstrated here is mostly in the **candidate** and in the **choice of
observable**, not in switching per se. The prize for switching on this workload is
small and bounded, and that bound was computed before the selector was built rather than
discovered after it succeeded.

All numbers at φ = A0 + 20 = 80.546 ms (A0 = 60.546 ms, the measured uncontended
input-age ceiling), 300 ms epoch, greedy solver, compaction and automerge forced off,
FireSim-measured cycles at an assumed 1 GHz. Single seed; schedules verified identical
across seeds where multiple seeds were run.

Reproduce:

```bash
python -m benchmarks.freshness_eval.headroom --deltas 5,20,50
python -m benchmarks.freshness_eval.adaptive --delta 20
python -m benchmarks.freshness_eval.plot_adaptive
```

Figures in `figures/freshness_adaptive/`:

| figure | shows |
|---|---|
| `plot4_candidate_ladder` | the ladder — validity up, soft utility down, monotone at every B |
| `plot5_switching_headroom` | the ceiling: +1 instance, only for targets ≤ 0.833; zero over B≤3 |
| `plot6_selector_timeline` | the selector on `step`: one epoch of a 2.7× overrun, then correct |
| `plot7_signal_saturation` | **the cause** — risk flat at 1.124 across B=1…4 |
| `plot8_signal_comparison` | downstream vs upstream signal per trajectory: every overrun removed |

Plot 7 exists because plot 6 cannot show the cause: `step` visits only B=0 and B=4,
so the flatness across B=1…4 is not in that trajectory's data.

---

## 1. The candidate ladder is real and monotone

| rung | mutations | B=0 | B=1 | B=2 | B=3 | B=4 | soft completed |
|---|---|---|---|---|---|---|---|
| C0 nominal | — | .933 | .633 | .400 | .220\* | .000\* | 0,1,2,3,4 |
| C1 defer12 | `soft_phase_ms=12` | .933 | .900 | .867 | .833 | .161\* | 0,1,2,3,3 |
| C2 +admit2 | `+admit_cap=2` | .933 | .900 | .867 | .867 | .867 | 0,1,2,2,2 |
| C3 +admit1 | `+admit_cap=1` | .933 | .900 | .900 | .900 | .900 | 0,1,1,1,1 |

`output_valid_rate`. **\* = the schedule overruns the 300 ms epoch**, so that rate is
computed over a longer trace with a different invocation count (41 at B=3, 64 at B=4,
against 30) and is **not comparable**. An overrun is itself a failure; the rate is not
quotable.

Strictly monotone in both directions at every burst: validity rises up the ladder, soft
utility falls. All three protective rungs pass the candidate-set gate.

**Quotable improvement**, restricted to the epoch-comparable bursts B=0–2: C1 gains
**+0.244** mean output-valid over nominal (+0.233 at φ=A0+5, +0.178 at φ=A0+50). Above
B=2 no comparison against nominal is valid, because the baseline's own schedule does not
fit the epoch there.

The `.933` ceiling is structural, not a policy limit: 28/30: two consumer invocations
fire before the first producer can finish even on an idle machine.

## 2. The opportunity for switching is at most one soft instance

`headroom.py` computes an upper bound assuming a selector with perfect observation, free
instantaneous switching, and no hysteresis — so no real selector can exceed it.

- **Over B=0–3: exactly zero gain at every validity target.** A single static rung is
  optimal everywhere.
- **Over B=0–4: exactly one soft instance of 10 offered** (3/3 rather than 2/3 at B=3),
  and only for targets ≤ 0.833. Above that, zero again.

The gain exists *only* because B=4 disqualifies C1 as a static choice via epoch overrun.
Remove B=4 from the operating range and the entire opportunity disappears.

## 3. The deployable selector realises none of it

Hysteretic selector on `risk = observed_max_age / φ`, entry 0.85/1.10, exit 0.70/0.95,
min-residency 1, cooldown 1, reacting to the previous epoch (the deployable case).

| trajectory | admissible | adaptive | best admissible static at ≥ adaptive's validity | gain |
|---|---|---|---|---|
| ramp `0,1,2,3,4,4,3,2,1,0` | yes | .907, 8/20 | C3: .907, 8/20 | **+0** |
| step `0,0,4,4,4,4,0,0,0,0` | **no** | — | — | — |
| oscillate `0,4,0,4,…` | **no** | — | — | — |
| sustained `4×10` | **no** | — | — | — |

On the one trajectory where it is admissible it **exactly ties** the most conservative
static. On the other three it is inadmissible: it runs C1 during a B=4 epoch, whose
schedule takes 805.9 ms against a 300 ms budget. The `oracle_contention_aware` strategy —
full knowledge of the current burst — also ties C3 on `ramp` (.907, 8/20).

The comparison is matched on validity deliberately. Comparing adaptive against the
highest-utility static regardless of validity would penalise it for being more
conservative, which is not the question asked.

## 4. Why it failed: the observable saturates

`max_input_age`, the selector's only input, measured per rung and burst (risk = age/φ):

| rung | B=0 | B=1 | B=2 | B=3 | B=4 |
|---|---|---|---|---|---|
| C0 nominal | 0.75 | 1.25 | 1.62 | 1.62\* | 6.21\* |
| C1 defer12 | 0.75 | 1.12 | 1.12 | 1.12 | 5.10\* |
| C2 +admit2 | 0.75 | 1.12 | 1.12 | 1.12 | 1.12 |
| C3 +admit1 | 0.75 | 1.12 | 1.12 | 1.12 | 1.12 |

Under **any** protective rung the signal is **flat at 1.124 for B=1, 2, 3 and 4 alike**.
The selector cannot distinguish 65% offered load from 131%. The only values that
discriminate B=4 are starred — belonging to schedules that overrun — and are therefore
observable only *after* the overrun has already occurred.

**The mitigation masks the disturbance it is mitigating.** Protection works by pinning the
worst-case input age to one missed producer period; having done so, the age stops
reporting how much contention was offered.

This is not a threshold-tuning problem, and the obvious first guess is wrong. A threshold
sweep on `step` shows entry risks **≤ 0.752 do keep it admissible** — but 0.752 *is* the
B=0 observation, so such a selector escalates at zero contention, switches once, never
returns, and reproduces static C3's numbers exactly (.920, 4/16). It becomes safe by
ceasing to adapt. Every threshold above 0.752 takes one full epoch of the 806 ms overrun.
No setting both adapts and survives a step to B=4.

Secondarily, the failure mode of the lower rung is a cliff, not a slope: C1 does not
degrade gracefully at B=4, it overruns by 2.7×. Reactive control must absorb one epoch of
whatever it was late to avoid, and one epoch of that is fatal.

## 4b. An upstream signal fixes the safety failure and part of the utility one

The saturation diagnosis makes a prediction, so it was tested rather than left as an
explanation. Replacing the signal with the **offered soft work for the epoch about to
be scheduled** — available at the boundary because admission control happens there —
gives a monotone observable: offered gemmini utilisation 42.7 / 65.1 / 87.5 / 109.9 /
132.3 % for B = 0…4.

This is not an oracle. It is the request count, not the outcome: it says nothing about
the resulting validity, makespan or input age, all of which still depend on the
schedule chosen. (`oracle_contention_aware` is the oracle — it reads every candidate's
measured validity at that burst and picks the winner.)

Because the signal is available *at* the boundary rather than one epoch later, it
removes the lag as well, so this experiment addresses both failures at once and cannot
attribute the change to either alone. Separating them would need a workload whose low
rung degrades gracefully; this one's fails by a 2.7× overrun.

Thresholds are **swept, not chosen** — 21 settings including both degenerate ends —
because picking the value that separates the measured outcomes would be fitting a
one-parameter model to five points and reporting the fit as a result.

| trajectory | downstream selector | upstream (best safe threshold) | vs best admissible static at that validity |
|---|---|---|---|
| ramp | +0, and ties the most conservative static | valid .880, soft **16/20** | **+2** |
| step | **INADMISSIBLE** | valid .907, soft 8/16 | +0 |
| oscillate | **INADMISSIBLE** | valid .900, soft 10/20 | +0 |
| sustained | **INADMISSIBLE** | valid .867, soft 20/40 | +0 |

**Safety is fully restored:** 20 of 21 threshold settings are admissible on every
trajectory. The single inadmissible one is the degenerate "never escalate", which *is*
static C1 and reproduces its overrun.

**Utility is gained only where the trajectory spends time at intermediate contention.**
The +2 on `ramp` is exactly +1 in each of its two B=3 epochs and 0 in all eight others —
verified per epoch. B=3 is the only burst where the cheap rung both suffices and fits the
epoch. `step`, `oscillate` and `sustained` visit only B∈{0,4}, so they gain nothing.

This also restates the bound's units: the headroom figure of +1 is **per visit to B=3**,
not per trajectory. A ramp gains +2; a workload that never reaches B=3 gains 0.

**The honest caveat:** the winning threshold is *only* escalate-at-4. Escalating earlier
sheds work that did not need shedding (escalate-at-3 → 10/20); escalating later
reproduces the overrun. The optimum is a single point on this grid, not a plateau — a
two-point calibration, and far weaker evidence than the deferral plateau's 8 ms width.

## 5. Causes, in the plan's terms

1. **The protective mechanism is nearly free.** Deferral costs zero soft utility, so a
   permanently conservative rung is barely worse than the best per-burst choice. This
   bounds the prize at one instance before any selector exists.
2. **The control signal is uninformative** because it is measured downstream of the
   mitigation.
3. **The candidate set is coarse.** Four rungs across five contention levels, with C2 and
   C3 differing only in `admit_cap`.
4. **Epoch-level reaction is too slow for a cliff-shaped failure**, and this workload's
   lower rung fails by cliff.

Not supported as causes: selector implementation (it escalates correctly on the very next
epoch, and overhead is sub-microsecond per decision), and threshold placement (swept; see
above).

## 6. What this does and does not establish

**Does:** on this workload, with this candidate set and this observable, epoch-level
adaptive switching retains no more noncritical utility than the best safe static schedule,
and is actively unsafe under step changes in contention. The freshness divergence itself is
real and large (Gate A: nominal loses up to 0.733 of its output validity while meeting
100% of local deadlines), and a *static* protective schedule recovers most of it for free
(+0.244 at B≤2). The value demonstrated in this pass is in the **candidate**, not in the
**switching**.

**Does not:** show that freshness-aware adaptation is useless in general. The specific
blocker was a saturated observable, and §4b confirms the diagnosis by fixing it: an
upstream signal makes the selector safe everywhere and profitable where the trajectory
spends time at intermediate contention. What remains true is that the *size* of the prize
on this workload is small, because the protective mechanism is nearly free — to make
adaptation matter you need a workload where protection costs something substantial, so
there is real utility to reclaim. That is a workload-design question and the right next
target.

## 7. Limitations specific to this gate

- **Epoch composition, not simulation.** Trajectories are stitched from per-epoch measured
  cells, which assumes no work is in flight at a boundary. Verified for the admissible
  cells (makespan ~290.5 ms in a 300 ms epoch); for the overrunning cells the assumption
  fails, which is exactly why those strategies are reported as inadmissible instead of
  scored.
- **One seed, and the reason is structural rather than expedient.** `seed` reaches the
  config only as `scheduler.random_seed`, whose three consumers in this tree are
  synthetic runtime generation (unreachable here — the sweep runs profiled durations
  with `strict=True`), the `random_list` scheduler (never selected), and CP-SAT (which
  hardcodes 42). The schedulers used are deterministic. So five seeds produce five
  identical schedules, and reporting a seed band from them would imply a robustness
  check that did not happen. The robustness axes here are B and φ. Pinned by
  `test_sweep_integrity.SeedIsNotAVarianceSource`, and confirmed empirically in
  `results/freshness_seeds/` (5 seeds × 3 bursts × 2 policies, schedule digests
  compared). An earlier draft of this note claimed seed never reaches the config at
  all, which is false.
- **Four fixed trajectories**, chosen in advance and pinned by a test so they cannot be
  tuned until adaptive wins.
- Freshness is imposed and evaluated analytically, not observed: no DroNet→MLP dataflow
  exists at any layer, so `producer_instance_consumed` is inferred from schedule
  timestamps. Unchanged from Gate A.

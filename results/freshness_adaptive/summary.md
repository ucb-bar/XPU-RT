# Decision Gate B: does switching among precomputed schedules retain more utility?

**Answer: no, on this workload. The mechanism has not demonstrated its value, and the
reason is identifiable and measurable rather than a tuning failure.**

All numbers at φ = A0 + 20 = 80.546 ms (A0 = 60.546 ms, the measured uncontended
input-age ceiling), 300 ms epoch, greedy solver, compaction and automerge forced off,
FireSim-measured cycles at an assumed 1 GHz. Single seed; schedules verified identical
across seeds where multiple seeds were run.

Reproduce:

```bash
python -m benchmarks.freshness_eval.headroom --deltas 5,20,50
python -m benchmarks.freshness_eval.adaptive --delta 20
```

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
blocker is a saturated observable. A signal taken *upstream* of the mitigation — offered
queue depth, admitted-vs-offered soft count, or the producer's own start-time slack — does
not saturate, and is the obvious next thing to try. That is a design change rather than a
retune, and it is out of scope here.

## 7. Limitations specific to this gate

- **Epoch composition, not simulation.** Trajectories are stitched from per-epoch measured
  cells, which assumes no work is in flight at a boundary. Verified for the admissible
  cells (makespan ~290.5 ms in a 300 ms epoch); for the overrunning cells the assumption
  fails, which is exactly why those strategies are reported as inadmissible instead of
  scored.
- **One seed.** Schedules were verified identical across seeds for the cells that were
  swept with more than one, but the 5-seed requirement is not yet met.
- **Four fixed trajectories**, chosen in advance and pinned by a test so they cannot be
  tuned until adaptive wins.
- Freshness is imposed and evaluated analytically, not observed: no DroNet→MLP dataflow
  exists at any layer, so `producer_instance_consumed` is inferred from schedule
  timestamps. Unchanged from Gate A.

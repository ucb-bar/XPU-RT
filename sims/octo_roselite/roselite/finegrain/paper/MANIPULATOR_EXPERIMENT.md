# The manipulator experiment: scheduling a VLA policy on a heterogeneous SoC

*Full working writeup. Every number here is computed from the artefacts named in
§10 and was checked against them; this is intended as raw material to condense,
not as final prose.*

---

## 1. What the experiment asks

A vision-language-action policy running on an embedded SoC does not produce
actions instantaneously or on demand. It produces them **late** (an action
reflects an observation captured some milliseconds ago) and **intermittently**
(a new action arrives only every so often). Both properties are set by how the
network is scheduled across the chip's compute engines, not by the network
itself.

The question is whether that scheduling choice reaches the robot. Concretely:

1. Does a better schedule produce a better robot, measurably, end to end?
2. If so, through which of the two knobs — **observation latency** or **release
   period** — and does the answer depend on the task?
3. Can the best schedule be picked from properties of the schedule alone, or
   does it require closing an outer loop through the robot?

The experiment answers these by taking real schedules measured on a real SoC,
replaying their exact timing inside a physics simulator, and measuring what the
robot achieves.

---

## 2. The platform being scheduled

**Model.** Octo-small-1.5, INT8, **27.0 M parameters on-device**: a SmallStem16
convolutional stem, a 12-layer transformer trunk at width d=384, and the action
head. The frozen T5-base text encoder (109.6 M parameters, 80% of the full
model) runs **off-device**: the task instruction is fixed per episode, so its
embedding is precomputed once and reused. This was verified bit-exact — maximum
absolute error on the readout is 0.0. There is no ViT-B vision tower in this
configuration; the stem is the whole visual front end.

**Target.** Qualcomm QRB5165, scheduled across three compute engines: CPU, DSP
and HTA.

**Partition.** The `mix3` partition splits the network into **52 single-graph
contexts** — 26 CPU, 14 DSP, 12 HTA — executed as **710 dispatches** with **zero
context evictions**. Eviction-free execution matters: a context swap would inject
a latency spike that the periodic schedule cannot absorb.

**What a schedule is, from the robot's point of view.** The simulator receives
exactly two numbers per schedule:

| quantity | meaning |
|---|---|
| **observation latency** | age of the observation the action was computed from, at the moment the action is applied |
| **release period** | interval between successive actions becoming available |

Everything else about the schedule — which layer runs on which engine, in what
order, with what overlap — reaches the robot only through those two numbers. This
is what makes the study tractable and also what makes §8's result possible.

**The three headline schedules, all measured on the board:**

| schedule | latency | period | actions delivered in a 700 ms window |
|---|---|---|---|
| QNN CPU baseline (`mono`) | 677 ms | 677 ms | 1 |
| XPU-RT serial | 250 ms | 250 ms | 2 |
| XPU-RT pipelined (`p150/w300`) | 283 ms | 144 ms | 3 |

---

## 3. The robots and the tasks

Four tasks from SIMPLER-env, across two embodiments. The embodiment distinction
turns out to be the single most important variable in the whole study.

### 3.1 WidowX 250 S — `eggplant in basket`, `spoon on towel`

A 6-DOF arm built from ROBOTIS Dynamixel servos (XM430-W350 on waist, shoulder
×2, elbow ×2, forearm roll and wrist angle; XL430-W250 on wrist rotate and the
gripper). Simulated at **500 Hz** physics with a **40 ms actuation grid** —
a new command can be consumed every 40 ms.

| task | horizon | what success requires |
|---|---|---|
| put eggplant in basket | 600 ticks / **24.0 s** | grasp a deformable object and place it in a basket |
| spoon on towel | 300 ticks / **12.0 s** | grasp a small rigid object and place it on a target |

### 3.2 Google robot (Everyday Robots) — `pick coke can`, `close drawer`

Simulated at **513 Hz** physics but with a **333 ms actuation grid** — a command
is consumed only every 333 ms, roughly **8× coarser than the WidowX**.

| task | horizon | what success requires |
|---|---|---|
| pick coke can | 720 ticks / **28.8 s** | grasp and lift a rigid cylinder |
| close drawer | 1017 ticks / **39.6 s** | push a drawer from 200 mm open to ≤50 mm |

### 3.3 The saturation ratio, which explains most of what follows

Define **inferences per command** = (actuation interval) / (release period). It
says how many fresh actions the policy produces per action the robot can
actually consume.

| embodiment | actuation grid | at 125 ms period | at 685 ms period |
|---|---|---|---|
| WidowX | 40 ms | **0.32** | 0.06 |
| Google robot | 333 ms | **2.67** | 0.49 |

**The WidowX is never saturated.** Even the fastest schedule in the study
supplies only about one action for every three the arm could take, so every
improvement in cadence is consumed.

**The Google robot saturates at any competitive schedule.** At a 125 ms period
the policy produces 2.67 actions for every one the robot can use; 63% of them are
computed and discarded. Improving cadence past ~333 ms buys nothing, because
there is nowhere for the extra actions to go.

### 3.4 One structural property of failure, which shapes every metric

**A failed episode always runs to the full horizon.** Across the finished sweep's
**26,203 failed episodes** the tick count equals the task horizon with *zero*
variance — egg 600, spoon 300, coke 720, drawer 1017. There is no early
termination on failure in SIMPLER-env.

Consequences, all of which matter later:
* Mission time is only defined on successes, so it is *conditioned* and biased
  downward for arms that fail often (they are timed only on the episodes they
  could still win).
* Energy per episode conflates power with duration: a failing arm pays idle draw
  for the whole horizon. §6.4 separates the two.
* An episode-length contrast between a good and a bad schedule cannot be
  manufactured by picking a different episode.

---

## 4. The schedules evaluated

### 4.1 The curated ladder (9 arms, deep seeding)

Nine schedules spanning the achievable space, from an unreachable reference to
the un-scheduled baseline. Each named by its measured `latency / period` in ms.

| arm | latency | period | what it is |
|---|---|---|---|
| `ideal` (`lat0`) | 0 | 200 (WidowX) / 333 (Google) | zero-latency reference, *not* a ceiling — see §4.2 |
| `pipe110` | 385 | 111 | deepest pipelining: fastest cadence, **highest latency in the study** |
| `p105/300` | 259 | 125 | best all-round pipelined schedule |
| `p130/275` | 272 | 130 | |
| `p150/300` | 282 | 150 | |
| `pipe200` | 261 | 219 | shallow pipelining |
| `serial` | 283 | 283 | no overlap: period = latency |
| `fp32` | 555 | 555 | un-quantised, CPU only |
| `cpu int8` | 685 | 685 | **the QNN CPU baseline** |

Seeding: **30 seeds × 24 episodes** on eggplant, spoon and close drawer for the
six arms carried in the main figure; 10 seeds elsewhere.

### 4.2 Two naming traps worth stating explicitly

**The `ideal` arm is not an upper bound.** It has zero observation latency but
runs at a **200 ms release period** on the WidowX and 333 ms on the Google robot
— it is mid-ladder on cadence. It is a reference for *latency*, not a ceiling for
*performance*, and §7.3 shows a scheduled arm beating it on `pick coke can`.

**A schedule's name encodes its period, never its latency.** The two coincide
only for `serial`, `fp32` and `cpu int8`, where nothing overlaps. `pipe110` is
the cautionary case: named for a 111 ms period while paying **385 ms** of
latency, more than `serial`'s 283 ms. Deep pipelining buys cadence with latency.

### 4.3 The full plane (44 operating points, broad sweep)

The CP-SAT scheduler was run over a **12 × 9 grid** of (release period requested,
deadline window) = 108 cells, warm-started from a HEFT/EDF seed.

| outcome | cells |
|---|---|
| `OPTIMAL` | 60 |
| `FEASIBLE` | 11 |
| `INFEASIBLE` — no schedule exists | 25 |
| `UNKNOWN` — none found inside the budget | 12 |

The 71 solved cells collapse to **44 distinct (latency, cadence) operating
points**; the other 27 return a schedule identical to one already measured. Past
roughly a 220 ms period the deadline window stops binding — the schedule already
finishes inside it — so widening the window changes nothing. Of those 27, **26
are bit-identical to their twin at full float precision**, and the 27th differs
by 0.042 ms, which the pipeline's 0.1 ms formatting erases. They are the same
command, so they are propagated rather than re-simulated.

**Sweep size: 44 arms × 4 tasks × 10 seeds × 24 episodes = 42,240 episodes.**

### 4.4 Why the warm start matters

A greedy scheduler returns a schedule for all 108 cells, but those collapse to
only **12 distinct operating points**, spanning 233–353 ms of latency. The
warm-started CP-SAT solver solves fewer cells (71) but yields **44 distinct
operating points** spanning 233–396 ms.

**The greedy scheduler does not fail to schedule — it fails to differentiate.**
It produces a schedule everywhere and the same handful of schedules everywhere.
Search richness, not feasibility, is what the outer loop in §8 needs.

---

## 5. What is plotted, and why

### 5.1 Success rate

Fraction of 24 episodes completed successfully, computed per seed, then
aggregated across seeds. The per-seed rate is the unit of analysis because it is
the unit the intervals are taken over; a per-episode success is 0 or 1 and
carries no distribution.

### 5.2 Actuator energy — and why the obvious metric was wrong

This took two corrections, both worth recording.

**Correction 1: the original channel measured the wrong thing.** The first energy
metric integrated `robot.get_qf()`, which in this SAPIEN build is
`compute_passive_force(external=False)` — gravity plus Coriolis only. It is blind
to contact, to stall, and to drive torque *by construction*. The replacement
reconstructs the PD drive torque PhysX actually applies, per substep:

```
tau_drive = clip(K·(q_tgt − q_post) + D·(v_tgt − v_post), ±force_limit)
```

validated against the articulation's own equation of motion `M(q)q̈ = τ_drive` to
**0.024 N·m**. Direction was preserved by the correction; magnitude was
understated by roughly 17×.

**Correction 2: the corrected channel is still not joules.** The simulated PD
torque is not physical. On `egg/ideal` it runs **148 N·m rms** against **1.94
N·m** of gravity and Coriolis on the same trajectory — a factor of **76** — while
saturating 20% of substeps under a `force_limit` of 200 N·m, on a shoulder whose
two XM430-W350 servos stall at **8.2 N·m** combined. The simulator's gains are
tuned for tracking, not fidelity. Converting that integral at face value yields
**1.6 kW – 84 kW** against a 60 W supply.

**What is plotted instead** is a quasi-static lower bound, applied globally and
per episode:

```
E_J(episode) = c · ∫Σ τ_grav² dt  +  P_idle · T
```

with, for the WidowX, `c = 0.827 W/(N·m)²` and `P_idle = 4.61 W`. Both are
*derived from published ROBOTIS data*, not fitted: at stall, back-EMF vanishes,
so `R = V/I_stall` and `K = τ_stall/I_stall` close exactly from the published
stall torque/current pair (XM430-W350: 4.1 N·m at 2.3 A, 12 V). `P_idle` is the
published standby draw of all nine servos: 7 × 40 mA + 2 × 52 mA at 12 V.

For the Google robot there is no published actuator data at all — the URDF's
uniform `effort="10.0"` on wheels and fingers alike is a placeholder. Its
constant is therefore **balance-matched**: set so the arm has the same
dynamic/idle ratio as the WidowX (0.810) at a literature-anchored 40 W idle,
giving `c = 0.1321 W/(N·m)²`.

**Crucially, the plotted ratios are invariant to all of this.** Scaling `c`
scales every schedule within a workload equally, so relative energy is unchanged
by the calibration; only absolute joules depend on it. That is why the scatter is
normalised per workload and why the ratios can be quoted while the joules carry
a caveat.

### 5.3 Mission time

Completion time on successful episodes. Reported but not headlined, because §3.4
makes it structurally biased: slow arms are timed only on the episodes they still
win, so the true gap is *wider* than measured.

---

## 6. Result 1 — pipelining, and where its gain comes from

### 6.1 The mechanism

The three measured schedules differ in how much of the network is in flight at
once:

* **CPU baseline**: one engine, one inference at a time. 677 ms latency, 677 ms
  period, **1 action per 700 ms**.
* **Serial**: work distributed across CPU/DSP/HTA but no overlap between
  successive inferences. Latency drops to 250 ms and, because a serial chain
  cannot start the next inference until this one ends, the period drops with it.
  **2 actions per 700 ms.**
* **Pipelined**: successive inferences overlap across engines. Latency 283 ms —
  *worse* than serial — but the period drops to 144 ms. **3 actions per 700 ms.**

The pipelined schedule is the clearest statement of the trade: it accepts 33 ms
more latency than serial in exchange for halving the period.

### 6.2 What it buys the robot (curated ladder, WidowX, n=30)

| arm | period | eggplant | spoon |
|---|---|---|---|
| `ideal` | 200 | **54.9%** | **47.8%** |
| `p105/300` | 125 | 52.4% | 43.8% |
| `p150/300` | 150 | 49.4% | 37.2% |
| `pipe200` | 219 | 32.5% | 24.9% |
| `serial` | 283 | 19.3% | 13.2% |
| `cpu int8` | 685 | **1.7%** | **1.4%** |

From the un-scheduled baseline to the best pipelined schedule: **1.7% → 52.4% on
eggplant** and **1.4% → 43.8% on spoon** — a 31× and 31× increase in completed
tasks. Against `serial`, pipelining is worth **+33.1 points** on eggplant and
**+30.6** on spoon, purely from the cadence the overlap buys.

### 6.3 The gain is cadence, not latency

Across the full 44-point plane the two knobs separate cleanly, and they point
opposite ways:

| task | ρ(period, success) | ρ(latency, success) |
|---|---|---|
| eggplant | **−0.830** | +0.353 |
| spoon | **−0.843** | +0.315 |

Success is strongly, negatively associated with *period* and only weakly and
*positively* associated with latency. The positive latency correlation is not a
finding that staleness helps; it is the plane's geometry showing through. Across
the 44 points **Spearman(latency, period) = −0.62** — they are a scheduler Pareto
front, not a factorial grid, so an arm buys a shorter period by paying
observation age. Regressing success on latency alone reads backwards. Every
figure therefore plots period and carries latency separately.

`pipe110` is the cleanest single demonstration: it has the *worst* latency in the
study (385 ms, worse than serial's 283) and the best cadence (111 ms), and it
scores **45.4%** on eggplant against serial's **19.3%**.

### 6.4 What the energy gain actually is

Calibrated energy per episode rises monotonically down the ladder:

| arm | eggplant | spoon |
|---|---|---|
| `ideal` | 117 J | 86 J |
| `p105/300` | 133 J | 89 J |
| `serial` | 189 J | 102 J |
| `cpu int8` | 200 J (**1.71×**) | 123 J (**1.42×**) |

But decomposing the ratio into power × duration is more informative than the
ratio itself:

| task | duration | × power | = energy |
|---|---|---|---|
| eggplant | ×1.51 | ×1.09 | ×1.66 |
| spoon | ×1.06 | ×1.21 | ×1.29 |

**On eggplant, most of the baseline's extra energy is time, not effort.** It
fails, so it runs to the 24 s horizon instead of finishing at 15.9 s, and pays
idle draw for the extra 8 seconds. Spoon is the opposite balance: its horizon is
short, so duration barely moves and the effect is genuinely power (×1.21).

In absolute terms the actuation component is 3.03 W on `ideal` against a 4.61 W
idle floor, rising to 3.75 W on `cpu int8` — the schedule moves actuator power by
**0.72 W on eggplant and 1.80 W on spoon**. Real, but small; the honest headline
is that scheduling mostly saves energy *by finishing*.

### 6.5 Why the per-episode energy distribution is bimodal

On eggplant, per-episode energy separates by outcome: failures pile at ~1e6
(uncalibrated N²m²s) while successes sit at 1e4–1e5. **The expensive episodes are
the ones that flail without finishing.** Conditioning energy on success would
therefore hide precisely the effect being measured, which is why energy is
reported unconditioned.

---

## 7. Result 2 — the flow helps difficult tasks, not throughput-bound ones

### 7.1 The contrast

The same six schedules, same seeding, on the actuator-saturated task:

| arm | close drawer | vs `ideal`, paired |
|---|---|---|
| `ideal` | 40.0% | — |
| `p105/300` | 38.9% | −1.11 pt, p=0.523 |
| `p150/300` | 39.9% | −0.14 pt, p=0.944 |
| `pipe200` | 41.0% | +0.97 pt, p=0.623 |
| `serial` | 41.8% | +1.81 pt, p=0.331 |
| `cpu int8` | 36.8% | −3.19 pt, p=0.067 |

**No schedule differs significantly from the zero-latency reference.** At n=30,
with a per-arm SEM of 1.24 points, this is a *well-powered* null rather than an
underpowered one — the extension from n=10 to n=30 tripled the seeds and revealed
no structure, which is the informative outcome. Energy is flat to **1.00×**.

The one thing worth watching is `cpu int8` at p=0.067, hinting the un-scheduled
baseline may be marginally worse even here. Not significant, not claimed.

### 7.2 Why: the actuator, not the task

The mechanism is §3.3. At a 333 ms actuation grid the Google robot consumes one
action per 333 ms. Every schedule from `pipe110` (111 ms) through `serial`
(283 ms) supplies actions at least as fast as that, so they are
**indistinguishable to the robot** — the surplus is computed and discarded. Only
`cpu int8` at 685 ms actually starves the actuator, and that is exactly where the
one marginal effect appears.

A second, independent signature of saturation: **21.9% of close-drawer episodes
integrate to near-zero drive torque** — the arm never moves for the full 1017
ticks — and **none of those 1,749 episodes ever succeed**. The fraction is
schedule-dependent (ρ(period, zero-fraction) = −0.56) but the success rate is
not, because the policy's failure mode there is not a timing failure.

### 7.3 The result is not "google tasks are insensitive"

`pick coke can`, on the same embodiment and the same actuation grid, does
respond — but weakly and non-monotonically:

| arm | coke can |
|---|---|
| `ideal` | 37.9% |
| `pipe110` | **46.3%** |
| `p130/275` | 44.6% |
| `serial` | 35.4% |
| `fp32` | 15.4% |
| `cpu int8` | **10.4%** |

The degradation at the slow end is real (10.4% vs 37.9%), so scheduling *does*
reach this task. But `pipe110` — the worst-latency, best-cadence arm — **beats
the zero-latency reference by 8.4 points**, and the middle of the ladder is
essentially flat. Its across-arm resolution on the plane is 1.30× the per-arm
SEM, against 2.36× and 2.57× for the WidowX tasks.

**The honest framing** is therefore not "the flow helps hard tasks and not easy
ones", but: *the flow helps in proportion to how much of its output the actuator
can absorb.* The WidowX consumes everything it is given and shows a clean
monotone response; the Google robot discards most of it and shows a compressed,
noisy one.

---

## 8. Result 3 — the outer loop, and why analysis alone is not enough

### 8.1 The problem

Given 44 feasible operating points, which is best? The schedule metrics
themselves — latency and period — cannot answer it, because they are in
**direct competition** and the exchange rate depends on the task:

* Shorter period requires deeper pipelining, which **increases** latency
  (ρ(latency, period) = −0.62 across the plane).
* Which of the two dominates is task-dependent and *changes sign*:
  ρ(period, success) is **−0.830** on eggplant but **+0.461** on close drawer.
* The optimum is not at an extreme of either axis. Best-by-success points:

| task | best schedule | period | latency | success |
|---|---|---|---|---|
| eggplant | `g130_305` | 130 ms | 297 ms | 50.8% |
| spoon | `g130_275` | 131 ms | 271 ms | 44.2% |
| coke can | `g200_260` | 200 ms | 246 ms | 48.8% |
| close drawer | `g283_260` | 283 ms | 233 ms | 45.0% |

The two WidowX tasks peak at ~130 ms period — the fast end. **Both Google tasks
peak at the slowest end of the plane**, 200 and 283 ms. A single scalar objective
over (latency, period) cannot produce both answers.

### 8.2 What a purely analytical objective would get wrong

Any static cost function must commit to a weighting of latency against cadence
before seeing the robot. Three specific ways that fails here:

1. **Minimise latency** → picks `g220_260`-class points at 233 ms latency. Right
   for close drawer, badly wrong for eggplant, where the best point pays 297 ms.
2. **Minimise period** → picks `pipe110` at 111 ms. Right for coke can, wrong for
   the WidowX tasks, where 130 ms with lower latency beats it.
3. **Any fixed linear trade-off** → cannot produce a sign change in
   ρ(period, success) between embodiments, and so must be wrong on one of them.

The scheduler's own metrics also do not rank the points usefully: makespan,
utilisation and dispatch count are all monotone in pipeline depth, whereas
success is not.

### 8.3 What the outer loop does

The loop is: **enumerate the feasible schedule space → simulate each point end to
end → rank by the robot's own metric.** The measured plane is the ranking
function, replacing an analytical objective that cannot be written down.

This requires the search to be rich enough to be worth ranking, which is where
§4.4 matters: greedy scheduling offers 12 distinct points, warm-started CP-SAT
offers 44. It also requires the end-to-end measurement to be trustworthy, which
is where the seeding and the replication checks matter (§9).

### 8.4 Evidence that the loop is measuring signal, not noise

* **Independent replication.** The plane's success surface was measured twice,
  with different seed blocks and a different harness version. Pearson r =
  **+0.881** on eggplant and **+0.870** on spoon, offsets ≤1.0 point. Coke
  (+0.342) and drawer (+0.385) replicate weakly — as expected, since a flat
  surface has little structure to correlate.
* **Resolution.** Across-arm spread over per-arm SEM: eggplant **2.36×**, spoon
  **2.57×**, coke 1.30×, drawer 0.93×. The two WidowX surfaces are structure; the
  drawer surface is noise, and the figure says so.
* **The ranking is stable where it is resolved and unstable where it is not** —
  which is the correct behaviour for a measurement-driven loop, and is why the
  loop must report its own resolution alongside its ranking.

### 8.5 Success and energy do not trade

A natural worry about an outer loop optimising success is that it silently buys
success with energy. It does not:

| task | ρ(success, energy) | Pareto front, success-vs-energy |
|---|---|---|
| eggplant | **−0.959** | **1 of 44** |
| spoon | **−0.890** | **1 of 44** |
| coke can | −0.882 | 3 of 44 |
| close drawer | −0.719 | 2 of 44 |

On both WidowX tasks a **single operating point is best on both axes at once** —
a front of one is not a front. Choosing for success costs nothing in energy. (The
front sizes are recomputed on the *calibrated* energy axis; on the raw τ² axis
they are 1, 1, 4 and 5, so the conclusion does not depend on the calibration.)

One caveat on interpretation: because calibrated energy includes `P_idle × T`,
and failing episodes run to the horizon, part of this correlation flows through
duration rather than through torque. The *decision* it supports is unaffected —
the same schedule wins either way — but the mechanism is partly "finishing
sooner", not only "working less hard" (§6.4).

---

## 9. Threats to validity, and what is assumed

**Simulated torque is not physical** (§5.2). Every absolute joule figure is a
modelled lower bound. Ratios are safe; magnitudes are not.

**Google-robot energy constants are balance-matched, not sourced.** Everyday
Robots publishes nothing. The ratios are invariant to the choice, but absolute
joules for coke and drawer are proxy-determined.

**Mission time is conditioned on success** and biased downward for weak arms
(§3.4). The measured gap understates the true one.

**Harness nondeterminism.** ~20% of byte-identical invocations diverge, which is
why noise bands are empirical and seed-matched rather than analytic.

**The `ideal` arm is a latency reference, not a ceiling** (§4.2), and is beaten
outright on `pick coke can`.

**Seeding is uneven.** The curated ladder carries 30 seeds on eggplant, spoon and
close drawer; coke and the 44-point plane carry 10. Panel A and panel B of the
main figure therefore rest on different seed counts.

**Propagated cells are copies, not evidence.** 27 of 108 grid cells are filled
from a twin whose schedule is identical to within 0.04 ms; they are marked and
excluded from every statistic.

**12 grid cells are unresolved**, not infeasible: CP-SAT found no schedule inside
its budget. They sit at short periods (100–130 ms) and are reported as unknown
rather than as absent.

---

## 10. Artefacts

| what | where |
|---|---|
| curated ladder cells | `traces_torque3/<task>_<arm>[_rng<seed>]/energy2.json` |
| 44-point plane | `g5grid/plane3_runs/*/*/energy2.json` (mirror), `g5grid/plane3_archive/` (full record, 508 MB) |
| loader | `paper/plane3_lib.py` |
| numeric audit | `paper/plane3_audit.py`, `paper/PLANE3_SANITY.md` |
| energy correction | `ENERGY_AUDIT.md` |
| energy calibration | `calib/ENERGY_CALIBRATION.md`, `calib/calibrate.py`, `calib/calibrated_curated.tsv` |
| propagated-cell reasoning | `g5grid/PROPAGATED_CELLS.md` |
| CP-SAT grid | `paper/grid_warm.json`, `paper/grid_greedy.json`, `paper/grid_e2e_success.json` |
| main figure | `paper/layouts/cand_c21_three.py` → `.png`, caption `cand_c21_caption.tex` |
| figure design record | `paper/layouts/LAYOUT_OPTIONS.md` |

Total simulated evidence: **42,240 episodes** on the 44-point plane, plus the
curated ladder at 30 seeds × 24 episodes × 6 arms × 3 tasks and 10 seeds
elsewhere.

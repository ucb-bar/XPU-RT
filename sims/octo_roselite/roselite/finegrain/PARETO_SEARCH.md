# Is there a second Pareto axis? A search over the 44-arm plane.

**Short answer: no, and the "no" is the result.** Across 868 pairs of task-level metrics
on 44 schedules x 4 tasks x 20 seeds x 24 episodes, exactly one pair produces a frontier
that is both rich and honest, and its second axis is not a task metric at all — it is the
accelerator work the schedule spends. Every pair of two *task-quality* axes collapses to
one or two points on the tasks where scheduling matters, because on those tasks a better
schedule is better on every task axis at once. Two of the most plausible candidates
(speed-vs-reliability, and the grasp-to-placement funnel) are not merely weak: they run
the *wrong way*, with a single arm dominating all 43 others on both axes.

Everything below is reproducible from `paper/pareto_search_cache.json`
(built by `paper/make_pareto_cache.py`), and the three figures are
`paper/fig_pareto_compute.py`, `paper/fig_pareto_missiontime.py`,
`paper/fig_pareto_funnel.py`.

---

## 1. What the data actually is, and two traps in it

`g5grid/runs` holds **4,822** `summary.json`. Only **3,520** are distinct
`(task, arm, seed)` keys:

* **983 keys are fetched twice** into different shard directories. Verified byte-identical
  by md5, so first-wins deduplication is lossless — but counting files inflates every n
  by 37%.
* **319 files belong to the older 9-arm sweep** (`drawer_lat0`, `drawer_cpu685`, ...),
  whose names are not of the `g<period>_<window>` form. The arm regex is anchored
  (`^(egg|spoon|coke|drawer)_g\d+_\d+_rng\d+$`) to drop them.

After both: 44 arms x 4 tasks x 20 seeds x 24 episodes = **84,480 episodes**, no cell
missing, none over-full. Asserted in `make_pareto_cache.py`.

**Seed sets are not uniform on egg and coke.** 39 arms ran seeds `{100-109, 200-209}`
(egg) / `{120-129, 200-209}` (coke); the five arms `g130_275 … g130_350` ran
`{90-99, 200-209}`. All 44 share `200-209`. This looked alarming at first — pooling
success by seed made the 90-99 block look ~11 points easier on egg — but that pooling is
circular, since those seeds appear only for five fast arms. Restricting each of the five
to the ten shared seeds moves its egg success by **-0.4 / +4.6 / -5.4 / +2.9 / -1.7**
points, and neighbouring arms on the standard seed set move by as much in both directions
(`g140_290 -4.6`, `g130_400 -6.7`, `g140_320 +2.5`). The seed-set split is inside the
noise. **Checked and cleared**; all numbers below use the full 20 seeds.

**Every failure is a timeout.** Across all four tasks, **0 of the 52,460 failing
episodes** terminate before the horizon. Mission time is therefore only defined on successes, and any
*unconditioned* duration metric is an exact affine function of success rate. This kills a
whole family of candidate axes before it starts (see §5).

### Noise model

The band used throughout is **1.96 x the standard error over the 20 seeds**, averaged over
arms — a 95% interval on the seed mean, computed **per metric**, not borrowed. On success
it comes out at

| task | this band | band quoted by earlier figures |
|---|---|---|
| egg | ±4.01 pts | ±3.33 |
| spoon | ±3.66 pts | ±2.29 |
| coke | ±3.47 pts | ±2.08 |
| drawer | ±2.98 pts | ±2.08 (borrowed, provisional) |

Slightly wider than the house bands, i.e. **conservative**. Every claim below uses the
wider one.

Three counts are reported for each frontier, because they disagree and the disagreement
is the point:

* **raw** — arms nothing else dominates on the 20-seed means. Flattered by noise.
* **resolved** — walk the raw frontier from one end, keep a point only when it differs
  from the last kept point by more than the band on **both** axes. The honest count of
  distinguishable operating levels.
* **bootstrap** — resample the 20 seeds with replacement 2,000x, rebuild the frontier each
  time, report the median size (and how many arms stay on it in >=50% of resamples). What
  would survive a re-run of a nondeterministic harness.

---

## 2. Which tasks can carry a frontier at all

Success rate, end to end, in units of its own 95% band:

| task | success range | span / band | verdict |
|---|---|---|---|
| **egg** | 19.4 → 54.0 % | **8.6 bands** | schedule-sensitive |
| **spoon** | 15.4 → 42.7 % | **7.5 bands** | schedule-sensitive |
| coke | 34.6 → 46.5 % | 3.4 bands | marginal |
| **drawer** | 34.4 → 42.7 % | 2.8 bands | **insensitive — a control, not evidence** |

Drawer's `r(success, inference rate)` is **+0.16**, i.e. the *slow* arms are nominally
better, which is noise. Any frontier drawn on drawer is 44 points scattered inside one
band. Coke is halfway. **Only egg and spoon can test a trade-off hypothesis**, and that
is the standard applied below.

---

## 3. The one pair that trades: task success vs accelerator work

**`fig_pareto_compute.py` → `fig_pareto_compute.png`**

x = **inference rate**, accelerator invocations per second of mission time
(`n_inferences / (sim_ms/1000)`, per episode, averaged). y = success rate. Preference is
**up and left**.

| task | raw | resolved | bootstrap median | arms on frontier in ≥50% of bootstraps | r(success, rate) |
|---|---|---|---|---|---|
| **egg** | 10 | **5** | 10 [8–13] | **8** | +0.86 |
| **spoon** | 13 | **7** | 11 [8–15] | **8** | +0.77 |
| coke | 12 | 3 | 9 [6–13] | 4 | +0.67 |
| drawer | 6 | 1 | 4 [2–6] | 2 | +0.16 |

Egg frontier (slow → fast): `g283_260, g250_260, g220_260, g200_290, g180_275, g165_275,
g150_290, g140_400, g140_320, g130_275` — success 19.4 → 54.0% for 3.55 → 7.64 inferences
per second. Spoon: 13 arms, 15.4 → 42.7% over the same rate range.

**Why this one counts.** Success is an outcome of the rollout; inference rate is the duty
the schedule imposes on the accelerator. Neither appears in the other's definition, so
this is not the defect `fig_pareto_per_success.py` carries (its y-axis divides by the
success count that is also its x-axis). Both are read off the same episodes. And the trade
is physical: 7.6 Hz of a VLA forward pass is roughly twice the energy, thermal headroom
and bus contention of 3.5 Hz, on a part that also has to run perception.

**Three caveats, all of which the figure states.**

1. **The x-axis is not a measurement.** Achieved inference rate reproduces
   `1000 / (achieved cadence from arms.tsv)` to a **median 0.30% and a worst 1.39%**
   (the worst case is `g283_260` on spoon, where the 12 s horizon is short enough for the
   pipeline warm-up to bias the ratio). Its band is ±0.0016 Hz against an axis range of
   5.55 Hz — three and a half orders of magnitude. So the frontier is
   "vertical uncertainty only", which is *why* it bootstraps so well, and half the chart is
   the schmoo restated. Read it as **the price of the schmoo's x-axis**, not as a newly
   discovered tension.
2. **Resolved ≪ raw.** Ten raw frontier arms on egg span 8.6 success bands, so adjacent
   frontier points are ~0.9 bands apart: the *shape* is real, individual neighbours are
   not separable. Five levels are.
3. **Coke and drawer add nothing.** Their raw counts (12, 6) are the largest per band in
   the whole sweep and the most misleading; resolved they are 3 and 1.

### 3b. The near-duplicates that were not plotted

Two pairs match or beat the chosen one on raw count and were rejected as
restatements, not findings:

| pair | egg | spoon | why not plotted |
|---|---|---|---|
| success vs **inferences to complete a win** (`inf_succ`) | raw 11 / res 5 / 10 stable | raw 15 / res 6 / 11 stable | `r(inf_succ, inf_rate) = +0.945 (egg), +0.987 (spoon), +0.987 (coke), +0.996 (drawer)`. After regressing out inference rate only **3.3 of 10.0** units of its spread survive on egg and **1.6 of 9.5** on spoon. It is the compute axis wearing a duration hat. Also conditioned on success (survivorship). |
| mission-time-on-success vs inference rate | raw 10 / res 5 / 9 stable | raw 8 / res 4 / 6 stable | A genuine second frontier with *no* success on either axis — buy a shorter mission with more compute — but it works only on egg/spoon (coke `r=+0.18`, drawer `+0.70` go the other way) and its x-axis is the same cadence again. Worth a sentence in the paper, not a figure. |

---

## 4. The two hypotheses that failed hardest (both plotted)

### 4a. Speed vs reliability — **`fig_pareto_missiontime.png`**

The obvious second axis: a schedule that wins more by taking longer. The plane says the
opposite.

| task | r(success, mission time) | raw | resolved | arms dominated by the single best arm |
|---|---|---|---|---|
| **egg** | **-0.84** | 1 | **1** | **43 of 43** (`g130_275`) |
| **spoon** | **-0.88** | 2 | **1** | **41 of 43** (`g130_305`) |
| coke | **+0.09** | 4 | 2 | 15 of 43 |
| drawer | **+0.00** | 4 | 1 | 18 of 43 |

On eggplant a single arm has both the highest success rate and the shortest median
time-to-completion of all 44, and dominates *every* other arm on both axes at once. The
frontier is one point. Both effects have the same cause: an action computed from a fresher
observation moves the gripper toward where the object actually is, so it both succeeds
more often and gets there sooner.

Conditioning on success is unavoidable (§1: all failures are timeouts) and it **biases
against this conclusion** — a weak arm is scored only on the episodes it happened to win,
which are its easy ones, shortening its measured time. The effect survives the bias.

### 4b. The funnel — **`fig_pareto_funnel.png`**

The most physically plausible hypothesis in the brief: reaching is coarse and tolerant of
stale actions, while closing and placing needs freshness, so a fast schedule might buy
grasps it cannot convert.

Precursor stage (strictly contains success, verified arm by arm) vs completion:

| task | stage | r(stage, completion) | conversion across the 44 arms | raw / resolved | dominated by best |
|---|---|---|---|---|---|
| **egg** | `consecutive_grasp` 51.2→78.1% | **+0.97** | **37.8 → 69.7%, rising** | 2 / **1** | 42 of 43 |
| **spoon** | `consecutive_grasp` 39.2→64.6% | **+0.93** | **39.4 → 66.1%, rising** | 1 / **1** | **43 of 43** |
| coke | `grasped` 55.2→64.2% | +0.73 | 60.5 → 74.9%, rising | 1 / 1 | 43 of 43 |
| drawer | qpos ≤ 0.10 m, 48.1→56.0% | +0.67 | 69.8 → 80.8%, rising | 3 / 1 | 40 of 43 |

Not only do the stages move together — the **yield between them rises with the stage
rate**. A schedule that grasps more converts a *larger fraction* of those grasps. Freshness
helps the whole funnel, not one stage of it. On spoon a single arm dominates all 43 others
on both stages simultaneously.

(Note on coke: its own `consec_grasp` flag fires in only 20-30% of episodes against a
35-46% success rate, so it is the *stricter* event and cannot be the precursor; `grasped`
is used instead. This was caught by asserting nesting arm by arm.)

---

## 5. Everything else, and why it failed

### 5a. Axes that are success in disguise

On the two schedule-sensitive tasks, the correlation of each candidate with success rate:

| candidate | egg | spoon | verdict |
|---|---|---|---|
| `src_on_target` rate | **exactly equal** (max abs diff 0.00) | exactly equal | it *is* the success criterion |
| `conv_g2t` (place \| grasp) | +0.98 | +0.96 | plus a shared denominator |
| `flail` (moved the object and still failed) | **-0.98** | -0.92 | ≈ `moved − success`, and `moved` is near-constant |
| `cgrasp` | +0.97 | +0.93 | §4b |
| `conv_m2g` | +0.96 | +0.94 | shared denominator |
| `grasp` | +0.94 | +0.91 | §4b |

Nothing in this block is an independent objective. Pairing any two of them yields a
frontier of 1–3 raw points, 1 resolved.

### 5b. Axes with no defensible "better" direction — the trap

`gripper_frac_closed`, `mean_abs_trans`, `mean_abs_rot` all vary far outside noise
(7.0, 7.3, 4.0 bands on egg) and are near-perfectly correlated with success
(`r = +0.99, +0.97, -0.80`). **Declare one of them "lower is better" and a
half-the-plane frontier appears out of nothing.** Egg `stable-grasp rate vs
gripper_frac_closed` scores raw **22 of 44**, resolved 6, **11 arms** bootstrap-stable,
goodness-correlation −0.97 — the richest frontier in the entire sweep; egg
`success vs gripper_frac_closed` is close behind at raw **20**, resolved 6, **18 arms**
stable, −0.99. Both are worthless: an arm that closes the gripper more is the
arm that grasps more, which is *good*, and the "trade" is manufactured entirely by the sign
I chose. All eight such metrics are flagged `AMBIG` in the appendix and excluded from
every conclusion (the eighth is `moved_correct_obj`: displacing the target is both the
precursor to grasping it and the definition of knocking it about, and there is no way to
say which reading a number is reporting). This is the exact failure mode the brief warned against, and the sweep walks
straight into it if the direction of each axis is not argued for first.

### 5c. Disturbance / gentle failure

* `moved_wrong_obj` is **identically 0 on all 44 egg arms** — it is not even in egg's
  metric pool, since a constant has no frontier — and reaches 0.83% max on spoon
  (1.5 bands). There is no wrong-object disturbance to trade against.
* `flail_of_fail` (of the episodes it loses, the fraction where it at least moved the
  object) spans only 2.4 bands on egg / 2.7 on spoon, and correlates **-0.22 / -0.57** with
  success: the better arms fail *more gently*, not less. Frontier 2 raw, 1 resolved.
* Drawer's `qpos_fail` (how far the drawer is left open when it fails) spans 2.3 bands,
  `r = +0.19`. Noise.

### 5d. Consistency

`sr@sd` — the seed-to-seed standard deviation of success — is the one axis that is
genuinely *uncorrelated* with the mean (`r = +0.20` egg, `+0.13` spoon). But its own spread
is **1.6 bands** (egg) and **1.9** (spoon): it is not measurably different between arms.
Worst-seed success (`sr@min`) spans 4.3 bands but is `r = +0.83` with the mean and shares
its data outright. Neither survives.

### 5e. Unconditioned duration and per-episode compute

`inf_per_ep` and unconditioned mean time are structurally coupled to success, because
episodes stop early only on success (§1) and every failure runs to the horizon. On egg,
unconditioned mean ticks is exactly `600(1−p) + t_succ·p`. Pairing either against success
rate is plotting success against itself; they are flagged `SHARED` in the appendix. This is
the same defect as the existing energy-per-successful-trajectory chart.

### 5f. Cross-task trade-off (not in the brief; worth checking, also negative)

A robot that must do several tasks could face a real trade if the per-task optimum differs.
Across the 44 arms: `r(egg, spoon) = +0.90`, `r(egg, coke) = +0.59`, `r(spoon, coke) =
+0.60`. The drawer correlations are −0.24 to −0.27, but drawer's whole span is noise. Best
cross-task frontier: egg vs coke, **4 raw / 2 resolved**. There is no meaningful "different
schedules for different tasks" structure: `g130_275` / `g130_305` are at or near the top of
egg, spoon and coke simultaneously.

---

## 6. The traces: safety and motion quality (exploratory, 9 arms, one seed)

`traces_torque2/` covers the **older 9-arm sweep** (`lat0, pipe110fix, p105w300, p130w275,
p150w300, pipe200fix, serial283, fp32_555, cpu685`), one seed, 24 episodes — *not* the 44
`g<P>_<W>` arms, and no per-episode torque exists for the plane (the 1,760-run energy sweep
that would provide it reached 134/1760 and its outputs are not on this host). So nothing
here can produce a 44-arm frontier, and single-seed differences prove nothing. It was
computed anyway because "safety vs success" is the most plausible remaining place a trade
could hide.

Correlation with success across the 9 arms, per-episode medians:

| metric | egg | spoon | coke | reading |
|---|---|---|---|---|
| peak joint torque `max abs(qf)` | **-0.88** | -0.65 | **-0.92** | better arms are *gentler* |
| p99 end-effector acceleration | **-0.90** | -0.76 | -0.70 | gentler |
| peak end-effector speed | -0.87 | +0.14 | -0.59 | gentler |
| peak mechanical power `max Σ abs(τω)` | -0.90 | -0.64 | -0.88 | gentler |
| copper-loss energy `∫Στ²dt` | -0.89 | -0.88 | -0.86 | gentler (matches `fig_pareto_energy`) |
| **commanded action jerk** | **+0.85** | +0.67 | +0.80 | *worse* — but see below |
| commanded action sign reversals | +0.85 | +0.67 | +0.83 | same artifact |
| object net displacement | +0.88 | +0.77 | −0.66 | not a disturbance axis: on the widowx tasks the good arms move the object *on purpose* (into the basket / onto the towel); coke's sign flips because its success is a vertical lift, not a translation |

The only "aggression" axis that trades against success is the **commanded** action stream,
and it is a zero-order-hold artifact, not physics: mean step-to-step change in the commanded
action correlates with **1/cadence at r = +0.990 (egg) and +0.989 (spoon)**. A slow arm
holds the same action for many ticks, so its command stream is smooth by construction while
the arm flails. Every *realised* physical quantity — joint torque peak, end-effector
acceleration, mechanical power — moves the other way. **There is no safety-vs-success
trade; there is a safety-and-success alignment**, which is the same story the existing
energy chart tells.

---

## 7. Verdict

**868 pairs evaluated** — every pair of the per-task metric pools, with no pair dropped.
Of them:

* **513** pair at least one metric with **no defensible "better" direction**
  (`gripper_frac_closed`, `mean_abs_trans`, `mean_abs_rot`, `moved_correct_obj`, and the
  success/failure splits of the first three). Excluded — §5b shows they produce the
  *richest-looking* frontiers in the sweep and every one is manufactured by the sign
  chosen.
* **125 more** put one axis inside the other's definition (`SHARED`). Excluded.
* **230 remain clean.** Of those, **36 reach a resolved frontier of ≥4**.
* **Of those 36, exactly ZERO have a non-compute metric on both axes.**

That last line is the finding. Every rich, honest frontier on this plane has accelerator
work on one axis. The best clean non-compute pair anywhere in the sweep reaches **resolved
= 2**, and five of the top six involve `sr@sd`, whose own spread is under 2 noise bands
(the sixth is drawer `mission time vs residual gap`, on the task that responds to nothing).

> **On the tasks where scheduling matters, a better schedule is simply better on every
> task-level axis measured — success, grasp rate, conversion yield, time-to-completion,
> gentleness of failure, peak torque, peak end-effector acceleration and actuator energy.
> The only thing it costs is compute.**

That is not a disappointing answer; it is a clean one, and it is the argument for the
schedule work. It also means:

* **Keep** `fig_pareto_compute.png` as the plane's Pareto figure. 5–7 resolved levels on
  egg/spoon, 8 bootstrap-stable arms each — materially richer than the 1–2 point frontiers
  we have, with the honest caveat that its x-axis is a schedule parameter rather than a
  measurement.
* **Keep** `fig_pareto_missiontime.png` and `fig_pareto_funnel.png` as explicit negatives.
  A reader's first two objections to "the fast schedule just wins" are "it must be slower"
  and "it must grasp without placing"; both are answered with a frontier of one point and a
  single arm dominating 41–43 of 43.
* **Do not** replace `fig_pareto_per_success.py` with another per-success chart. Its defect
  (success in the denominator of its own y-axis) is real, and `inf_succ` — inferences spent
  inside a winning episode — is the clean substitute if one is wanted, though §3b shows it
  is 94–99.6% collinear with inference rate.
* **What would change this answer:** the 1,760-run energy sweep. Actuator energy is the one
  axis with a plausible independent claim (`r(success, ∫Στ²dt) ≈ -0.86` on 9 arms, one
  seed — so it too looks aligned rather than opposed, but nine points at one seed cannot
  settle it). If it ever lands for the 44 arms, re-run this search with energy in the pool
  before concluding again.

---

## Appendix A — all 868 pairs

Machine-readable: `paper/pareto_search_sweep.tsv`.

Columns: `raw` non-dominated arms of 44; `res` resolved (separated by more than the 95%
seed band on both axes); `boot` median frontier size over 2,000 seed bootstraps; `stbl`
arms on the frontier in ≥50% of bootstraps; `gcorr` correlation of the two axes after
orienting both so that larger is better (**negative = genuine tension**). Flags:
`AMBIG` = an axis with no defensible preferred direction, `SHARED` = one axis appears in
the other's definition, `compute` = one axis is accelerator work.

```
--- egg  (231 pairs) -------------------------------------------
x              y               raw res boot stbl  gcorr  flags
sr             grip             20   6   18   18  -0.99  AMBIG
cgrasp         grip             22   6   14   11  -0.97  AMBIG
flail          grip             16   6   14   11  -0.97  AMBIG
conv_m2g       grip             18   6   14    9  -0.96  AMBIG
inf_rate_hz    grasp             9   6    9    9  -0.82  compute
inf_per_ep     grasp            10   6    9    8  -0.72  compute
sr             trmag            18   5   16   13  -0.97  AMBIG
grasp          grip             15   5   12   11  -0.94  AMBIG
flail          trmag            17   5   13   10  -0.95  AMBIG
inf_succ       conv_m2g         14   5   11   10  -0.70  compute
sr             inf_succ         11   5   10   10  -0.70  SHARED,compute
inf_succ       flail            11   5   10   10  -0.73  compute
inf_per_ep     cgrasp           10   5    9   10  -0.73  compute
inf_succ       conv_g2t          9   5   10   10  -0.70  compute
conv_g2t       grip             15   5   13    9  -0.96  AMBIG
inf_succ       cgrasp           14   5   11    9  -0.68  compute
t_succ_ms      inf_rate_hz      10   5    9    9  -0.63  compute
inf_per_ep     conv_m2g         10   5    9    9  -0.75  compute
inf_per_ep     flail             9   5    8    9  -0.75  compute
inf_rate_hz    conv_m2g         12   5   11    8  -0.85  compute
inf_succ       grasp            11   5    9    8  -0.67  compute
inf_rate_hz    cgrasp           11   5   11    8  -0.85  compute
inf_rate_hz    conv_g2t         11   5   10    8  -0.84  compute
inf_rate_hz    flail            11   5   10    8  -0.87  compute
sr             inf_rate_hz      10   5   10    8  -0.86  compute
cgrasp         trmag            21   5   13    7  -0.94  AMBIG
t_succ_ms      grip             15   5   10    7  -0.87  AMBIG
inf_per_ep     conv_g2t          9   5    9    7  -0.72  compute
sr             inf_per_ep        8   5    8    7  -0.74  SHARED,compute
t_succ_ms      inf_per_ep        8   5    8    7  -0.46  SHARED,compute
grasp          trmag            13   4   11    8  -0.89  AMBIG
conv_g2t       trmag            11   4   13    8  -0.95  AMBIG
conv_m2g       trmag            16   4   13    7  -0.93  AMBIG
t_succ_ms      trmag            14   4   12    7  -0.90  AMBIG
t_succ_mean_ms inf_per_ep       10   4    8    7  -0.43  SHARED,compute
sr@min         grip              9   4    8    7  -0.80  AMBIG
t_succ_mean_ms inf_rate_hz      11   4    8    6  -0.61  compute
t_succ_ms      inf_succ          9   4    7    6  -0.35  SHARED,compute
t_succ_mean_ms inf_succ          9   4    8    6  -0.34  SHARED,compute
t_succ_mean_ms grip             15   4   10    5  -0.87  AMBIG
t_succ_mean_ms trmag            15   4   11    5  -0.91  AMBIG
sr@min         trmag             9   4    7    4  -0.74  AMBIG
t_succ_ms      trmag_succ       13   3   11    8  -0.71  AMBIG
inf_per_ep     romag            11   3    9    6  -0.81  AMBIG,compute
inf_succ       moved             8   3    6    6  -0.19  AMBIG,compute
inf_rate_hz    sr@min            6   3    8    6  -0.82  compute
inf_succ       romag            10   3    9    5  -0.78  AMBIG,compute
inf_rate_hz    romag             9   3    8    5  -0.85  AMBIG,compute
inf_per_ep     moved             9   3    7    5  -0.28  AMBIG,compute
grip           romag             8   3    8    5  -0.78  AMBIG
inf_per_ep     sr@min            6   3    6    5  -0.76  compute
inf_rate_hz    moved             7   3    6    4  -0.37  AMBIG,compute
trmag          romag             7   3    7    4  -0.70  AMBIG
inf_succ       sr@min            6   3    6    4  -0.72  SHARED,compute
grip           romag_fail       12   3    8    3  -0.76  AMBIG
moved          grip              5   3    6    3  -0.56  AMBIG
grip           trmag_fail       11   3    8    2  -0.81  AMBIG
trmag          trmag_fail       11   3    7    2  -0.76  AMBIG
trmag          romag_fail       11   3    7    2  -0.68  AMBIG
inf_succ       romag_fail       11   2    8    6  -0.75  AMBIG,compute
moved          flail_of_fail     9   2   10    6  -0.68  AMBIG,SHARED
t_succ_mean_ms trmag_succ       10   2   10    5  -0.74  AMBIG
sr             trmag_succ        9   2    6    5  -0.42  AMBIG
inf_per_ep     trmag_fail        9   2    8    5  -0.62  AMBIG,compute
inf_per_ep     romag_fail        8   2    8    5  -0.74  AMBIG,compute
sr             romag             4   2    4    5  +0.80  AMBIG
conv_m2g       trmag_succ        8   2    6    4  -0.38  AMBIG
flail          trmag_succ        8   2    5    4  -0.37  AMBIG
inf_succ       trmag_fail        9   2    7    3  -0.59  AMBIG,compute
inf_rate_hz    trmag_fail        9   2    8    3  -0.72  AMBIG,compute
inf_rate_hz    romag_fail        7   2    7    3  -0.79  AMBIG,compute
grasp          trmag_succ        7   2    6    3  -0.37  AMBIG
conv_g2t       trmag_succ        7   2    6    3  -0.38  AMBIG
moved          trmag             5   2    6    3  -0.58  AMBIG
romag          trmag_succ        5   2    5    3  -0.11  AMBIG
inf_succ       trmag_succ        3   2    3    3  -0.15  AMBIG,compute
inf_rate_hz    trmag_succ        3   2    3    3  +0.13  AMBIG,compute
inf_per_ep     trmag_succ        3   2    3    3  -0.05  AMBIG,compute
grip           trmag_succ        2   2    3    3  +0.46  AMBIG
cgrasp         trmag_succ        9   2    6    2  -0.41  AMBIG
moved          trmag_succ        7   2    6    2  -0.44  AMBIG
grip_fail      romag             5   2    5    2  -0.48  AMBIG
sr@sd          conv_g2t          4   2    4    2  -0.19  -
sr             sr@sd             3   2    4    2  -0.20  SHARED
trmag_fail     trmag_succ        8   2    6    1  -0.40  AMBIG
moved          romag_fail        7   2    4    1  +0.27  AMBIG
flail_of_fail  trmag_succ        6   2    4    1  +0.20  AMBIG
sr@sd          romag             5   2    4    1  -0.15  AMBIG
romag          trmag_fail        5   2    3    1  +0.79  AMBIG
t_succ_ms      sr@sd             3   2    4    1  -0.27  -
sr@sd          conv_m2g          3   2    3    1  -0.19  -
sr@sd          moved            10   2    5    0  -0.33  AMBIG
sr@min         moved             7   2    3    0  +0.34  AMBIG
sr@sd          trmag_fail        7   2    4    0  -0.08  AMBIG
conv_g2t       romag             4   1    4    4  +0.78  AMBIG
conv_g2t       romag_fail        4   1    4    4  +0.76  AMBIG
flail_of_fail  romag             3   1    4    4  +0.27  AMBIG
sr             grip_fail         5   1    4    3  -0.39  AMBIG
inf_succ       sr@sd             5   1    5    3  +0.03  compute
cgrasp         grip_fail         5   1    5    3  -0.44  AMBIG
t_succ_ms      romag             4   1    4    3  +0.59  AMBIG
conv_m2g       romag             4   1    4    3  +0.81  AMBIG
sr             romag_fail        3   1    4    3  +0.78  AMBIG
t_succ_mean_ms moved             3   1    3    3  +0.59  AMBIG
inf_rate_hz    sr@sd             3   1    4    3  +0.11  compute
inf_per_ep     sr@sd             3   1    4    3  +0.08  compute
moved          grasp             3   1    4    3  +0.54  AMBIG,SHARED
moved          flail             3   1    3    3  +0.39  AMBIG,SHARED
grasp          conv_g2t          3   1    3    3  +0.90  SHARED
cgrasp         conv_g2t          3   1    2    3  +0.92  SHARED
conv_m2g       romag_fail        3   1    3    3  +0.77  AMBIG
flail          romag             3   1    4    3  +0.80  AMBIG
grip_fail      trmag             2   1    4    3  +0.36  AMBIG
t_succ_ms      romag_fail        9   1    4    2  +0.52  AMBIG
t_succ_ms      grip_fail         8   1    4    2  -0.39  AMBIG
t_succ_mean_ms romag_fail        6   1    4    2  +0.51  AMBIG
conv_m2g       grip_fail         6   1    5    2  -0.45  AMBIG
trmag_succ     romag_fail        6   1    5    2  -0.12  AMBIG
flail_of_fail  trmag_fail        5   1    4    2  +0.23  AMBIG
sr@min         trmag_fail        4   1    3    2  +0.76  AMBIG
moved          grip_fail         4   1    4    2  -0.20  AMBIG
cgrasp         romag             4   1    3    2  +0.80  AMBIG
grip_fail      romag_fail        4   1    5    2  -0.49  AMBIG
t_succ_mean_ms conv_g2t          3   1    2    2  +0.79  -
t_succ_mean_ms romag             3   1    3    2  +0.55  AMBIG
inf_succ       flail_of_fail     3   1    4    2  -0.39  compute
inf_per_ep     flail_of_fail     3   1    4    2  -0.31  compute
moved          cgrasp            3   1    3    2  +0.57  AMBIG,SHARED
cgrasp         romag_fail        3   1    3    2  +0.76  AMBIG
grip_fail      trmag_succ        3   1    4    2  +0.05  AMBIG
sr             t_succ_mean_ms    2   1    2    2  +0.84  SHARED
sr             sr@min            2   1    2    2  +0.83  SHARED
sr             moved             2   1    2    2  +0.55  AMBIG,SHARED
sr             grasp             2   1    2    2  +0.94  SHARED
sr             cgrasp            2   1    2    2  +0.97  SHARED
sr             conv_g2t          2   1    1    2  +0.98  SHARED
sr             flail             2   1    2    2  +0.98  SHARED
sr             flail_of_fail     2   1    3    2  +0.22  SHARED
t_succ_ms      moved             2   1    3    2  +0.65  AMBIG
t_succ_ms      grasp             2   1    2    2  +0.82  -
t_succ_ms      cgrasp            2   1    2    2  +0.86  -
t_succ_ms      conv_g2t          2   1    2    2  +0.79  -
t_succ_ms      flail             2   1    2    2  +0.79  -
t_succ_mean_ms conv_m2g          2   1    2    2  +0.81  -
t_succ_mean_ms trmag_fail        2   1    3    2  +0.65  AMBIG
inf_succ       grip_fail         2   1    3    2  +0.32  AMBIG,compute
inf_rate_hz    grip_fail         2   1    3    2  +0.40  AMBIG,compute
inf_per_ep     grip_fail         2   1    4    2  +0.38  AMBIG,compute
sr@sd          flail_of_fail     2   1    3    2  +0.26  -
sr@sd          grip              2   1    4    2  +0.25  AMBIG
sr@sd          trmag             2   1    4    2  +0.23  AMBIG
moved          conv_m2g          2   1    3    2  +0.51  AMBIG,SHARED
moved          romag             2   1    4    2  +0.35  AMBIG
grasp          romag             2   1    3    2  +0.79  AMBIG
grasp          trmag_fail        2   1    3    2  +0.78  AMBIG
cgrasp         trmag_fail        2   1    3    2  +0.80  AMBIG
conv_g2t       conv_m2g          2   1    2    2  +0.92  -
conv_g2t       trmag_fail        2   1    3    2  +0.81  AMBIG
conv_m2g       flail             2   1    2    2  +0.95  -
flail          trmag_fail        2   1    3    2  +0.83  AMBIG
flail          romag_fail        2   1    3    2  +0.80  AMBIG
grip           grip_fail         2   1    4    2  +0.48  AMBIG
trmag          trmag_succ        2   1    3    2  +0.57  AMBIG
sr             t_succ_ms         1   1    2    2  +0.84  SHARED
sr             conv_m2g          1   1    2    2  +0.96  SHARED
t_succ_ms      conv_m2g          1   1    2    2  +0.84  -
t_succ_ms      trmag_fail        1   1    3    2  +0.69  AMBIG
t_succ_mean_ms cgrasp            1   1    2    2  +0.83  -
moved          conv_g2t          1   1    2    2  +0.53  AMBIG
t_succ_mean_ms grip_fail         7   1    4    1  -0.35  AMBIG
flail_of_fail  grip              7   1    5    1  -0.18  AMBIG
flail_of_fail  trmag             6   1    5    1  -0.15  AMBIG
grasp          grip_fail         5   1    5    1  -0.42  AMBIG
flail          grip_fail         5   1    4    1  -0.39  AMBIG
inf_rate_hz    flail_of_fail     4   1    4    1  -0.31  compute
sr@min         grip_fail         4   1    3    1  -0.40  AMBIG
sr@min         trmag_succ        4   1    4    1  -0.12  AMBIG
sr@sd          romag_fail        4   1    4    1  -0.08  AMBIG
conv_g2t       grip_fail         4   1    3    1  -0.34  AMBIG
grip_fail      trmag_fail        4   1    4    1  -0.32  AMBIG
sr@min         conv_g2t          3   1    2    1  +0.81  SHARED
sr@min         romag             3   1    3    1  +0.79  AMBIG
grasp          romag_fail        3   1    3    1  +0.74  AMBIG
conv_g2t       flail             3   1    2    1  +0.97  -
conv_g2t       flail_of_fail     3   1    3    1  +0.23  -
flail_of_fail  grip_fail         3   1    4    1  -0.06  AMBIG
t_succ_ms      t_succ_mean_ms    2   1    1    1  +0.95  SHARED
t_succ_ms      sr@min            2   1    2    1  +0.66  SHARED
t_succ_ms      flail_of_fail     2   1    2    1  -0.05  -
t_succ_mean_ms sr@sd             2   1    3    1  -0.28  -
sr@min         sr@sd             2   1    2    1  +0.12  SHARED
sr@min         conv_m2g          2   1    2    1  +0.85  -
sr@sd          grasp             2   1    3    1  -0.23  -
sr@sd          cgrasp            2   1    3    1  -0.21  -
sr@sd          flail             2   1    3    1  -0.15  -
moved          trmag_fail        2   1    3    1  +0.38  AMBIG
grasp          conv_m2g          2   1    2    1  +0.97  SHARED
cgrasp         conv_m2g          2   1    1    1  +0.99  SHARED
conv_m2g       flail_of_fail     2   1    3    1  +0.23  -
flail_of_fail  romag_fail        2   1    3    1  +0.35  AMBIG
sr             trmag_fail        1   1    2    1  +0.83  AMBIG
t_succ_mean_ms sr@min            1   1    1    1  +0.60  -
t_succ_mean_ms grasp             1   1    2    1  +0.78  -
t_succ_mean_ms flail             1   1    1    1  +0.79  -
t_succ_mean_ms flail_of_fail     1   1    2    1  +0.01  -
inf_succ       inf_rate_hz       1   1    1    1  +0.94  SHARED,compute
inf_succ       inf_per_ep        1   1    1    1  +0.98  SHARED,compute
inf_succ       grip              1   1    1    1  +0.63  AMBIG,compute
inf_succ       trmag             1   1    1    1  +0.58  AMBIG,compute
inf_rate_hz    inf_per_ep        1   1    1    1  +0.97  SHARED,compute
inf_rate_hz    grip              1   1    1    1  +0.82  AMBIG,compute
inf_rate_hz    trmag             1   1    1    1  +0.79  AMBIG,compute
inf_per_ep     grip              1   1    1    1  +0.68  AMBIG,compute
inf_per_ep     trmag             1   1    1    1  +0.64  AMBIG,compute
sr@min         grasp             1   1    2    1  +0.83  -
sr@min         cgrasp            1   1    1    1  +0.84  -
sr@min         flail             1   1    1    1  +0.84  SHARED
sr@min         flail_of_fail     1   1    2    1  +0.34  -
sr@sd          trmag_succ        1   1    4    1  +0.27  AMBIG
grasp          cgrasp            1   1    2    1  +0.97  SHARED
grasp          flail             1   1    2    1  +0.91  SHARED
grasp          flail_of_fail     1   1    3    1  +0.17  -
cgrasp         flail             1   1    2    1  +0.95  SHARED
cgrasp         flail_of_fail     1   1    3    1  +0.17  -
conv_m2g       trmag_fail        1   1    3    1  +0.81  AMBIG
flail          flail_of_fail     1   1    2    1  +0.38  SHARED
grip           trmag             1   1    1    1  +0.97  AMBIG
sr@sd          grip_fail         4   1    4    0  +0.13  AMBIG
trmag_fail     romag_fail        4   1    3    0  +0.77  AMBIG
sr@min         romag_fail        2   1    2    0  +0.77  AMBIG
romag          romag_fail        2   1    2    0  +0.93  AMBIG

--- spoon  (276 pairs) -----------------------------------------
x              y               raw res boot stbl  gcorr  flags
sr             inf_per_ep       12   7   10    9  -0.72  SHARED,compute
sr             inf_rate_hz      13   7   11    8  -0.77  compute
sr             inf_succ         15   6   12   11  -0.69  SHARED,compute
inf_per_ep     conv_m2g         13   5   12   12  -0.73  compute
inf_per_ep     grasp            11   5   10   11  -0.75  compute
inf_succ       conv_m2g         14   5   11   10  -0.70  compute
inf_succ       cgrasp           12   5   10    9  -0.67  compute
inf_rate_hz    conv_m2g         12   5   10    9  -0.78  compute
inf_per_ep     cgrasp           12   5   10    9  -0.70  compute
inf_succ       grasp            11   5   10    9  -0.72  compute
inf_rate_hz    cgrasp           11   5    9    9  -0.75  compute
inf_rate_hz    grasp            10   5    9    8  -0.78  compute
grasp          grip             17   5   12    7  -0.91  AMBIG
sr             grip             13   5   12    7  -0.94  AMBIG
conv_m2g       grip             12   5   11    6  -0.92  AMBIG
flail          grip             15   5   10    5  -0.87  AMBIG
inf_succ       conv_g2t         13   4   12    9  -0.69  compute
cgrasp         grip             10   4   12    9  -0.92  AMBIG
inf_succ       flail             9   4   10    9  -0.56  compute
inf_rate_hz    conv_g2t         11   4   10    8  -0.77  compute
inf_rate_hz    flail             8   4    9    8  -0.65  compute
conv_g2t       grip             12   4   10    7  -0.88  AMBIG
inf_rate_hz    trmag_fail       10   4    8    7  -0.53  AMBIG,compute
inf_per_ep     conv_g2t         10   4   10    7  -0.72  compute
inf_per_ep     trmag_fail       10   4    9    7  -0.49  AMBIG,compute
t_succ_mean_ms inf_rate_hz       9   4    9    7  -0.70  compute
inf_per_ep     flail             9   4    9    7  -0.59  compute
t_succ_mean_ms inf_per_ep        8   4    8    7  -0.65  SHARED,compute
t_succ_ms      grip             12   4    9    6  -0.84  AMBIG
t_succ_ms      inf_rate_hz       8   4    8    6  -0.70  compute
t_succ_ms      inf_per_ep        7   4    8    6  -0.65  SHARED,compute
inf_succ       trmag_fail        9   4    8    5  -0.47  AMBIG,compute
t_succ_ms      trmag_succ       12   3    9    7  -0.68  AMBIG
sr             trmag             7   3    8    7  -0.55  AMBIG
t_succ_mean_ms inf_succ          7   3    7    7  -0.59  SHARED,compute
t_succ_ms      grip_fail        12   3    7    6  -0.65  AMBIG
grasp          trmag            11   3    7    5  -0.42  AMBIG
inf_per_ep     moved             9   3    7    5  -0.43  AMBIG,compute
conv_m2g       trmag             9   3    8    5  -0.57  AMBIG
inf_succ       moved             8   3    7    5  -0.43  AMBIG,compute
sr             grip_fail         7   3    7    5  -0.69  AMBIG
t_succ_ms      trmag             7   3    7    5  -0.61  AMBIG
inf_rate_hz    moved             7   3    6    5  -0.42  AMBIG,compute
cgrasp         trmag             7   3    8    5  -0.53  AMBIG
t_succ_ms      inf_succ          6   3    7    5  -0.58  SHARED,compute
conv_m2g       grip_fail         6   3    7    5  -0.72  AMBIG
cgrasp         grip_fail        11   3    8    4  -0.77  AMBIG
flail          grip_fail        11   3    7    4  -0.60  AMBIG
t_succ_mean_ms trmag_succ       10   3    9    4  -0.66  AMBIG
conv_g2t       trmag_succ       10   3    7    4  -0.55  AMBIG
grasp          grip_fail         8   3    7    4  -0.76  AMBIG
cgrasp         trmag_succ        8   3    8    4  -0.47  AMBIG
t_succ_mean_ms trmag             6   3    7    4  -0.63  AMBIG
grasp          trmag_succ        9   3    7    3  -0.39  AMBIG
flail          trmag             9   3    7    3  -0.56  AMBIG
sr             trmag_succ        8   3    8    3  -0.55  AMBIG
t_succ_mean_ms grip_fail         7   3    7    3  -0.62  AMBIG
t_succ_mean_ms grip              6   3    8    3  -0.79  AMBIG
conv_m2g       trmag_succ        9   3    7    2  -0.52  AMBIG
flail          trmag_succ        9   3    7    2  -0.60  AMBIG
flail_of_fail  trmag             9   3    6    1  -0.44  AMBIG
conv_g2t       romag_fail        6   2    5    7  -0.05  AMBIG
inf_per_ep     romag             8   2    6    6  +0.22  AMBIG,compute
conv_g2t       grip_fail         6   2    6    5  -0.57  AMBIG
sr             romag_fail        5   2    5    5  -0.09  AMBIG
inf_per_ep     romag_fail        5   2    7    5  +0.20  AMBIG,compute
conv_g2t       trmag             5   2    7    5  -0.48  AMBIG
sr             romag             7   2    6    4  -0.02  AMBIG
conv_m2g       romag             7   2    5    4  -0.08  AMBIG
inf_rate_hz    romag             6   2    5    4  +0.22  AMBIG,compute
t_succ_mean_ms romag_fail        5   2    4    4  -0.18  AMBIG
inf_succ       romag_fail        5   2    6    4  +0.19  AMBIG,compute
grasp          romag_fail        5   2    5    4  -0.02  AMBIG
inf_rate_hz    trmag             4   2    4    4  +0.41  AMBIG,compute
cgrasp         romag_fail        4   2    5    4  -0.08  AMBIG
conv_m2g       romag_fail        4   2    5    4  -0.13  AMBIG
flail          romag_fail        4   2    4    4  -0.12  AMBIG
trmag_fail     trmag_succ        4   2    6    4  -0.39  AMBIG
grip           trmag_fail        9   2    7    3  -0.72  AMBIG
conv_g2t       romag             8   2    6    3  +0.03  AMBIG
flail_of_fail  grip              8   2    6    3  -0.54  AMBIG
cgrasp         romag             7   2    6    3  -0.04  AMBIG
t_succ_ms      romag             6   2    5    3  -0.11  AMBIG
inf_succ       sr@min            6   2    6    3  -0.56  SHARED,compute
inf_succ       romag             6   2    5    3  +0.21  AMBIG,compute
inf_per_ep     sr@min            6   2    5    3  -0.58  compute
sr@min         grip_fail         6   2    5    3  -0.42  AMBIG
moved          trmag_fail        6   2    4    3  +0.34  AMBIG
grasp          romag             6   2    6    3  +0.03  AMBIG
flail_of_fail  grip_fail         6   2    6    3  -0.31  AMBIG
grip           romag             6   2    5    3  -0.11  AMBIG
inf_rate_hz    sr@min            5   2    6    3  -0.60  compute
inf_rate_hz    romag_fail        5   2    5    3  +0.19  AMBIG,compute
trmag_fail     romag_fail        5   2    4    3  +0.34  AMBIG
t_succ_ms      romag_fail        4   2    4    3  -0.12  AMBIG
inf_succ       trmag             3   2    4    3  +0.33  AMBIG,compute
inf_per_ep     trmag             3   2    4    3  +0.37  AMBIG,compute
sr@sd          conv_g2t          9   2    5    2  -0.05  -
sr             sr@sd             7   2    5    2  -0.13  SHARED
sr@sd          trmag             7   2    5    2  +0.02  AMBIG
grip_fail      trmag_fail        7   2    5    2  -0.55  AMBIG
sr@min         grip              6   2    6    2  -0.57  AMBIG
sr@sd          grasp             6   2    5    2  -0.24  -
sr@sd          flail             6   2    5    2  -0.01  -
moved          romag_fail        6   2    5    2  +0.07  AMBIG
flail          romag             6   2    6    2  -0.00  AMBIG
flail_of_fail  trmag_succ        6   2    6    2  -0.51  AMBIG
grip_fail      romag             6   2    5    2  -0.17  AMBIG
sr@sd          conv_m2g          5   2    5    2  -0.21  -
flail_of_fail  romag_fail        5   2    4    2  -0.13  AMBIG
t_succ_mean_ms moved             4   2    4    2  +0.32  AMBIG
sr@min         trmag             4   2    5    2  -0.31  AMBIG
sr@min         romag_fail        4   2    3    2  +0.12  AMBIG
sr@sd          cgrasp            4   2    5    2  -0.22  -
grip           romag_fail        4   2    5    2  -0.00  AMBIG
grip_fail      romag_fail        3   2    4    2  -0.02  AMBIG
trmag_succ     romag_fail        3   2    4    2  +0.05  AMBIG
flail_of_fail  romag             7   2    5    1  +0.02  AMBIG
sr@min         trmag_succ        6   2    5    1  -0.25  AMBIG
sr@sd          flail_of_fail     5   2    5    1  +0.19  -
sr@sd          romag             5   2    4    1  +0.15  AMBIG
wrong_of_fail  romag_fail        5   2    3    1  -0.08  AMBIG
wrong          romag_fail        3   2    2    1  -0.11  AMBIG
sr@min         romag             5   2    4    0  +0.04  AMBIG
sr@min         sr@sd             4   2    3    0  +0.45  SHARED
wrong          wrong_of_fail     5   1   10    5  +0.99  SHARED
t_succ_mean_ms sr@sd             4   1    4    4  -0.11  -
moved          flail_of_fail     4   1    8    4  -0.57  AMBIG,SHARED
inf_succ       flail_of_fail     7   1    5    3  -0.22  compute
sr             moved             4   1    4    3  +0.34  AMBIG,SHARED
moved          conv_g2t          4   1    4    3  +0.21  AMBIG
moved          flail             4   1    5    3  -0.05  AMBIG,SHARED
moved          trmag             4   1    5    3  -0.06  AMBIG
moved          romag             4   1    5    3  -0.06  AMBIG
romag          trmag_fail        4   1    3    3  +0.40  AMBIG
sr             t_succ_mean_ms    3   1    2    3  +0.84  SHARED
t_succ_mean_ms sr@min            3   1    2    3  +0.44  -
t_succ_mean_ms cgrasp            3   1    3    3  +0.83  -
t_succ_mean_ms conv_m2g          3   1    2    3  +0.86  -
t_succ_mean_ms flail             3   1    2    3  +0.75  -
sr@sd          grip              3   1    4    3  +0.10  AMBIG
sr@sd          trmag_succ        3   1    5    3  -0.10  AMBIG
moved          conv_m2g          3   1    4    3  +0.32  AMBIG,SHARED
t_succ_mean_ms flail_of_fail     2   1    3    3  +0.45  -
trmag          trmag_succ        2   1    3    3  +0.60  AMBIG
inf_rate_hz    flail_of_fail     6   1    5    2  -0.30  compute
inf_per_ep     flail_of_fail     6   1    6    2  -0.25  compute
sr@sd          trmag_fail        6   1    5    2  -0.10  AMBIG
t_succ_mean_ms romag             5   1    5    2  -0.16  AMBIG
inf_succ       wrong_of_fail     4   1    2    2  -0.04  compute
inf_rate_hz    wrong_of_fail     4   1    2    2  -0.04  compute
inf_per_ep     wrong_of_fail     4   1    2    2  -0.05  compute
sr@sd          grip_fail         4   1    4    2  +0.02  AMBIG
moved          grasp             4   1    4    2  +0.51  AMBIG,SHARED
moved          trmag_succ        4   1    4    2  +0.04  AMBIG
t_succ_ms      sr@sd             3   1    4    2  -0.09  -
t_succ_ms      moved             3   1    4    2  +0.32  AMBIG
t_succ_ms      grasp             3   1    2    2  +0.81  -
t_succ_ms      flail_of_fail     3   1    3    2  +0.49  -
t_succ_mean_ms conv_g2t          3   1    2    2  +0.77  -
inf_succ       wrong             3   1    2    2  -0.06  compute
inf_rate_hz    wrong             3   1    2    2  -0.07  compute
inf_per_ep     wrong             3   1    2    2  -0.07  compute
moved          cgrasp            3   1    3    2  +0.48  AMBIG,SHARED
grip           trmag             3   1    3    2  +0.47  AMBIG
grip_fail      trmag             3   1    3    2  +0.31  AMBIG
trmag          trmag_fail        3   1    3    2  +0.08  AMBIG
sr             t_succ_ms         2   1    2    2  +0.88  SHARED
sr             trmag_fail        2   1    3    2  +0.73  AMBIG
t_succ_ms      t_succ_mean_ms    2   1    1    2  +0.96  SHARED
t_succ_ms      sr@min            2   1    2    2  +0.52  SHARED
t_succ_ms      cgrasp            2   1    2    2  +0.84  -
t_succ_ms      conv_m2g          2   1    2    2  +0.87  -
t_succ_ms      flail             2   1    2    2  +0.80  -
t_succ_ms      trmag_fail        2   1    3    2  +0.61  AMBIG
t_succ_mean_ms wrong             2   1    1    2  +0.06  -
t_succ_mean_ms wrong_of_fail     2   1    1    2  +0.03  -
t_succ_mean_ms grasp             2   1    2    2  +0.78  -
inf_succ       sr@sd             2   1    3    2  +0.17  compute
inf_succ       grip              2   1    2    2  +0.58  AMBIG,compute
inf_succ       grip_fail         2   1    2    2  +0.37  AMBIG,compute
inf_succ       trmag_succ        2   1    3    2  +0.11  AMBIG,compute
inf_rate_hz    grip              2   1    2    2  +0.66  AMBIG,compute
inf_rate_hz    grip_fail         2   1    2    2  +0.45  AMBIG,compute
inf_rate_hz    trmag_succ        2   1    3    2  +0.24  AMBIG,compute
inf_per_ep     grip              2   1    2    2  +0.61  AMBIG,compute
inf_per_ep     grip_fail         2   1    2    2  +0.40  AMBIG,compute
inf_per_ep     trmag_succ        2   1    3    2  +0.17  AMBIG,compute
grasp          trmag_fail        2   1    3    2  +0.69  AMBIG
cgrasp         trmag_fail        2   1    3    2  +0.65  AMBIG
conv_g2t       trmag_fail        2   1    2    2  +0.74  AMBIG
conv_m2g       trmag_fail        2   1    3    2  +0.64  AMBIG
flail          trmag_fail        2   1    2    2  +0.64  AMBIG
flail_of_fail  trmag_fail        2   1    3    2  +0.35  AMBIG
trmag          romag_fail        2   1    4    2  +0.52  AMBIG
romag          trmag_succ        2   1    4    2  +0.13  AMBIG
t_succ_mean_ms trmag_fail        1   1    3    2  +0.56  AMBIG
sr@min         conv_g2t          1   1    1    2  +0.64  SHARED
cgrasp         conv_g2t          1   1    2    2  +0.81  SHARED
conv_g2t       conv_m2g          1   1    2    2  +0.86  -
sr@min         moved             5   1    3    1  +0.18  AMBIG
sr@sd          romag_fail        5   1    4    1  +0.16  AMBIG
sr@sd          moved             4   1    5    1  -0.31  AMBIG
moved          grip              4   1    4    1  -0.31  AMBIG
moved          grip_fail         4   1    5    1  -0.33  AMBIG
sr@min         trmag_fail        3   1    2    1  +0.46  AMBIG
wrong_of_fail  trmag             3   1    2    1  -0.06  AMBIG
wrong_of_fail  romag             3   1    2    1  -0.22  AMBIG
t_succ_ms      conv_g2t          2   1    2    1  +0.83  -
inf_rate_hz    sr@sd             2   1    3    1  +0.16  compute
inf_per_ep     sr@sd             2   1    3    1  +0.16  compute
wrong          grip              2   1    2    1  +0.06  AMBIG
wrong          grip_fail         2   1    2    1  +0.06  AMBIG
wrong          trmag             2   1    2    1  -0.12  AMBIG
wrong          romag             2   1    2    1  -0.22  AMBIG
wrong          trmag_succ        2   1    2    1  -0.03  AMBIG
wrong_of_fail  grip              2   1    2    1  +0.12  AMBIG
wrong_of_fail  grip_fail         2   1    2    1  +0.10  AMBIG
wrong_of_fail  trmag_succ        2   1    2    1  +0.02  AMBIG
romag          romag_fail        2   1    3    1  +0.66  AMBIG
sr             sr@min            1   1    1    1  +0.62  SHARED
sr             wrong             1   1    1    1  -0.04  -
sr             wrong_of_fail     1   1    1    1  -0.10  SHARED
sr             grasp             1   1    1    1  +0.91  SHARED
sr             cgrasp            1   1    2    1  +0.93  SHARED
sr             conv_g2t          1   1    1    1  +0.96  SHARED
sr             conv_m2g          1   1    2    1  +0.94  SHARED
sr             flail             1   1    1    1  +0.92  SHARED
sr             flail_of_fail     1   1    3    1  +0.57  SHARED
t_succ_ms      wrong             1   1    1    1  +0.03  -
t_succ_ms      wrong_of_fail     1   1    1    1  -0.01  -
inf_succ       inf_rate_hz       1   1    1    1  +0.99  SHARED,compute
inf_succ       inf_per_ep        1   1    1    1  +1.00  SHARED,compute
inf_rate_hz    inf_per_ep        1   1    1    1  +1.00  SHARED,compute
sr@min         wrong             1   1    1    1  +0.01  -
sr@min         wrong_of_fail     1   1    1    1  -0.01  -
sr@min         grasp             1   1    1    1  +0.57  -
sr@min         cgrasp            1   1    1    1  +0.53  -
sr@min         conv_m2g          1   1    1    1  +0.53  -
sr@min         flail             1   1    1    1  +0.58  SHARED
sr@min         flail_of_fail     1   1    2    1  +0.42  -
wrong          grasp             1   1    1    1  -0.08  -
wrong          cgrasp            1   1    1    1  -0.07  -
wrong          conv_g2t          1   1    1    1  -0.04  -
wrong          conv_m2g          1   1    1    1  -0.06  -
wrong          flail             1   1    1    1  -0.13  -
wrong          flail_of_fail     1   1    2    1  -0.19  -
wrong_of_fail  grasp             1   1    1    1  -0.13  -
wrong_of_fail  cgrasp            1   1    1    1  -0.14  -
wrong_of_fail  conv_g2t          1   1    1    1  -0.09  -
wrong_of_fail  conv_m2g          1   1    1    1  -0.13  -
wrong_of_fail  flail             1   1    1    1  -0.19  -
wrong_of_fail  flail_of_fail     1   1    2    1  -0.22  -
grasp          cgrasp            1   1    1    1  +0.96  SHARED
grasp          conv_g2t          1   1    2    1  +0.82  SHARED
grasp          conv_m2g          1   1    2    1  +0.93  SHARED
grasp          flail             1   1    1    1  +0.75  SHARED
grasp          flail_of_fail     1   1    3    1  +0.33  -
cgrasp         conv_m2g          1   1    1    1  +0.97  SHARED
cgrasp         flail             1   1    2    1  +0.79  SHARED
cgrasp         flail_of_fail     1   1    3    1  +0.38  -
conv_g2t       flail             1   1    1    1  +0.93  -
conv_g2t       flail_of_fail     1   1    3    1  +0.65  -
conv_m2g       flail             1   1    2    1  +0.87  -
conv_m2g       flail_of_fail     1   1    3    1  +0.53  -
flail          flail_of_fail     1   1    2    1  +0.84  SHARED
grip           grip_fail         1   1    1    1  +0.87  AMBIG
grip           trmag_succ        1   1    2    1  +0.54  AMBIG
grip_fail      trmag_succ        1   1    2    1  +0.37  AMBIG
trmag          romag             1   1    2    1  +0.59  AMBIG
sr@sd          wrong_of_fail     4   1    3    0  -0.08  -
sr@sd          wrong             3   1    2    0  -0.11  -
wrong          trmag_fail        3   1    2    0  -0.05  AMBIG
wrong_of_fail  trmag_fail        3   1    2    0  -0.08  AMBIG
moved          wrong             2   1    2    0  +0.21  AMBIG
moved          wrong_of_fail     2   1    2    0  +0.20  AMBIG

--- coke  (190 pairs) ------------------------------------------
x              y               raw res boot stbl  gcorr  flags
sr             inf_succ          8   3    8    8  -0.66  SHARED,compute
t_succ_mean_ms trmag_succ        8   3    9    7  -0.73  AMBIG
inf_succ       conv_g2s          7   3    7    6  -0.41  compute
inf_succ       trmag_succ        7   3    7    6  -0.61  AMBIG,compute
inf_rate_hz    conv_g2s          7   3    7    6  -0.43  compute
inf_per_ep     trmag_succ        7   3    6    6  -0.58  AMBIG,compute
inf_succ       romag_fail        6   3    7    6  -0.49  AMBIG,compute
inf_rate_hz    trmag_succ        6   3    6    6  -0.55  AMBIG,compute
inf_succ       trmag_fail       10   3    7    5  -0.35  AMBIG,compute
inf_per_ep     trmag_fail       10   3    8    5  -0.40  AMBIG,compute
inf_per_ep     conv_g2s          9   3    6    5  -0.40  compute
inf_per_ep     romag_fail        7   3    7    5  -0.55  AMBIG,compute
inf_rate_hz    romag_fail        6   3    8    5  -0.54  AMBIG,compute
inf_rate_hz    trmag_fail       13   3    8    4  -0.38  AMBIG,compute
sr             inf_rate_hz      12   3    9    4  -0.67  compute
sr             inf_per_ep        8   3    7    4  -0.61  SHARED,compute
inf_rate_hz    cgrasp            8   3    7    4  -0.80  compute
inf_rate_hz    grasp             7   3    7    4  -0.68  compute
inf_succ       cgrasp           15   3    9    3  -0.82  compute
inf_rate_hz    romag            12   3    7    3  -0.52  AMBIG,compute
inf_per_ep     cgrasp           12   3    8    3  -0.80  compute
inf_succ       grasp             9   3    7    3  -0.68  compute
inf_per_ep     grasp             9   3    7    2  -0.64  compute
inf_per_ep     romag            10   2    8    5  -0.56  AMBIG,compute
sr             trmag             9   2    7    5  -0.38  AMBIG
t_succ_ms      trmag_succ        9   2    8    4  -0.51  AMBIG
t_succ_ms      inf_rate_hz       8   2    6    4  +0.18  compute
inf_succ       sr@sd             7   2    4    4  +0.09  compute
inf_succ       grip              7   2    6    4  -0.16  AMBIG,compute
inf_rate_hz    grip              7   2    6    4  -0.22  AMBIG,compute
inf_per_ep     sr@sd             7   2    4    4  +0.10  compute
inf_rate_hz    grip_fail         6   2    6    4  -0.04  AMBIG,compute
grip_fail      trmag_fail        6   2    6    4  -0.29  AMBIG
inf_succ       flail             5   2    5    4  -0.15  compute
inf_rate_hz    sr@sd             5   2    4    4  +0.08  compute
inf_per_ep     grip_fail         5   2    5    4  -0.00  AMBIG,compute
conv_g2s       trmag_succ        5   2    5    4  +0.29  AMBIG
inf_succ       sr@min            4   2    5    4  -0.31  SHARED,compute
conv_g2s       romag_fail        4   2    5    4  +0.35  AMBIG
inf_succ       romag            12   2    8    3  -0.54  AMBIG,compute
inf_per_ep     lift              8   2    5    3  -0.31  compute
inf_per_ep     grip              7   2    6    3  -0.17  AMBIG,compute
grip           romag             7   2    6    3  -0.29  AMBIG
grip           romag_fail        7   2    5    3  -0.03  AMBIG
grasp          trmag_succ        6   2    4    3  +0.42  AMBIG
conv_g2s       trmag_fail        6   2    5    3  +0.18  AMBIG
grip           trmag             6   2    7    3  -0.66  AMBIG
grip           trmag_fail        6   2    6    3  -0.30  AMBIG
grip_fail      romag_fail        6   2    6    3  -0.13  AMBIG
t_succ_ms      inf_succ          5   2    4    3  +0.33  SHARED,compute
t_succ_ms      inf_per_ep        5   2    5    3  +0.22  SHARED,compute
inf_rate_hz    lift              5   2    5    3  -0.33  compute
sr@sd          trmag             5   2    5    3  -0.21  AMBIG
sr@sd          trmag_succ        5   2    4    3  -0.03  AMBIG
inf_succ       grip_fail         4   2    5    3  -0.02  AMBIG,compute
sr@sd          trmag_fail        4   2    5    3  -0.28  AMBIG
grasp          conv_g2s          4   2    5    3  +0.35  SHARED
grasp          trmag             4   2    4    3  -0.11  AMBIG
grip           trmag_succ        4   2    6    3  -0.33  AMBIG
sr@sd          romag_fail        3   2    4    3  -0.18  AMBIG
lift           trmag_succ        3   2    4    3  -0.03  AMBIG
grip_fail      trmag_succ        3   2    4    3  -0.08  AMBIG
trmag_succ     romag_fail        3   2    4    3  +0.33  AMBIG
t_succ_ms      trmag             9   2    6    2  -0.32  AMBIG
t_succ_mean_ms trmag             8   2    6    2  -0.41  AMBIG
flail          romag_fail        8   2    6    2  -0.13  AMBIG
sr             romag             6   2    5    2  +0.16  AMBIG
sr             trmag_succ        6   2    5    2  +0.41  AMBIG
t_succ_ms      romag             6   2    5    2  -0.21  AMBIG
t_succ_mean_ms grasp             6   2    5    2  -0.29  -
inf_succ       lift              6   2    5    2  -0.32  compute
flail          trmag_fail        6   2    6    2  -0.28  AMBIG
t_succ_ms      conv_g2s          5   2    5    2  +0.04  -
inf_per_ep     sr@min            5   2    4    2  -0.28  compute
grip_fail      trmag             5   2    5    2  -0.41  AMBIG
grip_fail      romag             5   2    5    2  -0.10  AMBIG
inf_rate_hz    flail             4   2    5    2  -0.16  compute
grasp          romag             4   2    3    2  +0.31  AMBIG
inf_rate_hz    sr@min            3   2    4    2  -0.31  compute
inf_per_ep     flail             3   2    5    2  -0.11  compute
sr@min         trmag             3   2    4    2  -0.32  AMBIG
trmag_fail     trmag_succ        3   2    4    2  +0.48  AMBIG
lift           conv_g2s          7   2    5    1  -0.01  -
sr             t_succ_mean_ms    6   2    5    1  -0.26  SHARED
t_succ_mean_ms sr@sd             6   2    5    1  -0.03  -
t_succ_mean_ms conv_g2s          6   2    5    1  -0.06  -
sr             conv_g2s          5   2    3    1  +0.63  SHARED
lift           grip              5   2    4    1  +0.22  AMBIG
sr             t_succ_ms         4   2    5    1  -0.09  SHARED
sr@sd          lift              4   2    4    1  -0.07  -
sr@sd          romag             4   2    5    1  -0.25  AMBIG
conv_g2s       grip              6   2    4    0  +0.19  AMBIG
lift           flail             5   2    4    0  +0.08  -
conv_g2s       grip_fail         5   2    4    0  +0.10  AMBIG
t_succ_mean_ms flail             4   2    5    0  -0.11  -
lift           grip_fail         3   2    4    0  +0.15  AMBIG
t_succ_ms      sr@sd             6   1    5    4  +0.10  -
flail          romag             8   1    6    3  -0.13  AMBIG
flail          trmag             7   1    6    3  -0.40  AMBIG
sr             romag_fail        6   1    5    3  +0.27  AMBIG
t_succ_ms      cgrasp            5   1    5    3  -0.34  -
t_succ_ms      flail             5   1    5    3  +0.06  -
t_succ_mean_ms inf_succ          5   1    4    3  +0.38  SHARED,compute
t_succ_mean_ms inf_per_ep        5   1    4    3  +0.29  SHARED,compute
conv_g2s       trmag             5   1    5    3  -0.18  AMBIG
t_succ_ms      grip_fail         4   1    4    3  +0.15  AMBIG
sr@sd          grip              4   1    4    3  +0.08  AMBIG
t_succ_mean_ms lift              3   1    4    3  +0.05  -
sr@sd          grasp             3   1    4    3  -0.03  -
lift           trmag             3   1    4    3  -0.23  AMBIG
sr             trmag_fail        6   1    5    2  +0.09  AMBIG
t_succ_mean_ms romag             6   1    5    2  -0.40  AMBIG
cgrasp         conv_g2s          5   1    5    2  +0.40  SHARED
t_succ_mean_ms trmag_fail        4   1    4    2  -0.09  AMBIG
sr@sd          grip_fail         4   1    4    2  +0.18  AMBIG
grasp          cgrasp            4   1    3    2  +0.64  SHARED
t_succ_ms      trmag_fail        3   1    4    2  +0.08  AMBIG
t_succ_mean_ms grip              3   1    4    2  +0.37  AMBIG
t_succ_mean_ms grip_fail         3   1    4    2  +0.11  AMBIG
t_succ_mean_ms romag_fail        3   1    4    2  +0.06  AMBIG
grasp          flail             3   1    5    2  -0.10  SHARED
grasp          grip              3   1    4    2  +0.31  AMBIG
cgrasp         trmag_fail        3   1    3    2  +0.33  AMBIG
conv_g2s       romag             3   1    5    2  +0.21  AMBIG
trmag          trmag_fail        3   1    3    2  +0.65  AMBIG
sr             sr@sd             2   1    4    2  +0.07  SHARED
t_succ_ms      romag_fail        2   1    3    2  +0.14  AMBIG
sr@min         trmag_fail        2   1    3    2  -0.20  AMBIG
sr@min         romag_fail        2   1    3    2  -0.02  AMBIG
grasp          grip_fail         2   1    4    2  +0.06  AMBIG
cgrasp         flail             2   1    3    2  +0.21  SHARED
cgrasp         romag             2   1    3    2  +0.52  AMBIG
trmag          romag_fail        2   1    3    2  +0.34  AMBIG
inf_succ       trmag             7   1    5    1  -0.11  AMBIG,compute
inf_per_ep     trmag             7   1    5    1  -0.14  AMBIG,compute
flail          trmag_succ        7   1    4    1  +0.14  AMBIG
t_succ_mean_ms inf_rate_hz       6   1    4    1  +0.26  compute
inf_rate_hz    trmag             6   1    4    1  -0.07  AMBIG,compute
sr@sd          cgrasp            6   1    5    1  -0.20  -
t_succ_ms      grasp             5   1    5    1  -0.20  -
t_succ_mean_ms sr@min            5   1    4    1  -0.23  -
cgrasp         grip_fail         5   1    4    1  +0.02  AMBIG
t_succ_ms      grip              4   1    4    1  +0.31  AMBIG
cgrasp         grip              4   1    4    1  +0.11  AMBIG
sr             cgrasp            3   1    3    1  +0.64  SHARED
sr             lift              3   1    3    1  +0.33  SHARED
t_succ_mean_ms cgrasp            3   1    5    1  -0.42  -
sr@min         trmag_succ        3   1    3    1  +0.16  AMBIG
cgrasp         lift              3   1    4    1  +0.16  SHARED
cgrasp         romag_fail        3   1    3    1  +0.41  AMBIG
lift           romag             3   1    4    1  -0.05  AMBIG
conv_g2s       flail             3   1    3    1  +0.50  -
romag          trmag_fail        3   1    2    1  +0.76  AMBIG
romag          romag_fail        3   1    2    1  +0.71  AMBIG
sr             flail             2   1    2    1  +0.60  SHARED
sr             grip              2   1    2    1  +0.38  AMBIG
sr             grip_fail         2   1    3    1  +0.33  AMBIG
sr@min         conv_g2s          2   1    3    1  +0.27  -
sr@min         grip              2   1    2    1  +0.17  AMBIG
sr@min         romag             2   1    3    1  -0.04  AMBIG
sr@sd          conv_g2s          2   1    3    1  +0.06  -
grasp          lift              2   1    2    1  +0.30  SHARED
grasp          trmag_fail        2   1    2    1  +0.32  AMBIG
grasp          romag_fail        2   1    2    1  +0.41  AMBIG
cgrasp         trmag             2   1    3    1  +0.14  AMBIG
cgrasp         trmag_succ        2   1    3    1  +0.66  AMBIG
sr             grasp             1   1    2    1  +0.73  SHARED
inf_succ       inf_rate_hz       1   1    1    1  +0.99  SHARED,compute
inf_succ       inf_per_ep        1   1    1    1  +0.99  SHARED,compute
inf_rate_hz    inf_per_ep        1   1    1    1  +1.00  SHARED,compute
sr@min         grasp             1   1    2    1  +0.33  -
lift           trmag_fail        1   1    2    1  +0.01  AMBIG
lift           romag_fail        1   1    2    1  +0.09  AMBIG
flail          grip              1   1    3    1  +0.18  AMBIG
flail          grip_fail         1   1    3    1  +0.42  AMBIG
grip           grip_fail         1   1    2    1  +0.72  AMBIG
trmag          romag             1   1    2    1  +0.66  AMBIG
trmag          trmag_succ        1   1    2    1  +0.64  AMBIG
romag          trmag_succ        1   1    2    1  +0.66  AMBIG
trmag_fail     romag_fail        1   1    1    1  +0.84  AMBIG
t_succ_ms      lift              8   1    4    0  +0.04  -
t_succ_ms      sr@min            4   1    4    0  -0.07  SHARED
sr@sd          flail             4   1    4    0  +0.15  -
sr@min         lift              3   1    2    0  +0.11  -
t_succ_ms      t_succ_mean_ms    2   1    2    0  +0.79  SHARED
sr@min         cgrasp            2   1    3    0  +0.25  -
sr@min         flail             2   1    3    0  +0.28  SHARED
sr@min         grip_fail         2   1    2    0  +0.15  AMBIG
sr             sr@min            1   1    2    0  +0.45  SHARED
sr@min         sr@sd             1   1    1    0  +0.67  SHARED

--- drawer  (171 pairs) ----------------------------------------
x              y               raw res boot stbl  gcorr  flags
inf_per_ep     trmag             9   3    9    6  -0.80  AMBIG,compute
inf_succ       trmag             8   3    9    6  -0.79  AMBIG,compute
inf_rate_hz    trmag             8   3    9    6  -0.79  AMBIG,compute
inf_rate_hz    trmag_fail        8   3    7    6  -0.51  AMBIG,compute
inf_succ       romag             9   3    7    5  -0.64  AMBIG,compute
inf_rate_hz    romag             9   3    7    5  -0.64  AMBIG,compute
inf_per_ep     romag             9   3    7    5  -0.64  AMBIG,compute
inf_per_ep     trmag_fail       10   3    7    4  -0.50  AMBIG,compute
inf_rate_hz    trmag_succ        7   3    8    4  -0.78  AMBIG,compute
inf_succ       trmag_fail        6   3    7    4  -0.52  AMBIG,compute
inf_succ       trmag_succ       11   3    9    3  -0.78  AMBIG,compute
inf_per_ep     trmag_succ       10   3    8    3  -0.78  AMBIG,compute
inf_rate_hz    romag_fail        7   2    7    7  -0.47  AMBIG,compute
inf_per_ep     romag_fail        9   2    7    6  -0.46  AMBIG,compute
inf_succ       romag_fail        6   2    7    5  -0.48  AMBIG,compute
inf_per_ep     grip_fail         6   2    6    5  -0.15  AMBIG,compute
inf_succ       grip              9   2    7    4  -0.23  AMBIG,compute
inf_rate_hz    grip              9   2    7    4  -0.22  AMBIG,compute
inf_per_ep     grip              9   2    7    4  -0.23  AMBIG,compute
inf_rate_hz    grip_fail         8   2    6    4  -0.16  AMBIG,compute
t_succ_ms      trmag_succ        7   2    7    4  -0.65  AMBIG
t_succ_ms      qpos              4   2    4    4  +0.02  -
t_succ_ms      trmag             9   2    7    3  -0.61  AMBIG
inf_succ       grip_fail         9   2    6    3  -0.16  AMBIG,compute
t_succ_mean_ms trmag_succ        8   2    8    3  -0.73  AMBIG
inf_succ       qpos_fail         3   2    3    3  -0.06  compute
inf_rate_hz    qpos_fail         3   2    3    3  -0.07  compute
inf_per_ep     qpos_fail         3   2    3    3  -0.08  compute
qpos_fail      grip_fail         9   2    5    2  -0.10  AMBIG
t_succ_mean_ms trmag             7   2    7    2  -0.67  AMBIG
sr             qpos_fail         6   2    5    2  -0.19  SHARED
t_succ_ms      grip_fail         5   2    5    2  -0.13  AMBIG
t_succ_mean_ms romag_fail        5   2    5    2  -0.41  AMBIG
sr             grip_fail         4   2    4    2  +0.19  AMBIG
grip_fail      trmag             4   2    4    2  +0.17  AMBIG
inf_succ       qpos              3   2    4    2  +0.10  compute
inf_rate_hz    qpos              3   2    4    2  +0.11  compute
inf_per_ep     qpos              3   2    4    2  +0.13  compute
qpos           grip              3   2    4    2  -0.10  AMBIG
t_succ_mean_ms romag             9   2    6    1  -0.52  AMBIG
t_succ_ms      trmag_fail        7   2    5    1  -0.43  AMBIG
t_succ_mean_ms grip_fail         7   2    5    1  -0.17  AMBIG
t_succ_mean_ms grip              6   2    5    1  -0.29  AMBIG
qpos_fail      trmag             6   2    5    1  +0.17  AMBIG
grip_fail      romag_fail        6   2    4    1  +0.05  AMBIG
qpos           grip_fail         5   2    5    1  +0.13  AMBIG
qpos_fail      romag             5   2    5    1  +0.09  AMBIG
trmag_fail     trmag_succ        5   2    5    1  +0.33  AMBIG
qpos_fail      trmag_succ        4   2    4    1  +0.11  AMBIG
t_succ_ms      grip              8   2    5    0  -0.26  AMBIG
t_succ_ms      romag             7   2    6    0  -0.50  AMBIG
t_succ_ms      romag_fail        7   2    5    0  -0.41  AMBIG
qpos_med       grip_fail         6   2    4    0  +0.10  AMBIG
sr@sd          grip              5   2    4    0  +0.01  AMBIG
sr@min         qpos_p90          5   1    8    5  +0.06  -
sr             inf_succ          6   1    4    3  +0.15  SHARED,compute
sr             inf_per_ep        5   1    4    3  +0.19  SHARED,compute
t_succ_ms      qpos_med          4   1    4    3  +0.07  -
qpos_fail      trmag_fail        4   1    4    3  +0.05  AMBIG
sr             t_succ_mean_ms    3   1    5    3  -0.04  SHARED
qpos_fail      romag_fail        3   1    5    3  -0.09  AMBIG
sr             inf_rate_hz       6   1    4    2  +0.16  compute
inf_succ       sr@sd             6   1    5    2  -0.23  compute
qpos_med       trmag             6   1    5    2  -0.25  AMBIG
sr             trmag             5   1    5    2  -0.30  AMBIG
inf_rate_hz    sr@sd             5   1    5    2  -0.22  compute
inf_per_ep     sr@sd             5   1    5    2  -0.22  compute
grip_fail      trmag_fail        5   1    4    2  +0.18  AMBIG
sr             t_succ_ms         4   1    5    2  -0.00  SHARED
sr             grip              4   1    4    2  -0.09  AMBIG
t_succ_mean_ms trmag_fail        4   1    5    2  -0.48  AMBIG
t_succ_mean_ms qpos_fail         3   1    4    2  -0.07  -
inf_succ       qpos_med          3   1    4    2  +0.18  compute
inf_rate_hz    qpos_med          3   1    4    2  +0.19  compute
inf_per_ep     qpos_med          3   1    4    2  +0.22  compute
qpos           trmag             3   1    5    2  -0.19  AMBIG
qpos           romag_fail        3   1    4    2  +0.09  AMBIG
qpos_fail      grip              3   1    4    2  -0.06  AMBIG
grip_fail      trmag_succ        3   1    4    2  +0.23  AMBIG
t_succ_mean_ms inf_succ          2   1    2    2  +0.79  SHARED,compute
t_succ_mean_ms inf_rate_hz       2   1    3    2  +0.74  compute
t_succ_mean_ms inf_per_ep        2   1    3    2  +0.75  SHARED,compute
sr@sd          romag_fail        6   1    4    1  +0.01  AMBIG
qpos           trmag_succ        6   1    4    1  -0.10  AMBIG
sr             sr@sd             5   1    4    1  +0.07  SHARED
sr             trmag_succ        5   1    4    1  -0.19  AMBIG
sr@sd          grip_fail         5   1    4    1  +0.12  AMBIG
sr@sd          trmag             5   1    4    1  +0.16  AMBIG
sr@sd          trmag_fail        5   1    4    1  +0.10  AMBIG
qpos           trmag_fail        5   1    3    1  +0.23  AMBIG
qpos_med       grip              5   1    4    1  -0.11  AMBIG
sr             romag             4   1    4    1  -0.02  AMBIG
sr             romag_fail        4   1    3    1  +0.19  AMBIG
t_succ_ms      sr@sd             4   1    5    1  -0.21  -
t_succ_mean_ms sr@sd             4   1    5    1  -0.26  -
sr@sd          qpos              4   1    4    1  +0.12  -
sr@sd          qpos_fail         4   1    4    1  +0.17  -
grip_fail      romag             4   1    4    1  +0.15  AMBIG
t_succ_mean_ms sr@min            3   1    2    1  -0.30  -
sr@sd          qpos_med          3   1    5    1  -0.06  -
qpos_med       trmag_fail        3   1    3    1  +0.19  AMBIG
grip           romag             3   1    3    1  +0.34  AMBIG
grip           trmag_fail        3   1    4    1  +0.28  AMBIG
sr             trmag_fail        2   1    3    1  +0.25  AMBIG
t_succ_ms      sr@min            2   1    2    1  -0.21  SHARED
t_succ_ms      qpos_fail         2   1    3    1  +0.02  -
t_succ_mean_ms qpos              2   1    4    1  -0.06  -
t_succ_mean_ms qpos_med          2   1    4    1  +0.03  -
inf_succ       sr@min            2   1    2    1  -0.20  SHARED,compute
inf_rate_hz    sr@min            2   1    2    1  -0.19  compute
inf_per_ep     sr@min            2   1    2    1  -0.17  compute
sr@min         qpos_fail         2   1    2    1  +0.14  -
sr@min         grip_fail         2   1    2    1  +0.08  AMBIG
qpos           qpos_fail         2   1    2    1  +0.47  SHARED
grip           grip_fail         2   1    2    1  +0.49  AMBIG
grip           trmag             2   1    3    1  +0.39  AMBIG
grip           trmag_succ        2   1    3    1  +0.31  AMBIG
grip           romag_fail        2   1    3    1  +0.33  AMBIG
t_succ_ms      inf_succ          1   1    2    1  +0.76  SHARED,compute
t_succ_ms      inf_rate_hz       1   1    2    1  +0.70  compute
t_succ_ms      inf_per_ep        1   1    2    1  +0.70  SHARED,compute
inf_succ       inf_rate_hz       1   1    1    1  +1.00  SHARED,compute
inf_succ       inf_per_ep        1   1    1    1  +1.00  SHARED,compute
inf_succ       qpos_p90          1   1    1    1  -0.06  compute
inf_rate_hz    inf_per_ep        1   1    1    1  +1.00  SHARED,compute
inf_rate_hz    qpos_p90          1   1    1    1  -0.07  compute
inf_per_ep     qpos_p90          1   1    1    1  -0.07  compute
qpos_p90       grip_fail         1   1    1    1  -0.15  AMBIG
trmag          trmag_succ        1   1    2    1  +0.81  AMBIG
qpos_med       trmag_succ        8   1    4    0  -0.16  AMBIG
qpos           romag             5   1    4    0  +0.02  AMBIG
romag          romag_fail        5   1    2    0  +0.83  AMBIG
trmag_succ     romag_fail        5   1    4    0  +0.27  AMBIG
sr             qpos              4   1    2    0  +0.77  SHARED
sr@min         trmag_succ        4   1    2    0  +0.15  AMBIG
sr@sd          romag             4   1    4    0  +0.10  AMBIG
qpos_med       qpos_fail         4   1    3    0  +0.32  SHARED
qpos_med       romag             4   1    4    0  -0.09  AMBIG
trmag          trmag_fail        4   1    3    0  +0.72  AMBIG
trmag          romag_fail        4   1    3    0  +0.64  AMBIG
romag          trmag_fail        4   1    2    0  +0.76  AMBIG
romag          trmag_succ        4   1    3    0  +0.57  AMBIG
sr@min         trmag             3   1    3    0  +0.07  AMBIG
sr@min         trmag_fail        3   1    2    0  +0.18  AMBIG
sr@sd          trmag_succ        3   1    4    0  +0.20  AMBIG
qpos           qpos_med          3   1    2    0  +0.91  SHARED
trmag_fail     romag_fail        3   1    2    0  +0.90  AMBIG
sr             sr@min            2   1    1    0  +0.38  SHARED
sr             qpos_med          2   1    2    0  +0.76  SHARED
t_succ_ms      t_succ_mean_ms    2   1    2    0  +0.89  SHARED
sr@min         sr@sd             2   1    1    0  +0.62  SHARED
sr@min         grip              2   1    2    0  +0.04  AMBIG
sr@min         romag             2   1    2    0  +0.14  AMBIG
sr@min         romag_fail        2   1    2    0  +0.09  AMBIG
qpos           qpos_p90          2   1    1    0  +0.04  SHARED
qpos_med       romag_fail        2   1    3    0  +0.06  AMBIG
qpos_p90       trmag             2   1    1    0  -0.02  AMBIG
qpos_p90       trmag_fail        2   1    1    0  +0.05  AMBIG
qpos_p90       trmag_succ        2   1    1    0  -0.05  AMBIG
trmag          romag             2   1    2    0  +0.80  AMBIG
sr             qpos_p90          1   1    1    0  +0.05  SHARED
t_succ_ms      qpos_p90          1   1    1    0  +0.02  -
t_succ_mean_ms qpos_p90          1   1    1    0  -0.05  -
sr@min         qpos              1   1    1    0  +0.39  -
sr@min         qpos_med          1   1    2    0  +0.22  -
sr@sd          qpos_p90          1   1    1    0  -0.09  -
qpos_med       qpos_p90          1   1    1    0  -0.03  SHARED
qpos_fail      qpos_p90          1   1    1    0  +0.00  SHARED
qpos_p90       grip              1   1    1    0  +0.03  AMBIG
qpos_p90       romag             1   1    1    0  +0.08  AMBIG
qpos_p90       romag_fail        1   1    1    0  +0.14  AMBIG
```

---

## Appendix B — noise check for every candidate axis

`range / band` is the end-to-end spread of the metric across the 44 arms divided by its own
95% seed band. **Below ~3 the arms are not measurably different on that metric** and no
frontier drawn on it is real. `r with success` is the plain correlation across the 44 arms;
values near ±1 mean the axis is success under another name.

### egg
| metric | min | max | range | 95% band | range/band | r with success |
|---|---|---|---|---|---|---|
| `sr` | 19.38 | 53.96 | 34.58 | 4.01 | 8.6 | +1.00 |
| `t_succ_ms` | 8380 | 1.125e+04 | 2870 | 558 | 5.1 | -0.84 |
| `inf_succ` | 39.74 | 86.1 | 46.36 | 3.42 | 13.6 | +0.70 |
| `inf_rate_hz` | 3.547 | 9.096 | 5.549 | 0.00164 | 3388.6 | +0.86 |
| `inf_per_ep` | 76.96 | 161.9 | 84.94 | 3.41 | 24.9 | +0.74 |
| `sr@min` | 0 | 37.5 | 37.5 | 8.82 | 4.3 | +0.83 |
| `sr@sd` | 6.688 | 11.96 | 5.272 | 3.32 | 1.6 | +0.20 |
| `grip` | 0.1858 | 0.2711 | 0.08525 | 0.0121 | 7.0 | +0.99 |
| `grip_fail` | 0.1292 | 0.1512 | 0.022 | 0.0116 | 1.9 | +0.39 |
| `trmag` | 0.008945 | 0.009888 | 0.0009426 | 0.000128 | 7.3 | +0.97 |
| `romag` | 0.03132 | 0.03298 | 0.001663 | 0.000414 | 4.0 | -0.80 |
| `trmag_fail` | 0.008173 | 0.008538 | 0.0003654 | 0.000114 | 3.2 | -0.83 |
| `romag_fail` | 0.03118 | 0.03338 | 0.002204 | 0.00067 | 3.3 | -0.78 |
| `moved` | 86.67 | 94.38 | 7.708 | 2.45 | 3.1 | +0.55 |
| `wrong` | 0 | 0 | 0 | 0 | nan | +nan |
| `grasp` | 60.62 | 85.63 | 25 | 3.49 | 7.2 | +0.94 |
| `cgrasp` | 51.25 | 78.13 | 26.88 | 3.82 | 7.0 | +0.97 |
| `conv_g2t` | 37.21 | 69.54 | 32.33 | 5 | 6.5 | +0.98 |
| `conv_m2g` | 54.16 | 80.58 | 26.42 | 3.96 | 6.7 | +0.96 |
| `flail` | 37.5 | 67.29 | 29.79 | 4.3 | 6.9 | -0.98 |
| `flail_of_fail` | 80.41 | 90.49 | 10.08 | 4.14 | 2.4 | -0.22 |

### spoon
| metric | min | max | range | 95% band | range/band | r with success |
|---|---|---|---|---|---|---|
| `sr` | 15.42 | 42.71 | 27.29 | 3.66 | 7.5 | +1.00 |
| `t_succ_ms` | 8135 | 9845 | 1710 | 279 | 6.1 | -0.88 |
| `inf_succ` | 35.23 | 78.65 | 43.43 | 1.73 | 25.1 | +0.69 |
| `inf_rate_hz` | 3.583 | 9.093 | 5.511 | 0.00163 | 3377.6 | +0.77 |
| `inf_per_ep` | 41.69 | 98.28 | 56.59 | 0.816 | 69.3 | +0.72 |
| `sr@min` | 0 | 29.17 | 29.17 | 9.52 | 3.1 | +0.62 |
| `sr@sd` | 5.751 | 11.46 | 5.712 | 3 | 1.9 | +0.13 |
| `grip` | 0.3742 | 0.4605 | 0.08638 | 0.0116 | 7.4 | +0.94 |
| `grip_fail` | 0.3457 | 0.3954 | 0.04964 | 0.0154 | 3.2 | +0.69 |
| `trmag` | 0.009438 | 0.009895 | 0.0004563 | 0.000141 | 3.2 | +0.55 |
| `romag` | 0.03855 | 0.04015 | 0.001602 | 0.000715 | 2.2 | +0.02 |
| `trmag_fail` | 0.008832 | 0.009409 | 0.0005766 | 0.000174 | 3.3 | -0.73 |
| `romag_fail` | 0.0386 | 0.04148 | 0.002882 | 0.0012 | 2.4 | +0.09 |
| `moved` | 67.5 | 78.96 | 11.46 | 3.84 | 3.0 | +0.34 |
| `wrong` | 0 | 0.8333 | 0.8333 | 0.556 | 1.5 | +0.04 |
| `wrong_of_fail` | 0 | 1.286 | 1.286 | 0.823 | 1.6 | +0.10 |
| `grasp` | 48.54 | 76.04 | 27.5 | 3.79 | 7.3 | +0.91 |
| `cgrasp` | 39.17 | 64.58 | 25.42 | 3.79 | 6.7 | +0.93 |
| `conv_g2t` | 33.68 | 64.54 | 30.87 | 5.58 | 5.5 | +0.96 |
| `conv_m2g` | 51.39 | 80.29 | 28.9 | 4.14 | 7.0 | +0.94 |
| `flail` | 32.71 | 58.54 | 25.83 | 4.24 | 6.1 | -0.92 |
| `flail_of_fail` | 56.82 | 70.49 | 13.67 | 5.12 | 2.7 | -0.57 |

### coke
| metric | min | max | range | 95% band | range/band | r with success |
|---|---|---|---|---|---|---|
| `sr` | 34.58 | 46.46 | 11.88 | 3.47 | 3.4 | +1.00 |
| `t_succ_ms` | 8654 | 9804 | 1150 | 530 | 2.2 | +0.09 |
| `inf_succ` | 33.27 | 87.4 | 54.13 | 3.37 | 16.1 | +0.66 |
| `inf_rate_hz` | 3.568 | 9.106 | 5.537 | 0.00213 | 2595.6 | +0.67 |
| `inf_per_ep` | 74.64 | 180 | 105.4 | 3.87 | 27.3 | +0.61 |
| `sr@min` | 16.67 | 33.33 | 16.67 | 6.7 | 2.5 | +0.45 |
| `sr@sd` | 4.925 | 11.22 | 6.296 | 2.47 | 2.5 | -0.07 |
| `grip` | 0.1963 | 0.2246 | 0.02829 | 0.0115 | 2.5 | -0.38 |
| `grip_fail` | 0.1736 | 0.2141 | 0.04055 | 0.0175 | 2.3 | -0.33 |
| `trmag` | 0.0371 | 0.03928 | 0.002179 | 0.000834 | 2.6 | +0.38 |
| `romag` | 0.08552 | 0.08929 | 0.003775 | 0.00169 | 2.2 | -0.16 |
| `trmag_fail` | 0.03159 | 0.0329 | 0.001306 | 0.000474 | 2.8 | -0.09 |
| `romag_fail` | 0.07843 | 0.08438 | 0.005949 | 0.00179 | 3.3 | -0.27 |
| `grasp` | 55.21 | 64.17 | 8.958 | 2.95 | 3.0 | +0.73 |
| `cgrasp` | 20.21 | 30.42 | 10.21 | 3.54 | 2.9 | +0.64 |
| `lift` | 4.375 | 9.792 | 5.417 | 2.22 | 2.4 | +0.33 |
| `conv_g2s` | 65.69 | 85.93 | 20.25 | 7.27 | 2.8 | +0.63 |
| `flail` | 16.25 | 24.17 | 7.917 | 3.18 | 2.5 | -0.60 |

### drawer
| metric | min | max | range | 95% band | range/band | r with success |
|---|---|---|---|---|---|---|
| `sr` | 34.37 | 42.71 | 8.333 | 2.98 | 2.8 | +1.00 |
| `t_succ_ms` | 2.176e+04 | 2.466e+04 | 2900 | 1.08e+03 | 2.7 | +0.00 |
| `inf_succ` | 77.22 | 219.8 | 142.5 | 6.9 | 20.6 | -0.15 |
| `inf_rate_hz` | 3.539 | 9.103 | 5.564 | 0.000797 | 6983.2 | -0.16 |
| `inf_per_ep` | 112.1 | 296.4 | 184.3 | 2.89 | 63.8 | -0.19 |
| `sr@min` | 16.67 | 33.33 | 16.67 | 6 | 2.8 | +0.38 |
| `sr@sd` | 4.962 | 8.55 | 3.587 | 2.23 | 1.6 | -0.07 |
| `grip` | 0.4258 | 0.4475 | 0.02171 | 0.00932 | 2.3 | +0.09 |
| `grip_fail` | 0.3922 | 0.4338 | 0.04166 | 0.0163 | 2.6 | -0.19 |
| `trmag` | 0.04044 | 0.0424 | 0.001961 | 0.00049 | 4.0 | +0.30 |
| `romag` | 0.08899 | 0.09171 | 0.002721 | 0.00107 | 2.6 | +0.02 |
| `trmag_fail` | 0.03673 | 0.03854 | 0.001812 | 0.000718 | 2.5 | -0.25 |
| `romag_fail` | 0.08201 | 0.08624 | 0.004232 | 0.00197 | 2.1 | -0.19 |
| `qpos` | 0.1094 | 0.1206 | 0.01121 | 0.00398 | 2.8 | -0.77 |
| `qpos_med` | 0.0855 | 0.1146 | 0.02905 | 0.0119 | 2.4 | -0.76 |
| `qpos_fail` | 0.154 | 0.1656 | 0.01153 | 0.00505 | 2.3 | +0.19 |
| `qpos_p90` | 0.2 | 0.2016 | 0.001575 | 0.000342 | 4.6 | -0.05 |

Legend for the less obvious ones: `inf_succ` = accelerator invocations inside a winning
episode; `inf_rate_hz` = invocations per second of mission time; `sr@min` / `sr@sd` = worst
seed / seed-to-seed standard deviation of success; `flail` = moved the object and still
failed; `flail_of_fail` = the same as a fraction of failures only; `conv_g2t` = placed
given a stable grasp; `conv_m2g` = stable grasp given the object moved; `trmag` / `romag` =
magnitude of the commanded translation / rotation delta; `qpos` = residual drawer opening
in metres (success is ≤ 0.05).

---

## Reproducing

```
python paper/make_pareto_cache.py        # 4,822 files -> 3,520 runs -> per-seed cache
python paper/fig_pareto_compute.py       # the one real frontier
python paper/fig_pareto_missiontime.py   # negative: speed vs reliability
python paper/fig_pareto_funnel.py        # negative: grasp -> placement funnel
```

`paper/pareto_lib.py` holds the frontier, resolution and bootstrap machinery shared by the
three figures. Nothing here modifies an existing `fig_*.py`.

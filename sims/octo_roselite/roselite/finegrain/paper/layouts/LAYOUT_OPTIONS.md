# XPU-RT figure set — layout exploration and recommendation

A design exploration, not an analysis. 26 rendered candidates across the three stories
the paper has to tell, a shared visual vocabulary that makes events trackable between
them, and a ranked recommendation for which figure carries which story.

Everything lives in
`/scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/paper/layouts/`.
**No file outside this directory was modified.** The paper's live `fig_*.py` and `*.png`
were read as reference only.

```bash
cd /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/finegrain/paper/layouts
./render_all.sh          # palette gate + all 26 candidates + the contact sheet
```

Every candidate is a self-contained `cand_<story><n>_<name>.py` that re-derives its
numbers from local data at draw time and writes one PNG beside itself. Nothing ssh-es,
nothing runs on a GPU, nothing launches a simulator. The only external tool is a local
`ffmpeg` call that extracts frames from rollout videos already on disk into a 3 MB
cache under `.frames/`.

| file | what it is |
|---|---|
| `vocab.py` | **the shared vocabulary** — palette, glyphs, the release rail, and every data loader |
| `palette_check.py` | Python port of the dataviz skill's `validate_palette.js` (node is not installed here) |
| `cand_d2_keysheet.py` | the vocabulary **rendered** — the key sheet other figures point at |
| `make_contact_sheet.py` | all 26 candidates on one sheet for triage → `contact_sheet.png` |
| `render_all.sh` | re-render everything |
| `contact_sheet.png` | 1930×2193 triage sheet |

All candidates are drawn at **7.16 in wide** (double-column, ACM/IEEE) at 170 dpi.
Heights vary and are listed per candidate. No candidate uses `bbox_inches="tight"` —
margins are explicit, per the failure mode `LAYOUTS.md` already documents.

---

## 1. The shared visual vocabulary (deliverable)

**`cand_d2_keysheet.png`** (7.16 × 5.30 in) — render it once, put it in the appendix or
the supplementary, and let every other figure stop carrying its own legend.

### Colour — three channels, one job each, disjoint hue families

| channel | job | encoding | where |
|---|---|---|---|
| **compute lane** | categorical, n=3 | CPU `#4a3aa7` · DSP `#eb6834` · HTA `#1baf7a` | story A only |
| **release period** | sequential | blue ramp `#cde2fb → #0d366b`, light = fast | stories B and C |
| **observation age** | sequential | orange ramp, **colourbars only** | rare; age is normally an axis |
| **solver / outcome state** | status, reserved | good `#0ca30c`, critical `#d03b3b` + hatch for absent | never a series |

Blue is deliberately **not** a lane colour, so a lane bar and a schedule point can never
be confused across facing pages. **Task is never a colour** — four tasks are small
multiples with a monochrome label chip carrying the robot family and its actuation grid.

The two named reference arms sit *inside* the period ramp and are marked by
**annotation**, not by a hue of their own: `ideal` gets a hollow ring, `cpu685` gets a
status ring plus the words "QNN baseline". This is what stops a nine-colour arm palette
from existing at all.

### Gate results (`python3 palette_check.py`, exit 0)

```
LANE CPU/DSP/HTA   light, ALL-PAIRS   worst CVD ΔE 9.2 (deutan) · normal-vision ΔE 27.6   PASS
LANE dark steps    dark,  ALL-PAIRS   worst CVD ΔE 9.4 · normal-vision ΔE 24.6            PASS
PERIOD ramp        ordinal + continuous, monotone L, one hue (4° spread)                  PASS
LATENCY ramp       continuous                                                             PASS
```

Three findings worth recording, because they changed the design rather than being
worked around:

1. **HTA aqua is 2.74:1 on white.** The relief rule applies, so *every* lane mark in
   every candidate is direct-labelled. That is why the Gantt panels carry `CPU CPU_X`
   coloured tick labels instead of a legend box.
2. **There is no discrete latency step set.** A five-step orange ordinal ramp that
   clears the 2:1 floor lands on `#ff966e…#970000`, whose dark end collides with
   status-critical red. Rather than tune around it, latency was made **positional or
   continuous only** — the failing gate was removed, not hidden.
3. **success-green vs failure-red is ΔE 4.1 under deutan.** The skill fixes those two
   hexes, so the mitigation is the documented one: outcome is carried by **glyph first**
   (filled star vs cross), colour second. `palette_check.py` prints this as an
   informational FAIL with the mitigation spelled out rather than silently dropping it.

### Event glyphs — shape carries the event, ink is monochrome

`|` release · `▼` inference done · `·` actuation · `✚` gripper closes · `●` grasp held ·
`◆` contact · `★` success · `✗` failure · `///` infeasible · `\\\` unknown.

### Time-axis convention — and the one thing that makes figures trackable

x is **always** wall-clock time, left to right, zero at the window or episode start.
Milliseconds at schedule scale, seconds at episode scale. Both scales carry the same
**release rail**: a hairline with release ticks hanging *below* it and completion
triangles sitting *on* it.

> The offset is not decoration. When latency equals the period — which is exactly the
> un-pipelined baseline, the case the figure most needs to show — the two events fall on
> the same instant and a release tick drawn at the same y vanishes under the triangle.
> The first draft of the key sheet had this bug and the baseline row looked like it
> never released anything.

`vocab.release_rail()` draws it identically everywhere, which is what lets a reader
carry one triangle from **A7 → C9v2 → C10v2**.

---

## 2. Candidate index, ranked within each story

Rank key: **A** = recommend for the paper · **B** = strong, use if space · **C** =
works but superseded · **D** = instructive failure, keep the rejection reasoning.

### Story A — MECHANISM (how a better schedule is built)

| rank | candidate | in | encodes | does well | fails |
|---|---|---|---|---|---|
| **A** | `cand_a7_cascade_refined.png` | 3.95 | 4-row lane Gantt on one 700 ms window; the pipelined trace drawn **twice** — by lane and by inference index; right gutter with per-lane busy ms and duty | one trace answers two questions with no extra claim; the by-inference shades make overlap a *shape*; the gutter turns "the SoC is used" into 51/55/22% | at 700 ms the pipelined rows are dense; a reader counts inferences from the shades, not the bars |
| **A** | `cand_a5_partition_waffle.png` | 2.15 | 52 contexts as a waffle by lane · per-inference lane busy with the bottleneck marked · a 3-number stat block (710 dispatches, 0 evictions) | the smallest figure in the set and it lands three facts; the stat block is the right form for a lone headline number | the waffle breaks a lane block mid-row; the "bottleneck" callout nearly touches its own value label |
| B | `cand_a1_gantt_lanes.png` | 4.05 | the same Gantt, three schedules, lane colour only | honest and legible; the tiling is drawn as a hatched reconstruction | the pipelined row is a barcode — colour-by-lane cannot show overlap, which is the point of the row |
| B | `cand_a2_cascade_flow.png` | 3.35 | one inference as a ribbon walking CPU→DSP→HTA, then tiled | panel A is the picture a reader has in their head of "a cascade" | panel B recolours by lane and destroys the overlap story; segment labels do not fit the thin bars |
| C | `cand_a6_ladder_age.png` | 3.55 | in-flight ladder over 2 s **plus** the measured per-tick age for a whole episode, bracketed | the bridge idea — schedule scale and episode scale on one vocabulary | superseded: C9v2 does the episode half better and A7 does the schedule half better |
| C | `cand_a3_occupancy_area.png` | 2.55 | stacked-area lanes-busy over time, small multiples | the pipelined panel visibly reaches 2 while the others sit at 1 | dispatches are ~2 ms at 1 ms sampling, so it renders as a picket fence; needs a rolling occupancy window to be usable |
| **D** | `cand_a4_radial.png` | 2.75 | polar release clock — θ = phase in one period, one ring per release | — | **rejected.** All three schedules render as concentric rings that look identical; an arc past 360° reads as a closed circle, so "it wraps" — the entire point — is invisible. Polar destroys duration comparison and buys nothing back. |

### Story B — SEARCH (how the good schedule is found)

| rank | candidate | in | encodes | does well | fails |
|---|---|---|---|---|---|
| **A** | `cand_b1_grid_status.png` | 3.05 | 108-cell grid, CP-SAT vs greedy, colour = achieved age, hatch + glyph for INFEASIBLE / UNKNOWN, ring = one of the 44 | the contrast is instant: structure and absent regions on the left, twelve identical columns on the right; absent data is never a zero on the ramp | 4.9 pt cell values are at the print limit; the colourbar label needs two lines |
| **A** | `cand_b3_frontier.png` | 2.60 | achieved (period, age) scatter with the non-dominated front connected; greedy on identical axes | states ρ(age, period) = −0.62 at draw time and shows *why* it matters | panel 3 transposes the axes relative to panels 1–2 and its connector is spaghetti — **drop panel 3**, the point belongs in C8 |
| B | `cand_b2_funnel.png` | 2.85 | stage funnel 108 → 71 → 44 vs greedy 108 → 108 → 12, plus the collapse itself as connected dots | "3.7× more operating points" is legible in one glance; the collapse panel is the honest account of the 27 structural duplicates | label collisions (the second funnel's title lands on the first's last row); the collapse panel's legend sits on the data |
| **A** | `cand_b5_encoding_bakeoff.png` | 4.15 | **one dataset, six encodings** — requested-grid heatmap, achieved scatter, tricontour, hexbin, beeswarm-vs-period, and a bar of \|ρ\| per candidate x | the best *internal* artefact here: panels 1, 3 and 4 all look like a response surface and only 2 and 5 are supported by 44 samples on a 1-D frontier. Panel 6 is the whole argument in four bars | not a paper figure — it is the decision that produces one. Colourbar overlaps panel 3 |
| **D** | `cand_b4_parcoords.png` | 2.75 | parallel coordinates, 5 axes, 44 polylines | — | **rejected.** A hairball. Period-requested → period-achieved tracking is visible; nothing else is. With no intermediate tick values the axes are decoration. Superseded entirely by the SPLOM (C4). |

### Story C — IMPACT (what the schedule does to the robot)

| rank | candidate | in | encodes | does well | fails |
|---|---|---|---|---|---|
| **A** | `cand_c9v2_episode_timeline.png` | 3.30 | four schedules, one shared episode-time axis; age band + release rail + gripper-closed span + outcome glyph; right gutter with commands / mean age / aggregate success | **the trackability figure.** 33/48/85/35 commands, 80/157/405/1008 ms mean age, ★★✗✗, and the two failures running to the 24 s timeout is *mission time made visual*. Episode choice is a stated rule (median mean-age), not a pick | the two 24 s rows compress the two successful rows into the left fifth; release ticks merge into a band at high cadence |
| **A** | `cand_c10v2_sim_strip.png` | 3.10 | the same two rollouts as frames on a **common absolute time grid**, each strip over its own rail | the frames carry the story on their own — pipelined grasps and lifts by 4.5 s, baseline is still fumbling at 16 s; "episode ended at 5.6 s / 16 s before the baseline finishes" replaces a paragraph | frame sampling is 4 fps from the cache, so an instant is hit to ±0.25 s |
| **A** | `cand_c8_period_vs_latency.png` | 3.70 | success vs **period** (top) and vs **age** (bottom), 4 tasks; per-panel ρ computed at draw time; strong trends inked black, weak ones grey and dashed | the sign flips between rows — −0.81/−0.80 vs +0.35/+0.26 — so "a metric against latency reads backwards" is *shown*, not asserted. The two google panels are visibly flat | needed both rows to carry their own axis labels; a shared unlabelled x would have caused the exact confusion the figure prevents |
| **A** | `cand_c4_splom.png` | 4.55 | 6×6 scatterplot matrix: lower = scatter, upper = Spearman ρ on a diverging scale, diagonal = marginal + range | the single densest diagnostic in the set. period↔success −0.81, success↔energy −0.95, period↔energy +0.76, age↔success only +0.35, age↔makespan +0.89 — every claim the paper makes, in one grid | a diagnostic, not a narrative figure; the sign convention needs one caption line (negative ρ with success is *good* here) |
| **A** | `cand_c2_dumbbell.png` | 2.55 | baseline → best schedule as a dumbbell per task, three metric panels, bootstrap CI | real quantitative axes, so effects compare *between* tasks: +55 / +44 / +22 / **+0** pt. That last one is the actuator-saturation result stated as a number | drawer's two dots overlap at +0 pt; the marker order matters |
| B | `cand_c7_cooptimised.png` | 2.29 | success vs energy, points connected in period order | −0.95 on eggplant and −0.82 on spoon are unmistakable: up-and-left, no trade-off | the "fastest/slowest cadence" callouts collide with the ρ text; the connector is noise on the two google panels — draw it only where ρ is strong |
| B | `cand_c3_ladder_matrix.png` | 4.30 | the curated 9 arms × 4 tasks × 3 metrics, **ordered by release period** | reordering by period rather than by name surfaces a real finding — the "ideal" arm has zero observation age but a **200 ms period** (333 ms on google), so it lands *mid-ladder*, and three real schedules beat it: on eggplant p105/w300 57%, p130/w275 55% and p150/w300 53% against ideal's 51%. Period beats zero latency | log-scale tick labels bleed between panels; the ideal/baseline callouts land on the title. The direct replacement for `fig_metrics3.py`, but needs another pass |
| B | `cand_c5_energy_dists.png` | 4.05 | ridgeline / ECDF / violin / raw strip of the same per-episode energy sample | settles the question: the sample spans five decades, so ridgeline and violin are unreadable and **the ECDF wins**. The raw strip (1944 episodes) hides nothing | clipped labels at three edges; keep the ECDF panel only |
| C | `cand_c1_slopegraph.png` | 2.85 | baseline → best as a slope, per task, three metrics | the slope *is* the effect; endpoint values are the labels | no comparable axis between tasks, which is exactly what the saturation result needs. C2 dominates it |
| C | `cand_c6_bump.png` | 3.05 | rank of all 44 points on each task | finds something real: rank agreement is +0.88 within the widowx pair, **−0.12** within the google pair, +0.09 across families | 44 lines is a hairball even with only 10 inked; the finding is better delivered as those three numbers in C4/C8 |
| C | `cand_c9_episode_timeline.png` / `cand_c10_sim_strip.png` | 3.55 / 3.60 | first drafts | — | superseded by v2. Kept so the refinement is visible: v1 drew whichever episode was first (all four succeeded, which undercut the story) and clipped its own row labels |

### Composite

| rank | candidate | in | notes |
|---|---|---|---|
| **A** | `cand_d1_splash.png` | 5.90 | **Figure 1.** Three labelled bands: A the cascade + partition stat block, B the grid + frontier + funnel, C success-vs-period on both widowx tasks + the four-task dumbbell. Uses only vocabulary elements, carries no sentences. Full page width, ~0.55 column-height |
| **A** | `cand_d2_keysheet.png` | 5.30 | the vocabulary itself |

---

## 3. Recommended figure set

Six figures plus a key sheet. Every one is 7.16 in wide (double column).

| # | figure | source | height | carries |
|---|---|---|---|---|
| **1** | splash | `cand_d1_splash.py` | 5.90 in | A + B + C in one page. The reader's whole map |
| **2** | cascade | `cand_a7_cascade_refined.py` | 3.95 in | **MECHANISM.** Lanes, pipelining, per-lane duty. The by-inference panel is the payoff |
| **2b** | partition | `cand_a5_partition_waffle.py` | 2.15 in | inline strip beside §Implementation: 52 contexts, 710 dispatches, 0 evictions |
| **3** | search | `cand_b1_grid_status.py` + `cand_b3_frontier.py` panels 1–2 | ~3.4 in merged | **SEARCH.** 108 → 71 → 44 against greedy's 12, with INFEASIBLE / UNKNOWN hatched as absent |
| **4** | impact, aggregate | `cand_c8_period_vs_latency.py` | 3.70 in | **IMPACT.** Period dominates; age reads backwards; google is saturated |
| **5** | impact, per-schedule | `cand_c2_dumbbell.py` | 2.55 in | baseline → best on all three metrics, all four tasks, +55/+44/+22/+0 pt |
| **6** | the robot | `cand_c9v2_episode_timeline.py` **over** `cand_c10v2_sim_strip.py` | 3.30 + 3.10 in | **the trackable pair.** Same episodes, same rail, same glyphs, stacked on one page |
| **App.** | key sheet | `cand_d2_keysheet.py` | 5.30 in | the vocabulary |
| **App.** | diagnostics | `cand_c4_splom.py`, `cand_c5_energy_dists.py` (ECDF panel) | 4.55 / ~2 in | every ρ the text quotes, and the energy distribution behind the median |

**Ordering rationale.** Fig. 1 is the map. Figs. 2–3 are the systems contribution and
must come before any robot result, because §4's whole point is that the operating points
are *produced*, not chosen. Fig. 4 establishes which variable matters before Fig. 5
quotes a single schedule's delta — otherwise the reader has no reason to accept
`p105/w300` as "the best" one. Fig. 6 is last because it is the only figure that can be
misread as evidence on its own (n = 1 rollout per row); by then the aggregate is already
established and the row gutters print the n = 240 rate beside every rollout.

**Where the trackability lives.** A reader follows a single inference completion `▼` from
Fig. 2's rail (schedule scale, ms), into Fig. 6-top's rail (episode scale, s) where the
same triangle drops the age band, into Fig. 6-bottom where the frame above it shows what
the arm did with that action. One glyph, three zoom levels, one convention.

---

## 4. What this fixes in the current figures

The brief named "minimal whitespace, minimal long/full-sentence text on the plot" as the
thing to fix, and the current set violates it badly.

| current | problem | replacement |
|---|---|---|
| `fig_metrics3.png` | 12 axes, 12 titles, a **4-line italic paragraph** as caption, ~30% whitespace | `cand_c3_ladder_matrix` (same data, one shared ladder, prose removed) or `cand_c2_dumbbell` |
| `fig_schmoo.png` | a **7-line** caption paragraph carrying provenance, resolution limits and warm-start rationale | `cand_b1_grid_status` — provenance becomes the hatch + glyph, the rest goes to the LaTeX caption |
| `fig_e2e_heatmap.png` | an **11-line** caption block; four panels at 380 px tall | `cand_c8_period_vs_latency` |
| `fig_pareto3.png` | 8 panels, 3-line caption, error bars larger than the effect on the google tasks | `cand_c7_cooptimised` (widowx only) + `cand_c4_splom` in the appendix |
| `fig_trajectory.png` | a full-sentence *title*, per-panel event boxes, a 6-line caption | `cand_c10v2_sim_strip` |

The rule applied throughout: **the canvas carries labels, values and at most one short
annotation; every sentence moves to the LaTeX caption.** The longest string on any
recommended candidate is a header of the form
`MEASURED · 44 operating points × 4 tasks × 8-10 seeds × 24 episodes`, which is metadata,
not prose.

---

## 5. Rejected, with reasons

* **Radial / spiral release clock** (`cand_a4`). Three visually identical ring sets; the
  "arc wraps past 360°" that was supposed to *be* pipelining reads as a closed circle.
  Polar destroys duration comparison. Kept as a rendered rejection.
* **Parallel coordinates over the 44 points** (`cand_b4`). Hairball; superseded by the
  SPLOM, which shows the same pairwise structure with a readable statistic per cell.
* **Hexbin and tricontour on the achieved plane** (`cand_b5` panels 3–4). Both interpolate
  a surface that 44 samples on a 1-D frontier cannot support. Drawn deliberately so the
  rejection is visible rather than asserted.
* **Requested-grid heatmap as an E2E figure** (`cand_b5` panel 1). Fine for solver status
  (that *is* the grid), wrong for the robot metric — the simulator only ever saw the
  achieved `(age, cadence)` pair.
* **Ridgeline and violin for energy** (`cand_c5` panels 1 and 3). The sample spans five
  decades; both encodings collapse. ECDF is exact and needs no bandwidth.
* **A nine-colour arm palette.** The arms are ordered by a continuous quantity. They get
  one ramp; the two reference arms get annotation.
* **Colouring by task.** Four tasks are already separated positionally by faceting;
  spending a categorical channel on them would force a fourth hue family into a figure
  set that already needs three.
* **Ratios for a success rate.** `cpu685` succeeds ~2% of the time on eggplant, so a ratio
  reads "+2975%" and says nothing. Rates are reported in **points**, ratios in **%**.

---

## 6. Known gaps

* `cand_b3` panel 3 should be deleted and the merged Fig. 3 assembled from `b1` + `b3`
  panels 1–2 as one script; they are currently two files.
* `cand_c3_ladder_matrix` needs one more pass on its log-scale tick bleed before it can
  replace `fig_metrics3.py`.
* `cand_c9v2`'s time compression (two rows at 24 s, two at ~6 s) is truthful and is itself
  the mission-time result, but a detail/overview pair (0–7 s beside 0–24 s) would make the
  successful rows readable. Not built.
* **"Ideal" is not the ceiling, and the figures should say so.** The 0 ms-age reference
  arm runs at a 200 ms period on widowx and 333 ms on google, so three scheduled arms
  out-succeed it on eggplant (57 / 55 / 53% vs 51%) and `pipe110` out-succeeds it on coke
  (47% vs 38%). Only `cand_c3_ladder_matrix` currently exposes this, because it is the
  only candidate ordered by period; the splash's `ideal` ring sits at 200 ms on the x-axis
  for the same reason. Worth one caption sentence in whichever figure carries it.
* The plane was **1546 of 1760 cells / 37,104 episodes** at draw time. Every plane figure
  reduces through the paper's own `plane3_lib`, so re-running after the sweep finishes
  updates them all consistently; the ρ values quoted above will move in the last digit.
* Contact / grasp glyphs (`◆`, `●`) are defined in the vocabulary and used on the key
  sheet, but no candidate draws them from data yet — the per-tick contact series exist in
  `traces_torque3/*/series.npz` for the 36 curated cells and would slot straight into
  `cand_c9v2`'s rail.

---

# 7. Refinement round (A1, C10)

## A1v2 -- `cand_a1v2_gantt_lanes.png`  (supersedes A1)

* **The serial gap was an artefact, and it is gone.** A1 drew the MEASURED
  dispatch template (span 250.07 ms) but tiled it at the MODELLED period from
  the cost model (283.4 ms), opening 33 ms of phantom idle between invocations.
  A serial chain has period == latency by construction, so the measured
  rendering tiles at the template's own span and the invocations sit back to
  back. **Open question for the paper:** the simulator was fed 283.4 for
  `serial283`; A7 still prints 250/283. Pick one and reconcile both.
* **Pipeline stages are shaded.** Each dispatch takes its lane colour at full
  tone or 55% tint by the parity of the inference it belongs to. Lane identity
  survives and the interleave of two shades within a lane *is* the pipelining.
* **Age and period are brackets on the rail**, not words in a title -- they are
  intervals of the rail, so they are drawn as intervals. Where the two are equal
  (baseline, serial) they collapse to one bracket reading `age = period = N ms`,
  which is itself the statement that nothing overlaps.
* Height 2.46 in, down from 4.05 in.

## C10 v3 / v4 / v5 / v6 -- four fixes for "the keyframes don't match the timeline"

The defect in C10v2: frames sat in EQUAL-WIDTH columns but were labelled with
UNEQUAL times (0, 1.5, 3, 4.5, 6, 10, 16, 22 s), so the frame captioned 10.0 s
sat 62% across a rail running to 24 s -- it belongs at 42%.

| | file | idea | use when |
|---|---|---|---|
| **v3** | `cand_c10v3_sim_strip.png` | uniform frames, LEADER LINES to a shared absolute rail | the headline is WHAT THE ARM DID -- frames stay big |
| v4 | `cand_c10v4_sim_strip.png` | TRUE SCALE: frame width = real time, strip length = duration | the headline is DURATION -- one row visibly stops at 23% |
| v5 | `cand_c10v5_sim_strip.png` | normalised to 0-100% episode progress, with a true-scale duration bar in the footer | comparing behaviour stage-for-stage |
| **v6** | `cand_c10v6_sim_strip_spoon.png` | v3's layout on SPOON, a matched pair | **best pairing available** -- see below |

### Why the eggplant pairing is lopsided, and why it cannot be fixed by picking a
### different episode

A failure has no early termination. Across the finished sweep's **26,203 failed
episodes** the tick count is *exactly* the task horizon with zero variance:

| task | horizon (ticks / s) | failed episodes | success ticks |
|---|---|---|---|
| egg | 600 / 24.0 | 6,481 | 140-600 |
| spoon | 300 / 12.0 | 7,320 | 75-300 |
| coke | 720 / 28.8 | 6,080 | 82-712 |
| drawer | 1017 / 39.6 | 6,322 | 253-1009 |

So no eggplant baseline failure can finish sooner than 24.0 s, and a
"tosses the eggplant out and fails fast" rollout does not exist to be found.

**v6 solves it by changing task rather than episode.** Spoon's horizon is 12.0 s,
so its baseline failure (12.0 s) runs against a 7.0 s success -- 1.7x rather
than 4.3x, and both rows fill the canvas at full frame size. Both rollouts are
episode 21 of the same ladder run, so it is a **matched pair**: same initial
scene, same object pose, only the schedule differs. That is the strongest
available form of "the schedule caused this".

**Recommended pairing for the paper:** `cand_a1v2_gantt_lanes.png` above
`cand_c10v6_sim_strip_spoon.png` -- cascade mechanism into robot behaviour, one
event vocabulary, one rail, read top to bottom.

## 7b. Second refinement pass

**A1v2** — the glyph key moved off the first panel's title to sit beside the axis
it explains (lower right). Bracket labels dropped clear of the release ticks.
The period bracket is now measured **completion to completion, between the first
two**, so it no longer shares an x-range with the age bracket stacked above it.
2.62 in tall.

**C10v6** — compressed 4.10 in → **2.80 in**. Row labels moved off their own
caption lines and onto each strip's first frame, upper left, on an 80%-opaque
white plate; four short lines, because at this frame width a 19-character line
runs past the image and loses its last glyph. Leader lines thickened 0.55 → 1.15
and the frame-to-rail gap tightened. The x-axis says only `episode time (s)` —
the matched-pair provenance belongs in the caption, not on the canvas.

## 7c. D3 -- the stacked pair

`cand_d3_cascade_to_behaviour.png` (7.16 x 5.50 in) puts C10v6 over A1v2 on one
page: behaviour on top, the schedule that caused it underneath, read downward.
The glyph key is stated **once** at the top and governs both halves, so neither
panel carries its own legend.

Row labels in the strip are now a SINGLE line across the top of each image row,
drawn in figure coordinates rather than on the first frame's axes -- as a
four-line block it blotted out the one frame that shows the initial scene.

**The two halves deliberately do NOT share an x-axis**, and the axis labels say
which clock each is on: `episode time (s)` above, `schedule time (ms), one
baseline inference` below. The gantt is also inset (its lane labels need the
left margin) while the strip runs full width. That offset is worth keeping
rather than fixing: flush-aligned axes would invite the reader to map a position
in one half onto the other, and 12 s of robot against 700 ms of silicon is
exactly the ratio the figure exists to show.

## 7d. Gantt condensed further (A1v2, and the bottom half of D3)

Row titles moved out of a title band above each panel and into the **left
margin**, hand-wrapped over two or three lines and right-aligned beside the lanes
they name. Three title bands cost more vertical space than the lane bars
themselves, and removing them let `hspace` drop 0.52 -> 0.22 with no loss of
room for the rail brackets.

The x-axis reads **`schedule time (ms)`** and nothing else. "one baseline
inference" and "MEASURED on QRB5165" are caption facts, not axis facts.

Heights: A1v2 standalone **4.05 -> 2.62 -> 2.30 in**; D3 **5.50 -> 5.15 in**,
with the whole saving taken out of the gantt half.

## 7e. The gap under the frames was axes padding, not layout

The white band between each image strip and its leader lines was not spacing --
it was `imshow` letterboxing. The cached frames are 384x288 (4:3) and the frame
axes were very nearly square, so each image was padded ~0.10 in above and below
inside its own axes. Both strip figures now derive `FRAME_H` from the frame
aspect and lay the whole stack out in INCHES before converting to figure
fractions, so the images sit flush on their leader band and the geometry cannot
drift when the figure is resized.

Heights: C10v6 **2.80 -> 2.45 in**, D3 **5.15 -> 4.67 in** (0.21 in recovered per
image row). Terminology: the brackets and strip labels now read **latency**
rather than **age** throughout.

Still inconsistent: `cand_d2_keysheet.png` labels its orange ramp "observation
age" while `vocab.py` already calls the helper `c_latency`. Unify if the paper
adopts "latency".

## 8. C11 -- widowx highlight (success + energy, four panels)

`cand_c11_widowx_highlight.png` (7.16 x 2.28 in). Scene-major: eggplant success,
eggplant energy, spoon success, spoon energy -- each scene's pair reads as one
statement, that the same move up the ladder raises success AND lowers energy.

Bars are ordered and coloured by RELEASE PERIOD, which the finished plane shows
to be the dominant axis on widowx (rho = -0.83 egg, -0.84 spoon). Ordering by
latency would scramble it: across the schedule set the two are anti-correlated
at rho = -0.62.

BOX AND STRIP, as `fig_energy.py` drew it originally: box and whiskers for
median and quartiles, a white diamond for the mean (the statistic the headline
ratios quote), and the individual seeds jittered on top. The unit is the
PER-SEED value, n = 10 -- the unit the interval was taken over. Plotting all 240
episodes instead would show a much wider cloud than the statistic the panel
reports, because per-episode energy spans two to five orders of magnitude within
a single cell.

Axes are clamped to their physical bounds rather than autoscaled: both
quantities are bounded below at zero and a success rate is bounded above at 100,
and a box plot left to autoscale runs the success axis into negative territory.

Energy is `t2_drive_arm_sus` as a fraction of the QNN baseline, unconditioned on
success -- conditioning keeps only the episodes an arm won, which on eggplant
cpu685 is 1 in 60, and hides the flailing that costs the energy.

COLOUR is the ORIGINAL curated-sweep palette, lifted verbatim from
`paper/fig_metrics3.py` (also used by fig_pareto3 and fig_energy), so this
figure sits beside them without a reader relearning which colour is which arm:
grey ideal `#7f8c8d`, teal-green pipelined `#0e6655 -> #45b39d` (darkest =
fastest cadence), amber/orange serial `#d68910 / #e67e22`, red CPU-only
`#c0392b / #7b241c`. Ordered by period, that palette happens to run
teal -> grey -> amber -> red monotonically, which is why no legend is needed.

Ticks read `name` over `period/latency`, BOTH MEASURED, and STAGGER over two
rows so they can stay horizontal -- upright rotation costs more vertical space
than a second row does. The pair of numbers is not decoration: an arm's name
carries its PERIOD (or its period/window request) and never its latency, and the
two coincide only for the serial and CPU-only arms, where nothing overlaps. The
un-pruned `pipe110` is the cautionary case -- named for a 111 ms period while
paying 385 ms of latency, the HIGHEST of any schedule in the sweep, more than
serial's 283. Printed this way the ideal arm reads honestly as `200/0`: zero
latency, but a 200 ms cadence.

Each success panel carries TWO END-STATE FRAMES stacked at the top right,
success over failure, badged with a check and a cross -- the two outcomes the
bars are counting, taken from the same rollouts the strip figures use, so a
reader who has seen D3 recognises them. The panels take extra y-margin so the
cards sit in headroom rather than on an error bar. The badge corner is set PER
SCENE (`BADGE`), to whichever corner that rollout leaves empty: top right on
eggplant, where the sink and basket fill the left; top left on spoon, where the
table runs to the right.

PRUNED to six bars. Four pipelined points on one ladder is repetition, not
evidence; p105/300, p150/300 and pipe200 carry the cadence trend on their own.
Dropped: `pipe110fix` and `p130w275` (redundant pipelined points) and `fp32_555`
(fp32-vs-int8 is a quantisation question, not a scheduling one). The ideal arm
leads on the LEFT as the yardstick -- by period it would land mid-ladder.

The 0 ms arm is drawn where it actually sits -- at a 200 ms period, MID-ladder,
with three scheduled arms above it on eggplant. It is not a ceiling and the
figure does not draw it as one; on spoon it genuinely does win, and that shows
too.

### 8b. Why C11 has fewer points than the original, and C11b

The original `fig_energy.py` plotted one point PER EPISODE at a SINGLE seed:
**24 points per arm**, and across its nine arms that is the ~216-point strip the
figure is remembered for. It was never hundreds per arm.

C11 plots **10 points per arm** because its unit is the per-SEED value, and it
shows six arms rather than nine -- 60 points against 216, which is where the
apparent thinning comes from.

The original's unit is available here at TEN TIMES the original density (10
seeds x 24 = 240 episodes per arm) and `cand_c11b_widowx_episodes.py` draws it.
It is not the recommendation: per-episode energy spans **4,600x to 9,400x from
p5 to p95** on a scheduled arm, so the cloud needs a log axis, and on a log axis
every arm's box overlaps every other. The arm-level effect the figure exists to
show disappears into within-arm spread. C11b is worth keeping precisely because
it makes that spread visible -- note how tight cpu685 is (p5-p95 only 4x: every
episode flails) against the scheduled arms' bimodal cheap-success/expensive-
failure mix.

Success is per-seed in both, and has to be: a per-episode success point is 0 or 1.

## 9. C12 -- mission time against success rate

`cand_c12_time_vs_success.png` (7.16 x 2.86 in). The third pairing: C11 shows
success and energy move together, this shows completion time does too, so the
ladder is not trading speed for reliability anywhere except at its very top.

Points are joined in RELEASE-PERIOD order, so the line IS the ladder and its
direction is the finding -- every step down it moves DOWN and RIGHT, less likely
to finish and slower when it does. Better is up and left, marked once per panel
in the corner the data leaves empty.

The one real trade in the set is at the top of the eggplant ladder: `ideal`
finishes ~1.0 s faster than `pipe 125` (8.7 s vs 9.7 s) but succeeds 6.2 points
less often. Everywhere else the two axes agree.

MISSION TIME IS CONDITIONED ON SUCCESS and cannot be otherwise -- a failure has
no completion time. This biases the slow arms DOWNWARD, because they are timed
only on the episodes they could still win, so the true gap is WIDER than drawn.
The episode count is printed at every point and `cpu int8` is drawn hollow: its
time rests on 4 successful episodes out of 240 on both scenes.

n = 30 seeds x 24 episodes (seeds 100-129).

## 10. C13 -- is there a Pareto front? (whole plane, three objective pairs)

`cand_c13_plane_pareto.png` (7.16 x 5.85 in). 3 objective pairs x 4 tasks, all 44
operating points, 10 seeds, 42,240 episodes. Rings mark the non-dominated set.

**Answer: no, not on widowx.** success-vs-energy has a front of **1 of 44** on both
eggplant and spoon -- one schedule is simply best on both axes. success-vs-time
has a front of 2, and on eggplant those two are separated by **0.01 s**
(g130_275 at 45.4%/9.54 s, g130_305 at 50.8%/9.55 s), three orders of magnitude
below the seed noise. A front of one is not a front.

The correlations say the same thing without the rings: rho(success, energy) =
**-0.94** egg, **-0.85** spoon; rho(time, energy) = +0.60 / +0.64. Better
schedules are better on every axis at once, so the trade C12 hints at is local to
the top of the curated ladder and does not survive to the full plane.

The google tasks show 4-5 point fronts, which is NOT a richer trade-off: their
success surfaces are unresolved (across-arm spread 1.30x the per-arm SEM on coke,
0.93x on drawer), so those fronts are drawn through noise. Read them as absence
of structure.

METHOD CAVEAT, stated on the figure: domination is computed on POINT ESTIMATES.
Within the n=10 uncertainty many more points are statistically indistinguishable
from the front, so the rings are a best guess, not a resolved set.

## 11. C14 -- the plane as heatmaps, all four workloads, no highlight

`cand_c14_plane_heatmaps.png` (7.16 x 4.95 in). The same 44 operating points C13
scatters, laid back onto the grid the scheduler searched: release period across,
deadline window down, three metric rows x four workloads. No Pareto rings, no
best-cell marker -- the surface is the message.

NO IN-CELL NUMBERS, deliberately. At this panel width a cell is ~9.5 pt across
and a four-character value needs ~9 pt, so the columns touch and the surface
vanishes under the digits. The numbered version of the same data is
`paper/fig_plane3_progress.png`; this is the shape-reading companion to it.

Colour is a magnitude in every row, so each panel is a sequential single hue on
its own scale -- tasks differ in baseline difficulty and a shared scale would
paint that difficulty as schedule sensitivity. One hue per metric so an energy
panel is never mistaken for a success panel. Grey is "no feasible schedule at
that (period, window)": absent data, never a zero on the ramp.

What the surfaces show: eggplant and spoon have a clear diagonal ridge -- success
concentrated at short periods, energy climbing to the right -- while coke and
drawer are visibly mottled, which is what an unresolved surface looks like
(across-arm spread 1.30x the per-arm SEM on coke, 0.93x on drawer).

## 12. C15 -- eggplant: success-vs-energy beside the plane it lives on

`cand_c15_egg_pair.png` (7.16 x 2.62 in). Two views of ONE quantity: what success
costs in actuator energy across the 44 operating points (left), and where in the
schedule space those points sit (right).

ONE COLOUR MEANING, ONE COLOURBAR. Both panels encode SUCCESS RATE on the same
Blues ramp with the same limits, so they link by eye -- find a dark point on the
left, find the dark cells on the right, and you have read that operating point's
schedule off the map. Colouring the scatter by release period instead would put
two different sequential blues in one figure and force the reader to hold which
is which. The scatter's y position and its colour being the same variable is the
price of that linkage, and is deliberate.

rho(success, energy) = -0.94, the tightest coupling in the sweep: better
schedules are not buying success with energy, they are spending less of it.

## 13. C16 -- the complete plane, and why the long-period corner needs no GPU

`cand_c16_plane_complete.png` (7.16 x 5.05 in). 44 measured + 27 propagated
(hatched) + 37 grey = all 108 cells of the CP-SAT grid, three metrics, four
workloads.

**The dominated corner is not missing data.** Past ~220 ms the deadline window
stops binding -- the schedule already finishes inside it -- so CP-SAT returns the
same schedule whatever window you ask for. The 27 propagated cells collapse to
FIVE distinct simulator commands:

| twin | latency | cadence | cells it covers |
|---|---|---|---|
| g220_260 | 232.7 | 220.0 | 8 |
| g250_260 | 232.7 | 250.0 | 8 |
| g283_260 | 232.7 | 283.0 | 8 |
| g200_400 | 257.9 | 200.0 | 2 |
| g150_350 | 319.0 | 150.0 | 1 |

Checked at the BIT level, not just the physical one: **26 of 27 are
bit-identical** to their twin at full float precision, and the one that is not
(`g150_400`, 0.042 ms) becomes identical under the 0.1 ms formatting the job
scripts apply. **0 of 27 are bit-different as the pipeline runs them.**
`trace_eval2.py` receives nothing but those two numbers. Simulating all 27 at 4
tasks x 10 seeds would be 1,080 runs (~364 lane-hours, ~$370) re-measuring five
commands under 27 labels -- a study of harness nondeterminism, not of the plane.
Full reasoning in `g5grid/PROPAGATED_CELLS.md`.

The one genuinely arbitrary thing was the dedup threshold: `make_e2e_success.py`
folds a cell into a twin at `|dlat|+|dcad| <= 0.6 ms`, which is bookkeeping and
not physics on a 40 ms tick. It is why `g150_400` (0.042 ms away) was folded
while `g150_450` (1.348 ms away) stayed its own point. That choice decides which
representative is measured; it changes no value, because the representative's
measurement applies to every cell sharing its command.

Hatch, not a dimmed fill: the fill is a value on the scale and dimming it would
read as a lower value. Every panel's colour scale and printed range is computed
on MEASURED cells only -- asserted in code, not merely intended.

The grey cells now carry TWO marks, because they are different claims:
`x` INFEASIBLE (25, no schedule exists) and `?` UNKNOWN (12, none found within
the CP-SAT budget). The 12 are a reported outcome of the search and are left as
such; they sit at SHORT periods, 100-130 ms.

## 14. C17 / C17b -- all four workloads in one pair

`cand_c17_all_tasks_pair.png` and `cand_c17b_all_tasks_loge.png`
(7.16 x 3.30 in). Left half: success against energy, all four workloads
overlaid. Right half: the same 44 schedules per workload plotted ON the plane as
FOUR SMALL MULTIPLES.

**Small multiples replaced a quartered grid.** The first version fitted four
scales into one grid by splitting every cell into quadrants. It worked, but it
asked the reader to segment each cell by eye before reading anything. Four
separate panels cost the same space and are read directly.

**The mini planes use CONTIGUOUS CELLS, not markers.** The (period, window) grid
is categorical -- a set of requests the scheduler was given, not a continuous
quantity -- so cells that abut say "these are the choices", while scattered
markers wrongly imply the space between them was sampled. Ticks are every other
row and column: 12 periods and 9 windows do not fit a panel this size.

**Each panel keeps its own gradient, in its own hue, over its own success range,
with its own colourbar.** A shared scale would flatten the two google workloads,
whose entire ranges (33-49%, 36-45%) are narrower than eggplant's alone.

**Two versions of the left panel, and they answer different questions:**

* **C17 -- normalised.** Energy divided by each workload's own cheapest
  schedule, so every series starts at 1.0. Shapes become comparable; the ladder
  inside each workload reads sharply. Magnitudes are gone by construction.
* **C17b -- absolute, log axis.** Raw integrals kept. The ~70x separation
  BETWEEN workloads becomes visible and is itself a result: eggplant sits near
  1e6 N2m2s, spoon near 1e5, the two google tasks pile together near 1e4. The
  cost is that the within-workload spread is compressed, so each ladder reads
  less sharply than in C17.

Use C17 to compare shapes, C17b to compare costs. C17b also makes plain why the
normalisation was needed at all: on a LINEAR absolute axis three of the four
series would collapse onto a vertical line.

Palette validated all-pairs by `palette_check.py`: worst CVD separation dE 8.5
(deutan), worst normal-vision dE 16.8, all four inside the lightness band. A
first candidate set failed at dE 7.8 on the green/orange pair, so the orange
became red. Marker shape repeats the workload in the overlay and the panel title
carries it on the right, so neither half is colour-alone.

A STAR marks each workload's best schedule by success rate, on BOTH halves -- at
its (energy, success) in the overlay and at its (period, window) in the mini
plane -- so the same schedule can be found in either view. White-filled with the
workload's hue as the edge, because the best cell is by construction the most
saturated one and a hue-filled star would vanish into it. Taken over MEASURED
cells only.

| workload | best schedule | success | period / latency |
|---|---|---|---|
| eggplant | g130_305 | 50.8% | 130 / 297 ms |
| spoon | g130_275 | 44.2% | 131 / 271 ms |
| coke can | g200_260 | 48.8% | 200 / 246 ms |
| drawer | g283_260 | 45.0% | 283 / 233 ms |

The widowx pair peaks at ~130 ms period; both google workloads peak at the
SLOWEST end of the plane. That split is the saturation result, visible without a
statistic.

DRAW ORDER is deliberate: coke and drawer are laid down first and the two widowx
workloads over them. The google clouds are dense and nearly vertical, and drawn
last they bury the eggplant and spoon ladders that carry the result. The legend
is re-ordered back to reading order, so draw order never leaks into it.

Hatched cells are propagated from their twin; faint x and ? mark infeasible and
unsolved-in-budget.

## 15. C18 -- C11 with the energy panels in JOULES

`cand_c18_widowx_calibrated.png` (7.16 x 2.48 in). Same four-panel scene-major
layout as C11; the success panels are unchanged, the energy panels are rebuilt on
`calib/calibrated_curated.tsv`.

**Why C11's energy axis had to go.** `t2_drive_arm_sus` is a real PD drive-torque
integral but it is not joules, and its ratios are not energy ratios, because the
SIMULATED torque is not physical. On egg/ideal the PD drive runs **148 N.m rms**
against **1.94 N.m** of gravity+Coriolis on the same trajectory, saturating 20%
of substeps, under a `force_limit` of 200 N.m on a shoulder whose two
XM430-W350s stall at **8.2 N.m** combined -- 24x the real ceiling. Converting at
face value gives 1.6 kW-84 kW against a 60 W supply.

**Calibration is GLOBAL, applied PER EPISODE.** The constants are derived once
from published ROBOTIS data and then transform every episode, so the seed
distribution survives the conversion instead of collapsing to one aggregate
number per arm:

    M3_J(episode) = 0.8267 * t2_qf_sub_arm + 4.608 * duration_s
                    [W/(N.m)^2, R/K^2]      [W, published standby]

Reduced as everything else in this set is: median over an arm's 24 episodes, then
one point per seed -- so the panel is box + strip like its neighbour. That
reproduces `calib/calibrated_curated.tsv` exactly (egg/ideal 117.0 J,
egg/cpu685 200.1 J, spoon/ideal 86.0 J, spoon/cpu685 122.6 J).

**The MEAN over episodes does not reproduce it** -- egg/cpu685 comes out at
6371 J against the table's 200 J, because a few episodes carry enormous gravity
integrals. The median is what makes the estimator robust, and the same choice is
already load-bearing elsewhere in this sweep.

**The supply-rail UPPER bound is deliberately not drawn.** It is one number per
arm -- `series.npz` is missing for seeds 101-109, so there is no per-seed spread
to show -- and putting it on the axis forced a log scale that squashed the
distribution the panel exists to show. It belongs in the caption: 763-1440 J on
eggplant, 197-720 J on spoon (`calibrated_curated.tsv`, column `M2_J`). The axis
is linear and the ladder is legible.

**What changed:**

| | C11 quoted | C18 |
|---|---|---|
| eggplant, cpu int8 vs ideal | 3.36x | **1.71-1.89x** |
| spoon, cpu int8 vs ideal | 34.3x | **1.43-3.65x** |

The ORDERING survives on both widowx scenes -- eggplant's is identical across all
nine arms -- so every schedule conclusion stands; only the magnitudes move. On
the google tasks the ordering does NOT survive calibration (Spearman(t2, M3) is
+0.26 coke, -0.04 drawer), which is a further reason not to quote energy there.

C11 is kept as the tau-squared version for comparison, but C18 is the one to
publish. Anything quoting a tau-squared ratio AS an energy ratio needs restating.

## 16. C19 -- C17b with the energy axis calibrated, and captions moved out

`cand_c19_all_tasks_calibrated.png` (7.16 x 3.30 in). C17b's layout with the
left axis in JOULES instead of N2m2s. The four mini planes are unchanged --
success rate needs no calibration.

Calibration is global and applied per episode, then reduced with the same
estimator as everything else (median over an arm's 24 episodes, mean over
seeds). widowx constants are DERIVED from ROBOTIS' published stall
torque/current; google's are a PROXY.

**The google series are drawn HOLLOW**, for two disqualifying reasons: their
constants are a proxy and P_idle is ~92% of the result, so the numbers are
proxy-determined rather than measured; and calibration does not preserve their
ordering (Spearman(tau^2, calibrated) = +0.26 coke, -0.04 drawer, against +0.92
and +0.77 on widowx). Filled = calibrated from published data, hollow =
indicative only. That distinction did not exist on the tau^2 axis, where all
four looked equally solid.

**Captions are no longer baked into the canvas.** C18 and C19 write theirs to
`cand_c18_caption.tex` and `cand_c19_caption.tex` for the LaTeX float. Removing
the two footer lines from C18 also let it shrink 2.48 -> 2.30 in.

What the calibrated axis shows that the normalised one could not: the four
workloads occupy nearly disjoint ENERGY bands in real units -- spoon ~86-125 J,
eggplant ~104-200 J, and the two google tasks around 1-2 kJ, the latter almost
entirely idle draw over much longer episodes.

## 17. C20 -- C18 over C19 as one page

`cand_c20_combined.png` (7.16 x 5.62 in), caption in `cand_c20_caption.tex`.
Panel A is the curated widowx ladder (6 arms x 30 seeds, success and calibrated
energy per scene, with end-state insets); panel B opens it out to all 44
operating points and all four workloads.

**Two colour systems in one figure, and the A/B split is what keeps them apart.**
In A colour is the ARM -- grey reference, teal pipelined, amber serial, red CPU
only -- because the panels are per-scene and the arm is what varies. In B colour
is the WORKLOAD, because the panels are per-plane and the workload is what
varies. Stacked without the lettered section headers, one hue would be asked to
mean two things (red is both `cpu int8` and `coke can`). The headers make the two
halves read as what they are: two figures sharing a page and a calibration.

Everything else carries over unchanged -- calibrated energy applied per episode,
google series hollow because their constants are a proxy and calibration does not
preserve their ordering, propagated cells hatched, best-by-success starred in
both halves of B.

Supersedes C18 and C19 as a single float; keep those two if the paper wants them
in separate places.

## 18. C21 -- three workloads: six ladder panels over a four-panel plane row

`cand_c21_three.png` (7.16 x 4.62 in), caption in `cand_c21_caption.tex`.
A: the curated ladder as SIX panels -- success and calibrated energy for
eggplant, spoon and close drawer. B: the same three on the full plane as a
scatter plus one plane each, four panels in a row.

**Coke excluded, drawer kept.** Between the two google workloads drawer is the
one that says something -- it is the actuator-saturated case and its flatness IS
the finding. Coke repeats it with less contrast.

**Drawer's energy panel is labelled `energy — idle only`, in warning colour.**
Its calibration uses the google PROXY constants, and with P_idle = 40 W over
~39 s episodes the idle term alone accounts for essentially the entire ~1580 J.
The panel measures episode length, not actuation. Drawing it unlabelled would
invite a reader to see a schedule effect in a constant; leaving it out would
imply the measurement failed. Labelling it is the only honest third option.

Reading A left to right, the contrast is the whole argument: eggplant and spoon
fall 55->3% and 50->3% across the ladder while their energy climbs; drawer sits
at 35-41% throughout and its energy is flat to 2%.

### 18b. Google constants rebalanced

`calib/` paired two proxies from different machines: a Harmonic Drive FHA-14C-100
for the conversion constant and a Kinova/Franka envelope for P_idle. That put a
Franka-class 40 W idle next to a 3 W actuation term, so idle was 93% of the
google result and the actuation signal vanished under an assumed constant. The
drawer energy panel measured episode length, not actuation.

Since this study is about how end-to-end robot behaviour responds to scheduling
rather than about absolute watts for an unspecified robot, c_google is now set so
the arm's dynamic/idle BALANCE matches the WidowX's measured 0.810, keeping the
40 W idle:  c = 0.0125 * (0.810 * 40) / 3.07 = **0.1321 W/(N.m)^2**, 10.6x the
old value. google now sits at 30-31 W actuation against 40 W idle (43%), beside
the WidowX's 45%.

**This changes absolute joules for the google tasks and nothing else.** Scaling c
scales every arm equally, so the across-arm ratios are EXACTLY invariant: drawer
actuation max/min is 1.0269 under either constant, coke 1.2404. The findings --
drawer flat, coke nearly so -- do not rest on the proxy at all. The series stay
hollow and the panel title reads `energy (proxy)` to mark that only the scale is
assumed.

### 18c. close_drawer extended to n=30, and its snapshots are real

120 runs (6 arms x seeds 110-129) took close drawer from n=10 to n=30, matching
eggplant and spoon. 6/6 arms clean at exactly 30 seeds x 24 episodes,
contiguous 100-129.

**Tripling the seeds did not reveal structure -- which is the useful outcome.**
Success spans 36.8-41.8% and NO arm differs significantly from the 0 ms
reference on a paired test over the shared seeds:

| arm | vs ideal | p |
|---|---|---|
| p105w300 | -1.11 pt | 0.523 |
| p150w300 | -0.14 pt | 0.944 |
| pipe200fix | +0.97 pt | 0.623 |
| serial283 | +1.81 pt | 0.331 |
| cpu685 | -3.19 pt | 0.067 |

Per-arm SEM fell 2.2 -> 1.24 pts. The drawer panel now carries a WELL-POWERED
null rather than an underpowered one, which is what it needs to be, since it is
the figure's evidence that scheduling stops mattering once the actuator
saturates. The one thing worth watching is `cpu685` at p=0.067 -- a hint that
even here the un-scheduled baseline may be slightly worse, but not significant
and not claimed.

**Drawer's snapshots are now a real rollout pair** (`job_drvid.sh`, 8 episodes
with video). They come from the SAME arm, p105w300 episodes 1 and 0, because at
n=30 no schedule succeeds where another fails -- the pair shows the task's two
outcomes, drawer shut with the arm withdrawn against drawer still hanging open,
not a schedule effect. Eggplant and spoon keep scheduled-success-vs-baseline-
failure, where the contrast is real.

The video job initially failed silently: `ffmpeg` is not installed on the workers
and mediapy needs it to encode. The monitor counted only mp4s, so a hard failure
and "not started yet" looked identical for 40 minutes. Both the job's `rc=` line
and the monitor's filter should carry failure signatures, not just success.

### 18d. Scatter switched to relative energy; proxy tags removed

**Relative, not absolute.** On the absolute axis the workloads sit in disjoint
bands -- spoon ~90 J, eggplant ~150 J, the google tasks ~1.5-2.5 kJ -- so each
series collapsed to a vertical line and the within-workload ladder, the thing
actually being compared, was unreadable. Normalised to each workload's own
cheapest schedule they all span **1.0-1.4x** and overlay directly on a linear
axis: egg 1.00-1.38, coke 1.00-1.26, spoon 1.00-1.14, drawer 1.00-1.07.

**Proxy tags dropped, markers uniform.** An earlier version hollowed the google
series and titled drawer's panel `energy (proxy)`. That singled out one pair as
modelled and implied the others were measured, which is false: the WidowX
constants are DERIVED from published stall specs rather than measured on the arm,
and the whole quantity is a quasi-static lower bound. Provenance is now a caption
matter, where the real distinction (datasheet-derived vs balance-matched) can be
stated without the figure implying a two-tier reliability that does not exist.

The ratios plotted are invariant to every constant anyway -- scaling c scales all
of a workload's schedules equally -- so the figure's content does not depend on
the calibration at all. Only absolute joules do, and those now appear only in
panel A.

### 18e. Counts off the canvas; snapshots moved above the planes

Seed and operating-point counts removed from the section headers and scene
titles -- they are caption facts, not axis facts, and they were the only text on
those lines. The caption now carries them in bold instead.

The rollout pairs moved out of panel A's success panels and into a new row above
the three planes in panel B. In A they were thumbnails competing with the boxes
for the same corner, which forced 1.4-1.7x headroom on every success panel and
still clipped the shortest one. In B they get a row of their own and a full
column of width each, so the eggplant in the basket, the spoon on the towel and
the shut drawer are actually visible. Success left, failure right, badged.

Panel A's success panels drop to 1.12x headroom with the insets gone.

### 18f. Units into the titles; A/B gap closed

Panel A's y-axis labels removed and the unit folded into each subplot title
(`success (%)`, `energy (J)`). Six panels each spending width on a rotated label
that repeats its own title is pure margin -- dropping them let the left margin go
0.046 -> 0.030 and `wspace` 0.50 -> 0.38, so every box got wider.

"calibrated" came out of the title too: it is a caption fact, and the long form
`energy, calibrated (J)` ran off the right edge on the sixth panel.

The A/B gap closed from ~0.33 in to ~0.10 in by raising the B header and the
gridspec together, now that A's staggered tick labels are the only thing between
the two blocks.

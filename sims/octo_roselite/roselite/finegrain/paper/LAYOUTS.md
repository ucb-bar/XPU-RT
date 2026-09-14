# Figure layouts — schedules and trajectories

Four new figures, each a standalone `fig_*.py` that recomputes every number from source
data at plot time and writes one PNG beside itself.

| script | PNG | what it has to say |
|---|---|---|
| `fig_schedule.py` | `fig_schedule.png` | one lane for 677 ms → three lanes but serial → pipelined |
| `fig_lane_balance.py` | `fig_lane_balance.png` | the lanes are not equally loaded; that gap is the remaining headroom |
| `fig_trajectory.py` | `fig_trajectory.png` | the grasp collapses down the latency ladder and the push does not |
| `fig_grasp_failure.py` | `fig_grasp_failure.png` | the object still gets moved; it just stops getting grasped |

```bash
cd paper
for f in fig_schedule fig_lane_balance fig_trajectory fig_grasp_failure; do python3 $f.py; done
# each prints every number it drew, so the figure can be checked against stdout
```

Board traces are read from `/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs`,
overridable with `OCTO_REPRO_RUNS`. Sim data is read relative to the script
(`../traces_torque`, `../g5fine/runs`).

`fig_operating_points.py`, `fig_pareto.py`, `fig_metrics.py` and `fig_energy.py` are
untouched.

---

## Conventions kept from the existing four

* Family colours: ideal `#7f8c8d`, pipelined `#16a085`, serial `#e67e22`,
  CPU-only `#c0392b`.
* MEASURED / MODELLED / PREDICTED stated in words on axis labels, titles and captions.
  Everything in these four is MEASURED; the only derived quantities are labelled as
  bounds (`fig_lane_balance` panel B) or reconstructions (`fig_schedule` panel 2).
* An italic grey caption under every figure carrying the caveat that actually matters,
  not a generic disclaimer.
* Medians where the distribution is heavy-tailed, with the reason given.
* `n` printed wherever it varies, and coloured red where it is small.
* Nothing hardcoded. Latencies, cadences, per-lane busy, success rates, observation
  ages and the drawer's success threshold are all read or derived from the runs' own
  `summary.json` / trace CSV at plot time.

## Conventions added

* **No `bbox_inches="tight"`.** All four use explicit `left/right/top/bottom` margins and
  a plain `fig.savefig(path)`. `fig_energy.py` documents the failure mode this avoids:
  an artist outside the axes counts toward a tight bbox and once produced a
  61,744-pixel-tall image. With fixed margins the canvas size is a property of
  `figsize` and `dpi` alone. Final sizes: 2460×1410, 2640×930, 2639×1508, 2610×990 px.
* **Replicate selection is by median, never by recency.** `plot_gantt_multi.py` picks
  the *newest* log matching a glob, so adding a board run silently redraws a published
  figure — `REPRODUCE_SCHEDULES.md` §9 warns about exactly this. `fig_schedule.py`
  instead parses every replicate of a configuration and draws the one whose
  per-inference span is closest to that group's median, printing the whole spread.
* **`n` means independent board processes**, not dispatches. The between-process noise
  floor is ~15% (`OCTO_INT8_QRB5165.md` §5.4), so a single process is a sample of one.

---

## (a) `fig_schedule.py` — hardware-lane occupancy

### Layout

Four stacked panels on **one shared x-axis whose width is the CPU-only baseline's own
measured inference span (677.1 ms)**. Three lane rows per panel (CPU / DSP / HTA,
labelled with the machine id as well, because the trace's `kind` column is the machine
and `backend_label` is the segment). A right-hand gutter carries, per lane, busy ms and
duty cycle, plus one line giving that schedule's action rate and its ratio to the
baseline. A `▼` above each panel marks every inference completion — the instant a fresh
action exists, which is the only thing the closed loop downstream cares about.

Panel 4 is the pipeline ladder for panel 3: one row per in-flight inference over the
same window, so overlap is legible as rows rather than as colour.

### Why this frame

The unit of comparison is **"in the time the CPU-only baseline needs for one inference,
how many actions did this schedule deliver and how much of the SoC moved?"** That makes
the answer to all three of the brief's first three points readable off one axis:
1 action / 2 actions / 4 actions, and DSP+HTA idle 100% → all three lanes used but
never simultaneously → all three lanes used simultaneously.

### Specific decisions

* **Tiling in panel 2, drawn as a reconstruction.** The 3-way trace contains one
  inference (71 dispatches, one instance) because a serial schedule cannot start the
  next until this one ends — the very fact the panel exists to show. The window is
  filled by repeating the measured template at its own measured period, with repeat 1
  solid and the tiled repeats translucent + hatched, a dashed period boundary at each
  tile, and a legend entry that says "TILED". The tiling is exact for a strictly serial
  chain (verified: max concurrent inferences = 1) up to the ~15% process noise, and the
  caption prints the four measured replicates (239–270 ms) so the reader can size it.
* **The dependency staircase.** Thin grey connectors from the end of dispatch *k* to the
  start of *k+1*, in the trace's own start order, drawn on the real repeat only. This is
  what makes "no lane overlap" self-evident instead of asserted: 70 connectors zig-zag
  between lanes and never once run in parallel.
* **A 4-shade teal cycle for inference index**, not a 10-colour qualitative palette. At
  most 3 inferences are ever in flight (asserted from the trace at plot time), so two
  concurrent inferences can never share a shade, and the panel keeps its family colour.
* **The gate is checked, not assumed.** `runtime_main.cpp` busy-waits each dispatch to
  its scheduled start unless `XPURT_NO_GATE=1`; §5.1 shows a run that dispatched 0 of 71
  segments still clocking the predicted makespan. The script recomputes
  `gate_done_ms - dep_wait_done_ms` per dispatch and prints the held count on the
  figure (0 of 802).

### Rejected

* **The draft's dense barcode** (`plot_gantt_multi.py`, 710 dispatches over the full
  1114/1518 ms wall). Unreadable, as the brief says — and worse, its utilisation
  percentages are diluted by ramp-up and drain, so the pipelined panel reports numbers
  that describe the run's ends rather than its steady state.
* **Three panels each at its own natural time span.** The throughput comparison is the
  whole point and cannot be read off three different x-scales.
* **Colour-by-segment in the pipelined panel.** Eight segments × ten instances says
  nothing about overlap. Segment identity is moved to `fig_lane_balance`, where it is
  quantitative rather than decorative.
* **Drawing one serial inference and leaving 65% of the panel empty.** Honest but it
  throws away the comparison; explicit, visually distinct tiling keeps both.
* **"Overlap %" as the headline** (the draft's per-panel title). It conflates "some pair
  of lanes overlapped at some instant" with "throughput improved". Actions delivered per
  fixed window is the quantity that survives contact with the robot.
* **A utilisation pie / donut.** No time axis, and every claim here is about time.
* **`bbox_inches="tight"`** — see above.

---

## (a, continued) `fig_lane_balance.py` — where the headroom is

Three panels, six MEASURED schedules of the same network ordered by measured cadence.

* **A — per-inference lane work**, each lane's bar split into a stable part and an
  *excess*. Excess is defined per run as the time a dispatch spends above **that run's
  own median duration for its segment**, summed. The result: the CPU lane's stable part
  is 65–74 ms in every schedule while its total goes 67 → 89 ms. All of the growth is
  excess. That matches `REFINED_SCHEDULE.md` §4.2 — the `posta` thread-pool stall, a
  runtime defect no scheduler can remove.
* **B — the cadence floor.** Each schedule's measured cadence against two bounds
  computed from its own lane work: the bottleneck lane (what this placement allows) and
  total-work / 3 (perfect balance). The gap between them is the headroom.
* **C — where the growth is *not*.** Median dispatch duration for the segment that
  defines each lane. DSP `pre2` gets ~27% cheaper once the lane is kept busy
  (7.24 ms serial → 5.3–5.6 ms pipelined) — the accelerator is warm. CPU `posta`'s
  median does **not** move at all (3.89–4.34 ms across every schedule), which is the
  point: panel A's CPU growth is entirely tail, so it is a stall and not a cost.
  HTA `mlp` sits at 2.2–2.9 ms except in pipe200, whose single process reads 3.94 ms;
  at n=1 that is inside the noise floor and is not read as a trend.

### Specific decisions

* **Excess is threshold-free.** The writeup counts `posta` dispatches over 10 ms. That
  needs a magic number, and it reports a smaller figure than the median-excess
  definition (9.8 vs 18.7 ms/inference for p150/w300). The figure states its own
  definition in the legend so the two are not confused.
* **`gc105ruy*` excluded from the `gc105` group.** They are `XPURT_RUYCAP` sweep points
  — a different *runtime* configuration — so they are not replicates.
* **Panel B's "perfect balance" line is labelled a bound, not a plan**, in the caption:
  it assumes work can be moved between lanes at these costs, and it cannot be exactly —
  a segment costs differently on a different backend and several refuse to compile at
  all (the backend-support table in `OCTO_INT8_QRB5165.md`).
* **Lane colours** (CPU `#c0392b`, DSP `#2980b9`, HTA `#7d3c98`) deliberately differ
  from the family colours; family identity is carried by the bar fill in panel B and the
  x-tick label colour elsewhere. Reusing `#c0392b` for the CPU lane is intentional — the
  CPU-only baseline *is* the CPU lane.

### Rejected

* **Means.** Dispatch durations are bimodal (~3.9 ms and ~21 ms modes) and the process
  noise floor is ~15%; a mean tracks whichever process happened to stall more.
* **Including the CPU-only monolith.** 675 ms on one lane flattens every other bar, and
  it has no cadence to compare. Stated in the caption rather than silently dropped.
* **A full per-segment stack for all three lanes.** Eight segments × three lanes × six
  schedules is a wall. Panel C carries the segment story with three lines.
* **Max latency anywhere in this figure.** §4.1 shows two consecutive processes running
  the *identical* schedule whose median span moves 1% and whose max moves 24%. Cadence
  and median are reproducible; max is not, so max is not drawn.

---

## (b) `fig_trajectory.py` — the rollouts in their scenes

### Layout

Rows are tasks (eggplant = GRASP, close-drawer = PUSH), columns are the four schedule
families at their measured observation age. Each panel: the tick-0 frame lightened 38%,
the end-effector path coloured by observation age on a shared 0–1400 ms scale, the
object's own path in blue, events anchored to whatever they are actually about, and a
compact event box in whichever corner the drawn elements use least.

### The three draft faults, fixed

1. **Object events anchored to the object.** `moved_correct_obj` is a fact about the
   object (it travelled > 3 cm). It is drawn on the object's path, with a dashed leader
   to the nearest **robot link centre of mass** and the measured 3D distance printed.
   Arm events (gripper close, grasped, held) are drawn on the arm. The measured
   justification is in `fig_grasp_failure` panel B: median end-effector–object distance
   at that instant is 1.4 cm on the ideal arm and 4.8 cm on the CPU-only arm, and the
   worst single case in this dataset is **16.1 cm** (serial283, episode 2) — nothing was
   touching it.
2. **The object has its own path.** With a tick-0 background and an end-effector path
   only, a figure silently asserts the scene never moved. Object start (hollow) and end
   (filled) markers and the total travel in cm are printed.
3. **No inline event labels.** Corner box, and the corner is chosen by testing the four
   corner *rectangles* the box would actually occupy — see "verification" below for why
   a quadrant test was not enough.

Plus: the leader searches **all 14 widowx links**, not the gripper, and the arm's pose
at the object-motion instant is drawn as a dashed thin polyline of the links that
project inside the frame.

### A fourth fault the draft had that the brief did not list

`NONDETERMINISM.md` records three byte-identical invocations of the same configuration
returning False / True / False, and concludes that **no single episode's outcome can
carry a claim**. The draft put `SUCCESS` / `FAILURE` in the panel title as the headline.
Every panel here prints the arm's aggregate rate over all 24 configs *and*, separately
and secondarily, the outcome of the one rollout drawn.

### Specific decisions

* **Episode choice is disclosed on the figure.** Grasp row = one of the 4 configs where
  every arm registers an object-motion event (so the anchoring is comparable across the
  ladder). Push row = one of the 3 configs where all four rollouts succeed (which is the
  point of that row). The per-arm rates printed on every panel are over all configs and
  are not selected.
* **"Same scene" is verified geometrically at plot time**, and the script asserts it.
  Initial end-effector positions agree exactly and object positions to ≤ 0.082 mm. See
  the data-integrity note below for why a pixel hash is the wrong test.
* **`pipe200` is the pipelined exemplar in both rows**, so the row-to-row comparison
  holds one thing fixed; also it is the only pipelined arm whose trace directory agrees
  with its own sweep on all 24 episodes.
* **`traces_torque`, not `traces_energy2`**, so these panels describe literally the same
  episodes as `fig_energy.py`. The script reproduces that figure's path quirk (the
  eggplant baselines are stored without the `egg_` prefix).
* **The drawer row carries no object path and no gripper event.** `obj_xyz` is all NaN
  for google_robot and the evaluator exposes only the drawer joint position. The panel
  prints that joint value against a threshold **derived from the data**: over the 96
  episodes of the four arms drawn, every success has qpos ≤ 0.050 m and every failure
  ≥ 0.052 m.

### Rejected

* **Draft C, two arms overlaid on one scene.** The draft itself notes that two paths
  separating does not isolate latency, because two runs of the *same* arm separate too.
  Drawing them overlaid invites the reader to make that inference anyway.
* **Drafts A and B** (single arm, colour by time). One arm cannot show a ladder, and
  time is recoverable from the event box.
* **A per-panel "observation age vs time" strip under each scene.** Redundant once the
  path is coloured by age and the panel prints the arm's mean age; it would have cost
  25% of the vertical space.
* **A solid full kinematic chain.** The widowx base is mounted outside this camera's
  view, so proximal COMs project at or beyond the frame edge and a solid polyline reads
  as a *path*. Drawn dashed, thin, low-contrast and clipped to in-frame links — while
  the nearest-link *search* still runs over every link, in 3D.
* **A gripper-close marker on the drawer row.** The `grip` channel's sign convention is
  only established for the widowx tasks; the google_robot trace runs −0.04 → 0.75, so
  the widowx crossing test would have produced a marker that means nothing.
* **A random episode.** Disclosing the selection rule is more useful than pretending
  there wasn't one.

---

## (b, continued) `fig_grasp_failure.py` — the mechanism, over every episode

Three panels: the divergence (240 episodes/arm), the distance (24 episodes/arm), the
contacting link (24 episodes/arm). It exists so the ladder in `fig_trajectory.png` reads
as a measurement rather than an anecdote, and it is where the 16.1 cm case lives.

* Object moved 86.7% → 70.4% while grasped falls 84.2% → 10.4% and success follows
  *grasped*, not *moved*.
* Median end-effector–object distance at first object motion: 1.4 → 1.2 → 1.6 → 3.3 →
  5.0 → 4.8 cm up the ladder.
* Share of object-motion events whose nearest link is a **finger** rather than the
  gripper frame: 5% → 5% → 0% → 17% → 30% → 32%.

### Specific decisions

* **Categorical x, ordered by age — not an age axis.** A continuous age axis produces a
  sawtooth (serial283 at 405 ms scores below pipe110fix at 421 ms) that implies success
  is a function of age, which `fig_operating_points.py` exists to refute. The axis label
  says so and the caption repeats it.
* **Design-corrected intervals**, the same clustering estimator as `fig_metrics.py`, so
  the two figures' error bars mean the same thing.
* **n is printed as `n/24`** in the tick labels, and the caption states the direction of
  the bias it creates: an arm that never moves the object contributes no distance, which
  biases panel B *downward* for the slow arms — against the effect shown.
* **"Further up the arm" is kept in panel C's legend even though it is 0% everywhere.**
  That the search covered elbows and found none is the finding.

### Rejected

* **Means for the distance.** One 16.1 cm case in 128 would move them.
* **Including the drawer in panel A.** It has no object-motion events at all.
* **Reusing `fig_metrics.py`'s per-task success grid.** Already published; this figure
  adds the mechanism, not another view of the outcome.

---

## Numbers that did not reproduce, and what the figures print instead

Everything below is recomputed at plot time from the archived traces; the figures print
the recomputed value, not the value in the brief or the writeup.

| claim as given | recomputed | note |
|---|---|---|
| "HTA ~30% loaded, CPU and DSP near 55%" | **CPU 55.1 / DSP 53.3 / HTA 20.3%** on `p150w300_...164100` (the single run the sim arms use); **59.9 / 54.9 / 22.2%** as the median over its n=3 replicates; **70.2 / 64.9 / 24.5%** on the recommended greedy p105/w300 (n=4) | "near 55%" matches p150/w300. HTA is **20–25%** of the release interval, not 30%. 30.6 is the HTA's **ms per inference**; 29.6% is *pipe110*'s HTA duty cycle. The qualitative point — HTA carries about a third of what CPU and DSP carry — is unaffected and is what the figures say. |
| "126–146 ms idle per lane" (§5.2, from a 227.1 ms wall) | **140–209 ms** on the median-span archived replicate (CPU 183, DSP 140, HTA 209 of a 250 ms inference) | The four archived `ungated_*` traces are all longer than the 227.1 ms trace §5.2 quotes. Direction and magnitude hold; the exact range does not reproduce. `fig_schedule` prints its own. |
| p150w300 cadence 150.4 ms | **146.5 ms** median over n=3 | 150.4 is the single archived run the sim arms were configured from. |
| serial283 latency 283.4 ms | archived replicates span **239.4–269.7 ms**, median 255.0 | Already flagged in `REPRODUCE_SCHEDULES.md`'s troubleshooting table — "a 3-way re-run lands at 250-265, not 283.4". |
| pipe110 cadence 111.4 ms | **98.2 ms** from the trace | Span 385.1 ms matches. |
| "in one measured case the arm was 15.8 cm away" | **16.1 cm** (serial283, episode 2, tick 220; nearest link `left_finger_link` at 12.4 cm) | Marked on `fig_grasp_failure` panel B. |

## Data-integrity findings

* **`pipe200_gated_*.log` is not actually gated.** 0 of its 355 dispatches were held by
  the start-time gate (`gate_done_ms == dep_wait_done_ms`), despite the filename. It is
  therefore safe to use. **`mono_gated_*.log` genuinely is** — 16 of 21 dispatches held —
  so `fig_schedule` globs `mono_2*.log` and excludes it.
* **`traces_torque/drw_pipe110` disagrees with itself.** Its stored per-episode
  `ep*_trace.json` outcomes differ from that run's own `summary.json` on 8 of 24
  episodes; every other run in the set agrees 24/24. This is the mechanism
  `NONDETERMINISM.md` §3 describes — the per-episode traces come from their own runs, not
  from the sweep. Consequence: the rollout badge uses the trace's own flag, the aggregate
  uses the summary, and `drw_pipe110` is not drawn. The counts are printed on the figure.
* **Tick-0 background PNGs are not byte-identical across arms** for the same episode
  config, but the scenes are. For `drw` ep12, `lat0` vs `pipe110` differ on 326 of
  327,680 pixels by at most 28/255; initial end-effector positions match exactly and
  object positions to ≤ 0.082 mm. A pixel hash rejects identical scenes; the geometric
  test is the right one and is asserted in the script.
* **The drawer task exposes no object.** `obj_xyz` is all NaN and the only evaluator stat
  is the drawer joint position, so the two rows of `fig_trajectory` are deliberately
  asymmetric and say why.

## Verification log — found by rendering and looking

Every fault below was in a rendered PNG, not in review of the code.

* `fig_schedule` panel 1 reported **"0 inferences in the window"** — completions were
  compared against absolute trace time while the window started at the trace's own zero.
  Fixed by re-zeroing each trace to its first dispatch.
* `fig_schedule` panel 2's gutter showed per-*inference* busy against a per-*window*
  percentage (HTA "6.1%"). Given an explicit denominator and a printed note that the
  percentage is per 250 ms inference.
* `fig_schedule`'s gutter text and the headroom callout sat on the axis spine; the lane
  y-limits were opened and the text re-anchored.
* `fig_lane_balance` panel A **clipped its tallest bar** (DSP serial, 110 ms) behind the
  legend, because the y-limit was scaled to the CPU maximum rather than the maximum over
  all lanes.
* `fig_lane_balance` panel B's "+N slack" labels landed on the bound markers. Moved,
  given a white bbox, renamed "above bound", and the serial row's legend entry corrected
  — a serial schedule has a span, not a release period, so "slack" was wrong there.
* `fig_lane_balance` panel C had CPU `posta` and HTA `mlp` coincide at 3.94 ms on
  pipe200, hiding one marker under the other. Distinct markers per lane.
* `fig_grasp_failure` panel B ran its **y-axis to −2.6 cm** to make room for the `n=`
  labels. A distance cannot be negative; `n` moved into the tick labels and the axis
  starts at 0.
* `fig_grasp_failure`'s caption **overflowed the canvas on both sides**; reflowed from 3
  long lines to 7 and the figure grew 0.7 in so the caption stopped colliding with panel
  A's x-label.
* `fig_trajectory`'s panel titles overlapped horizontally across all four columns at the
  first draft width; shortened and re-set at 8.6 pt.
* `fig_trajectory`'s event box **covered the path-start marker** in two panels. A
  quadrant occupancy test was not enough — the box is wider than half a panel, so the
  emptiest quadrant can still be occluded. Replaced with a test over the four corner
  *rectangles* the box would actually cover, with single glyphs (path starts, object
  endpoints) weighted 40× because losing one loses the reading.
* `fig_trajectory`'s arm skeleton was drawn solid and read as a fifth path crossing the
  scene. Made dashed, thin and low-contrast, and clipped to the links that project
  inside the frame.
* All four canvases were checked for size after every change: 2460×1410, 2640×930,
  2639×1508, 2610×990.

# Every figure, and what verifies it

The question this answers is not "does the gate pass" but "for each figure we show, is every
section of it re-derived from the data". It is generated material read once and written down:
`scripts/verify_showdown_figure.py --coverage` emits the per-panel table, and the sweep behind the
counts below ran that verifier over **every** sidecar in `results/codesign_feedback/refined/`, not
only the ones the gate names.

---

## 1. The figures the paper carries

Matched against what the paper's `.tex` **includes**, not against what sits in its `plots/`
directory: a figure can be cut from the text and leave its file behind, and counting the file
reports a figure the paper does not carry. `scripts/figure_index.py` does the match by sha256 of
every included graphic against every render under `results/`.

The paper includes **21 graphics**. Four come from this work, and since 2026-09-28 **all four are
byte-identical to a render here**:

| paper figure | repo render | verified |
|---|---|---|
| `fig_hil_showdown` | `warehouse_showdown_cam30_solver_placed` | **every panel**, and **76/76 numbers load-bearing** under `mutation_audit.py` |
| `fig_hil_showdown_rate_gain` | `..._rate_gain` | as above, 76/76 |
| `fig_loop_overview` | `loop_overview` | **both bands** — `verify_loop_overview.py`, 49 checks, 0 failed. The paper's copy is this render since 2026-09-28. |
| `fig_cores_yolo` | `cores_yolo_service_derived_flat` | **every sourced point** — `verify_cores_yolo.py`, 0 failed, and 164/164 numbers load-bearing. The paper's copy is this render since 2026-09-28. |

The ROS 2 tier of each figure: `docs/Baselines/ros_baseline_tiers.md`.

The remaining 17 are from other work — the Qualcomm and QNN figures, the sharding set, the solver
frontier and Pareto plots, `fig_smolvla`, the feedback-channel figures and three diagrams. None has
a byte match under `results/`, and none is this campaign's to verify.

**`fig_schedule_evolution` is no longer in the paper.** Paper commit `aadc83d` ("Replace the
schedule-evolution figure with the full loop overview, double column") removed the
`\includegraphics`; `plots/fig_schedule_evolution.{png,pdf}` are orphan files left behind. The
repo's `schedule_evolution_mega` still matches those orphan bytes and is still verified
(`verify_schedule_evolution.py`, 29 checks, 0 failed, extended to the `short` and `tall` siblings),
so nothing is lost if the paper takes it back — it just is not carried today.

That figure also needed `--height 4.35` to reproduce: the documented command omitted it and drew
the same numbers at 3.7 in. `scripts/make_all_codesign_figures.sh` now passes it.

---

## 2. All 77 sidecars under `refined/`

| class | count | what it means |
|---|---|---|
| in the gate | 28 | verified on every run, **0 FAIL**, every panel reached |
| verified by a producer-specific script | 19 | not showdown composites, so `verify_showdown_figure.py` does not know them; each is re-derived by re-running the script that drew it (see below) |
| stale inputs | 21 | drawn against files that have since changed — the figure is not wrong, it is no longer re-derivable to the same numbers |
| no verifier | **0** | — |
| panel disagreement | **0** | — |

`scripts/verify_showdown_figure.py --all`: **38 ok, 21 stale, 0 unverifiable.**
The 21 stale renders are the only figures left that nothing re-derives.

### The producer-specific verifiers

Each re-runs the script that drew the figure into a temporary directory and compares the sidecar it
emits, field by field, with the committed one. That covers every number without restating any of the
arithmetic, so the check cannot drift from the renderer.

| verifier | stems | what it re-derives |
|---|---|---|
| `verify_schedule_evolution.py` | `schedule_evolution_{mega,short,tall}` | all four panels' makespan, drawn deadline misses and per-network misses |
| `verify_loop_overview.py` | `loop_overview` | both bands' cells, levers and all four evolution stages' misses |
| `verify_hil_feedback.py` | `hil_feedback_{a90,a90h,a120h,a120e,b5,close_120}` | per round: solver, calibration multiplier, every network's window misses, control-gap mean, camera→control median, predicted misses |
| `verify_control_rate_response.py` | `control_rate_response{,_cp3,_v2}` | every drawn point's n, completions, cadence and mean gates, through `flight_quarantine` |
| `verify_ros_effort_ladder.py` | `ros_effort_ladder_v2` | every rung's harts, control rate, camera→goal latency and flight tally |
| `audit_showdown_claims.py` | `audit_showdown_claims` | recomputes its own contrasts and exits non-zero when the envelope's floor contrast is not a positive, significant step |
| `verify_cores_yolo.py` | `cores_yolo_service_derived_{published,published_a24p5,flat,flat_a22}` | every sourced point against the schedule it names, both published cruise-label triples, and that each CP-SAT schedule packs to the tightest deadline attempted |

`verify_showdown_figure.py` carries the map in `VERIFIED_ELSEWHERE`, so asking it about one of these
sidecars names the verifier that owns it rather than reporting no verifier.

Two notes on what these found.

**`ros_effort_ladder` pools its flight column over `campaign*/campaign*.csv` by glob**, so the column
grows whenever a campaign directory is added. Nine landed after the figure was drawn, and the
XPU-RT rung moves from 29/192 to 44/300. Hashing the inputs a render recorded cannot see this: all
33 recorded files are unchanged while the population is larger. `verify_ros_effort_ladder.py`
therefore compares the input *set* too, `ros_effort_ladder_v2` is the render against today's
population, and the original stays as drawn (`--stem ros_effort_ladder` reports it).

**Ten renders were being marked as disagreeing with their own data when they agree with it exactly.**
The displayed-flight check asserted two things at once: that a panel's numbers match the flight
behind it, which is correctness, and that the pair reads the way a display figure should — XPU-RT
completing, the baseline crashing mid-course. Several renders break the second deliberately (an
`_xpu_3of4` variant exists to show a non-completing XPU-RT flight). The correctness half is still
asserted and still fails when a panel's stated gate count differs from the dump; the convention is
reported.

### Stale inputs (21) — disposition

The campaign CSVs grew, `ros_traced/summary.csv` was regenerated and the Gantt sidecars were rebuilt
after these were drawn. None is gated and none is in the paper. They are **not re-rendered**: a
re-render draws today's population, which is a different figure from the one that was published or
reviewed, so replacing them would destroy the record rather than fix it.

Grouped by the script that drew each, which is the honest discriminator — several are not showdown
composites at all but a different figure kind:

| producer | stems | disposition |
|---|---|---|
| `showdown_v3_figure` | `warehouse_showdown_v3`, `..._paper`, `..._paper_tall1005_{board,inb,none}`, `..._tall1000_board`, `..._tall1005_{board,inb,none}` (9) | the v3 form, superseded wholesale by the paper forms and then by the five audited stems. Kept as the record of what v3 showed. |
| `showdown_paper_figure` | `warehouse_showdown_cam45_unpinned_best` | **superseded by `warehouse_showdown_cam45_unpinned_best_v2`**, which is in the gate and re-derives |
| | `warehouse_showdown_cam45_unpinned_seed1000` | same arms at seed 1000; the current form of that pairing is `showdown_45hz_pinned_vs_rosdefault_s1000`, in the gate |
| | `warehouse_showdown_paper_cal17` | a calibration variant with no successor; nothing draws from it |
| `showdown_final_figure` | `warehouse_showdown_final` | an earlier full composite, no successor |
| `showdown_paper10_figure` | `warehouse_showdown_paper10` | an earlier 10-panel composite, no successor |
| `showdown_atlas` | `warehouse_showdown_atlas` | the 11,352-flight atlas — the best-evidenced flight figure here on population, and outside the audit. Stale only because the campaigns it pools grew. |
| `story_figures:*` | `course_progress`, `crash_position`, `latency_waterfall`, `rate_speed_map`, `ros_ladder`, `seed_pairs` (6) | a different figure kind, not showdown composites. All six re-render from `scripts/story_figures.py <name>`; none is used by the paper or the audited set. |

Every one of the eight producers still exists in `scripts/`, so any of these can be redrawn against
today's data on request — the choice not to is deliberate, not a missing capability.

## 2b. Which recorded numbers are load-bearing

A green gate says the checks agree with the figures. It does not say the checks *could* disagree.
`scripts/mutation_audit.py` settles that by mutation: every number in every verified sidecar is
perturbed in turn, the owning verifier re-run, and the result recorded as detected or not.

**5049 of 5145 numbers across 55 figures are load-bearing, and 54 of the 55 verify every number
they record** — including both figures the paper carries, at 76/76 each. The first sweep measured
3602; the difference is checks added for what it found unchecked: `ctrl_gap_mean_ms`, the panel B
rate axis and its two markers, the Gantt bar counts, a `series` never compared with the
`provenance` recording where its points came from, and — reusing `moments_for()` and the registry,
which were there all along — `clearance_m`, every callout's `step`, and `camera_hz`.

The last seven closed on 2026-09-29, and each had been written off too early. `scene_mean_gates.xpu`
was guarded on the ROS arm also being present, and the ladder renders name a ROS trace that scene
campaign never flew — so a real number rode along unverified behind a null one. The Gantt bars'
`drawn` count was compared only as "no more than the dispatches", which any smaller value satisfies;
it is now re-derived by calling the producer's own `merge_dispatches`. And the solved rung's
`cores_carrying_work` was simply never compared, by a verifier that re-runs its producer anyway.

**All 96 that remain belong to `audit_showdown_claims`**, which is a script's own regenerated output
rather than a figure sidecar: mutating it and re-running simply rewrites it, so mutation says
nothing there. **No number a figure records is unchecked.**

`docs/Artifact/mutation_audit.md` is the per-figure table and the remaining gaps by field;
`scripts/mutation_audit.py --check` fails if any figure's split changes.

---

## 3. Per-panel coverage of a gated showdown figure

`verify_showdown_figure.py --metrics <sidecar> --coverage` prints this per figure:

```
  panel            checked  asserted only  failed
  A                     15              0       0
  a-d                    5              0       0
  B                      1              0       0
  C                      1              0       0
  D                      1              0       0
  E-H                    5              0       0
  I                     21              0       0
  whole figure           4              0       0
```

*asserted only* counts checks that record where a number came from without recomputing it. Panel I
held the last two: its per-hart busy percentages were recorded with their source and never
re-derived. They now come back from the tracked sampler CSV over the window the renderer used —
8 numbers per row, 50 Gantt sidecars, all matching — so a gated showdown figure has none left.

## 4. What is still not checked

### `fig_cores_yolo` — 10 of 30 points have no schedule behind them

The figure is now derived rather than held as literals: `scripts/cores_yolo_service.py` computes
every point as the median per-frame YOLO response of a schedule under `schedules/`, writes a
sidecar naming that schedule and hashing it, and reproduces both published cruise-label triples
exactly — 1.04× / 0.90× / 0.79× at the 24 ms anchor and 1.07× / 0.91× / 0.80× at 24.5.
`scripts/verify_cores_yolo.py` checks every sourced point back against those schedules and is
gated. `docs/Evaluation/cores_yolo_reproduction.md` is the page; `scripts/attic/fig_fair_v6.py` preserves the
untracked script that drew the paper's bytes.

What remains not checked, because no data exists to check it against:

* **Eight ROS points.** `scheduled_ros_pin_{predicted,board}.json` is one 1-hart pin, not a sweep —
  which is what the caption's "core-independent" means. The published curve nevertheless varies by
  5 ms across five widths, and no schedule here produces its K4–K7 values. They are drawn only in
  the `--ros published` renders and are marked `as published; no schedule in this repository
  produces it` in the sidecar. `--ros flat` draws the one measurement and has two unsourced points
  instead of ten.
* **Two CP-SAT points.** K5 and K7 of the AOT row were never solved; the published figure
  interpolated them and said so, and the sidecar records the interpolation.

And one property of the figure that is now asserted rather than left implicit: **the CP-SAT rows
come from a deadline search, and each schedule packs to the deadline it was given** (within 0.02 ms
at all eight points, which `verify_cores_yolo.py` checks). The whole search is tracked, so the
verifier also asserts that each plotted deadline is the tightest one *attempted* at that width —
and reports the three points (board K7, board K8, AOT K4) where only one was ever attempted, so
nothing claims a tighter one would have failed. The level of that curve is therefore set by where
the search stopped, not by how fast the solver makes the chain run, while the greedy and ROS curves
are achieved service. Four of the five `cbp` solves also report a nonzero `deadline_miss_count`;
the sidecar carries it per point.

The 1.0× anchor is a chosen budget — 24.0 in the paper's copy, 24.5 in `refined/cores_yolo_service.png`,
22 in the paper's prose — and which curves fall inside the band follows from it. It is now a required
flag, drawn on the figure and recorded in the sidecar. Renders exist at all three.

### The rest

* The paper's copies of `fig_loop_overview` and `fig_cores_yolo` are earlier renders than the repo's.
  `scripts/figure_index.py` matches every file in `$XPURT_PAPER_PLOTS` against every render here by
  content, so which of the paper's figures this tree currently carries is printed rather than
  assumed: three of them, byte for byte.
* The 21 stale-input renders of 2. Each was drawn against a smaller population; re-rendering one
  produces different numbers, which is why it is recorded rather than refreshed.
* Flights are not bit-reproducible across machines; every figure reproduces by recipe, not by bytes.


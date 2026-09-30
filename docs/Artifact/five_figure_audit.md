# Are these five figures legitimate, and can they be rebuilt?

Five warehouse-showdown figures are the set under review. This page is the answer to both questions,
panel by panel: what each one claims, which check re-derives it, and what could not be rebuilt.

Nothing here is a summary of the checks — it is a map into them. `verify_showdown_figure.py --metrics
<sidecar> --coverage` prints the same panel table for any figure, generated from the checks
themselves, so this page cannot drift from the code without the table disagreeing.

```bash
bash artifact/verify_no_hardware.sh                     # the whole set, 28 stems, no hardware
bash scripts/render_audited_set.sh                      # rebuild all five from recorded inputs
bash scripts/repro_clean_clone.sh                       # rebuild them in a clean clone (Section 4)
```

---

## 1. The five

All under `results/codesign_feedback/refined/`. The stem names the camera rate, the XPU-RT arm, the
ROS 2 arm and the episode seed, in that order.

| stem | camera | XPU-RT | ROS 2 | displayed | census (completed/12, mean gates) |
|---|---|---|---|---|---|
| `showdown_36hz_solver_vs_rosallhart_s1006` | 36 Hz | solver-placed, 25.8 ms, 100.4 Hz | two YOLO pools + a nav pool, every hart, 37.4 ms, 36.0 Hz | 4/4 vs **2/4**, gate frame | 4/12 · 2.67 — 0/12 · 1.83 |
| `showdown_45hz_pinned_vs_rosdefault_s1000` | 45 Hz | `a_cpsat_hard`: CP-SAT on `wh_chain45_solve.json`, a spec with no placement keys (the stem and some pages call it "pinned"; `xpurt_arm_ranking.md` lists its placement as the solver's choice), 56.9 ms, 96.2 Hz | unpinned default, 242.1 ms, 39.1 Hz | 4/4 vs **2/4**, gate frame | 4/12 · 2.58 — 0/12 · 1.08 |
| `showdown_45hz_pinned_vs_rosdefault_s1003` | 45 Hz | same | same | 4/4 vs 1/4, crate | 4/12 · 2.58 — 0/12 · 1.08 |
| `showdown_45hz_pinned_vs_rospinned_s1011` | 45 Hz | `a_cpsat_hard`, as for `s1000`, 56.9 ms, 96.2 Hz | static 6-core partition, 56.8 ms, 38.5 Hz | 4/4 vs 2/4, crate | **4/12 · 2.75 — 4/12 · 2.83** |
| `showdown_45hz_solver_vs_rospinned_s1007` | 45 Hz | solver-placed, 27.5 ms, 103.5 Hz | static 6-core partition, 56.8 ms, 38.5 Hz | 4/4 vs 2/4, crate | 3/12 · 2.50 — 0/12 · 1.92 |

Reproduction pages: [`showdown_cam36_allcores_reproduction.md`](../Evaluation/showdown_cam36_allcores_reproduction.md),
[`showdown_cam45_ros_unpinned_reproduction.md`](../Evaluation/showdown_cam45_ros_unpinned_reproduction.md),
[`showdown_cam45_static6_reproduction.md`](../Evaluation/showdown_cam45_static6_reproduction.md),
[`showdown_cam45_solver_placed_reproduction.md`](../Evaluation/showdown_cam45_solver_placed_reproduction.md).
`s1003` has none of its own; see §3.

**`showdown_45hz_pinned_vs_rospinned_s1011` is the falsification test, not a win.** Its own legend
reports the baseline completing as often as the scheduled arm and reaching a slightly higher mean
gate count. Give ROS 2 a careful static partition at 45 Hz and the gap closes; the displayed seed
shows it crashing, the population does not. No caption may read that figure as a win.

## 2. What is checked, panel by panel

Every row is re-derived from the files the render read — the display dumps, the campaign CSVs, the
energy CSV, the Gantt sidecars and the board traces behind them. 51 checks per figure.

| panel | what it claims | what re-derives it |
|---|---|---|
| **A** top-down | both flights, their gates, the scene, the gains, the clearance, both control rates, the census in the legend | the two dumps and the scene-census CSVs: gate counts, layout seed, episode seed, `moment_scale`, the baseline's recorded ending, `eff_cmd_hz`, and the injected latency against the arm's registry value; the backdrop is re-composed from `ov_seq` and its plate std compared |
| **a–d** strips | four callouts, each a chase / FPV+YOLO / cross-ToF frame at a stated time | each callout's `t_s` against `t_s[step]` in its own dump, and that a captured frame exists within one stride of that step — the strip is `argmin abs(frame_steps - step)`, so a callout with no nearby capture would draw a different moment |
| **B** envelope | success against control rate over 300 injected-rate flights | k/n per rate against `hil_ablation.csv`, read through the same quarantine the panel now draws through |
| **C** course B | the floor reproduces on 120 flights over gates the policy never saw | k/n per rate against `hil_ablation_courseB.csv` |
| **D** ladder | mean commanded moment (measured) and propulsive power (modelled), relative to XPU-RT | per-condition n and both ratios against the energy CSV, and that the 1× arm is the one the panel draws as 1× |
| **E–H** telemetry | body rate, nav goal heading, forward speed, velocity through the gates | the four means **and** a per-series digest of all 11 drawn curves, recomputed through the same `showdown_final_figure.telemetry_series` the render draws from |
| **I** Gantt | both rows' camera→control, lateness, and which harts each network ran on | the board traces themselves: chain median, frames late, drawn lanes and traced harts against the executed schedule, the sha256 of trace / sampler / manifest, the executed table's own sha256 against the manifest, the solver name, and that both arms ran the same staged YOLO IR |
| whole figure | — | every input file's sha256 unchanged since the render; no silent numeric fallback was used; every ms and Hz in drawn text is a registry value |

Two of these were added by this audit and had **no** coverage before: panel H was checked by nothing
at all, and E/F/G were checked only at their means — a mean survives a curve coming from a different
flight, the digest does not.

## 3. What cannot be rebuilt, and why

* **`showdown_45hz_pinned_vs_rosdefault_s1003`'s flights have no producer.** Its display pair is
  `campaign_v2/display_lat_c1.4/{xpu,ros}_s1003_figdata`; no script in the repository writes that
  directory, `git log -S"display_lat_c1.4" -- scripts/` is empty across every branch, and the Isaac
  logs beside it record no argv. The dumps are archived with their sha256, so the figure **re-renders**
  from a clean clone — it is the flights that are not regenerable. Its render flags are recovered from
  its sidecar, which `scripts/render_audited_set.sh` says in a comment rather than implying.
* **No flight is bit-reproducible.** Isaac is not deterministic across machines, so re-flying tests the
  recipe, not the bytes. That is why the display dumps are archived rather than regenerated, and why
  panel A's legend reports a twelve-seed census rather than resting on the one displayed flight.
* **The board runs need the K1.** Every camera→control number, control cadence and Gantt row is a
  measurement of one binary on one board. The traces are archived; `scripts/verify_board_traces.py`
  requires each one a documented check opens to be tracked, or in an archive with the bytes on disk
  (407 paths today).
* **The `p45free` flights were flown at 28.3 ms** injected latency while the pooled board median is
  27.46 ms — 0.85 ms *against* the scheduled arm. Both values are in the dumps and the registry.

## 4. The clean-clone test

The in-place checks run in a tree where every input happens to be present. They cannot say whether a
reviewer who clones the branch and unpacks `archive_v3/` has what a rebuild needs.
`scripts/repro_clean_clone.sh` does that: clone, unpack only the named tars, re-render through
`scripts/render_audited_set.sh`, then compare every regenerated sidecar against the committed one
field by field. Identical is the pass; any difference is printed in full.

It also runs the repo-wide checks inside the clone, and says in its own log when it had to reuse the
existing virtualenv because no package index was reachable — in that case the run tests the data and
the commands, not that the environment rebuilds from `requirements-host.txt`.

### The 2026-09-24 run, at the 32 checks the gate then had

The host venv could not be built (no package index was reachable), so that run tested the data and
the commands rather than the environment. All six stems rendered and every regenerated sidecar was
identical to the committed one, including the board-derived panel I rows and the eleven panel E–H
digests; `measured_timing.py --verify` passed inside the clone, so the board constants re-derived
there too.

Getting there took four runs, and each failure was a real hole:

1. **The first run reported all six identical, and was meaningless.** The clone checks out the
   committed figures, so a render that fails leaves them in place and the comparison reads them back.
   Every render had in fact died. The script now deletes the audited stems before rendering.
2. **What they died on was `frames/frame_061.npz`.** `scripts/archive_dumps.sh` packed only
   `figure_data.npz` — on the reasoning, written in its own comment, that the frames beside it are
   read by no render. Panels a–d draw them. `verify_archived_dumps.py` could not see this because a
   sidecar records the npz it hashes, so a tar of npz files alone passed every check while the figure
   could not be drawn at all. The archiver now packs whole dump directories, and the check counts
   frames per dump.
3. **Four executed schedule tables were never force-added** past the blanket `schedules/*` ignore —
   the `scheduled_wh_chain*_free_*_clamped.json` family, which is exactly the solver-placed arms. In
   the clone, panel I could not be checked against the table the board ran.
4. **Two checks fail in a clone for a reason that is not a defect**: `verify_archived_dumps.py` and
   `verify_board_traces.py` ask whether an untracked input is in an archive, and the archives are
   gitignored, so a clone has none of its own. They are now pointed at the real archive directory,
   which is what a reviewer has beside a clone.

### The 2026-09-29 run, at 48

Five verifiers, 68 newly tracked schedules and four `cores_yolo` renders have landed since, so a
clone now exercises 48 checks rather than 32. `/scratch` had freed enough to take a ~32 GB clone,
and a package index was reachable, so this run **built the host venv from `requirements-host.txt`**
— it tests the environment as well as the data, which the earlier run could not.

All six stems render from the documented commands and every regenerated sidecar is **identical to
the committed one**, 236 to 254 fields each. The repo-wide checks inside the clone read **18 passed,
2 failed**, and both failures are the one known gap (below).

It took three attempts, and the first two failed on inputs rather than on numbers — which is the
point of running it at all:

* **30 schedules the six feedback figures read** were still caught by the blanket `/schedules/*`
  ignore, so `verify_hil_feedback.py` could not re-run a single producer. Found by recording every
  file the producers open, rather than by guessing at names; the same method found the five int8
  graph IR files under `ModelBlaster/build/`, which that submodule gitignores as build output and
  which are therefore in neither a clone nor the offline bundle. They are archived as
  `modelblaster_ir_v1.tar`.
* **Eighteen tracked symlinks named an absolute path into the working tree**, and
  `energy_dumps_v1.tar` carried the same eighteen. A checkout reproduces a symlink verbatim, so in a
  clone they dangled; the archive then put its own absolute copies back over the relative ones, and
  `tar -x` aborted on the first file that landed on one — so the clone got **no energy dumps at
  all**, and the only sign of it was one line of tar output. The links are relative now, the archive
  carries none, and `verify_archived_dumps.py` fails on any archived symlink.
* **Two checks passed here and failed there for reasons that were not defects in the artifact.**
  `verify_doc_commands.py` called `scripts/env.local.sh` a missing script when it is the file the
  reader writes; a cited path whose `.example` is tracked is now skipped as a template.
  `build_implementation_bundles.py --check` compared sidecars byte for byte, so re-rendering in a
  different directory made six bundles "stale" on a timestamp and six input paths; it now allows for
  when and where a sidecar was written, and nothing else.

**The two remaining failures are one gap.** Both `pytest` (collecting
`tests/test_ime_profile_from_picks.py`) and the nine `verify_doc_commands.py` entries name paths
inside `ModelBlaster/`, which comes up empty in a clone because the pinned commit is not on
`origin`. `artifact/history/modelblaster.bundle` carries it offline — `git clone -b <branch>` from
the bundle restores `ed776fd1` with 1739 files — and closing it properly needs a push, which this
work does not do.

`flight_quarantine.py` used to be a third: the campaign logs it cited were untracked and in no
archive. The cited lines are now tracked excerpts carrying each log's sha256, and it passes in the
clone.

### What a replication costs, and why

The `.git` history is what dominates, not the checkout: the
clone is now shallow — with `file://`, because git silently ignores `--depth` for a local-path clone
and copies the whole object store while reporting success.

Measured 2026-09-27: **5.15 GB of
`results/` is tracked** across 15,203 files, and the energy-run `.npz` are **not**
tracked at all. What is large is the working tree, not the checkout — `results/` is ~142 GB on
disk, of which 26 GB is the `archive_v3` tars and the rest is campaign and energy run output that
a clone never receives. `scripts/relevance_audit.py` gives the directory-by-directory account of it.


## 5. Findings this audit produced

Each is a property of how a figure's numbers are derived, found by checking rather than by reading.

1. **The Gantt window depends on the workload spec, which is now a required argument.**
   `make_measured_gantt_pair.py` defaulted `--spec` to a 40 ms perception period; at 45 Hz the spec's
   window is 66.67 ms. Scored against each, the same board run reads 0/44 and 44/44 frames late, from
   byte-identical bytes. Against the 45 Hz spec the **pinned baseline misses no frame at all**; against
   the 40 ms default the earlier pair drew it missing 765 of 769. `--spec` is required, which covers the
   scheduled arm too, whose manifest states no rate for the older guard to catch.
2. **`p45freer`'s recorded chain was 28.30 ms**, above all three replicates (27.454 / 27.464 / 27.455).
   Every other arm records the pooled median; this one did not, and passed on a 1.0 ms tolerance.
3. **Three cadence traces had no recorded cut command**, and a fourth's documented command produced no
   file — `--warmup-ms 3000` is the ROS figure, and an `xpurt_long` run is under a second, so it
   discarded the whole trace while the script ignored the non-zero exit. `xpu_p45det.csv` has never
   existed as a result. All six now have a command, each verified to reproduce its committed file byte
   for byte.
4. **`verify_board_traces.py` did not cover the ROS re-derivation inputs** — zero `ctrl_gaps.csv` and
   zero `summary.csv` among its 236 paths, though `summary.csv` is what every ROS constant is
   re-derived from. 407 paths now.
5. **Panels B and C drew one population and recorded another**, the quarantine applied on one path and
   not the other. They agree today (0 flights quarantined in either grid); they can no longer part.
6. **Panel D collapsed from four bars to two** at 36 Hz because `ARM_NAMES` listed the solver arm once
   per camera rate, so a new rate's arm sorted last and a ladder would have normalised to greedy.
7. **The archives could not rebuild the figures**, and the check that exists to say so could not see
   it — see §4. This is the one that matters most for artifact evaluation: the gate was green and the
   artifact was not reproducible.

## 6. Where to look next

`docs/Artifact/artifact_checklist.md` §5 is the standing record of known gaps, including the ones above that
are recorded rather than fixed. `verify_showdown_figure.py --all` is the backlog over every historical
figure and is not a verdict on this set — run `--metrics` on the figure you care about.

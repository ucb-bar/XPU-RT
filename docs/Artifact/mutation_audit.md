# Which recorded numbers are load-bearing

`artifact/verify_no_hardware.sh` passing says the checks agree with the figures. It does
not say the checks *could* disagree: a check that reads a field and compares it to itself
passes forever, and a number nothing reads looks exactly like a number that is verified.

`scripts/mutation_audit.py` separates the two. For every numeric leaf of every verified
sidecar it changes the value by more than any tolerance, re-runs the verifier that owns
that figure, and records whether the verifier failed.

```bash
scripts/mutation_audit.py            # the sweep, ~1-2 h
scripts/mutation_audit.py --check    # fails if any figure's split has changed
scripts/mutation_audit.py --doc      # regenerate this page from the recording
```

## The measurement

**5049 of 5145** numbers across **55 figures** are load-bearing: perturbing
one makes a check fail. **96** are recorded and unchecked, and
**54 of 55** figures verify
every number they record.

One entry dominates the residue and is not a figure: `audit_showdown_claims` is the output
of a script that recomputes it, so perturbing a value and re-running simply regenerates
the file. Mutation cannot say anything about a self-regenerating output, and its numbers
are checked by the script's own Fisher tests rather than by comparison. Excluding it,
**0**
numbers across the real figures remain unchecked.

22 further sidecars were skipped because their verifier already fails — the
stale-input renders of `figure_verification_inventory.md` §2. A check that is already
failing cannot be mutation-tested.

Undetected is not automatically a defect. A sidecar legitimately records provenance and
counts kept for the record. What matters is that the two populations are named, so a
reader knows which numbers a green gate stands behind.

## Per figure

| figure | verifier | load-bearing | unchecked |
|---|---|---|---|
| `audit_showdown_claims` | `audit_showdown_claims.py` | 0/96 | 96 |
| `control_rate_response` | `verify_control_rate_response.py` | 211/211 | 0 |
| `control_rate_response_cp3` | `verify_control_rate_response.py` | 237/237 | 0 |
| `control_rate_response_v2` | `verify_control_rate_response.py` | 231/231 | 0 |
| `cores_yolo_service_derived_flat` | `verify_cores_yolo.py` | 164/164 | 0 |
| `cores_yolo_service_derived_flat_a22` | `verify_cores_yolo.py` | 164/164 | 0 |
| `cores_yolo_service_derived_published` | `verify_cores_yolo.py` | 156/156 | 0 |
| `cores_yolo_service_derived_published_a24p5` | `verify_cores_yolo.py` | 156/156 | 0 |
| `hil_feedback_a120e` | `verify_hil_feedback.py` | 41/41 | 0 |
| `hil_feedback_a120h` | `verify_hil_feedback.py` | 41/41 | 0 |
| `hil_feedback_a90` | `verify_hil_feedback.py` | 41/41 | 0 |
| `hil_feedback_a90h` | `verify_hil_feedback.py` | 41/41 | 0 |
| `hil_feedback_b5` | `verify_hil_feedback.py` | 59/59 | 0 |
| `hil_feedback_close_120` | `verify_hil_feedback.py` | 4/4 | 0 |
| `ros_effort_ladder_v2` | `verify_ros_effort_ladder.py` | 155/155 | 0 |
| `schedule_evolution_short` | `verify_schedule_evolution.py` | 18/18 | 0 |
| `schedule_evolution_tall` | `verify_schedule_evolution.py` | 18/18 | 0 |
| `showdown_36hz_solver_vs_rosallhart_s1006` | `verify_showdown_figure.py` | 120/120 | 0 |
| `showdown_36hz_solver_vs_rosallhart_s1006_ladder` | `verify_showdown_figure.py` | 126/126 | 0 |
| `showdown_45hz_pinned_vs_rosdefault_s1000` | `verify_showdown_figure.py` | 126/126 | 0 |
| `showdown_45hz_pinned_vs_rosdefault_s1003` | `verify_showdown_figure.py` | 126/126 | 0 |
| `showdown_45hz_pinned_vs_rospinned_s1011` | `verify_showdown_figure.py` | 126/126 | 0 |
| `showdown_45hz_solver_vs_rospinned_s1007` | `verify_showdown_figure.py` | 126/126 | 0 |
| `warehouse_showdown_cam30_allcores_s1001_xpu_3of4` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_s1003` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_s1007` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_v2_s1001_xpu_3of4` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_v2_s1003` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_v2_s1007` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_allcores_v3_s1003` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_solver_placed` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_solver_placed_rate_gain` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_solver_placed_rate_gain_v2` | `verify_showdown_figure.py` | 120/120 | 0 |
| `warehouse_showdown_cam30_solver_placed_v2` | `verify_showdown_figure.py` | 120/120 | 0 |
| `warehouse_showdown_cam30_static6` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam30_threeway` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_cam36_allcores_s1001` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam36_allcores_s1003` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam36_allcores_s1003_ladder` | `verify_showdown_figure.py` | 80/80 | 0 |
| `warehouse_showdown_cam36_allcores_s1007` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam36_allcores_s1007_ladder` | `verify_showdown_figure.py` | 80/80 | 0 |
| `warehouse_showdown_cam36_allcores_s1009` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam36_allcores_s1009_ladder` | `verify_showdown_figure.py` | 80/80 | 0 |
| `warehouse_showdown_cam36_navpool_s1000_xpu_3of4` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam36_navpool_s1005_xpu_2of4` | `verify_showdown_figure.py` | 76/76 | 0 |
| `warehouse_showdown_cam45_solver_placed` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_cam45_static6` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_cam45_unpinned_best_s1007` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_cam45_unpinned_best_v2` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_paper_allcores` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_paper_allcores_merged` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_paper_ladder` | `verify_showdown_figure.py` | 82/82 | 0 |
| `warehouse_showdown_paper_r30` | `verify_showdown_figure.py` | 70/70 | 0 |
| `warehouse_showdown_paper_r36` | `verify_showdown_figure.py` | 70/70 | 0 |
| `warehouse_showdown_paper_submitted` | `verify_showdown_figure.py` | 70/70 | 0 |

## What is still unchecked, by field

Grouped across figures, so a gap that repeats in 38 renders reads as one gap.

**Every row below belongs to `audit_showdown_claims`**, which mutation cannot speak
about: it is a script's own regenerated output, so perturbing a value and re-running
rewrites the file. The `figures` column counts entries inside that one file, not
figures. No number a figure records is unchecked.

| field | figures | 
|---|---|
| `.COURSE A:20->100.from[]` | 2 |
| `.COURSE A:20->100.to[]` | 2 |
| `.COURSE A:20->25.from[]` | 2 |
| `.COURSE A:20->25.to[]` | 2 |
| `.COURSE A:20->33.from[]` | 2 |
| `.COURSE A:20->33.to[]` | 2 |
| `.COURSE A:20->50.from[]` | 2 |
| `.COURSE A:20->50.to[]` | 2 |
| `.COURSE A:25->100.from[]` | 2 |
| `.COURSE A:25->100.to[]` | 2 |
| `.COURSE A:25->33.from[]` | 2 |
| `.COURSE A:25->33.to[]` | 2 |
| `.COURSE A:25->50.from[]` | 2 |
| `.COURSE A:25->50.to[]` | 2 |
| `.COURSE A:33->100.from[]` | 2 |
| `.COURSE A:33->100.to[]` | 2 |
| `.COURSE A:33->50.from[]` | 2 |
| `.COURSE A:33->50.to[]` | 2 |
| `.COURSE A:50->100.from[]` | 2 |
| `.COURSE A:50->100.to[]` | 2 |
| `.COURSE B:25->100.from[]` | 2 |
| `.COURSE B:25->100.to[]` | 2 |
| `.COURSE B:25->33.from[]` | 2 |
| `.COURSE B:25->33.to[]` | 2 |

The sweep is what found them. Every field above was recorded by a render and read by
nothing; where one carried a claim, a check was added to the owning verifier and the
figure re-measured.


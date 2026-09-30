# Artifact — what to run, in what order

This directory is a signposted path through the repository. It holds no data of its own: every
step below names a script that is already in `scripts/`, a document that already records the
command, or a file that is already committed. Two thin drivers are the only new code
(`verify_no_hardware.sh` and `02_schedules/check_schedules.sh`); both call the existing scripts.

`docs/Artifact/artifact_checklist.md` is the authoritative inventory — what is stored and why, what is
generated and by what, what the checks are, and what the known gaps are. Read it alongside this
page; where the two differ, the checklist is the record.

---

## What needs what

| step | directory | needs |
|---|---|---|
| 0. verify the claims from the committed data | this page, §"No hardware" | **nothing** — host CPU only, ~45 s |
| 1. build and deploy the kernels and the ROS 2 node | [`01_board/`](01_board/) | K1 board + the cross toolchain |
| 2. check the executed schedules | [`02_schedules/`](02_schedules/) | **nothing** — host CPU only |
| 3. measure both runtimes on the board | [`03_measure/`](03_measure/) | K1 board |
| 4. fly the census, the display pair, the scene and the energy runs | [`04_fly/`](04_fly/) | GPU (Isaac Lab) |
| 5. render the figures and verify them | [`05_render/`](05_render/) | **nothing** — host CPU only |
| — the exact source of both repositories, offline | [`history/`](history/) | **nothing** |

A reviewer with neither a board nor a GPU can still do steps 0, 2 and 5: the flight CSVs and the
figure sidecars are committed, the raw board traces are archived outside git
(`docs/Artifact/external_data.md`, "Raw traces": fetch, unpack, then run the gate with `ARCHIVES=`),
and the checks re-derive every recorded number from them. Steps 1, 3 and 4 regenerate those inputs.

## Seeing the figures

`scripts/figure_index.py` writes a contact sheet of every render in
`results/codesign_feedback/refined/` — thumbnail, when it was written and by which script, whether it
carries a sidecar, whether the no-hardware check covers it, and which one the paper carries:

```bash
.venv/bin/python scripts/figure_index.py      # -> results/codesign_feedback/refined/INDEX.html
```

The page and its thumbnails are regenerable, so neither is committed. `docs/Artifact/artifact_checklist.md` §5
remains the record of what is current and what is backlog.

## Start here: a claim, and the one command that re-derives it

Pick a sentence from the evaluation and follow its row. Every command runs on a host CPU in seconds,
with no board and no GPU. `<v>` is `.venv/bin/python`; `R` is `results/codesign_feedback`.

| the claim | where it is drawn | what re-derives it |
|---|---|---|
| ROS 2 chains control to perception, so its command rate is the camera rate — 36–39 Hz — while XPU-RT commands at ~100 Hz off the same camera | every figure, panel A and panel I | `<v> scripts/verify_showdown_figure.py --metrics $R/refined/showdown_36hz_solver_vs_rosallhart_s1006_metrics.json` — both rates come from the flights, and panel I's from the board traces |
| the camera→control chain is 25.8 ms scheduled against 37.4 ms for a baseline using every hart | panel I | the same command; the row is re-derived from `xpurt_long/` and `ros_traced/`, sha256 included |
| below ~50 Hz of command rate the flight rarely completes — 0/60 at 20 Hz and 1/60 at 25 Hz against 23/60 at 50 Hz | panels B and C | the same command: k/n per rate against `hil_ablation.csv` (300 flights) and `hil_ablation_courseB.csv` (120, unseen gates). **The grid varies rate and latency together** — one latency per rate, 20 Hz↔48 ms through 100 Hz↔8 ms — so it shows the floor, not which of the two causes it; and its annotated `+38 pts · p<0.001` is the 20→50 Hz contrast, where the drawn arms are bracketed by 33→100 (+13 pts, p=0.16). |
| the starved baseline thrashes — far more commanded moment per second | panel D | the same command: per-condition ratios against the campaign's energy CSV |
| a baseline given a careful static partition **ties** at 45 Hz | `showdown_45hz_pinned_vs_rospinned_s1011`, panel A legend | `--metrics .../showdown_45hz_pinned_vs_rospinned_s1011_metrics.json` — 4/12 against 4/12, mean 2.75 against 2.83 |
| every board constant the figures print follows from a trace on disk | all of them | `<v> scripts/measured_timing.py --verify` → 0 DRIFT |
| and every such trace is actually present, tracked or archived | — | `<v> scripts/verify_board_traces.py` → 438 paths, 0 failed |
| the figures can be rebuilt from the repository and the archives alone | — | `bash scripts/render_audited_set.sh`, or `bash scripts/repro_clean_clone.sh` for the same thing in a fresh clone |
| 1492 of 4768 dispatches run on the K1 matrix engine, and the engine wins 49 of the 57 measured fused-conv shapes | not drawn (`docs/Artifact/artifact_checklist.md` §5) | `<v> scripts/verify_modelblaster_inputs.py` — both counts re-derived from the tracked schedule and the measured table, plus the submodule pin and the bundle that carries it |

Everything at once, ~45 s:

```bash
bash artifact/verify_no_hardware.sh        # 28 figure stems + 20 repo-wide checks
```

**What it does not cover.** It re-derives recorded numbers from recorded inputs. It does not rebuild
the kernels (needs the cross toolchain, step 1), re-measure the board (step 3), or re-fly anything
(needs a GPU, step 4) — and no flight is bit-reproducible in any case, so step 4 reproduces the
recipe rather than the bytes. [`../docs/five_figure_audit.md`](../docs/Artifact/five_figure_audit.md) is the
panel-by-panel account of what is checked and what is not, and §3 there lists what cannot be rebuilt
at all.

---

## Environments

`docs/Artifact/environment.md` §Environments is the one description; `docs/Artifact/REPRODUCE.md` §0 summarises it.
Two interpreters:

* the **host venv** (`requirements-host.txt`) — scheduler, ModelBlaster, figures, verifiers.
  Everything on this page that does not say "GPU" runs under `.venv/bin/python`.
* the **Isaac Sim / IsaacLab conda env** (`requirements-isaac.txt`) — the flight simulator only.
  `scripts/env.sh` reads it from `ISAAC_PY`; machine paths override in `scripts/env.local.sh`.

`scripts/env.sh` sets `REPO`, `SIM_TREE`, `HOST_PY`, `ISAAC_PY` and `RES`; the flight and render
drivers source it. Export `XPURT_CPSAT_WORKERS=0` before any CP-SAT solve run by hand.

The K1 board is reached over ssh as `$MODELBLASTER_K1_HOST` (default `k1`); `docs/K1/k1_board.md`
has board access.

---

## No hardware: verifying the claims

`docs/Artifact/artifact_checklist.md` §3 is the check list. It is reproduced here verbatim, and
`artifact/verify_no_hardware.sh` runs all of it except the board line in one command (~45 s).

```bash
.venv/bin/python -m pytest tests xpu-rt/tests -q          # the unit and property tests (8 skipped need hardware)
.venv/bin/python scripts/measured_timing.py --verify      # every timing constant re-derived from the board traces
.venv/bin/python scripts/flight_quarantine.py             # every quarantined batch still matches its evidence
.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import figure_constants as F;print(F.check_registry() or 'clean')"
.venv/bin/python scripts/verify_showdown_figure.py --metrics \
    results/codesign_feedback/refined/warehouse_showdown_paper_r36_metrics.json   # 0 FAIL
scripts/board_source_snapshot.sh --verify                 # needs the board: its sources match the repo
```

`measured_timing --verify` is the one that matters most: it re-reads the board traces and
recomputes every latency, control gap and late-frame count the figure prints, and reports DRIFT if
any recorded constant no longer follows from the data.

In one command, with the board line left out and the figure check run over each of the six figures
whose sidecars re-derive today:

```bash
artifact/verify_no_hardware.sh                 # ~45 s, host CPU only
SKIP_FIGURES=1 artifact/verify_no_hardware.sh  # the repo-wide checks only
```

What that printed here, 2026-09-17, with the flight campaigns still running in the background:

```
  pytest tests xpu-rt/tests                                  FAIL
  measured_timing.py --verify (board constants)              PASS
  flight_quarantine.py (faulted batches)                     PASS
  figure_constants.check_registry (display literals)         PASS
  executed_tables.py (schedule-table ledger)                 PASS
  verify_showdown_figure --metrics warehouse_showdown_paper_r36              PASS
  verify_showdown_figure --metrics warehouse_showdown_paper_submitted        PASS
  verify_showdown_figure --metrics showdown_45hz_pinned_vs_rosdefault_s1003                  PASS
  verify_showdown_figure --metrics warehouse_showdown_paper_ladder           PASS
  verify_showdown_figure --metrics warehouse_showdown_paper_allcores         PASS
  verify_showdown_figure --metrics warehouse_showdown_paper_allcores_merged  PASS
  verify_showdown_figure --metrics showdown_45hz_pinned_vs_rosdefault_s1000              PASS
```

`tests/test_showdown_figures.py::Registry::test_every_replayed_trace_has_an_arm` asserts that every
cadence trace on disk names an arm a figure could replay, and `figure_constants.check_registry()`
re-derives each registered arm's latency from the board traces behind it — so a trace that is
measured but drawn nowhere, such as the IME arm's `ctrl_traces/xpu_w2pg36ime.csv`, is still held to
the same provenance as one a figure uses. The first bullet of `docs/Artifact/artifact_checklist.md` §5 states
that arm's position.

`verify_showdown_figure.py --all` re-derives every sidecar in `refined/`, not only these six;
`docs/Artifact/artifact_checklist.md` §5 explains what the wider total counts and why it is a backlog rather
than the artifact's state.

---

## The order

1. **[`01_board/`](01_board/)** — generate the per-network kernels from the IR, build the sharded
   YOLO, push the ROS 2 node and the samplers, build the IME kernel. Board + cross toolchain.
2. **[`02_schedules/`](02_schedules/)** — the schedule tables the board actually executed, the
   codegen contract they have to satisfy, and the feasibility check. Host only.
3. **[`03_measure/`](03_measure/)** — run both runtimes on the K1 across the camera sweep, pull the
   traces back, and cut the cadence traces the flights replay. Board.
4. **[`04_fly/`](04_fly/)** — the flight census, the displayed pair, panel A's scene census and
   panel D's energy runs. GPU.
5. **[`05_render/`](05_render/)** — the render command per current figure and the verifier
   invocation that must return 0 FAIL. Host only.

**[`history/`](history/)** holds git bundles of this repository and of the ModelBlaster submodule at
the exact commits the results were produced at. Both branches are unpushed, so a plain clone cannot
fetch them; the bundles can. `ModelBlaster/` in a fresh clone comes up **empty** for that reason,
and one test under `tests/` cannot collect until it is filled:

```bash
git -C xpurt submodule init
git -C xpurt config submodule.ModelBlaster.url "$PWD/artifact/history/modelblaster.bundle"
git -C xpurt submodule update          # checks out 2b11d038 by sha
```

Clone a bundle with `-b <branch>`: that checks out the named branch whatever refs the bundle carries. Verified 2026-09-30: the ModelBlaster bundle restores `2b11d038`,
the pinned commit, with 1739 files. `artifact/history/README.md` has both routes.

## The documents this points at

| document | what it is |
|---|---|
| `docs/README.md` | the index of every page under `docs/`, grouped by topic |
| `docs/Artifact/artifact_checklist.md` | the authoritative inventory: stored vs generated, the checks, the known gaps |
| `docs/Baselines/ros_baseline_reproduction.md` | the ROS 2 baseline on the K1 end to end — staging, building, the arms, reading the results, scope of each claim |
| `docs/Baselines/ros_baseline_tiers.md` | the three ROS 2 baseline methods (analytical, calibrated analytical, measured), every ROS figure's tier, and each paper figure's tier with the render that replaces it |
| `docs/Baselines/ros_baseline_ladder.md` | every ROS 2 arm selectable by name (`scripts/ros_baseline.py`), ordered as an effort ladder: what each rung changes and costs, what the board measured, how to measure, replay and model it |
| `docs/Artifact/external_data.md` | the inputs the repository does not carry: the training datasets and the ViNT checkout (point at your own copies), and the raw board-trace archives (name, size, sha256, what reads them, how to unpack them and run the gate with `ARCHIVES=`) |
| `docs/Evaluation/showdown_rate_sweep_reproduction.md` | the rate-sweep showdown (`warehouse_showdown_paper_r36`), board to render, as one recipe |
| `docs/Evaluation/showdown_cam45_ros_out_of_box_reproduction.md` | the showdown against out-of-the-box ROS 2 at 45 Hz (the paper figure's configuration), both arms measured |
| `docs/Evaluation/showdown_analytical_reproduction.md` | the same figure with its ANALYTICAL baseline: the pinning model, its assumptions, its inputs, and which panels rest on it |
| `docs/K1/ime_kernel_reproduction.md` | the K1 matrix engine: why it was unreachable, the fused kernel, what it measures |
| `docs/Baselines/ros_arms_catalog.md` | every ROS 2 deployment measured on the board, generated from the run manifests |
| `docs/Evaluation/figure_runbook.md` | every headline figure: inputs and the exact command |
| `docs/Artifact/reproduction_full.md` | the second-form showdown study end to end |
| `docs/Artifact/REPRODUCE.md`, `docs/Artifact/environment.md` | the co-design loop on a new machine; the two interpreters |

# K3 walker feedback experiment

Measures how far the SpaceMiT K3 board run drifts from the schedule XPU-RT
predicts, and compares the fixes on the real board. Each condition fixes the
drift in a different place:

| Condition | What changes | Where |
|---|---|---|
| Before: static walker | nothing (the schedule as solved) | ModelBlaster walker, default |
| Ratio feedback | each op's cost x mean(measured / predicted), re-solve | `xpu-rt/duration_feedback.py` |
| Calibrated feedback | robust per-op costs + measured dispatch gap + blame-driven hart reservation, re-solve | `xpu-rt/calibration_feedback.py` |
| Dynamic walker | held periodic tasks move to idle cores of their kind at run time | ModelBlaster `--dynamic-walker` |
| Global pool | yolov8_nano's dispatches leave their planned cores for one shared ready queue | ModelBlaster `--pool-networks` |

Workload: `data/toplevel/networks_k3_mlp1600_dronet480_yolo64_singletons.json`:
mlp_control at 1600 Hz, dronet at 480 Hz, yolov8_nano at 64 Hz on the K3's X100 (CPUs 0-7) and
A100 (CPUs 8-15) clusters, one core per dispatch, greedy solver, one 62.5 ms hyperperiod.
Profiles: `gen/k3_board_profiles` (measured on the board).

## Results (K3, 2026-11-19)

Same build for every condition (ModelBlaster `5a3d0fe`), runs interleaved, board locked,
desktop services stopped. Every run's outputs are bit-exact against the goldens.
Misses are out of 100 / 30 / 4 instances. The discrepancy is the median of
actual-minus-predicted finish over all dispatches of the network.

| Condition | Runs | Misses mlp / dronet / yolo | Median discrepancy mlp / dronet / yolo (ms) |
|---|---|---|---|
| Before: static walker | 3 | 91-92 / 26 / 1 | +3.7 / +3.6 / +4.0 |
| Ratio feedback, round 1 | 2 | 90 / 25 / 1 | +3.0 / +3.0 / +3.4 |
| **Calibrated feedback, round 1** | 3 | **0 / 0 / 0** | **0.0 / +0.02 / -0.05** |
| **Calibrated feedback, round 2** | 2 | **0 / 0 / 0** | **0.0 / -0.13 / -0.22** |
| Dynamic walker (safety 1.0) | 2 | 0 / 0-1 / 0 | +0.04 / +0.4 / +1.9 |
| Global pool (yolo pooled) | 3 | 2 / 0 / 0 | +0.04 / +0.3 / n/a* |

\* pooled work has no planned slot, so its discrepancy is not meaningful.

- Calibrated feedback is the only condition where the board follows the predicted
  schedule. It reserves an X100 and an A100 hart each for mlp_control and dronet, which
  were being queued behind long yolov8_nano ops on shared harts.
- The dynamic walker and the global pool meet deadlines by moving work at run time,
  so the run no longer matches the plan. The pool needs no feedback round.
- Absolute numbers depend on the board's state: on 2026-11-18 the same static schedule
  missed 83-85 / 21-22, and that day's binary gives 91-92 / 26 today. Compare
  conditions within one campaign, not across days.

Figures (`figures/`): `summary_misses_and_discrepancy.png`, and predicted-vs-actual Gantt
charts for the first 12 ms of each condition (`gantt_*.png`).

## Running it

Prerequisites:
- a ModelBlaster checkout next to this repo (or `MODELBLASTER_ROOT`) whose
  `build/k3_xpurt/<model>/int8/` already holds the extracted IRs of mlp_control,
  dronet and yolov8_nano (extraction needs PyTorch; any earlier K3 build leaves them);
- an ssh alias for the board (`MODELBLASTER_K3_HOST`, default `k3`);
- `PY` = ModelBlaster's python, `PY_XPURT` = a python with XPU-RT's deps.

```bash
scripts/k3_walker_feedback/run_before_after_campaign.sh out/k3_walker_feedback      # ~10 min of board time
python scripts/k3_walker_feedback/score_k3_runs.py out/k3_walker_feedback/*_run* --modelblaster ../ModelBlaster
python scripts/k3_walker_feedback/plot_predicted_vs_actual.py --runs out/k3_walker_feedback
```

One schedule, one run: `run_schedule_on_k3.sh <schedule.json> <out_dir> [MB_XPURT_DYNAMIC=1 | MB_XPURT_POOL=net:period_ms ...]`.

### Board use

`k3_board_lock.sh` holds two locks for the whole campaign: a host `flock` on
`/tmp/k3_board.lock` (the `MODELBLASTER_BOARD_LOCK` convention) and a board-side
`flock` on the K3's `/tmp/k3_board.lock`, with a `/tmp/K3_IN_USE` notice. Both are
released on exit.

For the results above, the K3's desktop and background services were stopped first and
restarted afterwards:
- as root: `sddm cups cups-browsed fwupd`;
- in the login user's session, masked so they stay off across ssh logins:
  `bianbu-update-backend pipewire pipewire-pulse wireplumber filter-chain mpris-proxy`
  and the two pipewire sockets.

Stopping them did not change the results (91-92 misses either way). Record each
service's state before stopping it, and restore it after.

## Files

| File | Purpose |
|---|---|
| `run_schedule_on_k3.sh` | build one schedule with ModelBlaster and run it on the K3; copy stdout, trace and schedule |
| `run_before_after_campaign.sh` | all conditions, interleaved, under the board locks |
| `k3_board_lock.sh` | host + board locks (sourced) |
| `score_k3_runs.py` | misses, worst lateness, median discrepancy, bit-exact check per run |
| `plot_predicted_vs_actual.py` | predicted (top) vs actual (bottom) Gantt per condition, on the hart that actually ran each dispatch; summary figure |
| `../calibrated_feedback_loop.py` | one feedback round: solve -> board trace -> update (`--feedback calibrated|ratio`) |

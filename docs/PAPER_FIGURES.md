# Paper figures: reproduction status

Status of each evaluation figure when this repository is used **without** the lab
boards. "Requires boards" means the figure is built from measurements taken on
physical hardware (QRB5165, SpaceMiT K1, or a FireSim FPGA fleet); the scripts and
committed processed data are here, but the raw board runs are not, so the figure
cannot be regenerated end to end by an outside reader.

| Figure | Status | Notes |
|---|---|---|
| `solver_pareto_fix` | Reproduced | From the xpurt_rerun results table (`analysis/results.csv`, 77 workload-arms): CP-SAT warm-best 18.2% mean gain in 61.8 s, best-of-fast 10.0% in 1.66 s, 37x ratio. Offline, no board. A port over the QRB5165 sweep is `qnn_models/flow_c/sweeps/qrb5165_sched_algo_sweep10_*/analysis/plot_pareto_panels.py`. |
| `QNN_HEATMAP_2` | **Requires boards** | Panel B (44-point CP-SAT sweep) regenerates offline with `sims/octo_roselite/roselite/finegrain/paper/fig_e2e_heatmap.py`; the measured success and energy inputs come from board and simulator runs. |
| `QNN_spoon` | **Requires boards** | Two scripts in `sims/octo_roselite/roselite/finegrain/paper/layouts/`, each writing its PNG next to itself: `cand_c10v6_sim_strip_spoon.py` draws the top strip from simulator frames (`layouts/.frames/ladder_spoon_*`), and `cand_a1v2_gantt_lanes.py` the per-lane Gantt from QRB5165 run logs (`$OCTO_REPRO_RUNS`, default `qnn_models/octo/repro_runs`). Neither the frames nor those run logs are in the repository. |
| Isaac Lab flight results (abstract and conclusion: 10.0 vs 26.0 ms, 2.59x) | **Requires boards** | Trace replay in Isaac Lab on the Crazyflie track-steering task, `sims/isaaclab_tasks/track_steering_vision/` (Isaac Lab is the `sims/IsaacLab` submodule; environment setup in `docs/xpurt_env_setup.md`). The board traces, the replay campaign and its analysis scripts are in [ucb-bar/XPU-RT-deprecated](https://github.com/ucb-bar/XPU-RT-deprecated) (`dev`), not in this repository. |
| `vs_ROS` | **Requires boards** | ROS 2 baselines measured on QRB5165, SpaceMiT K1 and FireSim. |
| `fig_schedule_evolution` | **Requires boards** | K1 hardware feedback loop. |
| Architecture diagrams | n/a | Drawn by hand. |

The QRB5165 solver sweep (`qnn_models/flow_c/sweeps/qrb5165_20260829-200620`), `costmodel_validation` and `transfer_study` keep their raw board run logs, so `verify_provenance.py` checks all three repetitions of every point.

Regenerating the offline parts needs `matplotlib` and `numpy`; no GPU, board or
licence is required for the figures marked Reproduced.

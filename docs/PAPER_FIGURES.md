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
| `QNN_spoon` | **Requires boards** | Built from QRB5165 board traces (`fig_schedule.py`, `fig_lane_balance.py`, `fig_trajectory.py`). |
| `vs_ROS` | **Requires boards** | ROS 2 baselines measured on QRB5165, SpaceMiT K1 and FireSim. |
| `fig_schedule_evolution` | **Requires boards** | K1 hardware feedback loop. |
| Architecture diagrams | n/a | Drawn by hand. |

Regenerating the offline parts needs `matplotlib` and `numpy`; no GPU, board or
licence is required for the figures marked Reproduced.

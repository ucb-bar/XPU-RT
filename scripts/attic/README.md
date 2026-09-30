# scripts/attic — one-shot drivers and retired tools

These drivers were run once for the study and are kept for attribution; the documented
reproduction path is in `docs/Artifact/reproduction_full.md`. Nothing here is called by a tracked
script, test or document. They are kept as they ran (paths, waits on `*_DONE` markers,
sibling imports), so re-running one may need its working directory or `sys.path` adjusted.

Column key. *Results / tags*: the results directory under `results/codesign_feedback/`
(`R/`) the driver wrote, or the trace labels it passed to `scripts/run_xpurt_long.sh`
(`xpurt_long/trace_<label>_*.csv`) and `scripts/ros_traced_matrix.sh`
(`ros_traced/<rate>_<arm>_r<n>/`). *Documented*: the section that describes those results
(`M` = `docs/Evaluation/measurements_and_ablations.md`, `F` = `docs/Evaluation/figure_runbook.md`,
`R` = `docs/Artifact/reproduction_full.md`); `—` when no document cites them.

## Board queues (K1, this branch)

| Script | What it did | Results / tags | Documented |
|---|---|---|---|
| `board_highrate.sh` | Beyond the design rate: the XPU-RT layouts and every ROS 2 arm at 60/75/90 Hz. | xpurt_long `best60alt2 best60alt1 best45alt1 best75alt2 best90alt1`; ros_traced arms ship/spin/p3/multi/cspin/cp3/smte at 60 75 90 | R §1–2, M §1.5–1.6 |
| `board_highrate2.sh` | Complete traces for the high-rate XPU-RT layouts after ssh drops emptied the first pass. | xpurt_long (same labels), `xpurt_long/highrate2.log` | R §1 |
| `board_highrate2b.sh` | Second retry of `best60alt2 best75alt2 best90alt1`, after `RICH2_DONE`. | xpurt_long | R §1 |
| `board_more.sh` | Two cameras on both runtimes and the transformer block on the matrix engine. | xpurt_long `cam2alt1 rich45alt2ime`; ros_traced 45 Hz two-camera arms; rebuilds `ros_control_jitter/ros_mb_chain_traced.cpp` | R §1–2, M §1.5 |
| `board_queue3.sh` | Rebuild the traced node with the chained control mode, run the chained arms at 15/25/45 Hz, resume the ROS replicates. | ros_traced (chained arms) | R §2, M §1.6 |
| `board_rich.sh` | The heavier stack: YOLO 45 Hz + nav + control + ffn_block 10 Hz + dronet 30 Hz. | xpurt_long `rich25p4 rich45alt2`; ros_traced 25/45 Hz | R §1–2 |
| `board_rich3.sh` | Re-run of the heavier-stack XPU-RT layouts after `HIGHRATE2B_DONE`. | xpurt_long `rich45alt2 rich25p4` | R §1 |
| `board_sensitivity.sh` | Sensitivity rows on the idle board: QoS depth 1, 200 Hz control, background CPU hogs, a five-second run of the headline arm. | xpurt_long `best45alt2 best45alt2c200 best45alt2long`; ros_traced 25/45 Hz | M §1.5–1.6 |
| `board_smte_reps.sh` | Replicates of the multi-threaded-executor arm at the design rate. | ros_traced `smte` at 45 | M §1.6 |
| `board_stress.sh` | Combined load on both runtimes: two 45 Hz cameras and the heavier stack. | xpurt_long `cam2rich`; ros_traced `x2rp3` | M §1.5–1.6 |
| `board_stress2.sh` | The other two ROS 2 deployments under the combined load. | ros_traced 45 Hz | M §1.6 |
| `board_stress3.sh` | The multi-threaded executor under the combined load (two cameras share one YOLO callback group). | ros_traced `x2rmulti` | M §1.6 |
| `board_rates_rerun.sh` | Board runs for the tables whose clamp step had failed: 60/90/30 Hz chain and the shard-aware spec. | solver_v2, xpurt_long | R §3 |
| `board_vanilla4_qos1.sh` | The out-of-the-box graph with the 4-hart YOLO and keep-last-1 QoS. | ros_traced `vanilla4` at 45 90 | M §1.6 |
| `board_vanilla4x2.sh` | Traced node rebuilt with frame alternation; the pipelining-by-hand deployment across camera rates. | ros_traced `vanilla4x2` at 45 60 90 | M §1.6 |
| `xpurt_best_runs.sh` | The explicit best-effort XPU-RT placements, plus YOLO-alone timings on the E cluster and on all eight harts. | xpurt_long `best25p4 best45alt4 best45p4`; `ros_traced/yolo_standalone/` | R §1, M §1.5 |
| `xpurt_best_runs2.sh` | Re-run of the best-effort layouts the ssh drops emptied, plus two new 45 Hz layouts. | xpurt_long `best25p4 best45alt2 best45alt4b best45alt4s` | R §1 |
| `xpurt_runs_final.sh` | Clean SCHED_OTHER runs of the 25 Hz layout, a FIFO variant of the 45 Hz arm, two more alt2 replicates. | xpurt_long `best25p4 best45alt2 best45alt2b` | R §1, M §1.5 |
| `xpurt_runs_now.sh` | The clamped long25/long45 coupled-chain tables on the board. | xpurt_long `long25 long45` | R §1 |
| `xpurt_tiled_runs.sh` | XPU-RT on the tiled CP-SAT schedules, between the chained ROS arms and the replicates. | xpurt_long `tiled25 tiled45` | R §1 |
| `chain_rates.sh` | The chain spec at other camera rates: certificate (both solvers, 200 ms), one-second tables, board runs. | solver_v2 (`solve_stage2.sh`), `k1_board_calibration_yolo110.json` | R §3 |
| `stage2_b_chain.sh` | Spec B (90 Hz camera + heavier stack): both solvers on the one-second table, executed after spec A's board runs. | solver_v2 (`solve_stage2.sh wh_chain90_rich_solve b`) | R §3 |
| `queue_a90e.sh` | The execution-mean calibration variant on the 90 Hz chain's 200 ms table (a90e), sharing round 0 with a90h. | hil_feedback, solver_v2, xpurt_long | R §3, M §1.8c |
| `make_ros_partition_8core.py` | A ROS 2 baseline partition using all eight cores (YOLO on the P cluster, nav and control two each). | `cmp_coupled_cpsat_board_trace.csv` | — |
| `measured_chain_latency.py` | Camera→YOLO→nav→control chain latency from the executed trace rather than from a solve. | reads `cmp_coupled_cpsat_board_trace.csv` | — |
| `recost_coupled_on_board.py` | Dependency-honouring board re-cost of a coupled schedule (imports `recost_schedule_on_board.py`). | reads `k1_board_calibration.json` | — |
| `ros_chain_rate_sweep.sh` | What rate a real ROS 2 chain sustains on the board doing the real work. | `R/ros_chain_sweep/` | — |

## Flight campaigns and GPU queues (Isaac, this branch)

| Script | What it did | Results / tags | Documented |
|---|---|---|---|
| `campaign_ros_family.sh` | The out-of-the-box ROS 2 deployments in flight, replaying each arm's measured cadence across cruise speeds. | `R/campaign_v2/`, `ctrl_traces/ros_*` | M §1.7, R §4–5 |
| `campaign_xpu_solvers.sh` | The two solvers' tables in flight, replaying each solver's measured cadence. | `R/campaign_v2/`, `ctrl_traces/xpu_a_*` | M §1.7, R §4–5 |
| `campaign_markers.sh` | Writes the completion markers the follow-on scripts wait for. | `R/campaign_v2/*.log` | — |
| `campaign_v2_all.sh` | The whole trace-driven campaign as three parallel single-arm drivers. | `R/campaign_v2/` | M §1.7, F §3b |
| `campaign_v2_all_cal.sh` | The same campaign under the calibrated gain law (moment_scale = 0.5 / outputs-per-second). | `R/campaign_v2/` | M §1.7, M §3.6 |
| `cal_after_record.sh` | Runs `campaign_v2_cal_now.sh` once the display re-record has finished. | `R/campaign_v2/` | — |
| `gpu_queue.sh` | One GPU, strictly sequential: showdown campaign, then the follow-on campaigns in usable order. | `R/campaign/gpu_queue.log` | — |
| `gpu_queue2.sh` | After the main queue: the RoSE-style pipeline injection (control cadence + chain delay per arm). | `R/campaign/` | — |
| `grid_resume_final.sh` | The envelope grids (calibrated cells, then the 20 Hz cells under both policies) after the campaign drivers. | `R/gain_controlled/`, `R/energy_runs_v2/` | M §3.6 |
| `grid_resume_later.sh` | Resume the calibrated-gain grid and the 20 Hz envelope cells. | `R/gain_controlled/` | M §3.6 |
| `gain_controlled_rate_check.sh` | Does the rate effect survive gain calibration? | `R/gain_controlled/` | M §3.6 |
| `hil_ablation_moreseeds.sh` | Appends more seeds to the envelope CSV under identical conditions. | `R/hil_grid_more/` → `hil_ablation.csv` | F §2 |
| `run_strengthen_campaign.sh` | HIL data-strengthening campaign (canonical sim, new files only). | `R/hil_ablation_*.csv`, `R/strengthen_campaign/` | F §2 |
| `recover_and_finish.sh` | Idempotent recovery driver: waits for orphaned flights, resumes completed-cell skipping. | (resumes the grids above) | — |
| `queue_after_stronger.sh` | Slower end of the speed axis in the tall-people scene and twelve more seeds at 1.0 m/s. | `R/campaign_percep/`, `R/campaign_seeds24/` | M §1.8 |
| `decoupled_showdown.sh` | Control at full rate on both arms; perception staleness is what differs. | `R/decoupled/` | M §1.5 |
| `jitter_grounded_showdown.sh` | The showdown at each runtime's measured worst-case control rate. | `R/jitter_grounded/` | — |
| `mean_gap_showdown.sh` | The showdown at each runtime's mean measured control gap. | `R/mean_gap/` | — |
| `percep_latency_showdown.sh` | The showdown on perception latency modelled fully (delay and refresh). | `R/percep_latency/` | M §1.8 |
| `ros_fair_speed_sweep.sh` | At what cruise speed can real ROS 2 latency fly the course? | `R/fair_speed/` | — |
| `ros_grounded_showdown.sh` | The showdown at latencies grounded in measurement rather than in a model. | `R/ros_grounded/` | — |
| `worstcase_showdown.sh` | The showdown at each runtime's measured worst-case control response. | `R/worstcase/` | — |
| `display_v2.sh` | The two displayed flights of the selected cell, recorded (video + figure data), baseline re-flown until the rule's flight. | `R/campaign_v2/display/`, `refined/warehouse_showdown_v2` | F §3b |
| `display_v2_xpu.sh` | The displayed XPU-RT flight re-flown over the success seeds until one completes; then the composite. | `R/campaign_v2/display/` | F §3b |
| `display_v3_xpu_again.sh` | Re-record the XPU-RT display flight with the executed table's row label; composite again. | `R/campaign_v2/display/` | F §3b |

## Analysis and figure scripts superseded by the current figure set

| Script | What it did | Results / tags | Documented |
|---|---|---|---|
| `analyze_courseB.py` | Course-B envelope sweep against course A (cross-course generalisation, Wilson CIs, 25→50 Hz cliff). | reads `hil_ablation*.csv` | F §2 |
| `hil_logistic_fit.py` | Logistic response surface P(success) ~ log2(rate) + speed over every flight. | `R/hil_logistic_fit/` | — |
| `fig_strengthened.py` | Two-panel strengthened HIL story (cross-course, rate effect). | `R/hil_strengthened/` | — |
| `flight_effort_energy.py` | Control-effort and modelled-energy metrics per flight from recorded state traces. | `R/flight_effort_energy.csv` from `R/crash_verify/` | — |
| `hil_ablation_errorbars.py` | Envelope ablation with Wilson-CI error bars. | `R/hil_ablation_errorbars/` | F §2 |
| `compose_feedback_evolution.py` | The feedback loop iteration by iteration: Gantt after each round, stacked. | `R/feedback_evolution/` | — |

## The `flowc_*` cluster (QRB5165 / K1 feedback-knob study)

| Script | What it did | Results / tags | Documented |
|---|---|---|---|
| `flowc_contention_sweep.py` | Multi-model contention sweep on QRB5165: does the feedback knob pay once the board is busy? | `results/flowc_contention/` | `docs/Qualcomm/experiments/` |
| `flowc_contention_experiment.py` | Tune the QRB5165 cost model against a concurrent multi-model workload. | `results/flowc_contention/` | `docs/Qualcomm/experiments/` |
| `flowc_k1_contention_tune.py` | Tune a contention correction on the K1 exact-cycle runs. | `results/flowc_contention/k1_tune.json` | `docs/K1/k1_contention.md` |
| `flowc_overhead_ablation.py` | How much the slicing recommendation depends on runtime overhead. | (feeds the report below) | `docs/Qualcomm/experiments/overhead/` |
| `flowc_overhead_report.py` | Experiment log for the overhead ablation (`overhead_ablation.md/.jsonl`). | `docs/Qualcomm/experiments/overhead/` | same |
| `flowc_plot_overhead.py` | Regime-strip figures for the overhead ablation. | `docs/Qualcomm/experiments/overhead/` | same |
| `flowc_plot_before_after.py` | Before/after dumbbell figures for precision, slice, branch. | `docs/Qualcomm/experiments/stages/` | same |
| `flowc_residual_report.py` | Log and figures for the residual-feedback study (440 dispatches). | `docs/Qualcomm/experiments/residual/` | `docs/Qualcomm/qualcomm-qrb5165.md` |
| `flowc_size_sweep.py` | Does the winning feedback path depend on network size? | `results/flowc_size/` | `docs/Qualcomm/experiments/` |
| `flowc_stage_report.py` | Full experiment log and figures for the feedback-stage ladder. | `docs/Qualcomm/experiments/stages/` | same |
| `flowc_plot_findings.py` | Figures for the "knob inert" and "contention" findings. | `docs/Qualcomm/experiments/` | same |

## Pre-September tools

| Script | What it did | Results / tags | Documented |
|---|---|---|---|
| `packing_demo.py` | Examples of greedy and convex packing algorithms. | — | — |
| `testing.py` | Scratch file, no docstring. | — | — |
| `pie_smolvla_op_breakdown_gemm_gemv.py` | SmolVLA op-time pie with Linear split into GEMM and GEMV. | `plots/` | — |
| `plot_qnn_stack.py` | Block diagram of the QNN SDK stack on QRB5165. | `plots/qnn_sdk_stack.png` | — |
| `plot_fpga_lut_breakdown.py` | LUT breakdown of the three Saturn V128D128 prototype builds from Vivado reports. | `plots/` | — |
| `validate_gemmini_config.sh` | Wrapper to run after building a new Gemmini config / bitstream. | — | — |
| `plot_frontier.py` | The measured YOLOv8n frequency frontier at print size. | `plots/` | — |
| `worst_case_periodic_window_fraction.py` | Worst-case duration of a periodic task's layers versus its window. | stdout | — |
| `plot_exact_cycle_feedback.py` | Figure for the exact-cycle proof and repeated K1 corroboration. | `plots/` | — |
| `relocate_profile_provenance.py` | Make schedule profile provenance independent of the checkout directory. | rewrites `schedules/*.json` | — |

## Left in `scripts/` because a tracked file cites them

`dump_display_ros_attempts.sh`, `dump_display_serial.sh` (`docs/Evaluation/figure_runbook.md`);
`predict_conv_cost.py` (`ModelBlaster/tests/test_yolov8_input_size.py`);
`worst_case_nonperiodic_duration.py` (`xpu-rt/profile_metrics.py`, `data/banks/hardware_bank.json`);
`run_heterogeneous_schedule.py` (`qnn_scheduler/README.md`);
`flatten_schedule_aliases.py`, `serialize_instances.py`, `uartlog_to_profile.py`
(`runs/sweeps/fpga_20260829-195805/RUNBOOK.md`); `run_exact_cycle_feedback.py` (`.gitignore`, `pip_freeze.txt`).

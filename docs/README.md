# Documentation index

Every page under `docs/`, by topic folder. To evaluate the paper artifact, start at
[`../artifact/README.md`](../artifact/README.md): it orders the steps and says what each needs (the K1
board, a GPU, or nothing). [`Artifact/artifact_checklist.md`](Artifact/artifact_checklist.md) is the
inventory of what is stored, what is generated, and how each is checked.

| folder | what it covers |
|---|---|
| [`Artifact/`](#artifact) | setting up, reproducing, and verifying the artifact |
| [`Feature/`](#feature) | the co-design loop, the schedulers and solvers, the integrations |
| [`K1/`](#k1) | the SpaceMiT K1 board, its matrix engine, sharding and partitioning on it |
| [`Baselines/`](#baselines) | the ROS 2 baseline: its tiers, arms, ladder, models and reproduction |
| [`Evaluation/`](#evaluation) | the flight evaluation: showdown figures, ablations, arm rankings, figure recipes |
| [`Demo/`](#demo), [`Firesim/`](#firesim), [`Qualcomm/`](#qualcomm) | the spike, FireSim and QRB5165 flows |

A few files at the top of `docs/` contain only "Moved to …"; they are listed at the end so a link written
against the earlier flat layout still lands.

## Artifact

| page | what it is |
|---|---|
| [`Artifact/artifact_checklist.md`](Artifact/artifact_checklist.md) | what is stored, what is generated, the checks, the known gaps |
| [`Artifact/environment.md`](Artifact/environment.md) | the two interpreters and the machine paths (`scripts/env.sh`) |
| [`Artifact/xpurt_env_setup.md`](Artifact/xpurt_env_setup.md) | an `xpurt` conda environment that runs the forest-trail demo |
| [`Artifact/external_data.md`](Artifact/external_data.md) | the inputs the repository does not carry and how to point at your own copies |
| [`Artifact/REPRODUCE.md`](Artifact/REPRODUCE.md) | the AOT ↔ runtime ↔ HIL co-design loop on a new machine or target |
| [`Artifact/reproduction_full.md`](Artifact/reproduction_full.md) | the study end to end — host, simulator, board |
| [`Artifact/run_index.md`](Artifact/run_index.md) | every measured arm, what makes it different, how flights replay the board (generated) |
| [`Artifact/figure_verification_inventory.md`](Artifact/figure_verification_inventory.md) | every figure and the check that verifies it |
| [`Artifact/five_figure_audit.md`](Artifact/five_figure_audit.md) | the five showdown figures panel by panel: claim, check, rebuild |
| [`Artifact/mutation_audit.md`](Artifact/mutation_audit.md) | which recorded numbers the checks depend on (generated) |

## Feature

| page | what it is |
|---|---|
| [`Feature/the_loop.md`](Feature/the_loop.md) | every arrow of the compiler ↔ scheduler cycle and which script owns it |
| [`Feature/codesign_loop_reproduction.md`](Feature/codesign_loop_reproduction.md) | running the loop: the drivers, which stages need the board |
| [`Feature/feedback_loop_reference.md`](Feature/feedback_loop_reference.md) | the feedback loop: methodology, results, what can and cannot be claimed |
| [`Feature/w4_w5_inner_outer.md`](Feature/w4_w5_inner_outer.md) | what the inner and outer loops each do on the w4 and w5 workloads |
| [`Feature/board_calibration_codesign.md`](Feature/board_calibration_codesign.md) | board-calibrated co-design scheduling |
| [`Feature/board_and_model_gaps.md`](Feature/board_and_model_gaps.md) | two gaps between board and model, with the numbers that locate them |
| [`Feature/modelblaster_integration.md`](Feature/modelblaster_integration.md) | ModelBlaster ↔ XPU-RT: the two feedback channels |
| [`Feature/solvers.md`](Feature/solvers.md) | which solver and scheduler combinations exist and what each is for |
| [`Feature/scheduler_solver_study.md`](Feature/scheduler_solver_study.md) | survey, implementations and measurements of the scheduling solvers |
| [`Feature/solver_study_repro.md`](Feature/solver_study_repro.md) | redrawing the solver-study plots |
| [`Feature/scheduler_oracle_gap.md`](Feature/scheduler_oracle_gap.md) | how far the cheap schedulers are from optimal |
| [`Feature/freshness_eval_design.md`](Feature/freshness_eval_design.md) | design of the freshness-validity evaluation |

## K1

| page | what it is |
|---|---|
| [`K1/k1_board.md`](K1/k1_board.md) | running on the K1: commands, timings, compiler traps, recovery |
| [`K1/k1_modelblaster_xpurt_closed_loop.md`](K1/k1_modelblaster_xpurt_closed_loop.md) | profile → schedule → build → run → advise → rewrite on the K1 |
| [`K1/k1_contention.md`](K1/k1_contention.md) | contention on the K1, measured on the shipped path |
| [`K1/k1_cost_by_pred.md`](K1/k1_cost_by_pred.md) | what a dispatch pays to read data another hart wrote |
| [`K1/ime_kernel_reproduction.md`](K1/ime_kernel_reproduction.md) | the IME matrix engine on the deployed chain and the fused conv kernel |
| [`K1/nav_sharding.md`](K1/nav_sharding.md) | sharding the navigation network across harts |
| [`K1/partitioned_schedule.md`](K1/partitioned_schedule.md) | a spatially partitioned schedule for the deployed chain |

## Baselines

| page | what it is |
|---|---|
| [`Baselines/ros_baseline_tiers.md`](Baselines/ros_baseline_tiers.md) | the three methods (analytical, calibrated, measured), every ROS figure's tier, the paper figures' tiers |
| [`Baselines/ros_baseline_ladder.md`](Baselines/ros_baseline_ladder.md) | every arm selectable by name (`scripts/ros_baseline.py`), ordered by engineering effort |
| [`Baselines/ros_baseline_reproduction.md`](Baselines/ros_baseline_reproduction.md) | the baseline on the K1 end to end: staging, building, the arms, reading the results |
| [`Baselines/ros_arm_ranking.md`](Baselines/ros_arm_ranking.md) | every measured arm ranked per camera rate, and what each deployment choice buys |
| [`Baselines/ros_arms_catalog.md`](Baselines/ros_arms_catalog.md) | every deployment measured, from the run manifests (generated) |
| [`Baselines/ros_pinning_model_profiled.md`](Baselines/ros_pinning_model_profiled.md) | the Tier B model: the pinning recurrence costed from the board's own runs |
| [`Baselines/ros_with_ime.md`](Baselines/ros_with_ime.md) | the ROS 2 baseline given the matrix engine |

## Evaluation

| page | what it is |
|---|---|
| [`Evaluation/measurements_and_ablations.md`](Evaluation/measurements_and_ablations.md) | reproducing the board measurements and the ablations of the loop |
| [`Evaluation/figure_runbook.md`](Evaluation/figure_runbook.md) | every headline figure: inputs and the exact command |
| [`Evaluation/flight_environment.md`](Evaluation/flight_environment.md) | what the flights were flown in |
| [`Evaluation/xpurt_arm_ranking.md`](Evaluation/xpurt_arm_ranking.md) | which XPU-RT implementation of the chain is best, and at what |
| [`Evaluation/cores_yolo_reproduction.md`](Evaluation/cores_yolo_reproduction.md) | the cores × YOLO-service figure and what each point is |
| [`Evaluation/showdown_rate_sweep_reproduction.md`](Evaluation/showdown_rate_sweep_reproduction.md) | the showdown across camera rates, ROS 2 on all 8 cores |
| [`Evaluation/showdown_r30_reproduction.md`](Evaluation/showdown_r30_reproduction.md) | the showdown at a 30 Hz camera |
| [`Evaluation/showdown_cam30_solver_placed_reproduction.md`](Evaluation/showdown_cam30_solver_placed_reproduction.md) | 30 Hz, placement left to the solver |
| [`Evaluation/showdown_cam36_allcores_reproduction.md`](Evaluation/showdown_cam36_allcores_reproduction.md) | 36 Hz, against a baseline using every hart |
| [`Evaluation/showdown_cam45_ros_out_of_box_reproduction.md`](Evaluation/showdown_cam45_ros_out_of_box_reproduction.md) | 45 Hz, against ROS 2 as it ships, both arms measured |
| [`Evaluation/showdown_cam45_ros_unpinned_reproduction.md`](Evaluation/showdown_cam45_ros_unpinned_reproduction.md) | 45 Hz, against an unpinned ROS 2 deployment |
| [`Evaluation/showdown_cam45_static6_reproduction.md`](Evaluation/showdown_cam45_static6_reproduction.md) | 45 Hz, against a statically partitioned ROS 2 |
| [`Evaluation/showdown_cam45_solver_placed_reproduction.md`](Evaluation/showdown_cam45_solver_placed_reproduction.md) | 45 Hz, placement left to the solver |
| [`Evaluation/showdown_cam45_unpinned_best_reproduction.md`](Evaluation/showdown_cam45_unpinned_best_reproduction.md) | 45 Hz, on the solver-placed schedule |
| [`Evaluation/showdown_analytical_reproduction.md`](Evaluation/showdown_analytical_reproduction.md) | the showdown with the Tier A baseline: the model, its inputs, its panels |
| [`Evaluation/warehouse_hil_figure_reproduction.md`](Evaluation/warehouse_hil_figure_reproduction.md) | warehouse HIL navigation: flight → schedule → paper figure |
| [`Evaluation/warehouse_sensorfusion_reproduce.md`](Evaluation/warehouse_sensorfusion_reproduce.md) | the warehouse sensor-fusion navigation, HIL and K1 schedule experiments |

## Demo

| page | what it is |
|---|---|
| [`Demo/mlp_dronet_yolo_spike_reproduction.md`](Demo/mlp_dronet_yolo_spike_reproduction.md) | the three-network schedule on spike, no FireSim, no MOSEK licence |
| [`Demo/workload_specs.md`](Demo/workload_specs.md) | the fields of a `data/toplevel/*.json` workload spec and how each is read |
| [`Demo/replicate_forest_trail_demo.md`](Demo/replicate_forest_trail_demo.md) | the forest-trail navigation demo from a fresh clone |

## Firesim

| page | what it is |
|---|---|
| [`Firesim/end_to_end_xpurt_firesim.md`](Firesim/end_to_end_xpurt_firesim.md) | workload spec → schedule → FireSim run, with trace plots (Flow A) |

## Qualcomm

| page | what it is |
|---|---|
| [`Qualcomm/mlp_dronet_yolo_qnn_reproduction.md`](Qualcomm/mlp_dronet_yolo_qnn_reproduction.md) | the three networks on a QRB5165 through QNN (Flow C) |
| [`Qualcomm/qualcomm-qrb5165.md`](Qualcomm/qualcomm-qrb5165.md) | the SmolVLA vision encoder on the QRB5165 |
| [`Qualcomm/experiments/overhead/overhead_ablation.md`](Qualcomm/experiments/overhead/overhead_ablation.md) | QRB5165 overhead ablation |
| [`Qualcomm/experiments/residual/residual_report.md`](Qualcomm/experiments/residual/residual_report.md) | QRB5165 residual feedback |
| [`Qualcomm/experiments/stages/stage_ladder.md`](Qualcomm/experiments/stages/stage_ladder.md) | QRB5165 feedback-stage ladder |

## Pointer pages

`k1_board.md`, `k1_contention.md`, `k1_cost_by_pred.md`, `k1_modelblaster_xpurt_closed_loop.md`,
`partitioned_schedule.md` → `K1/`; `modelblaster_integration.md`, `solvers.md`, `the_loop.md`,
`scheduler_solver_study.md`, `solver_study_repro.md` → `Feature/`; `workload_specs.md` → `Demo/`;
`ros_baseline_reproduction.md` → `Baselines/`; `warehouse_hil_figure_reproduction.md` → `Evaluation/`.

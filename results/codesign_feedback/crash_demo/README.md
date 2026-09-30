# Warehouse showdown trace provenance

These are Isaac simulator traces used by `sims/scripts/compose_warehouse_showdown.py`.
They are qualitative flight evidence, not synchronized K1 board traces.

- `complete_figdata/`: selected successful run, 4/4 gates, 12.49 s. The log records
  `sched_latency=12.4 ms`, a 10 ms control step, and a 2-step zero-order hold
  (50 Hz effective command rate). Source log:
  `provenance/xpu_50hz_success.log`.
- `crash_figdata/`: deepest of five ROS-baseline trials, 1/4 gates, collision at
  4.69 s. The log records `sched_latency=30 ms`, a 10 ms control step, and a
  3-step zero-order hold (33 Hz effective command rate). Source log:
  `provenance/ros_33hz_crash.log`.

Both traces use the same learned navigation model, RL/MLP controller, YOLO
detector, warehouse course, and obstacle configuration. The board-calibrated
schedule comparison in the composed figure is a separate analytical panel; it
does not claim that these flight trajectories were recorded on the K1.

The publication-oriented schedule panel uses matched 5/6/12 YOLO/NAV/CTRL
horizons and compares the third YOLO release in:

- `schedules/scheduled__flight_deployed_matched_board_cpsat_profiled.json`
  (XPU-RT, feasible CP-SAT schedule, all eight harts), and
- `schedules/scheduled_ros_partition_deployed_matched_board.json`
  (ROS-style fixed partition on the same platform, six harts used by policy).

Generated assets are `results/codesign_feedback/warehouse_schedule_board.*`
(main-paper schedule panel) and
`results/codesign_feedback/warehouse_showdown_board.*` (combined overview).

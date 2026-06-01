---
name: sweep-schedulers
description: Profile multiple schedulers on one networks benchmark and compare makespan/deadline/granularity, with the advisor's diagnosis per scheduler. Use to pick or justify a scheduler on a real workload.
---

# sweep-schedulers

Run several schedulers on the same benchmark and compare them, with a deadline +
granularity diagnosis for each.

## Steps

1. **Run the sweep** (each combo emits a SchedulerReport; the driver advises on
   each and prints a comparison table):
   ```bash
   python3 scripts/profile_schedulers.py \
     --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
     --schedulers decomposed,heft,peft,edf,fifo,round_robin,critical_path,fastest_device \
     --deadline-us <N> --emit schedules/sweep.json
   ```
   - `decomposed`/`greedy`/`greedy_periodic` run as `--solver`; everything else
     runs as `--solver milp --scheduler <name>` (the registry path).
   - Add `--gantt` to print a terminal Gantt per scheduler.

2. **Interpret** the table for the user: which scheduler gives the best makespan,
   which meet the deadline, and what the advisor recommends for the rest
   (rebalance / coarsen / finer). Tie the recommendation to the chosen objective.

3. **Drill in** on a specific scheduler with `/diagnose-schedule` on its
   `schedules/scheduled_<...>_<scheduler>_report.json`.

## Rules

- Some registry schedulers need extra deps/licenses (mosek, gurobi, gnn, rl);
  the driver records failures and continues — report which ran and which didn't,
  don't hide skipped ones.
- Compare on a real benchmark (firesim profiles), not a toy one, when possible.

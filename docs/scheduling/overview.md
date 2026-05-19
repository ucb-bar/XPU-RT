# XPU-RT Scheduling Stack

XPU-RT ships a complete heterogeneous scheduling stack inherited from the
original UCB-BAR XPU-RT project, now integrated into the compiler generator.

The stack has four layers:

| Layer | What it does | Where it lives |
|---|---|---|
| **Workload model** | Operation / Job / Window / processing-times / transfer-times tensors over a heterogeneous machine set | `xpu_rt.scheduler.workload` |
| **MILP scheduler** | CVXPY-formulated MILP, minimum-makespan + fusion + cross-cluster cache penalties | `xpu_rt.scheduler.scheduler` |
| **QNN backend** | QRB5165 cost model + island-DAG scheduling for Qualcomm Hexagon NPU + GPU + HTA | `xpu_rt.targets.backends.qnn` |
| **Telemetry → calibration** | On-board telemetry (mean / median / p99 per dispatch) → typed dispatch hints fed back into the next compile | `xpu_rt.scheduler.feedback` + `xpu_rt.scheduler.streaming_feedback` |

## Solver story

The scheduler runs as a first-class
[`xpu_rt.solve` backend](solver-backend.md) — same envelope as placement,
memory planning, and proof discharge. MOSEK is the default MILP solver
(license auto-discovered from repo-local `mosek.lic`); HiGHS / SCIP / GLPK
serve as open-source fallbacks.

```python
from xpu_rt.scheduler import solve_makespan

t, alpha, fused, response = solve_makespan(workload, time_limit=30.0)
assert response.status.value in {"optimal", "feasible"}
```

`response` is a typed `SolverResponse` carrying `formulation_hash`,
`selected_backend`, `time_ms`, `objective_value` — same audit trail every
solver call in the compiler produces.

## Bridges to the compiler

| Compiler stage | Scheduling stack |
|---|---|
| `xpu_rt.stages.dispatch` (emits `xpu_rt.dispatch_id` per op) | `xpu_rt.scheduler.workload_factory.create_workload_from_dependencies(...)` |
| `xpu_rt.analysis.cost` (analytical roofline, M-21) | `xpu_rt.targets.backends.qnn.cost_table.CostTable.execute_us(...)` (empirical) |
| `xpu_rt.promotion` (gate ladder, M-22 calibration_status) | `xpu_rt.scheduler.feedback.derive_dispatch_hints(...)` |
| `runtime/tools/xpurt_scheduler_runner.c` | consumes `execution_plan.yaml` via `xpu_rt/runtime/schedule_json_view.py` |

## Reading order

1. [Two-cluster scheduler](two-cluster-scheduler.md) — the core MILP shape.
2. [QNN backend](qnn-backend.md) — QRB5165 cost model + island-DAG.
3. [Solver-backend integration](solver-backend.md) — how the CVX scheduler
   plugs into `xpu_rt.solve`.

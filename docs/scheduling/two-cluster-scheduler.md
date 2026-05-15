# Two-Cluster MILP Scheduler

The core scheduler is a Mixed-Integer Linear Program formulated in CVXPY.
It minimises makespan over a heterogeneous machine set, respecting

- per-(operation, machine-combination) processing times,
- pairwise transfer times between machines,
- a dependency DAG over operations,
- per-op time windows (`min_start_t`, `max_end_t`) for periodic workloads,
- hard infeasibility exclusions (e.g. an op the QNN backend can't run),
- a `processing_times_by_pred` tensor that linearises cross-cluster cache
  state penalties (CPU_P → CPU_E coming hot vs cold).

See the problem formulation in
[Bertsekas et al. 2023 §2.1](https://www.sciencedirect.com/science/article/pii/S037722172300382X#sec0014)
(the algorithmic basis the scheduler implements).

## Inputs

```python
from xpu_rt.scheduler.workload import Operation, Workload
import numpy as np

ops = [
    Operation(processing_times=[1.0, 2.0], operation_id=0, operation_name="conv0"),
    Operation(processing_times=[2.0, 1.5], operation_id=1, operation_name="bn0",
              predecessors=[0]),
    Operation(processing_times=[0.8, 1.0], operation_id=2, operation_name="relu0",
              predecessors=[1]),
]
transfer = np.array([[0.0, 0.05], [0.05, 0.0]])
wl = Workload(operations=ops, machines=["CPU_P", "CPU_E"], transfer_times=transfer)
```

## Solving

Two equivalent calls — the **direct** call is the original XPU-RT entry
point; the **envelope** call routes through `xpu_rt.solve` so every audit
hook fires.

```python title="Direct"
from xpu_rt.scheduler.scheduler import schedule

t, alpha, fused_wl, meta = schedule(wl, time_limit=30.0)
```

```python title="Envelope (recommended)"
from xpu_rt.scheduler import solve_makespan

t, alpha, fused_wl, response = solve_makespan(wl, time_limit=30.0)
print(response.status.value, response.objective_value, response.formulation_hash)
```

Both end up in MOSEK (when licensed) via CVXPY. The envelope path adds
typed status, formulation_hash, time_ms, selected_backend — and slots into
the same `compgen-solver-planning` MCP skill that probes placement and
memory.

## Outputs

| Variable | Shape | Meaning |
|---|---|---|
| `t` | `(num_ops,)` | Start time of each operation. |
| `alpha` | `(num_ops, num_machine_combinations)` | 0/1 assignment indicator. |
| `fused_wl` | `Workload \| None` | If `fusion_threshold` was set, the expanded post-fusion workload. |
| `response.objective_value` | float | Makespan in the same units as `processing_times`. |

## Fusion + packing

`scheduler.fusion.fuse_operations(workload, threshold)` collapses chains of
short ops into fused-operation supernodes before the MILP runs, reducing
binary variable count. `scheduler.packing.greedy_packing` /
`convex_packing` provide warm-start packings the MILP can refine.

## Schedule validation

`scheduler.schedule_validation.validate(t, alpha, workload)` confirms the
returned schedule respects every constraint (no overlap on the same
machine, all predecessors finish before successors start, every op is
placed on exactly one combination). Useful as a post-solve sanity gate in
CI.

## See also

- [QNN backend](qnn-backend.md) — drives the scheduler over Qualcomm
  Hexagon NPU + GPU + HTA islands using empirical QRB5165 latencies.
- [Solver-backend integration](solver-backend.md) — typed envelope details.
- `xpu-rt/python/xpu_rt/scheduler/scheduler.py` — full MILP formulation
  with section-by-section constraint logging (`debug_constraints=True`).

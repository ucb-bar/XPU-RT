# QNN Backend (Qualcomm Hexagon NPU)

The QNN backend slots into XPU-RT's `xpu_rt.targets.backends/` family
alongside `npu/` (Hexagon-MLIR open-source path) and `saturn_opu/`. It
owns the *empirical* cost model + island-DAG scheduling for the QRB5165
robotics SoC.

## Layout

```
xpu-rt/python/xpu_rt/targets/backends/qnn/
├── __init__.py
├── cost_table.py            # CostTable dataclass: 567 ops × QNN backends
├── transfer_model.py        # per-(src, dst, bytes, dtype) linear-fit transfer cost
├── island_dag.py            # Group ops into hardware-dispatch islands
├── scheduler.py             # schedule_groups(...) — drives the MILP over islands
├── seed_table_qrb5165.py    # bootstrap latencies before on-board profiling
├── plot.py                  # Gantt + DAG visualisation
├── qrb5165_costs.json       # empirical cost table (latest on-board profile)
└── README.md
```

## Cost table

Each op-kind / QNN-backend pair has an empirical `mean_us`, `median_us`,
`p99_us`. Load and query:

```python
from pathlib import Path
from xpu_rt.targets.backends.qnn.cost_table import CostTable

table = CostTable.load(Path("xpu-rt/python/xpu_rt/targets/backends/qnn/qrb5165_costs.json"))
print(table.device, table.qairt_sdk)            # qrb5165, 2.45.0.260326
print(len(table.execute))                       # 567 ops covered
cost_us = table.execute_us(op_kind="conv2d", qnn_backend="GPU", dtype="uint8")
```

The table is regenerated on-board by
`scripts/profile_qnn_per_dispatch.py`; that script runs a sweep of every
op-kind across every backend and updates the JSON in-place.

## Island DAG

QNN executes dispatch *groups* (islands) atomically — once a group is
handed to the backend driver, no preemption mid-island. `island_dag` walks
the IR's dispatch DAG and identifies maximal islands per backend, so the
MILP sees the right scheduling granularity.

```python
from xpu_rt.targets.backends.qnn import island_dag

groups = island_dag.identify_groups(dispatch_dag, backends=["CPU", "GPU", "HTA"])
for g in groups:
    print(g.backend, g.op_ids, g.start_predecessors)
```

## Transfer model

Cross-backend transfers (CPU↔GPU, GPU↔HTA, etc.) carry real cost. The
linear-fit transfer model captures
`time_us = fixed_overhead_us + bytes / bytes_per_us_mean` per
(src, dst, dtype) pair, fitted from `profile_transfers_on_board.py`.

```python
from xpu_rt.targets.backends.qnn.transfer_model import TransferModel

tm = TransferModel.from_cost_table(table)
us = tm.transfer_us(src="CPU", dst="GPU", n_bytes=4096, dtype="uint8")
```

## Scheduling

`schedule_groups(...)` is the QNN-specific wrapper around the two-cluster
MILP — it builds the Workload, plugs the cost-table + transfer-model
values into processing-times / transfer-times tensors, and calls
`xpu_rt.scheduler.solve_makespan` (which routes through `xpu_rt.solve`).

```python
from xpu_rt.targets.backends.qnn.scheduler import schedule_groups

result = schedule_groups(groups, table, transfer_model=tm, time_limit=30.0)
print(result.makespan_us, result.assignments)
```

## Bootstrap path (no on-board access yet)

If you don't have the QRB5165 device, `seed_table_qrb5165.seed()` returns
a hand-curated cost table sufficient to exercise the scheduler in CI and
in unit tests.

## See also

- [Two-cluster scheduler](two-cluster-scheduler.md) for the MILP itself.
- `xpu_rt/targets/cards/hexagon_npu.yaml` — declarative target card for the Hexagon NPU.

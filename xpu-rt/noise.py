"""
M16 helper — multiplicative Gaussian noise on processing_times for the
robustness benchmark. Also reusable by future runs that want to test
schedulers under uncertain cost models.

Single public entry point:

    add_processing_time_noise(workload, sigma_pct, rng=None) -> new_workload

Each per-(op, combo) processing_time is multiplied by
``max(0.05, 1 + N(0, sigma_pct/100))``. Infeasibility flags are preserved.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Operation, Workload


def add_processing_time_noise(
    workload: Workload,
    sigma_pct: float,
    rng: Optional[np.random.Generator] = None,
) -> Workload:
    """Return a NEW Workload with multiplicative Gaussian noise on every
    per-(op, combo) processing_time. ``sigma_pct`` is in percent (e.g. 25
    for ±25% one-sigma jitter). The original workload is not modified.
    """
    if rng is None:
        rng = np.random.default_rng()
    if sigma_pct <= 0:
        return workload

    # Deep-copy operations (cannot just shallow-copy because we mutate
    # processing_times list per-op).
    op_idx = {id(op): i for i, op in enumerate(workload.operations)}
    new_ops = []
    for op in workload.operations:
        sigma = sigma_pct / 100.0
        new_costs = []
        for k, c in enumerate(op.processing_times):
            if c >= 1e8 or k in op.infeasible_combinations:
                new_costs.append(c)
                continue
            factor = 1.0 + float(rng.normal(0.0, sigma))
            factor = max(0.05, factor)
            new_costs.append(float(c) * factor)
        new_op = Operation(
            processing_times=new_costs,
            operation_name=op.operation_name,
            operation_id=op.operation_id,
            job_id=op.job_id,
            min_start_t=op.min_start_t,
            max_end_t=op.max_end_t,
            deadline_us=op.deadline_us,
            skip_allowed=op.skip_allowed,
            infeasible_combinations=set(op.infeasible_combinations),
        )
        new_op.output_bytes = getattr(op, "output_bytes", 0)
        new_op.memory_region_preference = getattr(op, "memory_region_preference", None)
        new_ops.append(new_op)

    # Re-wire predecessors using index mapping.
    for old_op, new_op in zip(workload.operations, new_ops):
        for pred in old_op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is not None:
                new_op.add_predecessor(new_ops[pi])

    return Workload(
        new_ops,
        list(workload.machines),
        np.array(workload.transfer_times),
        job_names=list(workload.job_names),
        machine_combinations=[list(c) for c in workload.machine_combinations],
    )

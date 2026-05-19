"""Runtime planning and bundling subpackage (Stage 5).

Handles the final pipeline stage: generating execution plans, packaging
artifacts into deployable bundles, and local execution for testing.

Two-level scheduling architecture:
    Level 1 (compile-time): Per-workload execution plan generated here.
        - Placement: what runs on which device
        - Copy/sync: data movement between devices
        - Execution DAG: op ordering with dependencies
        - Memory plan: buffer allocation and lifetime

    Level 2 (runtime): Global multi-workload scheduling (future).
        - Admission control, priorities, queue assignment
        - Memory budgeting, backpressure, preemption

Optional adapters for IREE and PJRT backends are provided but not
required for the core pipeline.
"""

from __future__ import annotations

__all__: list[str] = []

"""Cluster-launch decision policy — Blackwell-specific.

Per bridge #108: ETC's per-stage cooperative grid-sync (~70 µs)
dominates per-task GEMM (~1 µs) on small-to-medium shapes.
Multi-block-per-task cooperation via ``cuLaunchKernelEx`` cluster
attribute amortizes the sync over a cluster instead of the whole
grid.

Expected impact: cluster-sync runs in ~5 µs vs ~70 µs grid-sync.
At 144 tasks per FFN and 3 stages, that's roughly 14× headroom on
the synchronization cost — enough to push Diamond past the 1.2×
gate and bring MLP-1 within striking distance.

Wave 1.6 ships the **wiring** — the probe detects support, the
schedule emits `cluster_dim`, the runtime launches with cluster
attributes. The body emitter side (using `cute::cluster_sync()`,
distributed shared memory across a cluster) is a follow-up that
arrives once we measure the wiring's standalone impact.

This file documents the policy decisions; the actual numeric
defaults live in the autotune probe + the BackendChoice dataclass.
Import this module to get the documented constants alongside the
runtime values.
"""

from __future__ import annotations

# Default cluster shape for Blackwell. The conservative (2, 1, 1)
# means 2 thread blocks form a cluster — they share distributed
# shared memory and synchronize via cluster_sync(). At 188 SMs
# (sm_120) and 256 tasks per FFN this means ~128 clusters of 2,
# half the per-stage sync cost vs single-block tasks.
# Larger clusters (4, 1, 1) and (8, 1, 1) are tunable. They
# amortize sync further but require:
# - Bigger smem budget per cluster (4× / 8× the per-block smem).
# - Body emitter to actually use cluster-distributed memory; a
#   non-cluster-aware body in a 4-block cluster wastes 3 of them.
# Wave 1.6 starts at 2; Wave 1.6b will autotune.
DEFAULT_CLUSTER_DIM = (2, 1, 1)


# Maximum cluster size on Blackwell. ``cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags``
# returns this; we set a conservative ceiling here for the autotune
# probe to use as the search bound.
MAX_CLUSTER_SIZE = 8


# Smem budget multiplier per cluster member. Blackwell's 100 KB
# optin-smem ceiling is per-block; cluster-distributed smem doesn't
# multiply that, but the body's own per-block smem allocation does
# scale with cluster_dim if the body uses cluster-shared regions.
CLUSTER_SMEM_OVERHEAD_FRACTION = 0.05


def is_cluster_eligible(arch: str) -> bool:
    """Pure helper — does this arch support cluster-launch?

    NVIDIA Hopper (sm_90+) introduced cluster_launch but Wave 1.6
    enables it ONLY on Blackwell (sm_100/sm_120) where bridge #108
    confirmed the perf impact. Hopper-cluster enablement is a
    follow-up; this function will widen when that lands.
    """
    a = arch.lower().lstrip("sm_").rstrip("a")
    return a in {"100", "120"}

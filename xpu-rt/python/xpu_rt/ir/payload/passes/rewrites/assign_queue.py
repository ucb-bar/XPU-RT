"""``assign_queue`` -- assign a queue_id to each region.

Reconstruction of XLA's ``queue_id`` annotation (from the Stream
assignment pipeline). Zero external references; CompGen owns the
rewrite.

Operates on :class:`ExecutionPlan`: fills in each
``RegionPlacement.queue`` based on the region's device + a simple
load-balanced round-robin across ``num_queues_per_device``.

Respects the **launch-order constraint**: two regions with a
dependency edge must not collide on the same queue UNLESS the
earlier one is guaranteed to be done (serial ordering within a
queue implicitly handles that).

Policy (static, deterministic):

- Group regions by device.
- Topologically sort each group by ``dependency_edges``.
- Round-robin across ``num_queues_per_device`` queues in topo order.
- The same queue can carry multiple non-interfering regions
  (serial execution within a queue is the gate).

Config:

- ``num_queues_per_device`` -- how many hardware queues to use.
- ``queue_prefix`` -- naming scheme; queue names become
  ``f"{queue_prefix}{device_tag}_{idx}"``.

LLM-tool signature:

    tool_name="assign_queue"
    wraps_pass="CompGen:XLAStreamQueueId"
    invent_slot="runtime/queue_assignment"
    policy="RoundRobinPerDevice"
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from compgen.runtime.execution_plan import ExecutionPlan


@dataclass(frozen=True)
class AssignQueueConfig:
    num_queues_per_device: int = 2
    queue_prefix: str = "q"
    overwrite_existing: bool = False


@dataclass
class AssignQueueStats:
    regions_seen: int = 0
    regions_assigned: int = 0
    regions_already_assigned: int = 0
    queues_in_use: set[str] = field(default_factory=set)


def _topo_sort(region_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Topologically sort region_ids subject to (from, to) edges.

    When cycles or missing nodes appear, fall back to the original
    order — we don't want the queue assigner to crash on malformed
    plans.
    """
    in_order = {rid: i for i, rid in enumerate(region_ids)}
    adj = defaultdict(list)
    indegree = defaultdict(int)
    for frm, to in edges:
        if frm in in_order and to in in_order:
            adj[frm].append(to)
            indegree[to] += 1
    queue = deque([rid for rid in region_ids if indegree[rid] == 0])
    out: list[str] = []
    while queue:
        node = queue.popleft()
        out.append(node)
        for nxt in adj[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(out) != len(region_ids):
        # Cycle or orphan -> fall back to original order.
        return list(region_ids)
    return out


def _safe_device_tag(device: str) -> str:
    # Normalize device string to something queue-name-safe.
    return "".join(c if c.isalnum() else "_" for c in device) or "unknown"


def run_assign_queue(
    plan: ExecutionPlan,
    *,
    config: AssignQueueConfig | None = None,
) -> AssignQueueStats:
    cfg = config if config is not None else AssignQueueConfig()
    stats = AssignQueueStats()

    if cfg.num_queues_per_device < 1:
        raise ValueError("num_queues_per_device must be >= 1")

    # Group regions by device, preserving original relative order.
    by_device: dict[str, list[str]] = defaultdict(list)
    for rp in plan.region_placement:
        by_device[rp.device].append(rp.region_id)

    edges = [(e.from_region, e.to_region) for e in plan.dependency_edges]

    # Build queue assignment per device.
    assignments: dict[str, str] = {}
    for device, region_ids in by_device.items():
        ordered = _topo_sort(region_ids, edges)
        dev_tag = _safe_device_tag(device)
        for i, rid in enumerate(ordered):
            qidx = i % cfg.num_queues_per_device
            qname = f"{cfg.queue_prefix}{dev_tag}_{qidx}"
            assignments[rid] = qname

    # Apply.
    for rp in plan.region_placement:
        stats.regions_seen += 1
        if rp.queue and not cfg.overwrite_existing:
            stats.regions_already_assigned += 1
            stats.queues_in_use.add(rp.queue)
            continue
        new_q = assignments.get(rp.region_id)
        if new_q is None:
            continue
        rp.queue = new_q
        stats.queues_in_use.add(new_q)
        stats.regions_assigned += 1

    return stats


__all__ = [
    "AssignQueueConfig",
    "AssignQueueStats",
    "run_assign_queue",
]

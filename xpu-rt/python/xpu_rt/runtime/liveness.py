"""Buffer liveness analysis + interference graph for ExecutionPlan.

Consumers are the W6 passes:

- ``plan_buffers`` uses the interference graph to pick a greedy
  coloring that minimizes peak memory in each domain.
- ``alias_io_buffers`` uses the interference graph to test whether
  two buffers can legally alias (non-overlapping lifetime + same
  memory space + compatible ownership).
- ``insert_copies`` uses the liveness report to detect cross-memory
  boundaries that need a materialized copy.

Everything here operates on an ``ExecutionPlan`` and returns pure
dataclasses -- no xDSL IR mutation, no disk I/O.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from compgen.runtime.execution_plan import (
    ExecutionPlan,
)


@dataclass
class BufferLiveness:
    """Per-buffer liveness record."""

    buffer_id: str
    memory_space: str
    size_bytes: int
    first_use_tick: int
    last_use_tick: int
    persistent: bool


@dataclass
class LivenessReport:
    """Program-wide liveness summary.

    ``live_at`` is an exclusive-upper-bound map: ``live_at[tick]`` is
    the set of buffer_ids whose ``[first_use_tick, last_use_tick]``
    interval contains ``tick``.
    """

    per_buffer: dict[str, BufferLiveness] = field(default_factory=dict)
    live_at: dict[int, set[str]] = field(default_factory=dict)
    peak_tick: int = 0
    peak_live_count: int = 0
    peak_live_bytes: int = 0

    def interval_for(self, buffer_id: str) -> tuple[int, int]:
        b = self.per_buffer[buffer_id]
        return (b.first_use_tick, b.last_use_tick)

    def lifetimes_overlap(self, a: str, b: str) -> bool:
        la = self.per_buffer[a]
        lb = self.per_buffer[b]
        if la.persistent or lb.persistent:
            return True
        return not (la.last_use_tick < lb.first_use_tick or lb.last_use_tick < la.first_use_tick)


@dataclass
class InterferenceGraph:
    """Undirected graph whose edges indicate lifetime overlap.

    Encoded as a mapping ``nodes[x] = set-of-neighbours``.
    """

    nodes: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, buffer_id: str) -> None:
        self.nodes.setdefault(buffer_id, set())

    def add_edge(self, a: str, b: str) -> None:
        if a == b:
            return
        self.nodes.setdefault(a, set()).add(b)
        self.nodes.setdefault(b, set()).add(a)

    def neighbours(self, buffer_id: str) -> set[str]:
        return self.nodes.get(buffer_id, set())

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        total = sum(len(neighbours) for neighbours in self.nodes.values())
        return total // 2


def compute_liveness(plan: ExecutionPlan) -> LivenessReport:
    """Compute per-buffer + per-tick liveness for a plan.

    Runs in ``O(N * T)`` where ``N`` is the number of buffers and
    ``T`` is the span of the longest lifetime. For the plans we see
    in practice (< 10k buffers, < 100k ticks) this is fine; if
    scalability becomes an issue we can switch to an event-list
    sweep over ``(first_use_tick, enter)`` / ``(last_use_tick+1,
    exit)`` tuples.
    """
    report = LivenessReport()
    max_tick = 0

    for b in plan.buffers:
        report.per_buffer[b.buffer_id] = BufferLiveness(
            buffer_id=b.buffer_id,
            memory_space=b.memory_space,
            size_bytes=b.size_bytes,
            first_use_tick=b.lifetime.first_use_tick,
            last_use_tick=b.lifetime.last_use_tick,
            persistent=b.lifetime.persistent,
        )
        if b.lifetime.last_use_tick > max_tick:
            max_tick = b.lifetime.last_use_tick

    if not plan.buffers:
        return report

    live_at: dict[int, set[str]] = defaultdict(set)
    for b in plan.buffers:
        lo = b.lifetime.first_use_tick
        hi = b.lifetime.last_use_tick
        if b.lifetime.persistent:
            lo = 0
            hi = max_tick
        for t in range(lo, hi + 1):
            live_at[t].add(b.buffer_id)
    report.live_at = dict(live_at)

    by_bytes: dict[str, int] = {b.buffer_id: b.size_bytes for b in plan.buffers}
    peak_tick = 0
    peak_bytes = 0
    peak_count = 0
    for tick, ids in live_at.items():
        total_bytes = sum(by_bytes[x] for x in ids)
        if total_bytes > peak_bytes or (total_bytes == peak_bytes and len(ids) > peak_count):
            peak_bytes = total_bytes
            peak_tick = tick
            peak_count = len(ids)
    report.peak_tick = peak_tick
    report.peak_live_bytes = peak_bytes
    report.peak_live_count = peak_count
    return report


def compute_interference_graph(
    liveness: LivenessReport,
    *,
    only_same_memory_space: bool = True,
) -> InterferenceGraph:
    """Build the interference graph from a liveness report.

    ``only_same_memory_space``: when ``True`` (the default and the
    right choice for greedy coloring), two buffers only interfere if
    they live in the same memory domain -- a tile-local scratchpad
    buffer is never aliased with a DRAM buffer.
    """
    graph = InterferenceGraph()
    ids = list(liveness.per_buffer.keys())
    for buf_id in ids:
        graph.add_node(buf_id)
    for i, a in enumerate(ids):
        la = liveness.per_buffer[a]
        for b in ids[i + 1 :]:
            lb = liveness.per_buffer[b]
            if only_same_memory_space and la.memory_space != lb.memory_space:
                continue
            if liveness.lifetimes_overlap(a, b):
                graph.add_edge(a, b)
    return graph


def greedy_color(graph: InterferenceGraph) -> dict[str, int]:
    """Greedy coloring by degree-descending order.

    Returns a ``{buffer_id: color_index}`` mapping where any two
    neighbouring buffers have distinct color indices. The coloring is
    deterministic: ties are broken by ``buffer_id`` lexicographic
    order.
    """
    ordered = sorted(
        graph.nodes,
        key=lambda n: (-len(graph.nodes[n]), n),
    )
    color_of: dict[str, int] = {}
    for node in ordered:
        forbidden = {color_of[nb] for nb in graph.nodes[node] if nb in color_of}
        c = 0
        while c in forbidden:
            c += 1
        color_of[node] = c
    return color_of


__all__ = [
    "BufferLiveness",
    "InterferenceGraph",
    "LivenessReport",
    "compute_interference_graph",
    "compute_liveness",
    "greedy_color",
]

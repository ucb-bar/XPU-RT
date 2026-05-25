"""
Post-schedule memory planner.

Given ``(workload, t, alpha)`` produced by any scheduler, this module:

  1. Computes per-buffer liveness intervals:
       buffer.live_start = producer.end
       buffer.live_end   = max(consumer.start for all consumers)
     A buffer with no consumers stays live until the end of the schedule.

  2. Assigns each buffer to a memory region (DRAM by default; scratchpad if
     ``op.memory_region_preference == "scratchpad"`` AND scratchpad has room).

  3. Allocates storage within each region via one of three policies:
       - ``no_reuse``         : every buffer gets a fresh slot
       - ``greedy_first_fit`` : place each buffer in the first slot whose
                                live interval does not overlap; reuse slots
                                when possible
       - ``size_aware_best_fit`` : among non-overlapping slots, pick the one
                                with the smallest waste vs requested size

The planner returns a ``MemoryPlan`` dict suitable for JSON serialization,
containing peak DRAM/scratchpad bytes, reuse count, top-N hot buffers, and a
per-buffer slot assignment table for the timeline plot.

Buffer sizes are read from ``op.output_bytes`` (default 0); buffer-annotation
overrides may be applied via ``annotations: Dict[op_name, bytes]``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workload import Workload


# ----------------------------------------------------------------------------
# Liveness
# ----------------------------------------------------------------------------


@dataclass
class Buffer:
    name: str
    producer_op_idx: int
    size_bytes: int
    live_start: float
    live_end: float
    region: str = "DRAM"  # or "scratchpad" or "device_local"

    @property
    def lifetime(self) -> float:
        return max(0.0, self.live_end - self.live_start)


def _op_finish(workload: Workload, t: np.ndarray, alpha: np.ndarray, i: int) -> float:
    combos = workload.get_machine_combinations()
    k = int(np.argmax(alpha[i]))
    dur = float(workload.operations[i].get_duration_for_combination(
        k, combos, workload.machines))
    return float(t[i]) + dur


def compute_liveness(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    annotations: Optional[Dict[str, int]] = None,
) -> List[Buffer]:
    """Build a Buffer per Op that has non-zero output_bytes."""
    annotations = annotations or {}
    ops = workload.operations
    op_idx = {id(op): i for i, op in enumerate(ops)}
    # Schedule end (max finish across all ops) — used for buffers with no consumers.
    schedule_end = max((_op_finish(workload, t, alpha, i) for i in range(len(ops))),
                       default=0.0)

    buffers: List[Buffer] = []
    # First pass: find consumers per producer.
    consumers_of: Dict[int, List[int]] = {i: [] for i in range(len(ops))}
    for i, op in enumerate(ops):
        for pred in op.get_predecessors():
            pi = op_idx.get(id(pred))
            if pi is not None:
                consumers_of[pi].append(i)

    for i, op in enumerate(ops):
        size = int(annotations.get(op.operation_name or f"op{i}",
                                   getattr(op, "output_bytes", 0)))
        if size <= 0:
            continue
        live_start = _op_finish(workload, t, alpha, i)
        cons = consumers_of[i]
        if cons:
            # Buffer is needed until the LAST consumer STARTS reading it.
            # (After read begins, the consumer takes a reference; the buffer
            #  can then be overwritten once the consumer finishes its access.)
            # For our purposes we conservatively use the latest consumer start.
            live_end = max(float(t[c]) for c in cons)
        else:
            # Sink output — kept alive until end of schedule (real systems
            # would copy it out at schedule end).
            live_end = schedule_end
        if live_end < live_start:
            live_end = live_start  # zero-lifetime (no consumer)
        buffers.append(Buffer(
            name=op.operation_name or f"op{i}_out",
            producer_op_idx=i,
            size_bytes=size,
            live_start=live_start,
            live_end=live_end,
            region=getattr(op, "memory_region_preference", None) or "DRAM",
        ))
    return buffers


# ----------------------------------------------------------------------------
# Allocators
# ----------------------------------------------------------------------------


@dataclass
class Slot:
    region: str
    size_bytes: int  # maximum size used by any buffer assigned here
    buffers: List[Buffer] = field(default_factory=list)

    def overlaps(self, b: Buffer) -> bool:
        for existing in self.buffers:
            if not (existing.live_end <= b.live_start or b.live_end <= existing.live_start):
                return True
        return False


def allocate(
    buffers: List[Buffer],
    policy: str,
    region_capacities: Optional[Dict[str, int]] = None,
) -> Tuple[List[Slot], Dict[Buffer, Slot]]:
    """Assign every buffer to a slot in its region.

    Returns (slots, mapping). If a buffer cannot fit in its preferred region's
    capacity, it spills to DRAM (recorded via mapping[b].region change).
    """
    region_capacities = region_capacities or {}
    # Process in order of decreasing size to make best-fit decisions meaningful.
    order = sorted(range(len(buffers)),
                   key=lambda i: -buffers[i].size_bytes)
    slots: List[Slot] = []
    mapping: Dict[int, Slot] = {}
    # If a region is over-capacity, downstream we'll re-route to DRAM.

    def _try_assign(b: Buffer, region: str) -> Optional[Slot]:
        candidates = [s for s in slots if s.region == region and not s.overlaps(b)]
        if not candidates:
            return None
        if policy == "no_reuse":
            return None  # force a fresh slot below
        if policy == "size_aware_best_fit":
            # Pick the slot whose existing size is closest to b.size_bytes
            # (smallest |slot.size - b.size|), preferring slots already large
            # enough to not need growth.
            def key(s):
                if s.size_bytes >= b.size_bytes:
                    return (0, s.size_bytes - b.size_bytes)
                return (1, b.size_bytes - s.size_bytes)
            return min(candidates, key=key)
        # default greedy_first_fit
        return candidates[0]

    for i in order:
        b = buffers[i]
        s = _try_assign(b, b.region) if policy != "no_reuse" else None
        if s is None:
            s = Slot(region=b.region, size_bytes=b.size_bytes)
            slots.append(s)
        else:
            if b.size_bytes > s.size_bytes:
                s.size_bytes = b.size_bytes
        s.buffers.append(b)
        mapping[id(b)] = s

    # Spill if region capacity exceeded.
    spills: List[Buffer] = []
    for region, cap in region_capacities.items():
        region_slots = [s for s in slots if s.region == region]
        peak = sum(s.size_bytes for s in region_slots)
        if peak <= cap:
            continue
        # Spill the largest slot(s) until under cap.
        region_slots.sort(key=lambda s: -s.size_bytes)
        for s in region_slots:
            if peak <= cap:
                break
            # Move this slot's buffers to DRAM.
            old_region = s.region
            s.region = "DRAM"
            for b in s.buffers:
                b.region = "DRAM"
                spills.append(b)
            peak -= s.size_bytes  # frees from this region
    return slots, {id(buffers[i]): mapping[id(buffers[i])] for i in range(len(buffers))}


# ----------------------------------------------------------------------------
# Plan summary
# ----------------------------------------------------------------------------


def plan_memory(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    *,
    annotations: Optional[Dict[str, int]] = None,
    policy: str = "greedy_first_fit",
    region_capacities: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Compute liveness + allocate + summarize."""
    buffers = compute_liveness(workload, t, alpha, annotations=annotations)
    slots, _mapping = allocate(buffers, policy, region_capacities)

    by_region: Dict[str, int] = {}
    for s in slots:
        by_region[s.region] = by_region.get(s.region, 0) + s.size_bytes

    # Compute peak instantaneous usage by sweeping over time.
    # Events: (time, +/-size).
    events: List[Tuple[float, int, str]] = []
    for s in slots:
        for b in s.buffers:
            events.append((b.live_start, +s.size_bytes if b is s.buffers[0] else 0, s.region))
    # Simpler: per-slot peak is sum of slot sizes — that's the conservative measure.
    # Compute time-instantaneous peak per region using buffer-level events:
    peak_inst: Dict[str, int] = {}
    for region in by_region:
        events_r = []
        for s in slots:
            if s.region != region:
                continue
            for b in s.buffers:
                events_r.append((b.live_start, +b.size_bytes))
                events_r.append((b.live_end, -b.size_bytes))
        events_r.sort()
        cur = 0
        peak = 0
        for _, delta in events_r:
            cur += delta
            if cur > peak:
                peak = cur
        peak_inst[region] = peak

    reuse_count = sum(len(s.buffers) - 1 for s in slots if len(s.buffers) > 1)
    hot = sorted(buffers, key=lambda b: -(b.size_bytes * max(1.0, b.lifetime)))[:10]

    return {
        "policy": policy,
        "num_buffers": len(buffers),
        "num_slots": len(slots),
        "reuse_count": reuse_count,
        "by_region_slot_total_bytes": by_region,
        "by_region_peak_instantaneous_bytes": peak_inst,
        # Aliases for compatibility with metrics.py field names.
        "peak_dram_bytes": peak_inst.get("DRAM", 0),
        "peak_scratchpad_bytes": peak_inst.get("scratchpad", 0),
        "buffer_reuse_count": reuse_count,
        "hot_buffers": [
            {"name": b.name, "size_bytes": b.size_bytes,
             "lifetime_us": b.lifetime, "region": b.region}
            for b in hot
        ],
        "slots_summary": [
            {"region": s.region, "size_bytes": s.size_bytes,
             "n_buffers": len(s.buffers),
             "buffer_names": [b.name for b in s.buffers]}
            for s in slots
        ],
    }


# ----------------------------------------------------------------------------
# Timeline plot
# ----------------------------------------------------------------------------


def render_memory_timeline(
    workload: Workload,
    t: np.ndarray,
    alpha: np.ndarray,
    save_path: str,
    annotations: Optional[Dict[str, int]] = None,
    region_capacities: Optional[Dict[str, int]] = None,
    title: str = "Memory timeline",
    policy: str = "greedy_first_fit",
):
    """Two-panel figure:
       (top)    instantaneous memory usage by region over time
       (bottom) per-buffer lifetime Gantt, coloured by allocated slot
                (shows the reuse pattern visually)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    buffers = compute_liveness(workload, t, alpha, annotations=annotations)
    if not buffers:
        return None
    slots, mapping = allocate(buffers, policy, region_capacities)

    regions = sorted({b.region for b in buffers})
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 9), constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )

    # --- Top: instantaneous usage by region ---
    for region in regions:
        events: List[Tuple[float, int]] = []
        for b in buffers:
            if b.region != region:
                continue
            # +size MUST be processed before -size at the same timestamp so
            # zero-lifetime buffers register as a non-negative spike.
            events.append((b.live_start, 0, +b.size_bytes))
            events.append((b.live_end, 1, -b.size_bytes))
        events.sort()
        ts: List[float] = []
        vals: List[int] = []
        cur = 0
        for time_, _ord, delta in events:
            ts.append(time_)
            vals.append(cur)
            cur += delta
            ts.append(time_)
            vals.append(cur)
        vals_mb = [v / (1024 * 1024) for v in vals]
        ax_top.plot(ts, vals_mb, label=region, drawstyle="steps-post", linewidth=1.5)
        if region_capacities and region in region_capacities:
            cap_mb = region_capacities[region] / (1024 * 1024)
            ax_top.axhline(cap_mb, linestyle="--", color="red",
                           label=f"{region} cap={cap_mb:.2f}MB")
    ax_top.set_ylabel("Memory (MB)")
    ax_top.set_title(f"{title} — instantaneous usage")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # --- Bottom: buffer-lifetime Gantt coloured by slot ---
    cmap = plt.get_cmap("tab20")
    slot_to_idx = {id(s): i for i, s in enumerate(slots)}
    for i, b in enumerate(buffers):
        s = mapping[id(b)]
        color = cmap(slot_to_idx[id(s)] % 20)
        width = max(b.live_end - b.live_start, 1.0)
        ax_bot.barh(i, width, left=b.live_start,
                    color=color, edgecolor="black", linewidth=0.3, alpha=0.85)
        ax_bot.text(b.live_start, i, f" {b.name} ({b.size_bytes/1024:.1f}KB)",
                    fontsize=6, va="center")
    ax_bot.set_yticks(range(len(buffers)))
    ax_bot.set_yticklabels([b.name for b in buffers], fontsize=6)
    ax_bot.invert_yaxis()
    ax_bot.set_xlabel("Time (us)")
    ax_bot.set_title(f"Buffer lifetimes (policy={policy}, slots={len(slots)})  "
                     f"— bars sharing a colour share storage")
    ax_bot.grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path

"""Same-network adjacent auto-merge pass.

Operates on an emitted schedule fixture (dict already serializable to
JSON via postprocessing.output_scheduled_json). The dual of compaction:
where compaction.py slides slack out, automerge collapses back-to-back
same-network dispatches on the same core into a single fused dispatch
boundary at *schedule time*, so the rendered Gantt reflects what the
runtime can actually fuse cheaply (one worker handshake instead of two).

This is the schedule-level analogue of the IR-level fusion that
`pipeline/apply_fusion_hint.py` performs in ModelBlaster. The IR pass
needs a curated fused KernelSpec to exist; this pass needs no curated
kernel — it just collapses two adjacent dispatch boundaries into one
trace event, and the harness already knows how to dispatch a
back-to-back kernel pair behind a single worker handshake.

Conflict checks (must ALL pass to merge op_i, op_j):
  1. Same `job_name` (network instance).
  2. Same `hardware_target`.
  3. op_j's only producer-among-dispatches is op_i (or none): no
     non-op_i dispatch in `op_j.dependencies` produces a value op_j
     reads.
  4. The gap between op_i.end and op_j.start on the same core is
     below `max_gap_us` (default: 50 µs — conservative).
  5. No other dispatch's `dependencies` lists op_i but not op_j (i.e.,
     nothing outside the pair reads op_i's output before op_j ends).

When all five pass:
  - Drop op_j from the fixture.
  - Stretch op_i's duration so `op_i.start + op_i.new_duration =
    op_j.start + op_j.duration` (preserve the original end time —
    that's the runtime cost regardless of whether we accounted for one
    or two handshakes).
  - Mark op_i with `merged_with: [op_j.id, ...]` and `is_fused: true`
    so the renderer can hatch the bar.
  - Rewrite every downstream dispatch's `dependencies` entry from
    op_j → op_i.
  - Update `time_dependency` references the same way.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Optional


# Periodic instances are named <base><instance_idx>_dispatch_<id>.
# The instance idx is a non-empty digit run sitting directly before the
# literal '_dispatch_' marker. Non-periodic networks (e.g. yolov8_nano)
# whose base names happen to contain digits do NOT have a digit run
# DIRECTLY before '_dispatch_' — the regex anchors on that boundary so
# 'yolov8_nano_dispatch_3' returns None rather than mis-parsing '8'.
_INSTANCE_RE = re.compile(r"^(.+?[^\d])(\d+)_dispatch_")


def _instance_idx_from_name(dispatch_name: str) -> Optional[int]:
    """Extract periodic-instance index from a dispatch name like
    'mlp_control0_dispatch_5' → 0, 'yolov8_nano_dispatch_3' → None.

    Returns None for non-periodic names (no '<base><digits>_dispatch_'
    pattern) — including networks whose base name itself contains
    digits (yolov8_nano).
    """
    m = _INSTANCE_RE.match(dispatch_name)
    if m is None:
        return None
    try:
        return int(m.group(2))
    except (TypeError, ValueError):
        return None


def automerge_adjacent(
    fixture: dict[str, Any],
    max_gap_us: float = 50.0,
    saved_handshake_us: float = 5.0,
) -> dict[str, Any]:
    """Return a new fixture with adjacent same-network dispatches merged.

    Args:
      fixture: Schedule fixture as produced by `output_scheduled_json`,
        with top-level keys `dispatches: {name: {...}}` and
        `metadata: {makespan, machines, ...}`.
      max_gap_us: Maximum µs of slack between op_i.end and op_j.start
        on the same hw target for the pair to be considered "adjacent".
        Above this we don't merge — the runtime gain is dominated by
        the gap, not by the handshake elimination.
      saved_handshake_us: How much we shorten the merged dispatch by
        (one fewer worker handshake than the unmerged pair). Conservative
        default; bumping it claims more savings than the runtime can
        deliver, so prefer low.

    Returns:
      A new fixture dict (input not mutated).
    """
    out = copy.deepcopy(fixture)
    dispatches: dict[str, dict] = out.get("dispatches", {})
    if not dispatches:
        return out

    # Index entries by core, time-ordered. We mutate in passes — each
    # pass merges at most one pair per core, so we loop until stable.
    while True:
        merged_this_pass = False
        by_core: dict[str, list[tuple[str, dict]]] = {}
        for name, entry in dispatches.items():
            ht = _hw_key(entry.get("hardware_target"))
            by_core.setdefault(ht, []).append((name, entry))
        for ht, items in by_core.items():
            items.sort(key=lambda kv: float(kv[1].get("start_time", 0.0)))
            for k in range(len(items) - 1):
                ni, ei = items[k]
                nj, ej = items[k + 1]
                if not _can_merge(ni, ei, nj, ej, dispatches, max_gap_us):
                    continue
                _do_merge(ni, ei, nj, ej, dispatches, saved_handshake_us)
                merged_this_pass = True
                break  # restart the outer loop — indices invalidated.
            if merged_this_pass:
                break
        if not merged_this_pass:
            break

    # Recompute makespan after the merge.
    if dispatches and "metadata" in out:
        out["metadata"]["makespan"] = max(
            float(e.get("start_time", 0.0)) + float(e.get("duration", 0.0))
            for e in dispatches.values()
        )
        out["metadata"]["num_operations"] = len(dispatches)

    return out


def _hw_key(hw_target: Any) -> str:
    """Normalize hardware_target (string or list) to a hashable key."""
    if isinstance(hw_target, list):
        return ",".join(hw_target)
    return str(hw_target) if hw_target is not None else ""


def _can_merge(
    ni: str,
    ei: dict,
    nj: str,
    ej: dict,
    dispatches: dict[str, dict],
    max_gap_us: float,
) -> bool:
    # 1. Same job_name (same network instance). This is already the
    #    instance-aware check: periodic networks are expanded by
    #    workload_factory so each instance has a unique job_name like
    #    'mlp_control0', 'mlp_control1', ... Two ops from different
    #    instances therefore differ here and the merge is refused.
    if ei.get("job_name") != ej.get("job_name"):
        return False
    # 1a. Defensive instance-suffix check (Phase A3). Even if the
    #     job_name slipped past (e.g. legacy fixtures without instance
    #     expansion), refuse to merge dispatches whose names parse to
    #     different instance indices. This is a belt-and-suspenders
    #     guard against cross-instance merges.
    inst_i = _instance_idx_from_name(ni)
    inst_j = _instance_idx_from_name(nj)
    if inst_i is not None and inst_j is not None and inst_i != inst_j:
        return False
    # 1b. Defensive max_end_t check. If either op carries a
    #     deadline_overrun_us flag (Phase A2 marking) past the band, do
    #     not merge — the resulting fused dispatch would inherit the
    #     overrun and obscure the violation. Merge only ops that BOTH
    #     fit their band.
    if ei.get("deadline_miss") or ej.get("deadline_miss"):
        return False

    # 2. Same hardware target.
    if _hw_key(ei.get("hardware_target")) != _hw_key(ej.get("hardware_target")):
        return False

    # 3. op_j's only in-fixture predecessor is op_i (or none).
    j_deps = list(ej.get("dependencies", []))
    if j_deps and not all(d == ni for d in j_deps):
        # j depends on some other in-flight dispatch besides op_i —
        # merging would create a cross-network sync barrier inside the
        # merged dispatch. Refuse.
        return False

    # 4. Gap below threshold.
    i_end = float(ei.get("start_time", 0.0)) + float(ei.get("duration", 0.0))
    j_start = float(ej.get("start_time", 0.0))
    gap = j_start - i_end
    if gap < -1e-6:  # overlap → solver bug, refuse
        return False
    if gap > max_gap_us:
        return False

    # 5. Nothing else reads op_i's output but op_j.
    #    (Anything depending on ni other than nj would lose its
    #     producer; bail out to be safe.)
    for n_other, e_other in dispatches.items():
        if n_other in (ni, nj):
            continue
        if ni in e_other.get("dependencies", []):
            return False

    return True


def _do_merge(
    ni: str,
    ei: dict,
    nj: str,
    ej: dict,
    dispatches: dict[str, dict],
    saved_handshake_us: float,
) -> None:
    i_start = float(ei.get("start_time", 0.0))
    j_end = float(ej.get("start_time", 0.0)) + float(ej.get("duration", 0.0))

    # New duration: end of pair minus i.start, minus one saved handshake.
    new_duration = max(0.0, (j_end - i_start) - saved_handshake_us)
    ei["duration"] = new_duration

    # Carry the fusion metadata.
    merged_with = list(ei.get("merged_with", []))
    merged_with.append(nj)
    ei["merged_with"] = merged_with
    ei["is_fused"] = True

    # Carry j's module_name if i didn't have one (and as a list for
    # renderers that want to label both halves).
    if "module_name" in ej:
        ei.setdefault("merged_module_names", []).append(ej["module_name"])

    # Drop op_j and rewire downstream deps from j → i.
    del dispatches[nj]
    for entry in dispatches.values():
        deps = entry.get("dependencies", [])
        if nj in deps:
            entry["dependencies"] = [ni if d == nj else d for d in deps]
            # Deduplicate (a downstream op might have already listed ni
            # because of compaction).
            seen: set[str] = set()
            entry["dependencies"] = [
                d for d in entry["dependencies"]
                if not (d in seen or seen.add(d))
            ]
        if entry.get("time_dependency") == nj:
            entry["time_dependency"] = ni


def automerge_savings(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Summary of what the pass collapsed.

    Returns:
      {
        "dispatches_before": int,
        "dispatches_after":  int,
        "pairs_merged":      int,
        "makespan_before":   float,
        "makespan_after":    float,
        "saved_us":          float,  # before - after (handshake only)
      }
    """
    d_b = before.get("dispatches", {})
    d_a = after.get("dispatches", {})
    n_b = len(d_b)
    n_a = len(d_a)
    mk_b = before.get("metadata", {}).get("makespan", _max_end(d_b))
    mk_a = after.get("metadata", {}).get("makespan", _max_end(d_a))
    return {
        "dispatches_before": n_b,
        "dispatches_after": n_a,
        "pairs_merged": n_b - n_a,
        "makespan_before": float(mk_b) if mk_b is not None else 0.0,
        "makespan_after": float(mk_a) if mk_a is not None else 0.0,
        "saved_us": float(mk_b - mk_a) if mk_b is not None and mk_a is not None else 0.0,
    }


def _max_end(dispatches: dict[str, dict]) -> float:
    if not dispatches:
        return 0.0
    return max(
        float(e.get("start_time", 0.0)) + float(e.get("duration", 0.0))
        for e in dispatches.values()
    )

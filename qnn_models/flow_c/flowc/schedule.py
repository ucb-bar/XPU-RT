"""Stage 4 — schedule ingest, via modelblaster, joined with the tile map.

`modelblaster.pipeline.ingest_xpurt_schedule.load()` does the whole
front half verbatim: parse the schedule, split job names into
(network, instance), resolve machine slots against the core registry,
priority-topologically order the table, and rewire both intra-job
`dependencies` and cross-job `time_dependency` edges into in-table entry
ids.  Flow C adds one join on top — entry (network, dispatch_id, kind) →
the context binary and graph name that actually execute it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import mb
from .bindings import BindingSet


@dataclass(frozen=True)
class QnnCore:
    """The Flow-C-only half of a registry core entry."""
    name: str
    kind: str
    harts: tuple[int, ...]
    lib: str
    label: str
    ctx_suffix: str
    exec_serialized: bool
    gate_spin_us: int
    sched_fifo_prio: int
    exec_cores: tuple[int, ...]


def load_qnn_cores(registry_path: str) -> dict[str, QnnCore]:
    with open(registry_path) as f:
        doc = json.load(f)
    out = {}
    for c in doc["cores"]:
        q = c.get("qnn")
        if not q:
            continue
        out[c["kind"]] = QnnCore(
            name=c["name"], kind=c["kind"], harts=tuple(c.get("harts", [])),
            lib=q["lib"], label=q["label"], ctx_suffix=q.get("ctx_suffix", ""),
            exec_serialized=bool(q.get("exec_serialized", False)),
            gate_spin_us=int(q.get("gate_spin_us", 200)),
            sched_fifo_prio=int(q.get("sched_fifo_prio", 0)),
            exec_cores=tuple(q.get("exec_cores", [])))
    return out


@dataclass
class FlowCEntry:
    entry_id: int
    network: str
    instance: int
    binding_id: int
    binding_name: str
    kind: str
    core_name: str
    hart: int
    harts: tuple[int, ...]
    backend_label: str
    backend_lib: str
    ctx: str
    graph: str
    exec_serialized: bool
    gate_spin_us: int
    sched_fifo_prio: int
    exec_cores: tuple[int, ...]
    n_ir_ops: int
    ir_first: int
    ir_last: int
    start_time_ms: float
    duration_ms: float
    deps: tuple[int, ...]
    time_dep: int


def ingest(schedule_path: str, bsets: dict[str, BindingSet], irs: dict[str, dict],
           registry_path: str, slot_to_kind: dict[str, str]) -> list[FlowCEntry]:
    reg = mb.core_registry().load(registry_path)
    mb.install_slot_map(slot_to_kind)
    ing = mb.ingest_xpurt_schedule()

    # modelblaster validates every (network, dispatch_id) against the IR it
    # is given.  Ours is the coarse IR — one op per binding — because that
    # is the dispatch space the schedule was solved in.
    from .artifacts import coarse_ir
    irs_by_network = {net: coarse_ir(bset, irs[net]) for net, bset in bsets.items()}

    entries = ing.load(schedule_path, irs_by_network, reg,
                       cpu_p_kind=slot_to_kind.get("CPU_P", ""),
                       cpu_e_kind=slot_to_kind.get("CPU_E", ""))

    qnn_cores = load_qnn_cores(registry_path)
    out: list[FlowCEntry] = []
    for e in entries:
        bset = bsets.get(e.network)
        if bset is None:
            raise KeyError(f"schedule references unknown network {e.network!r}")
        binding = bset.by_id(e.dispatch_id)
        qc = qnn_cores.get(e.core_kind)
        if qc is None:
            raise KeyError(f"registry core kind {e.core_kind!r} has no `qnn` block")
        bb = binding.backends.get(e.core_kind)
        if bb is None:
            raise ValueError(
                f"schedule put {binding.name} on {e.core_kind}, but no {e.core_kind} "
                f"context is declared for it in the binding manifest. Either build "
                f"one or re-emit the profile so the scheduler stops seeing that cell.")
        out.append(FlowCEntry(
            entry_id=e.entry_id, network=e.network, instance=e.instance,
            binding_id=binding.id, binding_name=binding.name,
            kind=e.core_kind, core_name=e.core_name, hart=e.hart,
            harts=qc.harts,
            backend_label=qc.label, backend_lib=qc.lib,
            ctx=bb.ctx, graph=bb.graph, exec_serialized=qc.exec_serialized,
            gate_spin_us=qc.gate_spin_us, sched_fifo_prio=qc.sched_fifo_prio,
            exec_cores=qc.exec_cores,
            n_ir_ops=binding.n_ops(),
            ir_first=binding.first, ir_last=binding.last,
            start_time_ms=e.start_time_ms, duration_ms=e.duration_ms,
            deps=tuple(e.deps_entry_ids), time_dep=e.time_dep_entry_id))
    return out


def summarize(entries: list[FlowCEntry]) -> str:
    from collections import Counter
    nets = Counter(e.network for e in entries)
    kinds = Counter(f"{e.kind}({e.backend_label})" for e in entries)
    ctxs = sorted({(e.ctx, e.graph, e.backend_label) for e in entries})
    makespan = max((e.start_time_ms + e.duration_ms) for e in entries) if entries else 0.0
    lines = [f"{len(entries)} entries  predicted makespan {makespan:.3f} ms",
             f"  per network: {dict(nets)}",
             f"  per lane:    {dict(kinds)}",
             f"  contexts:    {len(ctxs)}"]
    for ctx, graph, label in ctxs:
        lines.append(f"    {label:4} {ctx}  ::{graph}")
    return "\n".join(lines)

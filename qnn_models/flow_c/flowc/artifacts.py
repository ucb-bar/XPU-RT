"""Stage 3 — the XPU-RT contract artifacts, written with modelblaster's code.

Two files per (network, backend) cell, in the exact shapes
`xpu-rt/profile_loader.py` and `workload_factory.py` read:

  gen/qnn_vmfb/<net>/<target>/<hw>/<net>.<quant>/<...>_dispatch_graph.json
  gen/profile/<hw>/<target>/<net>/<net>.<quant>/topo_0/results.csv

The dispatch graph is emitted by modelblaster's own
`pipeline.emit_dispatch_graph.emit()`, fed a *coarse IR* — one synthetic op
per binding — so the scheduler's dispatch space is the tile space the
runtime can actually execute.  The profile CSV is emitted by
modelblaster's `pipeline.profile_writer.write_profile()`, so Flow C rows
carry the same IREE columns plus the same provenance fields (`source`,
`cpu`) as a spike or FireSim run.
"""

from __future__ import annotations

import json
import os
import tempfile

from . import mb
from .bindings import BindingSet, Feasibility

# A capability-excluded cell still needs a number, because
# xpu-rt's Operation carries a processing time per machine and has no
# "forbidden" flag.  Rather than a hand-typed constant (the 100_000 µs
# sentinel this flow used to carry), derive it from the measured data and
# label it in the CSV's provenance column so it is greppable.
EXCLUDED_FACTOR = 100.0


def coarse_ir(bset: BindingSet, ir: dict) -> dict:
    """One synthetic op per binding: the schedulable view of the network."""
    ids = [b.id for b in bset.bindings]
    if ids != list(range(len(ids))):
        raise ValueError(f"{bset.network}: binding ids must be 0..N-1, got {ids}")
    by_did = {o["dispatch_id"]: o for o in ir["ops"]}
    ops = []
    for b in bset.bindings:
        kinds = sorted({by_did[d]["op"] for d in b.op_ids() if d in by_did})
        ops.append({
            "name": b.name,
            "op": "qnn_graph",
            "dispatch_id": b.id,
            "depends_on": list(b.depends_on),
            "hardware_target": "any",
            "n_ir_ops": b.last - b.first + 1,
            "ir_op_kinds": kinds,
        })
    return {"name": bset.network, "quant": bset.quant, "version": 1, "ops": ops}


def emit_dispatch_graphs(bset: BindingSet, ir: dict, out_root: str,
                         target: str, hw_labels: list[str]) -> list[str]:
    cir = coarse_ir(bset, ir)
    written = []
    with tempfile.TemporaryDirectory() as td:
        ir_path = os.path.join(td, "coarse_graph.json")
        with open(ir_path, "w") as f:
            json.dump(cir, f)
        emit = mb.emit_dispatch_graph()
        for hw in hw_labels:
            written.append(emit.emit(ir_path, out_root, target, hw))
    return written


def emit_profiles(bset: BindingSet, feas: list[Feasibility], measurements: dict,
                  out_root: str, target: str,
                  kind_to_hw: dict[str, str]) -> tuple[list[str], list[str]]:
    """One results.csv per machine kind, rows in binding order.

    Returns (paths, warnings).  A cell with no measurement is written with
    a derived exclusion cost and `source=qnn-excluded`, and the reason is
    returned as a warning — never silently.
    """
    pw = mb.profile_writer()
    feas_by_id = {f.binding.id: f for f in feas}
    cells = measurements.get("cells", {})
    paths, warnings = [], []

    for kind, hw in kind_to_hw.items():
        records = []
        for b in bset.bindings:
            key = f"{bset.network}/{b.name}"
            measured = cells.get(key, {}).get(kind)
            blockers = feas_by_id[b.id].allowed.get(kind, [])
            if measured is None:
                peers = [v for v in cells.get(key, {}).values() if v is not None]
                excluded = EXCLUDED_FACTOR * (max(peers) if peers else 1000.0)
                why = (f"registry says {kind} cannot run {blockers}" if blockers
                       else f"no context built for {kind}")
                warnings.append(f"{key} on {kind}: excluded ({why}); "
                                f"cost set to {excluded:.0f} us")
                records.append({"name": f"{b.name}#excluded", "op": "qnn_graph",
                                "shape": f"ir_ops={b.last - b.first + 1}",
                                "cycles": excluded})
            else:
                records.append({"name": b.name, "op": "qnn_graph",
                                "shape": f"ir_ops={b.last - b.first + 1}",
                                "cycles": measured})
        meta = pw.ProfileMeta(
            model=bset.network, quant=bset.quant, backend=hw,
            cores=[0], cpu=target, clock_mhz=1.0,      # 1 MHz => 1 "cycle" == 1 us
            source=f"qnn-{measurements.get('statistic', 'mean')}"
                   f"@{measurements.get('captured_at', 'unknown')}")
        paths.append(pw.write_profile(records, meta, out_root=out_root))
    return paths, warnings


def emit_workload_spec(networks: list[dict], out_path: str, target: str,
                       slot_to_hw: dict[str, str], gen_root: str = "gen",
                       time_limit: int = 60, comment: str = "") -> str:
    """Write the data/toplevel/networks_*.json the scheduler consumes."""
    machines = {slot.lower(): 1 for slot in slot_to_hw}
    profile_hw = {slot.lower(): hw for slot, hw in slot_to_hw.items()}
    doc = {
        "_comment": comment,
        "hardware": {
            "machines": machines,
            "profile_hw": profile_hw,
            "profile": {"target": target, "topo_tag": "topo_0",
                        "topo_tag_override": False, "gen_root": gen_root},
            "p_core_speedup": 1.0,
        },
        "scheduler": {"random_seed": 42, "solver_verbosity": 0,
                      "time_limit": time_limit, "use_profiled": True,
                      "prune_periodic": True,
                      "restrict_makespan_to_nonperiodic": True},
        "networks": {},
        "edges": [],
    }
    for i, net in enumerate(networks):
        entry = {
            "id": i,
            "identifier": net["name"],
            "dispatch_deps_path": net["dispatch_deps_path"],
        }
        if net.get("period"):
            entry["period"] = net["period"]
            entry["window_duration"] = net.get("window_duration", net["period"])
        doc["networks"][net["name"]] = entry
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    return out_path

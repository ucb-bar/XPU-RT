"""Stage 2 — bindings: the tile map from IR ops to QNN graphs.

ModelBlaster's codegen maps one IR dispatch_id to one generated kernel
function, so its schedulable unit is a single op.  QNN's only execution
primitive is `QnnGraph_execute` on a pre-compiled graph, so Flow C's
schedulable unit is a *binding*: a contiguous run of IR dispatch ids that
was compiled into one graph, together with the per-backend context binary
that holds it.

That makes the binding manifest the tiling knob for this target.  Finer
tiles buy the scheduler freedom (the yolov8n head can leave HTA) and cost
one dispatch each (~0.5 ms of FastRPC round trip on dsp); coarser tiles
are cheaper to dispatch but pin more of the network to one lane.  Every
tile boundary also has to exist as a real sub-DLC, so re-tiling costs a
compile — which is why this is a manifest and not a runtime flag.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import lowering


@dataclass(frozen=True)
class BackendBinding:
    kind: str            # registry core kind: hta / dsp / cpu / gpu
    ctx: str             # context-binary filename in the board's ctx dir
    graph: str           # graph name inside that context


@dataclass
class Binding:
    id: int
    name: str
    first: int                       # inclusive IR dispatch_id
    last: int                        # inclusive
    source_onnx: str = ""            # the ONNX this tile was compiled from
    depends_on: tuple[int, ...] = ()
    backends: dict[str, BackendBinding] = field(default_factory=dict)
    lowering: tuple[str, ...] = ()   # transforms the compiled graph carries
    lowering_note: str = ""
    derived_from: str = ""           # provenance when the range was not hand-written

    def op_ids(self) -> range:
        return range(self.first, self.last + 1)


@dataclass
class BindingSet:
    network: str
    ir_spec: str
    quant: str
    graph_json: str | None
    bindings: list[Binding]
    partition: str = ""                       # partition JSON the ranges came from
    handoff_tensors: tuple[str, ...] = ()     # boundary tensors it declares

    def by_id(self, bid: int) -> Binding:
        for b in self.bindings:
            if b.id == bid:
                return b
        raise KeyError(f"{self.network}: no binding id {bid}")


def partition_groups(partition_path: str) -> list[tuple[str, int, int]]:
    """Contiguous hardware_target runs in a partition JSON.

    This is the actual source of a multi-tile cut: the partitioner wrote a
    per-op hardware_target and the sub-DLC builder sliced on the same
    boundaries. Reading the runs back beats trusting split_summary, which
    is a derived field that can drift from the ops list it summarises.
    """
    with open(partition_path) as f:
        doc = json.load(f)
    ops = [o for o in doc.get("ops", []) if o.get("dispatch_id") is not None]
    ops.sort(key=lambda o: o["dispatch_id"])
    groups: list[tuple[str, int, int]] = []
    for o in ops:
        t = o.get("hardware_target", "any")
        if groups and groups[-1][0] == t and o["dispatch_id"] == groups[-1][2] + 1:
            groups[-1] = (t, groups[-1][1], o["dispatch_id"])
        else:
            groups.append((t, o["dispatch_id"], o["dispatch_id"]))
    return groups


def load(path: str, ir: dict | None = None, repo_qnn_root: str = "") -> BindingSet:
    with open(path) as f:
        doc = json.load(f)
    ir_block = doc.get("ir", {})
    n_ops = len(ir["ops"]) if ir else None
    partition_rel = doc.get("partition", "")
    partition_path = (os.path.join(repo_qnn_root, partition_rel)
                      if partition_rel and repo_qnn_root else partition_rel)
    groups = partition_groups(partition_path) if partition_path else []
    handoff = ()
    if partition_path:
        with open(partition_path) as f:
            handoff = tuple(json.load(f).get("split_summary", {})
                            .get("handoff_tensors", []))
    bindings: list[Binding] = []
    for entry in doc["bindings"]:
        spec = entry.get("ops", "all")
        derived = ""
        if spec == "all":
            if n_ops is None:
                raise ValueError(f"{path}: ops='all' needs the IR to resolve")
            first, last = ir["ops"][0]["dispatch_id"], ir["ops"][-1]["dispatch_id"]
            derived = "whole network (ops='all')"
        elif isinstance(spec, dict) and "from_partition" in spec:
            if not groups:
                raise ValueError(f"{path}: ops.from_partition needs a `partition` path")
            g = int(spec["from_partition"]["group"])
            if g >= len(groups):
                raise ValueError(
                    f"{path}: partition has {len(groups)} contiguous target group(s), "
                    f"binding asks for group {g}")
            target, first, last = groups[g]
            derived = (f"{os.path.basename(partition_path)} group {g} "
                       f"(hardware_target={target})")
        else:
            first, last = int(spec["first"]), int(spec["last"])
            derived = "hand-written range"
        bindings.append(Binding(
            id=int(entry["id"]), name=entry["name"], first=first, last=last,
            source_onnx=entry.get("source_onnx", ""), derived_from=derived,
            depends_on=tuple(entry.get("depends_on", [])),
            backends={k: BackendBinding(k, v["ctx"], v["graph"])
                      for k, v in entry["backends"].items()},
            lowering=tuple(entry.get("lowering", doc.get("lowering", []))),
            lowering_note=entry.get("lowering_note", doc.get("lowering_note", "")),
        ))
    bindings.sort(key=lambda b: b.id)
    return BindingSet(network=doc["network"], ir_spec=ir_block.get("source", ""),
                      quant=ir_block.get("quant", "int8"),
                      graph_json=ir_block.get("graph_json"), bindings=bindings,
                      partition=partition_path, handoff_tensors=handoff)


def load_all(bindings_dir: str) -> dict[str, dict]:
    """Read every manifest without resolving `ops: all` (no IR yet)."""
    out = {}
    for fn in sorted(os.listdir(bindings_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(bindings_dir, fn)) as f:
            doc = json.load(f)
        out[doc["network"]] = {"path": os.path.join(bindings_dir, fn), "doc": doc}
    return out


# --------------------------------------------------------------------------
# Capability checking — the registry, not a sentinel cost, decides where a
# binding may run.
# --------------------------------------------------------------------------
@dataclass
class Feasibility:
    binding: Binding
    allowed: dict[str, list[str]]     # kind -> [] (ok) or [blocking op kinds]
    n_ir_ops: int = 0                 # ops in the tile before lowering
    n_lowered_ops: int = 0            # ops the compiled graph actually holds
    lowering_log: tuple[str, ...] = ()

    def kinds_ok(self) -> list[str]:
        return [k for k, blockers in self.allowed.items() if not blockers]


def check(bset: BindingSet, ir: dict, registry) -> list[Feasibility]:
    """For each binding, ask the registry which core kinds can run *every*
    op in the tile.  Returns per-kind blocking op lists (empty = feasible)."""
    by_did = {o["dispatch_id"]: o for o in ir["ops"]}
    kinds = sorted(registry.by_kind)
    out: list[Feasibility] = []
    for b in bset.bindings:
        tile = [by_did[d] for d in b.op_ids() if d in by_did]
        lowered, log = lowering.apply(tile, list(b.lowering))
        allowed: dict[str, list[str]] = {}
        for kind in kinds:
            caps = set()
            for c in registry.by_kind[kind]:
                caps |= set(c.capabilities)
            blockers = sorted({o["op"] for o in lowered if o["op"] not in caps})
            allowed[kind] = blockers
        out.append(Feasibility(b, allowed, n_ir_ops=len(tile),
                               n_lowered_ops=len(lowered), lowering_log=tuple(log)))
    return out


def reconcile(feas: list[Feasibility], strict: bool = True) -> list[str]:
    """Cross-check declared backends against capability results.

    A binding that declares a context for a kind the registry rejects is a
    hard error — one of the two is wrong and silently trusting either one
    is how a 100-second sentinel gets into a cost model.  The reverse (a
    feasible kind with no context built yet) is just unbuilt coverage.
    """
    problems, notes = [], []
    for f in feas:
        for kind, bb in f.binding.backends.items():
            blockers = f.allowed.get(kind)
            if blockers is None:
                problems.append(f"{f.binding.name}: declares kind {kind!r} "
                                f"which the registry does not define")
            elif blockers:
                problems.append(
                    f"{f.binding.name}: declares a {kind} context ({bb.ctx}) but "
                    f"the registry says {kind} cannot run {blockers}")
        for kind in f.kinds_ok():
            if kind not in f.binding.backends:
                notes.append(f"{f.binding.name}: {kind} is capable but no context "
                             f"is built — scheduler will not see this cell")
    if problems and strict:
        raise ValueError("binding/registry mismatch:\n  " + "\n  ".join(problems))
    return notes


# --------------------------------------------------------------------------
# Artifact cross-check — does the declared tile match the thing that was
# actually compiled?
#
# The binding says "IR ops F..L are executed by graph G in context C". The
# runtime never checks that: it looks up (ctx, graph) and executes. So a
# re-slice upstream can leave the manifest describing a cut that no longer
# exists, and the only symptom would be a wrong op count in the trace or a
# capability verdict computed over the wrong ops. When the manifest names
# the ONNX a tile was compiled from, count its nodes and say so.
# --------------------------------------------------------------------------
def verify_against_artifacts(bset: BindingSet, qnn_root: str,
                             python_exe: str | None = None) -> list[str]:
    import subprocess
    todo = [(b, os.path.join(qnn_root, b.source_onnx))
            for b in bset.bindings if b.source_onnx]
    if not todo:
        return []
    missing = [p for _, p in todo if not os.path.exists(p)]
    if missing:
        return [f"source_onnx not found: {', '.join(os.path.relpath(m, qnn_root) for m in missing)}"]
    script = (
        "import json,sys,onnx\n"
        "out={}\n"
        "for p in sys.argv[1:]:\n"
        "    m=onnx.load(p, load_external_data=False)\n"
        "    out[p]={'nodes':len(m.graph.node),\n"
        "            'inputs':[i.name for i in m.graph.input],\n"
        "            'outputs':[o.name for o in m.graph.output]}\n"
        "print(json.dumps(out))\n")
    exe = python_exe or "python3"
    try:
        res = subprocess.run([exe, "-c", script] + [p for _, p in todo],
                             capture_output=True, text=True, check=True)
        info = json.loads(res.stdout)
    except Exception as e:                       # onnx not importable, etc.
        return [f"artifact check skipped ({type(e).__name__}: no onnx reader)"]

    findings = []
    def norm(n):                                  # ONNX names vs IR names
        return n.replace("/", "_").replace(".", "_").strip("_")
    for b, path in todo:
        got = info[path]
        declared = b.last - b.first + 1
        if got["nodes"] != declared:
            findings.append(
                f"{b.name}: declares {declared} IR ops ({b.derived_from}) but "
                f"{os.path.basename(path)} holds {got['nodes']} nodes "
                f"(delta {got['nodes'] - declared:+d}) — expected when the IR and the "
                f"artifact come from different converter routes; "
                f"runtime/build_qnn_to_onnx_namemap.py is the bridge if you need "
                f"op-level correspondence")
        for t in bset.handoff_tensors:
            produced = any(norm(t) == norm(o) for o in got["outputs"])
            consumed = any(norm(t) == norm(i) for i in got["inputs"])
            if not (produced or consumed):
                continue
            findings.append(f"{b.name}: handoff tensor {t} "
                            f"{'produced' if produced else 'consumed'} — OK")
    return findings

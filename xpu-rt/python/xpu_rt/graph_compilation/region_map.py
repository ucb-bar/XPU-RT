"""Graph Analysis V2 (Milestone B) — IR-grounded region map.

Reads the lowered Payload IR (``01_payload_lowering/`` artifacts) plus
the captured FX metadata (``00_graph_capture/dynamo_partitions/*_meta.json``)
and produces three derived JSON views under ``02_graph_analysis/``:

- ``region_map.json``           — schema_version: ``region_map_v1``
- ``tensor_use_def_graph.json`` — schema_version: ``tensor_use_def_graph_v1``
- ``region_graph.json``         — schema_version: ``region_graph_v1``

The IR (``payload.mlir``) is the canonical source of truth; these JSON
views are *projections* of that IR for agent/LLM consumption. The audit
gate is that every ``compgen.region_id`` attribute observed in the IR
must appear as a region in ``region_map.json``.

This module does **not** modify ``FXImporter``, decompositions, capture,
pipeline, or runtime. It is a pure post-pass over already-lowered
artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# MLIR text parsing
# --------------------------------------------------------------------------- #

# Top-level op line. Captures result SSA names (optional), operand SSA names,
# the dialect.op stem, the inline-attr region_id (optional), and the trailing
# type annotation block.
#
# We anchor at non-blank line start; we tolerate leading whitespace.
_OP_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<results>%[A-Za-z0-9_]+(?:\s*,\s*%[A-Za-z0-9_]+)*)\s*=\s*)?
    (?P<dialect>[a-z][a-z0-9_]*)\.(?P<op>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)
_OPERAND_RE = re.compile(r"%([A-Za-z0-9_]+)")
_REGION_ID_ATTR_RE = re.compile(r'compgen\.region_id\s*=\s*"(?P<rid>[^"]+)"')
_DISPATCH_ID_ATTR_RE = re.compile(r'compgen\.dispatch_id\s*=\s*"(?P<did>[^"]+)"')
# Last ``: (...) -> tensor<...>`` annotation on the line — this is the result
# type we attribute to the SSA result(s). ``linalg.matmul`` and friends use
# structured ``ins/outs`` syntax with no leading ``:`` before the result; the
# fallback regex below picks the last ``tensor<...>`` token on the line.
_TYPE_ANNOT_RE = re.compile(r":\s*(?:\([^)]*\)\s*->\s*)?(?P<types>tensor<[^>]+>(?:\s*,\s*tensor<[^>]+>)*)\s*$")
_TENSOR_TYPE_RE = re.compile(r"tensor<(?P<dims>[^>]+)>")
_TRAILING_RESULT_TYPE_RE = re.compile(r"->\s*(?P<typ>tensor<[^>]+>)\s*$")
_FUNC_HEADER_RE = re.compile(r"^\s*func\.func\s+(?:private\s+)?@(?:\"[^\"]+\"|[^\s(]+)\s*\(")
_FUNC_RETURN_RE = re.compile(r"^\s*func\.return\s+(?P<ssa>%[A-Za-z0-9_]+(?:\s*,\s*%[A-Za-z0-9_]+)*)?")
_FUNC_FORWARD_RE = re.compile(r"^\s*func\.func\s+@forward\s*\((?P<args>[^)]*)\)\s*->")
_FUNC_ARG_RE = re.compile(r"%(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<typ>tensor<[^>]+>)")


_DTYPE_SIZES: dict[str, int] = {
    "f64": 8, "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1,
    "ui64": 8, "ui32": 4, "ui16": 2, "ui8": 1,
    "i1": 1,  # bool stored as 1 byte
}


def _parse_tensor_type(typ: str) -> tuple[list[int | None], str, int]:
    """Parse a ``tensor<DxNxf32>`` annotation into (shape, dtype, bytes).

    Dynamic dims (``?``) are kept as ``None`` in the shape; bytes is
    computed assuming dynamic dims contribute a factor of 1 (so the
    answer is a lower bound — flagged as ``shape_dynamic = True`` in
    the caller). Returns ``([], "", 0)`` if the type is not parseable.
    """
    m = _TENSOR_TYPE_RE.match(typ.strip())
    if not m:
        return [], "", 0
    dims = m.group("dims").split("x")
    if not dims:
        return [], "", 0
    dtype = dims[-1]
    shape: list[int | None] = []
    for d in dims[:-1]:
        if d == "?":
            shape.append(None)
        else:
            try:
                shape.append(int(d))
            except ValueError:
                shape.append(None)
    dtype_size = _DTYPE_SIZES.get(dtype, 4)
    numel = 1
    for d in shape:
        if d is None:
            continue
        numel *= max(d, 1)
    return shape, dtype, numel * dtype_size


# --------------------------------------------------------------------------- #
# Per-op record built by the MLIR walker
# --------------------------------------------------------------------------- #


@dataclass
class _ParsedOp:
    line_index: int
    op_name: str
    dialect: str
    op_stem: str
    region_id: str | None
    dispatch_id: str | None
    results: list[str] = field(default_factory=list)
    operands: list[str] = field(default_factory=list)
    callee: str | None = None
    result_types: list[str] = field(default_factory=list)


def _parse_payload_module(mlir_text: str) -> tuple[list[_ParsedOp], list[str], list[tuple[str, str]], list[str]]:
    """Walk one payload.mlir and return (ops_in_forward, forward_arg_ssa,
    forward_arg_typed, forward_return_ssa).

    Only ops *inside* ``func.func @forward`` participate in the region map
    — top-level ``func.func private @callee(...)`` declarations do not.
    """
    ops: list[_ParsedOp] = []
    forward_arg_typed: list[tuple[str, str]] = []  # (ssa, type)
    forward_return_ssa: list[str] = []
    forward_arg_ssa: list[str] = []

    in_forward = False
    for i, raw_line in enumerate(mlir_text.splitlines()):
        line = raw_line

        # Detect the start of @forward.
        if not in_forward:
            mfh = _FUNC_FORWARD_RE.match(line)
            if mfh:
                args = mfh.group("args")
                for am in _FUNC_ARG_RE.finditer(args):
                    forward_arg_typed.append((am.group("name"), am.group("typ")))
                forward_arg_ssa = [a for a, _ in forward_arg_typed]
                in_forward = True
            continue

        # Inside @forward.
        ret = _FUNC_RETURN_RE.match(line)
        if ret:
            ssa = ret.group("ssa") or ""
            for tok in ssa.split(","):
                tok = tok.strip()
                if tok.startswith("%"):
                    forward_return_ssa.append(tok[1:])
            in_forward = False
            continue
        # Skip ``}`` and blank lines and pure declarations.
        if not line.strip() or line.strip() in ("}",):
            continue

        m = _OP_LINE_RE.match(line)
        if not m:
            continue
        dialect = m.group("dialect")
        op_stem = m.group("op")
        op_name = f"{dialect}.{op_stem}"

        # Skip the ``func.return`` line (handled above) and ``builtin.module``.
        if op_name in ("func.return", "builtin.module"):
            continue

        results: list[str] = []
        if m.group("results"):
            for tok in m.group("results").split(","):
                tok = tok.strip()
                if tok.startswith("%"):
                    results.append(tok[1:])

        # Operands appear after the op stem and before any ``{`` attr block;
        # we use a permissive scan that accepts everything inside the first
        # ``(...)``. Some structured ops use ``ins(... : ...) outs(... : ...)``
        # form; we still capture ``%foo`` SSA references the same way.
        operands_raw = line[m.end():]
        # Cut off type annotation suffix to avoid grabbing %... that show up in
        # comments — there shouldn't be any but defensively keep it simple.
        operands: list[str] = []
        for om in _OPERAND_RE.finditer(operands_raw):
            name = om.group(1)
            if name in results:
                continue
            if name not in operands:
                operands.append(name)

        rid_m = _REGION_ID_ATTR_RE.search(line)
        rid = rid_m.group("rid") if rid_m else None
        did_m = _DISPATCH_ID_ATTR_RE.search(line)
        did = did_m.group("did") if did_m else None

        callee: str | None = None
        if op_name == "func.call":
            cm = re.search(r'func\.call\s+@(?:"(?P<q>[^"]+)"|(?P<u>[^"\s(]+))', line)
            if cm:
                callee = cm.group("q") or cm.group("u")

        # Result types: prefer the ``: (...) -> tensor<...>`` annotation;
        # fall back to ``-> tensor<...>`` (linalg.matmul / linalg.transpose
        # use this structured form); finally fall back to the last
        # ``tensor<...>`` token on the line.
        result_types: list[str] = []
        ta = _TYPE_ANNOT_RE.search(line)
        if ta:
            for tt in ta.group("types").split(","):
                result_types.append(tt.strip())
        else:
            tail = _TRAILING_RESULT_TYPE_RE.search(line)
            if tail:
                result_types.append(tail.group("typ"))
            else:
                # Last-resort: use the last tensor<...> on the line. This
                # only fires for ops we wouldn't otherwise type.
                tt_all = list(_TENSOR_TYPE_RE.finditer(line))
                if tt_all and results:
                    last = tt_all[-1]
                    result_types.append(line[last.start():last.end()])

        ops.append(
            _ParsedOp(
                line_index=i,
                op_name=op_name,
                dialect=dialect,
                op_stem=op_stem,
                region_id=rid,
                dispatch_id=did,
                results=results,
                operands=operands,
                callee=callee,
                result_types=result_types,
            )
        )

    return ops, forward_arg_ssa, forward_arg_typed, forward_return_ssa


# --------------------------------------------------------------------------- #
# Region kind heuristics
# --------------------------------------------------------------------------- #


def _region_kind(region_ops: list[_ParsedOp]) -> str:
    """Pick a region ``kind`` from its leading op."""
    if not region_ops:
        return "unknown"
    leader = region_ops[0]
    name = leader.op_name
    if name == "linalg.matmul":
        return "matmul"
    if name.startswith("linalg.conv"):
        return "conv"
    if name == "linalg.transpose":
        return "transpose"
    if name == "linalg.generic":
        return "generic"
    if name.startswith("linalg."):
        return f"linalg_{leader.op_stem}"
    if name == "tensor.empty":
        return "tensor_empty"
    if name == "func.call" and leader.callee is not None:
        c = leader.callee.lower()
        if "gelu" in c:
            return "elementwise_gelu"
        if "relu" in c:
            return "elementwise_relu"
        if "tanh" in c:
            return "elementwise_tanh"
        if "softmax" in c:
            return "softmax"
        if "layer_norm" in c or "layernorm" in c:
            return "layer_norm"
        if "batch_norm" in c or "batchnorm" in c:
            return "batch_norm"
        if "bias_add" in c:
            return "bias_add"
        if "embedding" in c:
            return "embedding"
        # Fall back to opaque_<callee_stem>; strip leading 'aten_' for terseness.
        stem = leader.callee.replace("<", "").replace(">", "").replace(" ", "_")
        return f"opaque_{stem}"
    return "unknown"


_HEAVY_KINDS = {"matmul", "conv"}
_REDUCTION_KINDS = {"matmul", "conv", "softmax", "layer_norm"}


def _estimate_region_cost(
    *,
    kind: str,
    output_bytes_total: int,
    input_bytes_total: int,
    output_numel: int = 0,
    reduction_dim: int = 1,
) -> tuple[int, int]:
    """Rough deterministic flops/bytes estimates. Real cost models are M-C.

    For matmul / conv we use ``2 * output_numel * reduction_dim`` (FMA
    count). For elementwise ops we use one flop per output element.
    """
    bytes_total = input_bytes_total + output_bytes_total
    if kind in _HEAVY_KINDS:
        flops = max(2 * output_numel * max(reduction_dim, 1), 1)
    elif kind == "softmax":
        flops = max(output_numel * 5, 1)
    elif kind in ("layer_norm", "batch_norm"):
        flops = max(output_numel * 8, 1)
    else:
        flops = max(output_numel, 1)
    return flops, bytes_total


# --------------------------------------------------------------------------- #
# Public dataclass returned by build_graph_analysis()
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GraphAnalysisResult:
    region_map_path: Path
    tensor_use_def_graph_path: Path
    region_graph_path: Path
    graph_analysis_report_path: Path
    region_count: int
    tensor_count: int
    edge_count: int
    is_dag: bool


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


def build_graph_analysis(run_dir: Path) -> GraphAnalysisResult:
    """Build the three Graph Analysis V2 JSONs under ``02_graph_analysis/``.

    Requires a completed ``00_graph_capture/`` and ``01_payload_lowering/``.
    """
    run_dir = Path(run_dir).resolve()
    pl_dir = run_dir / "01_payload_lowering"
    out_dir = run_dir / "02_graph_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_index = _read_json(pl_dir / "payload_index.json")

    # Optional v2 accounting — used to attach source_classification + fx_nodes.
    accounting: dict[str, Any] = {}
    acc_path = pl_dir / "fx_to_payload_accounting.json"
    if acc_path.exists():
        accounting = _read_json(acc_path)

    accounting_by_module: dict[str, dict[str, Any]] = {}
    for mod in accounting.get("modules", []):
        accounting_by_module[mod.get("module_id", "")] = mod

    all_regions: list[dict[str, Any]] = []
    all_tensors: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []

    for mod in payload_index.get("modules", []):
        if mod.get("status") == "skipped":
            continue
        module_id = mod["module_id"]
        payload_ref = mod["payload_mlir"]
        mlir_text = (run_dir / payload_ref).read_text(encoding="utf-8")
        ops, fwd_args, fwd_arg_typed, fwd_returns = _parse_payload_module(mlir_text)

        # Group ops by region_id. Ops without a region_id are still tracked
        # (e.g. ``tensor.empty`` ops emitted as scratch buffers); they get
        # synthetic region_ids ``_unmapped_<index>`` so the audit can see them.
        # ``op_to_rid`` maps each op (by line_index) to its assigned region_id
        # so consumers / edges always reference the correct synthetic name.
        regions: dict[str, list[_ParsedOp]] = {}
        op_to_rid: dict[int, str] = {}
        unmapped_counter = 0
        for op in ops:
            rid = op.region_id
            if rid is None:
                rid = f"_unmapped_{module_id}_{unmapped_counter}"
                unmapped_counter += 1
            regions.setdefault(rid, []).append(op)
            op_to_rid[op.line_index] = rid

        # Producer map: ssa_name → region_id.
        ssa_producer: dict[str, str] = {}
        for rid, region_ops in regions.items():
            for op in region_ops:
                for r in op.results:
                    ssa_producer[r] = rid
        # Function args are exogenous producers tagged ``input``.
        for arg in fwd_args:
            ssa_producer.setdefault(arg, "input")

        # SSA → result type (for tensor metadata).
        ssa_type: dict[str, str] = {}
        for op in ops:
            for r, t in zip(op.results, op.result_types, strict=False):
                ssa_type[r] = t
        for arg, t in fwd_arg_typed:
            ssa_type[arg] = t

        # Build region records.
        accounting_for_module = accounting_by_module.get(module_id, {})
        accounting_nodes = accounting_for_module.get("nodes", [])
        # Index region_id → list of FX-node names that mention this region_id
        # via their payload_ops.
        rid_to_fx: dict[str, list[str]] = {}
        rid_to_classification: dict[str, list[str]] = {}
        for n in accounting_nodes:
            for po in n.get("payload_ops", []):
                pid = po.get("region_id")
                if not pid:
                    continue
                rid_to_fx.setdefault(pid, []).append(n["fx_node"])
                rid_to_classification.setdefault(pid, []).append(n["classification"])

        # Severity ordering for source_classification aggregation.
        _CLASS_RANK = {
            "diagnostic_error": 0,
            "dropped_auxiliary_output": 1,
            "opaque_fallback": 2,
            "closed_by_registry": 3,
            "decomposed_structured": 4,
            "resolved_alias": 5,
            "placeholder": 6,
            "output": 7,
        }

        for rid, region_ops in regions.items():
            # Compute inputs/outputs of the region: SSA values consumed inside
            # but produced outside (= inputs); SSA values produced inside and
            # consumed outside the region or in func.return (= outputs).
            inside_results: set[str] = set()
            inside_operands: set[str] = set()
            for op in region_ops:
                for r in op.results:
                    inside_results.add(r)
                for o in op.operands:
                    inside_operands.add(o)
            inputs = sorted(inside_operands - inside_results)
            # Outputs: any inside-result SSA that is consumed elsewhere or
            # appears in func.return.
            outside_consumers: dict[str, list[str]] = {r: [] for r in inside_results}
            for op in ops:
                op_rid = op_to_rid[op.line_index]
                if op_rid == rid:
                    continue
                for o in op.operands:
                    if o in inside_results:
                        outside_consumers.setdefault(o, []).append(op_rid)
            for ret in fwd_returns:
                if ret in inside_results:
                    outside_consumers.setdefault(ret, []).append("output")
            outputs = sorted({k for k, v in outside_consumers.items() if v})

            # Bytes traffic + output numel for cost estimation.
            input_bytes = 0
            input_max_dim = 1
            for ssa in inputs:
                t = ssa_type.get(ssa, "")
                shape, _, b = _parse_tensor_type(t) if t else ([], "", 0)
                input_bytes += b
                if shape:
                    dims = [d for d in shape if isinstance(d, int) and d > 0]
                    if dims:
                        input_max_dim = max(input_max_dim, max(dims))
            output_bytes = 0
            output_numel = 0
            for ssa in outputs:
                t = ssa_type.get(ssa, "")
                shape, dtype, b = _parse_tensor_type(t) if t else ([], "", 0)
                output_bytes += b
                if shape:
                    n = 1
                    for d in shape:
                        if isinstance(d, int) and d > 0:
                            n *= d
                    output_numel += n

            kind = _region_kind(region_ops)
            flops, bytes_total = _estimate_region_cost(
                kind=kind,
                output_bytes_total=output_bytes,
                input_bytes_total=input_bytes,
                output_numel=output_numel,
                reduction_dim=input_max_dim,
            )
            arithmetic_intensity = flops / max(bytes_total, 1)

            # source_classification: most-severe over the FX nodes that
            # mention this region via payload_ops (v2 schema). When the
            # region was emitted by decomposition without an FX-node link
            # (e.g. tensor.empty scratch), fall back to ``decomposed_structured``
            # for linalg/tensor/arith ops or ``opaque_fallback`` for func.call.
            classes = rid_to_classification.get(rid, [])
            if classes:
                source_classification = sorted(
                    classes, key=lambda c: _CLASS_RANK.get(c, 99)
                )[0]
            else:
                if region_ops and region_ops[0].op_name == "func.call":
                    source_classification = "opaque_fallback"
                elif rid.startswith("_unmapped_"):
                    source_classification = "decomposed_structured"
                else:
                    source_classification = "decomposed_structured"

            region_record: dict[str, Any] = {
                "region_id": rid,
                "module_id": module_id,
                "kind": kind,
                "source_classification": source_classification,
                "fx_nodes": sorted(set(rid_to_fx.get(rid, []))),
                "payload_ops": [
                    {
                        "op_name": op.op_name,
                        "region_id": op.region_id,
                        "dispatch_id": op.dispatch_id,
                        "callee": op.callee,
                        "results": op.results,
                        "operands": op.operands,
                        "payload_ref": payload_ref,
                    }
                    for op in region_ops
                ],
                "inputs": [
                    {"tensor_id": f"{module_id}::{ssa}", "ssa": ssa}
                    for ssa in inputs
                ],
                "outputs": [
                    {"tensor_id": f"{module_id}::{ssa}", "ssa": ssa}
                    for ssa in outputs
                ],
                "estimated": {
                    "flops": flops,
                    "bytes": bytes_total,
                    "arithmetic_intensity": round(arithmetic_intensity, 6),
                },
                "gap_refs": [],
                "extension_refs": [],
            }
            all_regions.append(region_record)

        # Build tensor records for every SSA value that appears.
        all_ssa = set(ssa_producer.keys())
        for op in ops:
            for r in op.results:
                all_ssa.add(r)
            for o in op.operands:
                all_ssa.add(o)

        # Order SSA values by op line_index to compute reuse_horizon.
        first_use_index: dict[str, int] = {}
        # Producer line for each SSA (or 0 for func args).
        producer_line: dict[str, int] = {a: -1 for a in fwd_args}
        for op in ops:
            for r in op.results:
                producer_line.setdefault(r, op.line_index)
        # First *consumer* line for each SSA (any later op that lists it as operand).
        for op in ops:
            for o in op.operands:
                if o in producer_line and producer_line[o] < op.line_index:
                    first_use_index.setdefault(o, op.line_index)
        # Consumers list for each SSA.
        consumers: dict[str, list[str]] = {}
        for op in ops:
            for o in op.operands:
                rid_of_consumer = op_to_rid[op.line_index]
                consumers.setdefault(o, []).append(rid_of_consumer)
        for ret in fwd_returns:
            consumers.setdefault(ret, []).append("output")

        for ssa in sorted(all_ssa):
            t = ssa_type.get(ssa, "")
            shape, dtype, b = _parse_tensor_type(t) if t else ([], "", 0)
            producer_region = ssa_producer.get(ssa, "input")
            cons = consumers.get(ssa, [])
            consumer_count = len(cons)
            # reuse_horizon: line index of the FIRST consumer minus the
            # producer's line index. If there's no consumer, ``inf``
            # (recorded as -1 to keep the JSON integral).
            if ssa in first_use_index and ssa in producer_line:
                horizon = first_use_index[ssa] - producer_line[ssa]
            else:
                horizon = -1
            if producer_region == "input":
                lifetime = "input"
            elif "output" in cons:
                lifetime = "output"
            else:
                lifetime = "transient"
            # is_reduction_input: any consumer region's kind is in REDUCTION.
            cons_kinds = set()
            for crid in cons:
                if crid == "output":
                    continue
                # Look up that region's kind in the regions dict.
                if crid in regions:
                    cons_kinds.add(_region_kind(regions[crid]))
            is_reduction_input = bool(cons_kinds & _REDUCTION_KINDS)

            all_tensors.append(
                {
                    "tensor_id": f"{module_id}::{ssa}",
                    "module_id": module_id,
                    "ssa": ssa,
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": b,
                    "producer_region": producer_region,
                    "consumer_regions": sorted(set(cons)),
                    "consumer_count": consumer_count,
                    "reuse_horizon": horizon,
                    "producer_lifetime_class": lifetime,
                    "is_reduction_input": is_reduction_input,
                    "reduction_axis": None,
                }
            )

        # Edges between regions: for each producer-region → consumer-region pair
        # carrying any SSA value, emit one edge per (src, dst, tensor_id).
        for ssa, cons in consumers.items():
            src = ssa_producer.get(ssa)
            if src is None or src == "input":
                continue
            t = ssa_type.get(ssa, "")
            _, _, b = _parse_tensor_type(t) if t else ([], "", 0)
            for c in cons:
                if c == "output":
                    continue
                if c == src:
                    continue
                all_edges.append(
                    {
                        "src": src,
                        "dst": c,
                        "tensor_id": f"{module_id}::{ssa}",
                        "bytes": b,
                    }
                )

    # ------------------------------------------------------------------ #
    # Critical path (longest path by bytes through the region DAG)
    # ------------------------------------------------------------------ #
    region_ids = [r["region_id"] for r in all_regions]
    bytes_by_region: dict[str, int] = {r["region_id"]: r["estimated"]["bytes"] for r in all_regions}
    successors: dict[str, list[str]] = {r["region_id"]: [] for r in all_regions}
    indeg: dict[str, int] = {r["region_id"]: 0 for r in all_regions}
    for e in all_edges:
        if e["src"] in successors and e["dst"] in indeg:
            successors[e["src"]].append(e["dst"])
            indeg[e["dst"]] += 1

    # Topological sort (Kahn's). If we detect a cycle, mark is_dag=False
    # and skip the critical-path computation.
    topo: list[str] = []
    queue = [r for r in region_ids if indeg.get(r, 0) == 0]
    indeg_work = dict(indeg)
    while queue:
        n = queue.pop(0)
        topo.append(n)
        for s in successors.get(n, []):
            indeg_work[s] -= 1
            if indeg_work[s] == 0:
                queue.append(s)
    is_dag = len(topo) == len(region_ids)

    critical_path: list[str] = []
    if is_dag:
        # Longest path in DAG by node weight.
        best_score: dict[str, int] = {n: bytes_by_region.get(n, 0) for n in region_ids}
        best_prev: dict[str, str | None] = {n: None for n in region_ids}
        for n in topo:
            for s in successors.get(n, []):
                cand = best_score[n] + bytes_by_region.get(s, 0)
                if cand > best_score[s]:
                    best_score[s] = cand
                    best_prev[s] = n
        if best_score:
            end = max(best_score, key=lambda x: best_score[x])
            chain: list[str] = []
            cur: str | None = end
            while cur is not None:
                chain.append(cur)
                cur = best_prev[cur]
            chain.reverse()
            critical_path = chain

    # ------------------------------------------------------------------ #
    # Emit JSON artifacts
    # ------------------------------------------------------------------ #
    region_map_obj = {
        "schema_version": "region_map_v1",
        "totals": {
            "regions": len(all_regions),
            "modules": len(payload_index.get("modules", [])),
        },
        "regions": all_regions,
    }
    region_map_path = out_dir / "region_map.json"
    region_map_path.write_text(
        json.dumps(region_map_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    tensor_obj = {
        "schema_version": "tensor_use_def_graph_v1",
        "totals": {"tensors": len(all_tensors)},
        "tensors": all_tensors,
    }
    tensor_path = out_dir / "tensor_use_def_graph.json"
    tensor_path.write_text(
        json.dumps(tensor_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    region_graph_obj = {
        "schema_version": "region_graph_v1",
        "totals": {
            "nodes": len(all_regions),
            "edges": len(all_edges),
            "is_dag": is_dag,
        },
        "nodes": [
            {
                "region_id": r["region_id"],
                "module_id": r["module_id"],
                "kind": r["kind"],
                "flops": r["estimated"]["flops"],
                "bytes": r["estimated"]["bytes"],
            }
            for r in all_regions
        ],
        "edges": all_edges,
        "critical_path": critical_path,
    }
    region_graph_path = out_dir / "region_graph.json"
    region_graph_path.write_text(
        json.dumps(region_graph_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_obj = {
        "schema_version": "graph_analysis_report_v1",
        "stage_id": "graph_analysis",
        "status": "pass",
        "totals": {
            "regions": len(all_regions),
            "tensors": len(all_tensors),
            "edges": len(all_edges),
            "is_dag": is_dag,
            "critical_path_length": len(critical_path),
        },
        "llm_calls": 0,
    }
    report_path = out_dir / "graph_analysis_report.json"
    report_path.write_text(
        json.dumps(report_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    return GraphAnalysisResult(
        region_map_path=region_map_path,
        tensor_use_def_graph_path=tensor_path,
        region_graph_path=region_graph_path,
        graph_analysis_report_path=report_path,
        region_count=len(all_regions),
        tensor_count=len(all_tensors),
        edge_count=len(all_edges),
        is_dag=is_dag,
    )

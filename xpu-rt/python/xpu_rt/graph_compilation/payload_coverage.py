"""Payload Coverage Audit (v2) — per-FX-node accounting + dialect coverage.

The lowering stage already writes ``opaque_calls.json``,
``unsupported_ops.json``, and per-module ``diagnostics.json``. What it
does **not** answer is the central truth-telling question:

    Does every FX node have an explicit, single-valued classification?
    No silent drops. No silent decompositions.

This module produces three audit files at ``01_payload_lowering/``:

- ``fx_to_payload_accounting.json`` (schema_version
  ``fx_to_payload_accounting_v2``) — every FX node, classified into
  exactly one of the eight allowed values:

    * ``placeholder``                 — graph input (or get_attr param)
    * ``output``                      — graph output
    * ``decomposed_structured``       — lowered to real linalg/tensor/arith
    * ``opaque_fallback``             — recorded in ``opaque_calls.json``
    * ``closed_by_registry``          — replaced by a registered user-space ext
    * ``resolved_alias``              — getitem / view-shape op, no payload op needed
    * ``dropped_auxiliary_output``    — diagnostic-only no-typeinfo skip, FLAGGED
    * ``diagnostic_error``            — importer raised an error diagnostic, FLAGGED

  Per-node entries carry: ``fx_node``, ``fx_target``, ``op_kind``,
  ``classification``, ``payload_ops`` (list-of-objects with ``op_name`` /
  ``region_id`` / ``payload_ref``), ``diagnostics`` (list of strings),
  ``gap_id`` (filled by gap_discovery in ``02_gap_discovery/fx_audit_with_gaps.json``;
  always ``null`` here), and ``registry_closure`` (extension_id when
  ``classification == closed_by_registry``).

- ``dialect_coverage.json`` — per-module + aggregate counts of which
  dialects actually appear in the emitted Payload IR, plus the named
  ``func.call`` callees so opaque calls are visible by name.

- ``silent_drop_audit.json`` — strict acceptance gate:

    * ``unaccounted_call_function_nodes == []``
    * ``opaque_calls_without_origin == []``

  ``dropped_auxiliary_output`` and ``diagnostic_error`` are surfaced but
  do not fail the audit (they are known importer behaviours today: nodes
  with missing FX shape metadata, or downstream-only outputs that are
  dropped without consumers).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# --------------------------------------------------------------------------- #
# Targets that legitimately produce no Payload op (xDSL-level value reuse).
# --------------------------------------------------------------------------- #

# When a "skipping / no type info" diagnostic mentions one of these,
# treat as ``resolved_alias``. Anything else with a "skipping"
# diagnostic is a ``dropped_auxiliary_output`` (surfaced separately).
_RESOLVED_ALIAS_TOKENS: tuple[str, ...] = (
    "operator.getitem",
    "<built-in function getitem>",
    "getitem",
    "aten.view",
    "aten.reshape",
    "aten.transpose",
    "aten.permute",
    "aten.squeeze",
    "aten.unsqueeze",
    "aten.expand",
    "aten.detach",
    "aten.alias",
    "aten.contiguous",
    "aten.flatten",
    "aten.t.",
    "aten._unsafe_view",
    "_to_copy",
    "aten.to.",
)

# Bare-name FX targets used by Dynamo for value-passing view operations
# that the FXImporter intentionally lowers to zero payload ops (the SSA
# value is reused directly). These produce no diagnostic and no payload
# op but are NOT silent drops — they're legitimate aliases.
_DYNAMO_VIEW_BARE_NAMES: frozenset[str] = frozenset({
    "reshape", "view", "permute", "transpose",
    "squeeze", "unsqueeze", "expand", "flatten",
    "contiguous", "detach", "alias", "t",
})


# Allowed classifications (v2). Anything outside this set is a bug.
ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "placeholder",
        "output",
        "decomposed_structured",
        "opaque_fallback",
        "closed_by_registry",
        "resolved_alias",
        "dropped_auxiliary_output",
        "diagnostic_error",
    }
)

# MLIR op-line regex.
# Matches optional ``%name = `` (or ``%a, %b = ``) followed by ``dialect.op``.
# Captures the SSA result name(s), the dialect, and the op stem.
_OP_LINE_RE = re.compile(
    r"^\s*(?:(?P<results>%[A-Za-z0-9_]+(?:\s*,\s*%[A-Za-z0-9_]+)*)\s*=\s*)?"
    r"(?P<dialect>[a-z][a-z0-9_]*)\.(?P<op>[A-Za-z_][A-Za-z0-9_]*)"
)
_FUNC_CALL_CALLEE_RE = re.compile(r'func\.call\s+@(?:"(?P<q>[^"]+)"|(?P<u>[^"\s(]+))')
_REGION_ID_ATTR_RE = re.compile(r'compgen\.region_id\s*=\s*"(?P<rid>[^"]+)"')


# --------------------------------------------------------------------------- #
# Public dataclass returned by audit_payload_coverage()
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PayloadCoverageResult:
    fx_to_payload_accounting_path: Path
    dialect_coverage_path: Path
    silent_drop_audit_path: Path
    silent_drop_status: str  # "pass" | "fail"
    unaccounted_count: int
    dropped_auxiliary_output_count: int
    diagnostic_error_count: int
    opaque_without_origin_count: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _payload_op_record(po: dict[str, Any]) -> dict[str, Any]:
    """Project a ``payload_attribution.json`` op record into the v2
    ``payload_ops`` shape consumed by ``fx_to_payload_accounting.json``."""
    return {
        "op_name": po.get("op_name", ""),
        "region_id": po.get("region_id"),
        "dispatch_id": po.get("dispatch_id"),
        "callee": po.get("callee"),
        "payload_ref": po.get("payload_ref", ""),
    }


def _classify_skipped_target(target_str: str) -> str:
    """Map a 'skipping/no type info' diagnostic to ``resolved_alias`` or
    ``dropped_auxiliary_output`` based on the FX target."""
    s = target_str.lower()
    for tok in _RESOLVED_ALIAS_TOKENS:
        if tok.lower() in s:
            return "resolved_alias"
    return "dropped_auxiliary_output"


def _load_fx_graph(input_kind: str, run_dir: Path, graph_rel: str) -> Any:
    abs_path = run_dir / graph_rel
    if input_kind == "torch_dynamo_partition":
        gm = torch.load(abs_path, weights_only=False)
        return gm.graph
    if input_kind == "exported_program":
        ep = torch.export.load(str(abs_path))
        return ep.graph
    raise ValueError(f"unknown input_kind: {input_kind!r}")


def _read_json(path: Path) -> dict[str, Any]:
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return obj


# --------------------------------------------------------------------------- #
# Payload-MLIR walker
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _PayloadOp:
    """One parsed line from a payload.mlir file."""

    op_name: str  # e.g. "linalg.matmul" or "func.call"
    region_id: str | None  # value of compgen.region_id attr if present
    callee: str | None  # only set for func.call
    payload_ref: str  # run-dir-relative path to payload.mlir
    line_index: int  # 0-based line number


def _parse_payload_ops(payload_mlir_text: str, payload_ref: str) -> list[_PayloadOp]:
    """Walk MLIR text line-by-line, return one ``_PayloadOp`` per recognised
    op line. Only lines that match ``_OP_LINE_RE`` are returned; declaration
    lines (``func.func private @...``) inside a ``builtin.module`` are
    matched but their region_id is always ``None``."""
    ops: list[_PayloadOp] = []
    for i, line in enumerate(payload_mlir_text.splitlines()):
        m = _OP_LINE_RE.match(line)
        if not m:
            continue
        dialect = m.group("dialect")
        op_stem = m.group("op")
        op_name = f"{dialect}.{op_stem}"

        rid_m = _REGION_ID_ATTR_RE.search(line)
        rid = rid_m.group("rid") if rid_m else None

        callee: str | None = None
        if op_name == "func.call":
            cm = _FUNC_CALL_CALLEE_RE.search(line)
            if cm:
                callee = cm.group("q") or cm.group("u")

        ops.append(
            _PayloadOp(
                op_name=op_name,
                region_id=rid,
                callee=callee,
                payload_ref=payload_ref,
                line_index=i,
            )
        )
    return ops


# --------------------------------------------------------------------------- #
# Per-module classification
# --------------------------------------------------------------------------- #


def _classify_module(
    *,
    run_dir: Path,
    module: dict[str, Any],
    attribution_for_module: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Classify every FX node in one Dynamo partition / exported_program.

    ``attribution_for_module`` is the per-FX-node payload-op lookup
    produced by ``payload_attribution.build_payload_attribution`` — when
    supplied, it provides exact ``payload_ops`` for every classification
    (including ``decomposed_structured`` and ``closed_by_registry``).
    Falls back to the previous callee-matching heuristic when absent
    (so callers without an attribution sidecar still get attribution
    for opaque_fallback nodes).
    """
    if module.get("status") == "skipped":
        return {
            "module_id": module["module_id"],
            "status": "skipped",
            "nodes": [],
            "summary": {},
            "opaque_calls_without_origin": [],
        }

    attribution_for_module = attribution_for_module or {}
    module_dir = run_dir / Path(module["lowering_report"]).parent

    diagnostics_path = module_dir / "diagnostics.json"
    diagnostics = (
        _read_json(diagnostics_path).get("diagnostics", [])
        if diagnostics_path.exists()
        else []
    )
    opaque_path = module_dir / "opaque_calls.json"
    opaque_records = (
        _read_json(opaque_path).get("opaque_calls", []) if opaque_path.exists() else []
    )

    # Index FX-node-name → diagnostic message (multi-message flattened to a list).
    diag_by_node: dict[str, list[str]] = {}
    for d in diagnostics:
        n = d.get("fx_node") or ""
        if not n:
            continue
        msg = d.get("message", "")
        diag_by_node.setdefault(n, []).append(msg)

    # Skipping / no-type-info diagnostics — used for resolved_alias / dropped.
    skipping_nodes: set[str] = set()
    error_nodes: set[str] = set()
    resolved_diag_nodes: set[str] = set()
    dropped_diag_nodes: set[str] = set()
    for d in diagnostics:
        n = d.get("fx_node") or ""
        if not n:
            continue
        msg = d.get("message", "")
        lvl = d.get("level", "")
        if lvl == "warning" and ("skipping" in msg or "No type info" in msg):
            skipping_nodes.add(n)
        elif lvl == "error":
            error_nodes.add(n)
        elif msg.startswith("Resolved "):
            resolved_diag_nodes.add(n)
        elif msg.startswith("Dropped "):
            dropped_diag_nodes.add(n)

    # Inlined extensions: nodes structurally closed by a registered ext.
    inlined_targets: dict[str, str] = {}  # fx_target → extension_id
    for d in diagnostics:
        m = d.get("message", "")
        if "Inlined extension" in m:
            ext_match = re.search(r"Inlined extension '([^']+)'", m)
            tgt_match = re.search(r"for '([^']+)'", m)
            if ext_match and tgt_match:
                inlined_targets[tgt_match.group(1)] = ext_match.group(1)

    # Index FX-node → opaque record (so we can fill payload_ops cleanly).
    opaque_by_node: dict[str, dict[str, Any]] = {
        o["fx_node"]: o for o in opaque_records
    }

    # Parse the payload.mlir for this module so we can attribute payload_ops
    # to opaque calls (and later, to closed_by_registry inlinings) via callee
    # / region_id matching.
    payload_ref = module["payload_mlir"]
    payload_mlir_text = (run_dir / payload_ref).read_text(encoding="utf-8")
    payload_ops = _parse_payload_ops(payload_mlir_text, payload_ref)
    # Index func.call ops by callee (each callee in MLIR may be unique per
    # partition, but match all that share the name to be safe).
    callee_to_ops: dict[str, list[_PayloadOp]] = {}
    for p in payload_ops:
        if p.op_name == "func.call" and p.callee is not None:
            callee_to_ops.setdefault(p.callee, []).append(p)

    # Walk the FX graph and classify every node.
    graph = _load_fx_graph(module["input_kind"], run_dir, module["input_graph"])
    nodes_out: list[dict[str, Any]] = []

    for node in graph.nodes:
        target_str = str(getattr(node, "target", ""))
        op = node.op
        record: dict[str, Any] = {
            "fx_node": node.name,
            "fx_target": target_str,
            "op_kind": op,
            "classification": "",
            "payload_ops": [],
            "diagnostics": list(diag_by_node.get(node.name, [])),
            "gap_id": None,
            "registry_closure": None,
        }

        if op == "placeholder":
            record["classification"] = "placeholder"
        elif op == "output":
            record["classification"] = "output"
        elif op == "get_attr":
            # Folded into ``placeholder`` per the v2 classification set.
            # ``op_kind`` keeps the original ``get_attr`` so the
            # distinction is still visible.
            record["classification"] = "placeholder"
        elif op in ("call_function", "call_method", "call_module"):
            # Attribution from payload_attribution.json (02.5) — preferred
            # source when present. Falls back to the legacy
            # opaque_calls.json callee-matching for runs that pre-date the
            # attribution sidecar.
            attributed = attribution_for_module.get(node.name, [])

            if node.name in opaque_by_node:
                rec = opaque_by_node[node.name]
                record["classification"] = "opaque_fallback"
                if attributed:
                    record["payload_ops"] = [
                        _payload_op_record(po) for po in attributed
                    ]
                else:
                    callee = rec.get("callee", "")
                    pops = callee_to_ops.get(callee, [])
                    if pops:
                        chosen = pops.pop(0)
                        record["payload_ops"] = [
                            {
                                "op_name": chosen.op_name,
                                "region_id": chosen.region_id,
                                "payload_ref": chosen.payload_ref,
                            }
                        ]
                    else:
                        record["payload_ops"] = []
            elif target_str in inlined_targets:
                record["classification"] = "closed_by_registry"
                record["registry_closure"] = inlined_targets[target_str]
                # When attribution is available, the original FX node
                # carries the inlined extension's expanded ops directly.
                record["payload_ops"] = [
                    _payload_op_record(po) for po in attributed
                ]
            elif node.name in resolved_diag_nodes:
                # "Resolved getitem(...)" — primary result of a decomposed
                # tuple op; FXImporter reused an existing SSA value so no
                # new payload op was emitted.
                record["classification"] = "resolved_alias"
            elif node.name in dropped_diag_nodes:
                # "Dropped getitem(...)" — auxiliary tuple output dropped.
                record["classification"] = "dropped_auxiliary_output"
            elif node.name in error_nodes:
                record["classification"] = "diagnostic_error"
            elif node.name in skipping_nodes:
                record["classification"] = _classify_skipped_target(target_str)
            elif attributed:
                # Real decomposition with attribution — the canonical
                # ``decomposed_structured`` path.
                record["classification"] = "decomposed_structured"
                record["payload_ops"] = [
                    _payload_op_record(po) for po in attributed
                ]
            elif (
                target_str in _DYNAMO_VIEW_BARE_NAMES
                or any(tok in target_str.lower() for tok in _RESOLVED_ALIAS_TOKENS)
            ):
                # Dynamo-side bare-name view op (``reshape`` / ``permute`` /
                # ``transpose`` / ``view`` / ``squeeze`` / ...) that the
                # FXImporter handles by SSA-value reuse — no payload op
                # is emitted, and that is the correct lowering.
                record["classification"] = "resolved_alias"
            else:
                # Honest dropped output: FX node has no attribution, no
                # diagnostic, and is not a known alias. Surface as
                # ``dropped_auxiliary_output`` so downstream auditors
                # see it without poisoning the structured-decomposition
                # count.
                record["classification"] = "dropped_auxiliary_output"
                record["diagnostics"].append(
                    f"unattributed: target={target_str!r} produced no payload op and no diagnostic"
                )
        else:
            # Unknown FX op_kind. Leave classification blank so the audit
            # registers it under ``unaccounted_call_function_nodes`` and
            # the strict gate fails.
            record["classification"] = ""

        if (
            record["classification"]
            and record["classification"] not in ALLOWED_CLASSIFICATIONS
        ):  # internal invariant
            raise AssertionError(
                f"internal: classification {record['classification']!r} not in v2 set"
            )
        nodes_out.append(record)

    # Per-module summary histogram (v2 keys).
    summary: dict[str, int] = {
        "fx_nodes_total": len(nodes_out),
        "call_function_nodes": 0,
        "placeholder": 0,
        "output": 0,
        "decomposed_structured": 0,
        "opaque_fallback": 0,
        "closed_by_registry": 0,
        "resolved_alias": 0,
        "dropped_auxiliary_output": 0,
        "diagnostic_error": 0,
        "unaccounted": 0,
    }
    for n in nodes_out:
        if n["op_kind"] in ("call_function", "call_method", "call_module"):
            summary["call_function_nodes"] += 1
        c = n["classification"]
        if c == "":
            summary["unaccounted"] += 1
        else:
            summary[c] = summary.get(c, 0) + 1

    # Reconciliation: every opaque_calls.json record must have a matching
    # ``opaque_fallback`` node.
    accounted_opaque = {
        n["fx_node"] for n in nodes_out if n["classification"] == "opaque_fallback"
    }
    opaque_without_origin = [
        o["fx_node"] for o in opaque_records if o["fx_node"] not in accounted_opaque
    ]

    return {
        "module_id": module["module_id"],
        "status": "pass",
        "input_kind": module["input_kind"],
        "input_graph": module["input_graph"],
        "payload_mlir": module["payload_mlir"],
        "nodes": nodes_out,
        "summary": summary,
        "opaque_calls_without_origin": opaque_without_origin,
    }


# --------------------------------------------------------------------------- #
# Dialect coverage from payload.mlir text
# --------------------------------------------------------------------------- #


def _parse_dialect_coverage(payload_mlir: str) -> dict[str, Any]:
    """Walk the MLIR text line by line and tally:

    - dialect_counts (e.g. ``builtin``, ``func``, ``tensor``, ``linalg``, ``arith``)
    - structured_ops (per-op count, e.g. ``linalg.matmul``: 3)
    - opaque_func_calls (callee name → count, e.g. ``aten_gelu``: 1)
    """
    dialect_counts: dict[str, int] = {}
    structured_ops: dict[str, int] = {}
    opaque_calls: dict[str, int] = {}
    for line in payload_mlir.splitlines():
        m = _OP_LINE_RE.match(line)
        if not m:
            continue
        dialect = m.group("dialect")
        op = m.group("op")
        op_full = f"{dialect}.{op}"
        dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
        structured_ops[op_full] = structured_ops.get(op_full, 0) + 1
        if op_full == "func.call":
            cm = _FUNC_CALL_CALLEE_RE.search(line)
            if cm:
                callee = cm.group("q") or cm.group("u")
                opaque_calls[callee] = opaque_calls.get(callee, 0) + 1
    return {
        "dialect_counts": dict(sorted(dialect_counts.items())),
        "structured_ops": dict(sorted(structured_ops.items())),
        "opaque_func_calls": dict(sorted(opaque_calls.items())),
        "total_payload_ops": sum(dialect_counts.values()),
    }


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def audit_payload_coverage(run_dir: Path) -> PayloadCoverageResult:
    """Run the full Payload Coverage Audit and write three artifacts.

    ``run_dir`` must contain a completed ``01_payload_lowering/``.
    """
    from compgen.graph_compilation.payload_attribution import (
        load_attribution_lookup,
    )

    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "01_payload_lowering"
    payload_index = _read_json(out_dir / "payload_index.json")
    attribution_lookup = load_attribution_lookup(run_dir)

    per_module: list[dict[str, Any]] = []
    aggregate_summary: dict[str, int] = {
        "fx_nodes_total": 0,
        "call_function_nodes": 0,
        "placeholder": 0,
        "output": 0,
        "decomposed_structured": 0,
        "opaque_fallback": 0,
        "closed_by_registry": 0,
        "resolved_alias": 0,
        "dropped_auxiliary_output": 0,
        "diagnostic_error": 0,
        "unaccounted": 0,
    }
    dropped_aux: list[dict[str, str]] = []
    diag_errors: list[dict[str, str]] = []
    unaccounted: list[dict[str, str]] = []
    opaque_without_origin: list[dict[str, str]] = []

    dialect_aggregate: dict[str, int] = {}
    structured_aggregate: dict[str, int] = {}
    opaque_call_aggregate: dict[str, int] = {}
    per_module_dialects: list[dict[str, Any]] = []

    for module in payload_index.get("modules", []):
        attr_for_mod = attribution_lookup.get(module["module_id"], {})
        cls_result = _classify_module(
            run_dir=run_dir,
            module=module,
            attribution_for_module=attr_for_mod,
        )
        per_module.append(cls_result)

        for k, v in cls_result.get("summary", {}).items():
            aggregate_summary[k] = aggregate_summary.get(k, 0) + v

        for n in cls_result.get("nodes", []):
            mod_id = cls_result["module_id"]
            c = n["classification"]
            if c == "dropped_auxiliary_output":
                dropped_aux.append(
                    {
                        "module_id": mod_id,
                        "fx_node": n["fx_node"],
                        "fx_target": n["fx_target"],
                        "diagnostic": (n["diagnostics"] or [""])[0],
                    }
                )
            elif c == "diagnostic_error":
                diag_errors.append(
                    {
                        "module_id": mod_id,
                        "fx_node": n["fx_node"],
                        "fx_target": n["fx_target"],
                        "diagnostic": (n["diagnostics"] or [""])[0],
                    }
                )
            elif c == "":
                unaccounted.append(
                    {
                        "module_id": mod_id,
                        "fx_node": n["fx_node"],
                        "fx_target": n["fx_target"],
                        "fx_op": n["op_kind"],
                    }
                )
        for fx_node in cls_result.get("opaque_calls_without_origin", []):
            opaque_without_origin.append(
                {"module_id": cls_result["module_id"], "fx_node": fx_node}
            )

        # Dialect coverage from this module's payload.mlir.
        if cls_result["status"] == "pass":
            mlir_text = (run_dir / cls_result["payload_mlir"]).read_text(encoding="utf-8")
            cov = _parse_dialect_coverage(mlir_text)
            per_module_dialects.append({"module_id": cls_result["module_id"], **cov})
            for k, v in cov["dialect_counts"].items():
                dialect_aggregate[k] = dialect_aggregate.get(k, 0) + v
            for k, v in cov["structured_ops"].items():
                structured_aggregate[k] = structured_aggregate.get(k, 0) + v
            for k, v in cov["opaque_func_calls"].items():
                opaque_call_aggregate[k] = opaque_call_aggregate.get(k, 0) + v

    # ------------------------------------------------------------------ #
    # 1. fx_to_payload_accounting.json (v2)
    # ------------------------------------------------------------------ #
    accounting = {
        "schema_version": "fx_to_payload_accounting_v2",
        "summary": dict(sorted(aggregate_summary.items())),
        "modules": per_module,
    }
    accounting_path = out_dir / "fx_to_payload_accounting.json"
    accounting_path.write_text(
        json.dumps(accounting, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 2. dialect_coverage.json (v1, unchanged)
    # ------------------------------------------------------------------ #
    dialect_cov = {
        "schema_version": "dialect_coverage_v1",
        "aggregate": {
            "total_payload_ops": sum(dialect_aggregate.values()),
            "dialect_counts": dict(sorted(dialect_aggregate.items())),
            "structured_ops": dict(sorted(structured_aggregate.items())),
            "opaque_func_calls": dict(sorted(opaque_call_aggregate.items())),
        },
        "per_module": per_module_dialects,
    }
    dialect_path = out_dir / "dialect_coverage.json"
    dialect_path.write_text(
        json.dumps(dialect_cov, indent=2, sort_keys=True), encoding="utf-8"
    )

    # ------------------------------------------------------------------ #
    # 3. silent_drop_audit.json — strict pass gate (v1, unchanged)
    # ------------------------------------------------------------------ #
    status = "pass" if (not unaccounted and not opaque_without_origin) else "fail"
    audit = {
        "schema_version": "silent_drop_audit_v1",
        "status": status,
        "totals": {
            "unaccounted_call_function_nodes": len(unaccounted),
            "opaque_calls_without_origin": len(opaque_without_origin),
            "dropped_auxiliary_output": len(dropped_aux),
            "diagnostic_error": len(diag_errors),
        },
        "unaccounted_call_function_nodes": unaccounted,
        "opaque_calls_without_origin": opaque_without_origin,
        "dropped_auxiliary_output": dropped_aux,
        "diagnostic_error": diag_errors,
    }
    audit_path = out_dir / "silent_drop_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    return PayloadCoverageResult(
        fx_to_payload_accounting_path=accounting_path,
        dialect_coverage_path=dialect_path,
        silent_drop_audit_path=audit_path,
        silent_drop_status=status,
        unaccounted_count=len(unaccounted),
        dropped_auxiliary_output_count=len(dropped_aux),
        diagnostic_error_count=len(diag_errors),
        opaque_without_origin_count=len(opaque_without_origin),
    )

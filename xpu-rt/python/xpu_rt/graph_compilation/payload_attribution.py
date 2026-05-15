"""Structured Payload Attribution Hardening (Milestone 02.5, Option A).

For every FX call_function node lowered into Payload IR, record the
exact list of Payload ops produced by it. This closes the
attribution gap left by Milestones A+B, where ``decomposed_structured``
nodes had ``payload_ops = []``.

The strategy is **post-hoc, no FXImporter edit**:

- ``lower.py`` already records ``diagnostics.json`` per payload module
  with exact-count messages for every call_function node:

    * ``Opaque: <target> -> func.call @<callee>``                 — 1 op
    * ``Decomposed <target> -> N ops (regions: [...])``           — N ops
    * ``Inlined extension '<id>' for '<target>' (registry-driven)`` —
      followed by a sequence of synthetic ``ext_<hash>_*`` FX nodes
      each carrying its own ``Decomposed: ... -> M ops`` diagnostic;
      the original FX node's count is the sum.

- The FXImporter emits payload ops into ``func.func @forward`` in FX
  graph order. Walking diagnostics and forward-body ops in lockstep
  therefore deterministically partitions the payload ops between the
  FX nodes that produced them.

Outputs ``01_payload_lowering/payload_attribution.json`` (schema_version
``payload_attribution_v1``). Consumers: ``payload_coverage.py``
populates ``fx_to_payload_accounting.json`` per-node ``payload_ops``;
``region_map.py`` derives more accurate ``fx_nodes`` per region.

This module is **not** in the canonical compile path itself. It is a
post-pass over already-lowered artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Diagnostic-message parsing
# --------------------------------------------------------------------------- #

_OPAQUE_RE = re.compile(r"^Opaque: (?P<target>.+?) -> func\.call @(?P<callee>.+)$")
_DECOMPOSED_RE = re.compile(
    r"^Decomposed (?P<target>.+?) -> (?P<count>\d+) ops"
    r"(?:\s+\(regions: \[(?P<regions>[^\]]*)\])?"
)
_INLINED_RE = re.compile(
    r"^Inlined extension '(?P<extension_id>[^']+)' for '(?P<target>[^']+)' \(registry-driven\)"
)
_EXT_FX_NODE_RE = re.compile(r"^ext_([0-9a-f]+)_")


@dataclass
class _StreamEntry:
    """One contiguous block of payload ops attributed to a single FX node."""

    fx_node: str
    fx_target: str
    classification_hint: str  # "opaque" | "decomposed" | "inlined_extension"
    expected_op_count: int
    extension_id: str | None = None
    declared_regions: list[str] = field(default_factory=list)


def _build_attribution_stream(diagnostics: list[dict[str, Any]]) -> list[_StreamEntry]:
    """Walk per-module diagnostics in order, yield one entry per FX node.

    Inlined-extension blocks fold their child ``ext_<hash>_*`` decomposed
    diagnostics into the parent FX node's ``expected_op_count``.
    """
    out: list[_StreamEntry] = []
    i = 0
    while i < len(diagnostics):
        d = diagnostics[i]
        msg = d.get("message", "")
        fx = d.get("fx_node", "")
        if not isinstance(msg, str) or not isinstance(fx, str):
            i += 1
            continue

        m_op = _OPAQUE_RE.match(msg)
        if m_op:
            out.append(
                _StreamEntry(
                    fx_node=fx,
                    fx_target=m_op.group("target"),
                    classification_hint="opaque",
                    expected_op_count=1,
                )
            )
            i += 1
            continue

        m_dec = _DECOMPOSED_RE.match(msg)
        if m_dec:
            regions_raw = m_dec.group("regions") or ""
            regions = [
                r.strip().strip("'\"")
                for r in regions_raw.split(",")
                if r.strip()
            ]
            out.append(
                _StreamEntry(
                    fx_node=fx,
                    fx_target=m_dec.group("target"),
                    classification_hint="decomposed",
                    expected_op_count=int(m_dec.group("count")),
                    declared_regions=regions,
                )
            )
            i += 1
            continue

        m_inl = _INLINED_RE.match(msg)
        if m_inl:
            ext_id = m_inl.group("extension_id")
            target = m_inl.group("target")
            # Sum counts of immediately following ``Decomposed`` diagnostics
            # whose fx_node starts with ``ext_<hash>_``. Stop at the first
            # diagnostic that does not look like a child.
            j = i + 1
            total = 0
            while j < len(diagnostics):
                d2 = diagnostics[j]
                fx2 = d2.get("fx_node", "")
                msg2 = d2.get("message", "")
                if not isinstance(fx2, str) or not isinstance(msg2, str):
                    break
                if not _EXT_FX_NODE_RE.match(fx2):
                    break
                m_d2 = _DECOMPOSED_RE.match(msg2)
                m_o2 = _OPAQUE_RE.match(msg2)
                if m_d2:
                    total += int(m_d2.group("count"))
                elif m_o2:
                    total += 1
                else:
                    break
                j += 1
            out.append(
                _StreamEntry(
                    fx_node=fx,
                    fx_target=target,
                    classification_hint="inlined_extension",
                    expected_op_count=total,
                    extension_id=ext_id,
                )
            )
            i = j
            continue

        # Unknown / non-attribution diagnostic (e.g. plain warning) — skip.
        i += 1

    return out


# --------------------------------------------------------------------------- #
# Payload-MLIR forward-body walker
# --------------------------------------------------------------------------- #

_OP_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:(?P<results>%[A-Za-z0-9_]+(?:\s*,\s*%[A-Za-z0-9_]+)*)\s*=\s*)?
    (?P<dialect>[a-z][a-z0-9_]*)\.(?P<op>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)
_FUNC_FORWARD_RE = re.compile(r"^\s*func\.func\s+@forward\s*\(")
_FUNC_RETURN_RE = re.compile(r"^\s*func\.return\s")
_REGION_ID_ATTR_RE = re.compile(r'compgen\.region_id\s*=\s*"(?P<rid>[^"]+)"')
_DISPATCH_ID_ATTR_RE = re.compile(r'compgen\.dispatch_id\s*=\s*"(?P<did>[^"]+)"')
_FUNC_CALL_CALLEE_RE = re.compile(r'func\.call\s+@(?:"(?P<q>[^"]+)"|(?P<u>[^"\s(]+))')
_OPERAND_RE = re.compile(r"%([A-Za-z0-9_]+)")


@dataclass
class _ForwardOp:
    line_index: int
    op_name: str
    region_id: str | None
    dispatch_id: str | None
    callee: str | None
    results: list[str]
    operands: list[str]


def _parse_forward_ops(mlir_text: str) -> list[_ForwardOp]:
    """Return ops emitted *inside* ``func.func @forward`` (skip top-level
    ``func.func private`` declarations)."""
    ops: list[_ForwardOp] = []
    in_forward = False
    for i, line in enumerate(mlir_text.splitlines()):
        if not in_forward:
            if _FUNC_FORWARD_RE.match(line):
                in_forward = True
            continue
        if _FUNC_RETURN_RE.match(line):
            in_forward = False
            continue
        if not line.strip() or line.strip() == "}":
            continue
        m = _OP_LINE_RE.match(line)
        if not m:
            continue
        op_name = f"{m.group('dialect')}.{m.group('op')}"
        if op_name in ("func.return", "builtin.module"):
            continue
        results: list[str] = []
        if m.group("results"):
            for tok in m.group("results").split(","):
                tok = tok.strip()
                if tok.startswith("%"):
                    results.append(tok[1:])
        rid_m = _REGION_ID_ATTR_RE.search(line)
        did_m = _DISPATCH_ID_ATTR_RE.search(line)
        callee: str | None = None
        if op_name == "func.call":
            cm = _FUNC_CALL_CALLEE_RE.search(line)
            if cm:
                callee = cm.group("q") or cm.group("u")
        rest = line[m.end():]
        operands: list[str] = []
        for om in _OPERAND_RE.finditer(rest):
            name = om.group(1)
            if name in results:
                continue
            if name not in operands:
                operands.append(name)
        ops.append(
            _ForwardOp(
                line_index=i,
                op_name=op_name,
                region_id=rid_m.group("rid") if rid_m else None,
                dispatch_id=did_m.group("did") if did_m else None,
                callee=callee,
                results=results,
                operands=operands,
            )
        )
    return ops


# --------------------------------------------------------------------------- #
# Lockstep attribution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PayloadAttributionResult:
    path: Path
    total_attributed_ops: int
    total_unattributed_ops: int
    modules_with_count_mismatch: list[str]


def _attribute_module(
    *,
    module_id: str,
    payload_ref: str,
    diagnostics: list[dict[str, Any]],
    forward_ops: list[_ForwardOp],
) -> dict[str, Any]:
    """Walk diagnostic stream + forward ops in lockstep; return per-module
    attribution record."""
    stream = _build_attribution_stream(diagnostics)
    attributions: list[dict[str, Any]] = []
    op_idx = 0
    consumed = 0
    count_mismatch = False

    for entry in stream:
        claimed: list[dict[str, Any]] = []
        # Bound the claim by the available remaining ops; mark a mismatch
        # but never index past the end.
        n = entry.expected_op_count
        if op_idx + n > len(forward_ops):
            count_mismatch = True
            n = max(len(forward_ops) - op_idx, 0)
        for _ in range(n):
            op = forward_ops[op_idx]
            claimed.append(
                {
                    "op_name": op.op_name,
                    "region_id": op.region_id,
                    "dispatch_id": op.dispatch_id,
                    "callee": op.callee,
                    "results": op.results,
                    "operands": op.operands,
                    "line_index": op.line_index,
                    "payload_ref": payload_ref,
                }
            )
            op_idx += 1
        attributions.append(
            {
                "fx_node": entry.fx_node,
                "fx_target": entry.fx_target,
                "classification_hint": entry.classification_hint,
                "extension_id": entry.extension_id,
                "expected_op_count": entry.expected_op_count,
                "actual_op_count": len(claimed),
                "declared_regions": entry.declared_regions,
                "payload_ops": claimed,
            }
        )
        consumed += len(claimed)

    # Any remaining ops are unattributed — record them honestly so the
    # caller can investigate, but never silently drop.
    unattributed: list[dict[str, Any]] = []
    while op_idx < len(forward_ops):
        op = forward_ops[op_idx]
        unattributed.append(
            {
                "op_name": op.op_name,
                "region_id": op.region_id,
                "dispatch_id": op.dispatch_id,
                "callee": op.callee,
                "results": op.results,
                "operands": op.operands,
                "line_index": op.line_index,
                "payload_ref": payload_ref,
            }
        )
        op_idx += 1
        count_mismatch = True

    return {
        "module_id": module_id,
        "payload_ref": payload_ref,
        "totals": {
            "diagnostic_entries": len(stream),
            "forward_ops": len(forward_ops),
            "attributed_ops": consumed,
            "unattributed_ops": len(unattributed),
            "count_mismatch": count_mismatch,
        },
        "fx_attributions": attributions,
        "unattributed_ops": unattributed,
    }


def build_payload_attribution(run_dir: Path) -> PayloadAttributionResult:
    """Build ``01_payload_lowering/payload_attribution.json`` from the
    already-emitted lowering artifacts. Read-only against payload IR."""
    run_dir = Path(run_dir).resolve()
    pl_dir = run_dir / "01_payload_lowering"
    payload_index = json.loads(
        (pl_dir / "payload_index.json").read_text(encoding="utf-8")
    )

    per_module: list[dict[str, Any]] = []
    total_attributed = 0
    total_unattributed = 0
    mismatch_modules: list[str] = []

    for mod in payload_index.get("modules", []):
        if mod.get("status") == "skipped":
            continue
        module_id = mod["module_id"]
        payload_ref = mod["payload_mlir"]
        module_dir = run_dir / Path(mod["lowering_report"]).parent

        diagnostics_path = module_dir / "diagnostics.json"
        diagnostics: list[dict[str, Any]] = []
        if diagnostics_path.exists():
            doc = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            diagnostics = doc.get("diagnostics", [])

        mlir_text = (run_dir / payload_ref).read_text(encoding="utf-8")
        forward_ops = _parse_forward_ops(mlir_text)

        record = _attribute_module(
            module_id=module_id,
            payload_ref=payload_ref,
            diagnostics=diagnostics,
            forward_ops=forward_ops,
        )
        per_module.append(record)
        total_attributed += record["totals"]["attributed_ops"]
        total_unattributed += record["totals"]["unattributed_ops"]
        if record["totals"]["count_mismatch"]:
            mismatch_modules.append(module_id)

    obj = {
        "schema_version": "payload_attribution_v1",
        "totals": {
            "modules": len(per_module),
            "attributed_ops": total_attributed,
            "unattributed_ops": total_unattributed,
            "modules_with_count_mismatch": mismatch_modules,
        },
        "modules": per_module,
    }
    out_path = pl_dir / "payload_attribution.json"
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

    return PayloadAttributionResult(
        path=out_path,
        total_attributed_ops=total_attributed,
        total_unattributed_ops=total_unattributed,
        modules_with_count_mismatch=mismatch_modules,
    )


# --------------------------------------------------------------------------- #
# Helper consumed by payload_coverage.py
# --------------------------------------------------------------------------- #


def load_attribution_lookup(
    run_dir: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return a nested lookup: ``{module_id: {fx_node: [payload_op_record, ...]}}``.

    Returns ``{}`` if no payload_attribution.json has been emitted yet.
    Consumers should treat absence as "no attribution data" rather than
    fail; they may still rely on opaque_calls.json fallback.
    """
    run_dir = Path(run_dir).resolve()
    path = run_dir / "01_payload_lowering" / "payload_attribution.json"
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for mod in obj.get("modules", []):
        per_node: dict[str, list[dict[str, Any]]] = {}
        for a in mod.get("fx_attributions", []):
            per_node[a["fx_node"]] = list(a.get("payload_ops", []))
        out[mod["module_id"]] = per_node
    return out

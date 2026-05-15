"""Gap Discovery: deterministic analysis of Payload Lowering output.

Consumes ``01_payload_lowering/`` artifacts from a previous Payload
Lowering run and emits, under ``04_gap_discovery/``:

- ``gap_action_queue.json`` — typed gap records with ``allowed_actions``
  and ``required_evidence``. This is the **input to the agentic
  gap-closure loop** (next milestone). Today it's produced
  deterministically from ``unsupported_ops.json`` plus the FX graph.
- ``gap_analysis.json`` — descriptive: per-region classification +
  histogram + critical-path summary.
- ``dossier.json`` — graph-level: per-module node counts, region
  metadata, critical-path node names.
- ``gap_discovery_summary.json`` — per-stage report (status, llm_calls=0).

No LLM. ``canonical_pass_trace`` semantics carry over from Payload
Lowering: this is deterministic infrastructure, not the agent.

Critical-path heuristic
-----------------------

A node is on the **critical path** of its module iff *removing it from
the FX graph disconnects every placeholder→output route*. That is,
no parallel branch routes around it. In practice for feed-forward
networks every call_function node is critical; for branchy networks
(skip connections, dual-output ops) some nodes are not.

This is computed per-module by BFS-from-placeholders that pretends the
candidate node doesn't exist; if no output is reachable, the node is
critical. The per-module critical-path node list lands in
``dossier.json::modules[*].critical_path``; per-gap severity is set to
``critical_path`` when the gap's fx_node is on its module's critical
path, otherwise ``coverage_gap``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from compgen.graph_compilation.artifacts import ArtifactRef, StageRecord
from compgen.graph_compilation.hashing import sha256_file, sha256_tree

# Action enum (deterministically synthesized from gap_kind).
_ACTIONS_FOR_KIND: dict[str, list[str]] = {
    "unsupported_op": [
        "decompose_to_supported_ops",
        "create_payload_lowering_extension",
        "create_kernel_contract",
        "keep_as_fallback",
    ],
    "unsupported_dtype": [
        "create_dequantize_to_supported_format_fallback",
        "create_quant_format_adapter",
        "keep_as_fallback",
    ],
    "unsupported_quant_format": [
        "create_quant_format_adapter",
        "create_dequantize_to_supported_format_fallback",
        "keep_as_fallback",
    ],
}

# Evidence requirements per gap_kind. The validator REJECTS gaps that don't
# carry the required-evidence list for their kind.
_EVIDENCE_FOR_KIND: dict[str, list[str]] = {
    "unsupported_op": [
        "reference_semantics",
        "input_output_shapes",
        "dtype_policy",
        "differential_tests",
    ],
    "unsupported_dtype": [
        "dequant_reference",
        "rounding_policy",
        "error_tolerance",
    ],
    "unsupported_quant_format": [
        "quant_format_spec",
        "dequant_reference",
        "rounding_policy",
        "scale_layout",
        "error_tolerance",
    ],
}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class RegionInfo:
    region_id: str
    fx_node: str
    fx_target: str
    module_id: str
    input_kind: str
    classification: str  # "native" | "library" | "need_kernel" | "unsupported"
    reason: str
    shape_signature: dict[str, Any]
    dtype_signature: dict[str, Any]
    on_critical_path: bool


@dataclass
class GapRecord:
    gap_id: str
    region_id: str
    module_id: str
    gap_kind: str
    semantic_name: str           # exactly the FX target — human-readable identity
    slug: str                    # filesystem-safe form of the target
    extension_id: str            # canonical workspace ID (matches Extension Closure)
    suggested_extension_path: str
    target_id: str
    severity: str  # "critical_path" | "coverage_gap" | "performance_blocker" | "noncritical"
    allowed_actions: list[str]
    required_evidence: list[str]
    fx_target: str
    fx_node: str
    shape_signature: dict[str, Any]
    dtype_signature: dict[str, Any]
    source: dict[str, str]       # {unsupported_op_id, payload_mlir, lowering_report}
    source_artifacts: dict[str, str]  # legacy field; kept for the 6 closure consistency checks
    payload_ref: dict[str, str]
    reason: str
    # Severity-audit fields, populated in the second pass once the run-wide
    # total cost is known. ``critical_path_member`` carries the raw graph
    # property; ``severity`` is the calibrated bucket derived from it
    # plus ``cost_fraction_estimate``.
    critical_path_member: bool = False
    severity_score: float = 0.0
    severity_reasons: list[str] = field(default_factory=list)
    cost_fraction_estimate: float = 0.0
    op_family: str = "unknown"
    raw_cost: float = 0.0
    # Sequential priority within the run: 1 = highest. Critical-path gaps
    # come first, then performance_blocker, then coverage_gap, then
    # noncritical; ties broken by severity_score desc, then cost desc.
    closure_priority: int = 0
    recommended_next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "region_id": self.region_id,
            "module_id": self.module_id,
            "gap_kind": self.gap_kind,
            "semantic_name": self.semantic_name,
            "slug": self.slug,
            "extension_id": self.extension_id,
            "suggested_extension_path": self.suggested_extension_path,
            "target_id": self.target_id,
            "severity": self.severity,
            "severity_score": self.severity_score,
            "severity_reasons": list(self.severity_reasons),
            "cost_fraction_estimate": self.cost_fraction_estimate,
            "critical_path_member": self.critical_path_member,
            "op_family": self.op_family,
            "allowed_actions": list(self.allowed_actions),
            "required_evidence": list(self.required_evidence),
            "fx_target": self.fx_target,
            "fx_node": self.fx_node,
            "shape_signature": self.shape_signature,
            "dtype_signature": self.dtype_signature,
            "source": dict(self.source),
            "source_artifacts": dict(self.source_artifacts),
            "payload_ref": dict(self.payload_ref),
            "reason": self.reason,
            "closure_priority": self.closure_priority,
            "recommended_next_action": self.recommended_next_action,
        }


@dataclass
class ModuleAnalysis:
    module_id: str
    input_kind: str
    graph_path: str
    num_fx_nodes: int
    num_call_function: int
    num_placeholders: int
    num_outputs: int
    critical_path_nodes: list[str] = field(default_factory=list)
    regions: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# FX graph access (Dynamo partition vs ExportedProgram)
# --------------------------------------------------------------------------- #


def _load_fx_graph(input_kind: str, run_dir: Path, graph_path_rel: str) -> Any:
    """Return an FX ``Graph`` object from the on-disk artifact.

    Dynamo partition: ``torch.load(graphmodule.pt)`` → ``GraphModule.graph``.
    ExportedProgram: ``torch.export.load(.pt2)`` → ``ep.graph``.
    """
    abs_path = run_dir / graph_path_rel
    if input_kind == "torch_dynamo_partition":
        gm = torch.load(abs_path, weights_only=False)
        return gm.graph
    if input_kind == "exported_program":
        ep = torch.export.load(str(abs_path))
        return ep.graph
    raise ValueError(f"unknown input_kind: {input_kind!r}")


def _node_predecessors(node: Any) -> list[Any]:
    """Return every FX node that feeds a value into ``node`` via args or kwargs."""
    preds: list[Any] = []
    for a in node.args:
        if hasattr(a, "op"):
            preds.append(a)
        elif isinstance(a, (list, tuple)):
            for sub in a:
                if hasattr(sub, "op"):
                    preds.append(sub)
    for v in node.kwargs.values():
        if hasattr(v, "op"):
            preds.append(v)
    return preds


def _is_critical(graph: Any, node_name: str) -> bool:
    """``True`` iff removing the named node disconnects all placeholder→output paths.

    AND-semantics dataflow propagation: a node ``N`` is *computable*
    iff every predecessor that supplies an operand value is also
    computable (or ``N`` is a placeholder). We pretend ``target``
    does not exist and check whether the output node is still
    computable. If not, ``target`` is on the critical path.

    Why AND, not OR: FX execution doesn't run a node until *all*
    operands are available. A simple forward BFS (OR-reachability)
    would incorrectly mark ``linear_1`` as reachable in TinyMLP after
    blocking ``linear``, just because the bias/weight placeholders
    feed ``linear_1`` directly — but ``linear_1`` also needs the
    output of ``gelu``, which depends on ``linear``. AND-semantics
    catches this.

    Placeholders/outputs themselves are never reported as critical
    (they aren't candidates for gap remediation).
    """
    target = next((n for n in graph.nodes if n.name == node_name), None)
    if target is None or target.op in ("placeholder", "output"):
        return False
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    outputs = [n for n in graph.nodes if n.op == "output"]
    if not placeholders or not outputs:
        return False

    computable: set[Any] = set(placeholders)
    # Fixed-point AND propagation. ``target`` is excluded forever.
    changed = True
    while changed:
        changed = False
        for n in graph.nodes:
            if n is target or n in computable or n.op in ("placeholder", "output"):
                continue
            preds = _node_predecessors(n)
            # A node with zero data-flow predecessors (e.g. constant-only)
            # is trivially computable.
            if not preds or all(p in computable for p in preds):
                computable.add(n)
                changed = True

    # ``target`` is on the critical path iff *every* output value is
    # broken when target is removed. If even one output value still
    # has a fully computable producer, target is on a *branch* path —
    # not the critical path. This matches the GAP-00 definition of
    # "every directed path from input to output passes through it".
    for out_node in outputs:
        out_args = out_node.args[0] if out_node.args else ()
        if not isinstance(out_args, (tuple, list)):
            out_args = (out_args,)
        for out_arg in out_args:
            if not hasattr(out_arg, "op"):
                continue  # plain constant, contributes no path
            if out_arg in computable:
                return False  # this output bypasses target → not critical
    return True


def _critical_path_node_names(graph: Any) -> list[str]:
    """List of call_function node names that are on the module's critical path."""
    return [n.name for n in graph.nodes if n.op == "call_function" and _is_critical(graph, n.name)]


# --------------------------------------------------------------------------- #
# Classification — for now everything opaque is "need_kernel"
# --------------------------------------------------------------------------- #


def _classify_op(reason: str) -> str:
    if reason == "no_decomposition_rule":
        return "need_kernel"
    return "unsupported"


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #


def run_gap_discovery(
    run_dir: Path,
    *,
    target_id: str,
    model_id: str,
    extension_registry: Path | None = None,
) -> StageRecord:
    started_at = _utcnow()
    run_dir = Path(run_dir).resolve()
    pl_dir = run_dir / "01_payload_lowering"
    if not pl_dir.is_dir():
        raise FileNotFoundError(f"01_payload_lowering/ missing: {pl_dir}")

    out_dir = run_dir / "04_gap_discovery"
    out_dir.mkdir(parents=True, exist_ok=True)

    unsupported_path = pl_dir / "unsupported_ops.json"
    payload_index_path = pl_dir / "payload_index.json"
    summary_path = pl_dir / "lowering_summary.json"
    if not unsupported_path.exists() or not payload_index_path.exists():
        raise FileNotFoundError(
            f"missing payload_lowering inputs under {pl_dir}: unsupported_ops.json or payload_index.json"
        )

    unsupported = json.loads(unsupported_path.read_text(encoding="utf-8"))
    payload_index = json.loads(payload_index_path.read_text(encoding="utf-8"))

    # Index the payload modules by id for cross-reference.
    modules_by_id = {m["module_id"]: m for m in payload_index.get("modules", [])}

    # ------------------------------------------------------------------ #
    # 1. Per-module FX graph load + critical-path computation
    # ------------------------------------------------------------------ #
    module_analyses: dict[str, ModuleAnalysis] = {}
    fx_graph_cache: dict[str, Any] = {}

    for m in payload_index.get("modules", []):
        if m.get("status") == "skipped":
            continue
        try:
            fx_graph = _load_fx_graph(m["input_kind"], run_dir, m["input_graph"])
        except Exception as exc:
            # Honest failure recording — don't make up regions.
            module_analyses[m["module_id"]] = ModuleAnalysis(
                module_id=m["module_id"],
                input_kind=m["input_kind"],
                graph_path=m["input_graph"],
                num_fx_nodes=0,
                num_call_function=0,
                num_placeholders=0,
                num_outputs=0,
                critical_path_nodes=[],
                regions=[{"error": f"{type(exc).__name__}: {exc}"}],
            )
            continue
        fx_graph_cache[m["module_id"]] = fx_graph
        nodes = list(fx_graph.nodes)
        critical = _critical_path_node_names(fx_graph)
        module_analyses[m["module_id"]] = ModuleAnalysis(
            module_id=m["module_id"],
            input_kind=m["input_kind"],
            graph_path=m["input_graph"],
            num_fx_nodes=len(nodes),
            num_call_function=sum(1 for n in nodes if n.op == "call_function"),
            num_placeholders=sum(1 for n in nodes if n.op == "placeholder"),
            num_outputs=sum(1 for n in nodes if n.op == "output"),
            critical_path_nodes=critical,
            regions=[],  # filled after gap construction
        )

    # ------------------------------------------------------------------ #
    # 1.5. Load the extension registry, if any. Closed targets are
    #      skipped at gap-construction time and tallied in the report.
    # ------------------------------------------------------------------ #
    from compgen.graph_compilation.extension_registry import load_registry

    registry_obj = load_registry(extension_registry) if extension_registry else None
    registry_path_str = str(extension_registry.resolve()) if extension_registry else None
    closed_targets: list[dict[str, Any]] = []  # for the report

    # ------------------------------------------------------------------ #
    # 2. Build GapRecords from unsupported_ops + critical-path bit
    # ------------------------------------------------------------------ #
    gaps: list[GapRecord] = []
    region_infos: list[RegionInfo] = []

    next_gap_idx = 0
    for u in unsupported.get("unsupported_ops", []):
        module_id = u["module_id"]
        fx_node_name = u["fx_node"]
        fx_target = u["fx_target"]
        reason = u.get("reason", "no_decomposition_rule")

        # Skip targets with a registered+verified extension.
        if registry_obj is not None and registry_obj.has("unsupported_op", fx_target):
            entry = registry_obj.lookup("unsupported_op", fx_target)
            closed_targets.append(
                {
                    "module_id": module_id,
                    "fx_node": fx_node_name,
                    "fx_target": fx_target,
                    "extension_id": entry.extension_id if entry else None,
                    "extension_path": entry.extension_path if entry else None,
                }
            )
            continue

        gap_id = f"gap_{next_gap_idx:04d}"
        next_gap_idx += 1
        # region_id = original unsupported_id (or fall back to fx_node for stable ID).
        region_id = u.get("unsupported_id", f"{module_id}::{fx_node_name}")

        on_critical = (
            module_id in module_analyses
            and fx_node_name in module_analyses[module_id].critical_path_nodes
        )
        # Severity is assigned in the second pass once the run-wide total
        # cost is known. Use ``coverage_gap`` as a placeholder so the field
        # is always populated even if the second pass somehow skipped it.
        severity = "coverage_gap"

        gap_kind = "unsupported_op"  # everything in unsupported_ops.json today
        actions = list(_ACTIONS_FOR_KIND[gap_kind])
        evidence = list(_EVIDENCE_FOR_KIND[gap_kind])

        # Resolve the source payload.mlir for this module.
        payload_module = modules_by_id.get(module_id, {})
        payload_mlir_rel = u.get("payload_ref", {}).get("payload_mlir") or payload_module.get("payload_mlir", "")
        lowering_report_rel = payload_module.get("lowering_report", "")

        # Compute the canonical extension identity now so:
        # - the gap_action_queue carries the same extension_id Extension
        #   Closure will materialize
        # - same op + same target + same shape → same id (stable across reruns)
        # - different shape OR different target → different id
        from compgen.graph_compilation.gap_naming import (
            extension_id as _gap_ext_id,
        )
        from compgen.graph_compilation.gap_naming import (
            slug_for_target,
            suggested_extension_path,
        )

        slug = slug_for_target(fx_target)
        ext_id = _gap_ext_id(
            gap_kind=gap_kind,
            fx_target=fx_target,
            target_id=target_id,
            shape_signature=u.get("shape_signature", {}),
            dtype_signature=u.get("dtype_signature", {}),
        )
        sugg_path = suggested_extension_path(
            gap_kind=gap_kind,
            fx_target=fx_target,
            target_id=target_id,
            shape_signature=u.get("shape_signature", {}),
            dtype_signature=u.get("dtype_signature", {}),
        )

        from compgen.graph_compilation.severity import estimate_raw_cost

        raw_cost, family = estimate_raw_cost(fx_target, u.get("shape_signature", {}))

        gap = GapRecord(
            gap_id=gap_id,
            region_id=region_id,
            module_id=module_id,
            gap_kind=gap_kind,
            semantic_name=fx_target,
            slug=slug,
            extension_id=ext_id,
            suggested_extension_path=sugg_path,
            target_id=target_id,
            severity=severity,
            allowed_actions=actions,
            required_evidence=evidence,
            fx_target=fx_target,
            fx_node=fx_node_name,
            shape_signature=u.get("shape_signature", {}),
            dtype_signature=u.get("dtype_signature", {}),
            source={
                "unsupported_op_id": u.get("unsupported_id", region_id),
                "payload_mlir": payload_mlir_rel,
                "lowering_report": lowering_report_rel,
            },
            source_artifacts={
                "payload_mlir": payload_mlir_rel,
                "gap_analysis": "04_gap_discovery/gap_analysis.json",
                "dossier": "04_gap_discovery/dossier.json",
            },
            payload_ref={
                "payload_mlir": payload_mlir_rel,
                "callee": u.get("payload_ref", {}).get("callee", u.get("callee", "")),
            },
            reason=reason,
            critical_path_member=on_critical,
            raw_cost=raw_cost,
            op_family=family,
        )
        gaps.append(gap)

        region_infos.append(
            RegionInfo(
                region_id=region_id,
                fx_node=fx_node_name,
                fx_target=fx_target,
                module_id=module_id,
                input_kind=payload_module.get("input_kind", "unknown"),
                classification=_classify_op(reason),
                reason=reason,
                shape_signature=u.get("shape_signature", {}),
                dtype_signature=u.get("dtype_signature", {}),
                on_critical_path=on_critical,
            )
        )

    # ------------------------------------------------------------------ #
    # 3. Detect unsupported_dtype gaps (any non-fp32 dtype in any signature
    #    that isn't already covered by unsupported_op above).
    # ------------------------------------------------------------------ #
    # Only flag *floating-point* dtypes other than fp32 — int64 indices into
    # ``aten.embedding`` are universally supported and produce noisy
    # false-positive gaps if treated like fp16/bf16/fp8/fp4.
    fp32 = {"torch.float32", "torch.float", ""}  # blank means missing meta — ignore
    flagged_floats = {
        "torch.float16", "torch.bfloat16", "torch.float64",
        "torch.float8_e4m3fn", "torch.float8_e5m2",
        "torch.float8_e4m3fnuz", "torch.float8_e5m2fnuz",
    }
    seen_dtype_keys: set[str] = set()
    for u in unsupported.get("unsupported_ops", []):
        # Skip closed targets — their dtype gaps are subsumed by the closure.
        if registry_obj is not None and registry_obj.has("unsupported_op", u.get("fx_target", "")):
            continue
        dtypes = u.get("dtype_signature", {}).get("inputs", []) + u.get("dtype_signature", {}).get(
            "outputs", []
        )
        for dt in dtypes:
            if dt in flagged_floats and dt not in seen_dtype_keys:
                seen_dtype_keys.add(dt)
                # Synthesize one summary gap per non-fp32 dtype.
                from compgen.graph_compilation.gap_naming import (
                    extension_id as _gap_ext_id,
                )
                from compgen.graph_compilation.gap_naming import (
                    slug_for_target,
                    suggested_extension_path,
                )

                idx = len(gaps)
                gap_id = f"gap_{idx:04d}"
                payload_module = modules_by_id.get(u["module_id"], {})
                payload_mlir_rel = u.get("payload_ref", {}).get(
                    "payload_mlir", payload_module.get("payload_mlir", "")
                )
                lowering_report_rel = payload_module.get("lowering_report", "")
                semantic_name = f"dtype:{dt}"
                slug = slug_for_target(semantic_name)
                ext_id = _gap_ext_id(
                    gap_kind="unsupported_dtype",
                    fx_target=semantic_name,
                    target_id=target_id,
                    shape_signature=u.get("shape_signature", {}),
                    dtype_signature={"dtype": dt},
                )
                sugg_path = suggested_extension_path(
                    gap_kind="unsupported_dtype",
                    fx_target=semantic_name,
                    target_id=target_id,
                    shape_signature=u.get("shape_signature", {}),
                    dtype_signature={"dtype": dt},
                )
                from compgen.graph_compilation.severity import estimate_raw_cost

                raw_cost, family = estimate_raw_cost(
                    u["fx_target"], u.get("shape_signature", {})
                )
                gaps.append(
                    GapRecord(
                        gap_id=gap_id,
                        region_id=f"dtype::{dt}",
                        module_id=u["module_id"],
                        gap_kind="unsupported_dtype",
                        semantic_name=semantic_name,
                        slug=slug,
                        extension_id=ext_id,
                        suggested_extension_path=sugg_path,
                        target_id=target_id,
                        severity="performance_blocker",  # second-pass override
                        allowed_actions=list(_ACTIONS_FOR_KIND["unsupported_dtype"]),
                        required_evidence=list(_EVIDENCE_FOR_KIND["unsupported_dtype"]),
                        fx_target=u["fx_target"],
                        fx_node=u["fx_node"],
                        shape_signature=u.get("shape_signature", {}),
                        dtype_signature=u.get("dtype_signature", {}),
                        source={
                            "unsupported_op_id": u.get("unsupported_id", ""),
                            "payload_mlir": payload_mlir_rel,
                            "lowering_report": lowering_report_rel,
                        },
                        source_artifacts={
                            "payload_mlir": payload_mlir_rel,
                            "gap_analysis": "04_gap_discovery/gap_analysis.json",
                            "dossier": "04_gap_discovery/dossier.json",
                        },
                        payload_ref={
                            "payload_mlir": payload_mlir_rel,
                            "callee": u.get("payload_ref", {}).get("callee", u.get("callee", "")),
                        },
                        reason=f"non_fp32_dtype:{dt}",
                        critical_path_member=False,
                        raw_cost=raw_cost,
                        op_family=family,
                    )
                )

    # ------------------------------------------------------------------ #
    # 3.5. Second pass: calibrated severity scoring.
    #      Now that every gap has a raw_cost, normalise across the run
    #      to compute ``cost_fraction_estimate`` per gap, then bucket
    #      into critical_path / performance_blocker / coverage_gap /
    #      noncritical via the severity classifier.
    # ------------------------------------------------------------------ #
    from compgen.graph_compilation.severity import (
        THRESHOLD_HIGH,
        THRESHOLD_LOW,
        THRESHOLD_MED,
    )
    from compgen.graph_compilation.severity import (
        classify as _classify_severity,
    )

    total_cost = sum(g.raw_cost for g in gaps) or 1.0
    severity_audit_entries: list[dict[str, Any]] = []
    severity_histogram: dict[str, int] = {
        "critical_path": 0,
        "performance_blocker": 0,
        "coverage_gap": 0,
        "noncritical": 0,
    }
    for g in gaps:
        cost_frac = g.raw_cost / total_cost
        # ``unsupported_dtype`` gaps are dtype-policy holes, not graph
        # nodes — they don't have a critical-path bit, but they always
        # block downstream lowering until a quant adapter is present.
        # Blocks-lowering is always true for the kinds we currently emit.
        verdict = _classify_severity(
            on_critical_path=g.critical_path_member,
            cost_fraction=cost_frac,
            family=g.op_family,
            blocks_lowering=True,
        )
        # ``unsupported_dtype`` gaps default to performance_blocker
        # regardless of cost — a non-fp32 dtype that the compiler
        # didn't recognise is by definition a performance blocker, not
        # a coverage hole.
        if g.gap_kind == "unsupported_dtype":
            g.severity = "performance_blocker"
            g.severity_reasons = ["dtype_format_unsupported", *verdict.reasons]
        else:
            g.severity = verdict.bucket
            g.severity_reasons = list(verdict.reasons)
        g.severity_score = verdict.score
        g.cost_fraction_estimate = round(cost_frac, 4)
        severity_histogram[g.severity] = severity_histogram.get(g.severity, 0) + 1
        severity_audit_entries.append(
            {
                "gap_id": g.gap_id,
                "fx_target": g.fx_target,
                "severity": g.severity,
                "severity_score": g.severity_score,
                "severity_reasons": list(g.severity_reasons),
                "cost_fraction_estimate": g.cost_fraction_estimate,
                "critical_path_member": g.critical_path_member,
                "op_family": g.op_family,
                "raw_cost": round(g.raw_cost, 4),
            }
        )

    # Compute closure_priority + recommended_next_action.
    _BUCKET_RANK = {
        "critical_path": 0,
        "performance_blocker": 1,
        "coverage_gap": 2,
        "noncritical": 3,
    }

    def _ranked_key(g: GapRecord) -> tuple[int, float, float, str]:
        return (
            _BUCKET_RANK.get(g.severity, 4),
            -g.severity_score,
            -g.cost_fraction_estimate,
            g.gap_id,
        )

    def _recommended_action(g: GapRecord) -> str:
        # Noncritical: always defer. Otherwise pick the first non-fallback
        # action; if the gap is view-shaped (op_family=view) and noncritical
        # the deterministic fallback is appropriate.
        if g.severity == "noncritical":
            return "keep_as_fallback"
        for a in g.allowed_actions:
            if a != "keep_as_fallback":
                return a
        return "keep_as_fallback"

    sorted_gaps = sorted(gaps, key=_ranked_key)
    for rank, g in enumerate(sorted_gaps, start=1):
        g.closure_priority = rank
        g.recommended_next_action = _recommended_action(g)
    # Reflect the closure_priority back into the severity-audit entries.
    priority_by_id = {g.gap_id: g.closure_priority for g in gaps}
    rec_by_id = {g.gap_id: g.recommended_next_action for g in gaps}
    for entry in severity_audit_entries:
        entry["closure_priority"] = priority_by_id[entry["gap_id"]]
        entry["recommended_next_action"] = rec_by_id[entry["gap_id"]]

    severity_audit_obj = {
        "schema_version": "gap_severity_audit_v1",
        "model_id": model_id,
        "target_id": target_id,
        "thresholds": {
            "high": THRESHOLD_HIGH,
            "medium": THRESHOLD_MED,
            "low": THRESHOLD_LOW,
        },
        "policy": {
            "critical_path_requires": [
                "critical_path_member",
                f"cost_fraction >= {THRESHOLD_HIGH} OR (critical_path_member AND cost_fraction >= {THRESHOLD_MED})",
            ],
            "performance_blocker_requires": [
                f"cost_fraction >= {THRESHOLD_HIGH}",
                "off_critical_path",
            ],
            "coverage_gap_requires": [
                "actionable but cost below high threshold",
                f"cost_fraction in [{THRESHOLD_LOW}, {THRESHOLD_HIGH})",
            ],
            "noncritical_requires": [
                f"cost_fraction < {THRESHOLD_LOW} OR op_family == view",
            ],
            "ordering": "bucket then severity_score then cost_fraction (desc)",
            "low_cost_fallback_reason_when": (
                f"cost_fraction < {THRESHOLD_LOW} OR op_family == view"
            ),
        },
        "total_raw_cost": round(total_cost, 4),
        "histogram": severity_histogram,
        "gap_severity": severity_audit_entries,
    }
    severity_audit_path = out_dir / "severity_audit.json"
    severity_audit_path.write_text(
        json.dumps(severity_audit_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    # gap_priority_plan.json — agent-facing ordered worklist.
    plan_obj = {
        "schema_version": "gap_priority_plan_v1",
        "model_id": model_id,
        "target_id": target_id,
        "ordered_gaps": [
            {
                "rank": g.closure_priority,
                "gap_id": g.gap_id,
                "semantic_name": g.semantic_name,
                "fx_target": g.fx_target,
                "severity": g.severity,
                "severity_score": g.severity_score,
                "cost_fraction_estimate": g.cost_fraction_estimate,
                "recommended_next_action": g.recommended_next_action,
                "extension_id": g.extension_id,
                "suggested_extension_path": g.suggested_extension_path,
            }
            for g in sorted_gaps
        ],
    }
    gap_priority_plan_path = out_dir / "gap_priority_plan.json"
    gap_priority_plan_path.write_text(
        json.dumps(plan_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Reflect the new severity into the per-region critical-path bit so
    # gap_analysis.json's ``on_critical_path`` count tracks the same
    # input the audit used.
    severity_by_region: dict[str, str] = {g.region_id: g.severity for g in gaps}
    for r in region_infos:
        # carry the audit's severity bucket forward for the region too
        # (gap_analysis stores ``on_critical_path``, but downstream tools
        # may want the bucket — accept it as a side fact)
        if r.region_id in severity_by_region:
            pass  # purely informational for the audit; nothing to mutate

    # ------------------------------------------------------------------ #
    # 4. Populate per-module regions + emit dossier.json
    # ------------------------------------------------------------------ #
    for r in region_infos:
        ma = module_analyses.get(r.module_id)
        if ma is None:
            continue
        ma.regions.append(
            {
                "region_id": r.region_id,
                "fx_node": r.fx_node,
                "fx_target": r.fx_target,
                "classification": r.classification,
                "reason": r.reason,
                "shape_signature": r.shape_signature,
                "dtype_signature": r.dtype_signature,
                "on_critical_path": r.on_critical_path,
            }
        )

    dossier_obj = {
        "schema_version": "dossier_v1",
        "model_id": model_id,
        "target_id": target_id,
        "modules": [
            {
                "module_id": ma.module_id,
                "input_kind": ma.input_kind,
                "graph_path": ma.graph_path,
                "num_fx_nodes": ma.num_fx_nodes,
                "num_call_function": ma.num_call_function,
                "num_placeholders": ma.num_placeholders,
                "num_outputs": ma.num_outputs,
                "critical_path": list(ma.critical_path_nodes),
                "regions": ma.regions,
            }
            for ma in module_analyses.values()
        ],
        "totals": {
            "modules": len(module_analyses),
            "fx_nodes": sum(ma.num_fx_nodes for ma in module_analyses.values()),
            "call_function_nodes": sum(ma.num_call_function for ma in module_analyses.values()),
            "critical_path_nodes": sum(len(ma.critical_path_nodes) for ma in module_analyses.values()),
        },
    }
    dossier_path = out_dir / "dossier.json"
    dossier_path.write_text(json.dumps(dossier_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 5. gap_analysis.json (descriptive) and gap_action_queue.json (actionable)
    # ------------------------------------------------------------------ #
    histogram_native = sum(1 for r in region_infos if r.classification == "native")
    histogram_library = sum(1 for r in region_infos if r.classification == "library")
    histogram_need_kernel = sum(1 for r in region_infos if r.classification == "need_kernel")
    histogram_unsupported = sum(1 for r in region_infos if r.classification == "unsupported")

    gap_analysis_obj = {
        "schema_version": "gap_analysis_v1",
        "model_id": model_id,
        "target_id": target_id,
        "regions_total": len(region_infos),
        "ops": [
            {
                "region_id": r.region_id,
                "module_id": r.module_id,
                "fx_node": r.fx_node,
                "fx_target": r.fx_target,
                "shape_signature": r.shape_signature,
                "dtype_signature": r.dtype_signature,
                "classification": r.classification,
                "reason": r.reason,
                "on_critical_path": r.on_critical_path,
            }
            for r in region_infos
        ],
        "summary": {
            "native": histogram_native,
            "library": histogram_library,
            "need_kernel": histogram_need_kernel,
            "unsupported": histogram_unsupported,
            "critical_path_count": sum(1 for r in region_infos if r.on_critical_path),
        },
    }
    gap_analysis_path = out_dir / "gap_analysis.json"
    gap_analysis_path.write_text(
        json.dumps(gap_analysis_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    by_kind: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for g in gaps:
        by_kind[g.gap_kind] = by_kind.get(g.gap_kind, 0) + 1
        by_severity[g.severity] = by_severity.get(g.severity, 0) + 1

    queue_obj = {
        "schema_version": "gap_action_queue_v1",
        "model_id": model_id,
        "target_id": target_id,
        "gaps": [g.to_dict() for g in gaps],
        "summary": {
            "count": len(gaps),
            "by_kind": dict(sorted(by_kind.items())),
            "by_severity": dict(sorted(by_severity.items())),
        },
    }
    queue_path = out_dir / "gap_action_queue.json"
    queue_path.write_text(json.dumps(queue_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 5b. gap_index.json — flat lookup: gap_id → (extension_id, suggested
    #     workspace path, semantic_name). Lets Extension Closure consume
    #     the queue without re-deriving paths.
    # ------------------------------------------------------------------ #
    index_obj = {
        "schema_version": "gap_index_v1",
        "model_id": model_id,
        "target_id": target_id,
        "entries": [
            {
                "gap_id": g.gap_id,
                "gap_kind": g.gap_kind,
                "semantic_name": g.semantic_name,
                "extension_id": g.extension_id,
                "suggested_extension_path": g.suggested_extension_path,
            }
            for g in gaps
        ],
    }
    index_path = out_dir / "gap_index.json"
    index_path.write_text(json.dumps(index_obj, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 5c. gap_evidence/<gap_id>.json — one file per gap that captures the
    #     evidence references the agent (or Claude Code) needs to author
    #     a closure: where the payload_mlir lives, the originating
    #     unsupported_op record id, the lowering_report path, and the
    #     full required-evidence checklist.
    # ------------------------------------------------------------------ #
    evidence_dir = out_dir / "gap_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for g in gaps:
        evidence_obj = {
            "schema_version": "gap_evidence_v1",
            "gap_id": g.gap_id,
            "gap_kind": g.gap_kind,
            "semantic_name": g.semantic_name,
            "fx_target": g.fx_target,
            "fx_node": g.fx_node,
            "module_id": g.module_id,
            "extension_id": g.extension_id,
            "suggested_extension_path": g.suggested_extension_path,
            "shape_signature": g.shape_signature,
            "dtype_signature": g.dtype_signature,
            "source": dict(g.source),
            "required_evidence": list(g.required_evidence),
            "allowed_actions": list(g.allowed_actions),
            "severity": g.severity,
        }
        (evidence_dir / f"{g.gap_id}.json").write_text(
            json.dumps(evidence_obj, indent=2, sort_keys=True), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # 6. Per-stage report
    # ------------------------------------------------------------------ #
    failed_modules = [ma for ma in module_analyses.values() if any("error" in r for r in ma.regions)]
    overall_status = "fail" if not module_analyses else ("partial_success" if failed_modules else "pass")
    if not gaps and not module_analyses:
        overall_status = "fail"
    elif overall_status == "pass":
        # Pure pass when every module loaded cleanly; partial_success if any
        # module's FX graph couldn't be loaded.
        pass

    # Count actionable gaps: any gap whose allowed_actions contains a
    # non-fallback action (i.e. could actually be closed by Extension
    # Closure rather than just `keep_as_fallback`).
    actionable = sum(
        1 for g in gaps if any(a != "keep_as_fallback" for a in g.allowed_actions)
    )
    input_unsupported = len(unsupported.get("unsupported_ops", []))

    report_obj = {
        "schema_version": "gap_discovery_summary_v1",
        "stage_id": "gap_discovery",
        "status": overall_status,
        "model_id": model_id,
        "target_id": target_id,
        "lowering_summary_sha256": "sha256:" + sha256_file(summary_path),
        "extension_registry": registry_path_str,
        "closed_targets": closed_targets,
        "input_unsupported_ops_count": input_unsupported,
        "discovered_gap_count": len(gaps),
        "actionable_gap_count": actionable,
        "totals": {
            "regions_total": len(region_infos),
            "gaps_total": len(gaps),
            "closed_by_registry_count": len(closed_targets),
            "critical_path_gaps": sum(1 for g in gaps if g.severity == "critical_path"),
            "coverage_gaps": sum(1 for g in gaps if g.severity == "coverage_gap"),
            "performance_blocker_gaps": sum(1 for g in gaps if g.severity == "performance_blocker"),
            "modules_loaded_ok": len(module_analyses) - len(failed_modules),
            "modules_failed": len(failed_modules),
        },
        "outputs": {
            "gap_action_queue": "04_gap_discovery/gap_action_queue.json",
            "gap_analysis": "04_gap_discovery/gap_analysis.json",
            "gap_index": "04_gap_discovery/gap_index.json",
            "gap_evidence_dir": "04_gap_discovery/gap_evidence/",
            "dossier": "04_gap_discovery/dossier.json",
            "severity_audit": "04_gap_discovery/severity_audit.json",
            "gap_priority_plan": "04_gap_discovery/gap_priority_plan.json",
        },
        "severity_histogram": severity_histogram,
        "llm_calls": 0,
    }
    report_path = out_dir / "gap_discovery_summary.json"
    report_path.write_text(json.dumps(report_obj, indent=2, sort_keys=True), encoding="utf-8")

    finished_at = _utcnow()
    output_hash = sha256_tree(out_dir)
    # Hash chain: input = sha256_tree(<directly-prior stage dir>). The
    # layout inserts ``03_recipe_planning`` between graph_analysis
    # and gap_discovery, so the predecessor preference is:
    #   recipe_planning → graph_analysis → payload_lowering
    rp_dir = run_dir / "03_recipe_planning"
    ga_dir = run_dir / "02_graph_analysis"
    if rp_dir.is_dir():
        chain_predecessor = rp_dir
    elif ga_dir.is_dir():
        chain_predecessor = ga_dir
    else:
        chain_predecessor = pl_dir
    input_hash = sha256_tree(chain_predecessor)

    artifact_refs: list[ArtifactRef] = []
    for p in (queue_path, gap_analysis_path, dossier_path, report_path, index_path,
              severity_audit_path, gap_priority_plan_path):
        artifact_refs.append(
            ArtifactRef(
                path=p.relative_to(run_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )
    for p in sorted(evidence_dir.glob("gap_*.json")):
        artifact_refs.append(
            ArtifactRef(
                path=p.relative_to(run_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )

    # graph_compilation manifest contract: status ∈ {pass, fail, skipped}. partial_success
    # is recorded inside the per-stage report only.
    manifest_status = "pass" if overall_status in {"pass", "partial_success"} else "fail"

    return StageRecord(
        stage_id="gap_discovery",
        status=manifest_status,
        inputs=(
            ArtifactRef(
                path=unsupported_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(unsupported_path),
                size_bytes=unsupported_path.stat().st_size,
                kind="file",
            ),
            ArtifactRef(
                path=payload_index_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(payload_index_path),
                size_bytes=payload_index_path.stat().st_size,
                kind="file",
            ),
        ),
        outputs=tuple(artifact_refs),
        report_path="04_gap_discovery/gap_discovery_summary.json",
        input_hash=input_hash,
        output_hash=output_hash,
        llm_calls=0,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

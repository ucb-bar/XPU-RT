"""MCP tools that expose the :class:`FusionPlan` to the agent.

These tools land the protocol now (schema + handlers + registration)
even though the interactive agent-in-the-loop fusion override flow
is a follow-up. The pipeline-level study (P1.7
``run_pipeline_comparison``) talks to the planner directly via the
Python API, not through these tools — but landing the schema lets a
future :file:`xpu-rt-fusion-plan.md` slash command query the plan
without us re-inventing the carrier shape.

Tools:

  * ``xpu_rt_emit_fusion_decision_request`` — given a serialized
    :class:`ContractGraph` (and a target identifier resolving to a
    :class:`HardwareEnvelope`), run :func:`plan_fusion` and return a
    JSON view: per-cluster members, planner rationale, planner
    estimated speedup, and the per-pair / per-cluster oracle
    verdicts.

  * ``xpu_rt_commit_fusion_decision_response`` — accept an agent's
    fusion-cluster override and persist it next to the source graph.
    Today this is a schema-validating recorder; the pipeline-level
    driver will honour overrides in a follow-up.

Handler shape matches the AGENT_DECISION_TOOLS / KERNEL_BLAST_TOOLS
convention: takes a :class:`SessionManager` plus kwargs, returns a
JSON-serialisable dict.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from xpu_rt.mcp.session import SessionManager


def _load_graph_view(path: Path) -> dict[str, Any]:
    """Read a serialized :class:`ContractGraph` from JSON.

    The on-disk schema is the minimal one the planner needs:
    ``{"nodes": {<op_id>: {"op_name": ..., "input_shapes": [...],
    "output_shapes": [...], "dtype": "i8"}}, "edges": [{"producer_id":
    ..., "consumer_id": ..., "operand_index": ..., "tensor_shape":
    [...], "dtype": "i8"}]}``. The pipeline-level driver writes this
    JSON next to the captured ``payload.mlir`` so downstream tools
    can reconstruct the graph without re-walking the IR.
    """
    if not path.is_file():
        raise FileNotFoundError(f"contract graph file not found: {path}")
    obj: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "nodes" not in obj or "edges" not in obj:
        raise ValueError(f"malformed contract graph at {path}: missing nodes/edges keys")
    return obj


def _envelope_for_target(target_id: str) -> Any:
    """Resolve a target id to a :class:`HardwareEnvelope`.

    Today we hard-code Gemmini (the only target the study runs
    against). A follow-up will pull this from the Target Card
    knowledge store, which is where the envelope of record lives.
    """
    from xpu_rt.kernels.contract_v3 import HardwareEnvelope

    if target_id == "gemmini_mx":
        return HardwareEnvelope(
            target_name="gemmini_mx",
            vector_lanes=16,
            scratchpad_bytes=256 * 1024,
            register_bytes=16,
            native_dtypes=("i8", "i32"),
            peak_bandwidth_gbps=8.0,
            register_quota_per_thread=256,
        )
    raise ValueError(
        f"no envelope wired for target_id={target_id!r}; "
        "this tool currently only supports 'gemmini_mx' — see the "
        "Target Card knowledge store for the follow-up plan."
    )


def _rebuild_graph_from_view(view: dict[str, Any]) -> Any:
    from xpu_rt.ir.payload.contract_graph import (
        ContractEdge,
        ContractNode,
        build_contract_graph_from_nodes,
    )
    from xpu_rt.ir.payload.contracts import CostEstimate, KernelContract, LayoutKind, LayoutRequirement

    nodes_list = []
    for nid, n_view in view["nodes"].items():
        c = KernelContract(
            op_name=n_view["op_name"],
            input_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR) for _ in n_view.get("input_shapes", [])],
            output_layouts=[LayoutRequirement(LayoutKind.ROW_MAJOR) for _ in n_view.get("output_shapes", [])],
            supported_dtypes={n_view.get("dtype", "f32")},
            cost=CostEstimate(),
            metadata={
                "input_shapes": [tuple(s) for s in n_view.get("input_shapes", [])],
                "output_shapes": [tuple(s) for s in n_view.get("output_shapes", [])],
                "region_id": nid,
                "dispatch_id": nid,
            },
        )
        nodes_list.append(ContractNode(op_id=nid, contract=c, op_name=n_view["op_name"], region_id=nid))
    edges = [
        ContractEdge(
            producer_id=e["producer_id"],
            consumer_id=e["consumer_id"],
            operand_index=int(e.get("operand_index", 0)),
            tensor_shape=tuple(e.get("tensor_shape", ())),
            dtype=str(e.get("dtype", "f32")),
            bytes_per_element=int(e.get("bytes_per_element", 4)),
        )
        for e in view["edges"]
    ]
    return build_contract_graph_from_nodes(nodes_list, edges)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def xpu_rt_emit_fusion_decision_request(
    sm: SessionManager,
    *,
    graph_json: str,
    target_id: str = "gemmini_mx",
) -> dict[str, Any]:
    """Run the fusion planner and return its plan as JSON.

    Args:
        graph_json: Path to a serialized ContractGraph JSON file.
        target_id: Target identifier (only ``gemmini_mx`` today).

    Returns:
        ``{"plan": {...}, "graph_path": ..., "target_id": ...}`` where
        plan exposes ``clusters`` (each with member_op_ids, rationale,
        estimated_speedup), ``estimated_speedup`` overall, and
        ``per_pair_verdicts`` so the agent can see why the planner
        joined or split each pair.
    """
    del sm  # unused — this tool operates on disk paths
    from xpu_rt.kernels.fusion_planner import plan_fusion

    path = Path(graph_json)
    view = _load_graph_view(path)
    graph = _rebuild_graph_from_view(view)
    envelope = _envelope_for_target(target_id)
    plan = plan_fusion(graph, envelope)
    return {
        "plan": {
            "envelope_target": plan.envelope_target,
            "estimated_speedup": plan.estimated_speedup,
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "member_op_ids": list(c.member_op_ids),
                    "rationale": c.rationale,
                    "estimated_speedup": c.estimated_speedup,
                }
                for c in plan.clusters
            ],
            "per_pair_verdicts": [
                {
                    "producer_id": p,
                    "consumer_id": c,
                    "decision": v.decision.value,
                    "est_speedup_ratio": v.est_speedup_ratio,
                    "reason": v.reason,
                }
                for (p, c, v) in plan.per_pair_verdicts
            ],
            "per_cluster_granularity": [
                {
                    "cluster_id": cid,
                    "granularity": gv.granularity.value,
                    "reason": gv.reason,
                    "chain_speedup_estimate": gv.chain_speedup_estimate,
                }
                for (cid, gv) in plan.per_cluster_granularity
            ],
        },
        "graph_path": str(path),
        "target_id": target_id,
    }


def xpu_rt_commit_fusion_decision_response(
    sm: SessionManager,
    *,
    graph_json: str,
    response: dict[str, Any],
    out_path: str | None = None,
) -> dict[str, Any]:
    """Persist an agent's fusion-cluster override.

    The response shape mirrors the emit-request output (the agent
    edits ``clusters`` to a different partition). Today we just
    validate the partition covers every node exactly once and write
    the override next to ``graph_json``; future P1.7 runs will read
    it back and skip the planner in favour of the agent's verdict.
    """
    del sm
    path = Path(graph_json)
    view = _load_graph_view(path)
    all_node_ids = set(view["nodes"].keys())

    clusters = response.get("clusters", [])
    seen: set[str] = set()
    for c in clusters:
        if "member_op_ids" not in c:
            return {"ok": False, "error": "cluster missing member_op_ids", "cluster": c}
        for m in c["member_op_ids"]:
            if m in seen:
                return {"ok": False, "error": f"node {m!r} appears in multiple clusters"}
            if m not in all_node_ids:
                return {"ok": False, "error": f"node {m!r} not in graph nodes"}
            seen.add(m)
    missing = all_node_ids - seen
    if missing:
        return {"ok": False, "error": f"clusters do not cover nodes {sorted(missing)}"}

    if out_path is None:
        out_path = str(path.with_suffix(path.suffix + ".fusion_override.json"))
    Path(out_path).write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "ok": True,
        "override_path": out_path,
        "graph_path": str(path),
        "cluster_count": len(clusters),
        "covered_nodes": len(seen),
    }


# --------------------------------------------------------------------------- #
# Tool registry entries
# --------------------------------------------------------------------------- #


FUSION_DECISION_TOOLS: list[dict[str, Any]] = [
    {
        "name": "xpu_rt_emit_fusion_decision_request",
        "description": (
            "Run the XPU-RT FusionPlanner over a serialised ContractGraph "
            "and return its plan + per-pair fusion-oracle verdicts + "
            "per-cluster granularity-oracle verdicts as JSON. Step 1 of "
            "an interactive fusion-decision override flow (the agent "
            "reads this view, picks a different partition, and commits "
            "via xpu_rt_commit_fusion_decision_response). The pipeline-"
            "level study uses the planner's verdict directly without "
            "going through this MCP path."
        ),
        "phase": "inspect",
        "handler": xpu_rt_emit_fusion_decision_request,
        "input_schema": {
            "type": "object",
            "properties": {
                "graph_json": {
                    "type": "string",
                    "description": "Path to a ContractGraph serialised as JSON.",
                },
                "target_id": {
                    "type": "string",
                    "default": "gemmini_mx",
                    "description": "Target identifier; today only 'gemmini_mx' is wired.",
                },
            },
            "required": ["graph_json"],
        },
    },
    {
        "name": "xpu_rt_commit_fusion_decision_response",
        "description": (
            "Persist an agent's fusion-cluster override next to the "
            "source ContractGraph. Validates that the response's "
            "clusters partition every node exactly once. Today this is "
            "a recorder; the pipeline-level driver will honour the "
            "override in a follow-up. Schema-validating only — does "
            "not run the planner."
        ),
        "phase": "transform",
        "handler": xpu_rt_commit_fusion_decision_response,
        "input_schema": {
            "type": "object",
            "properties": {
                "graph_json": {"type": "string"},
                "response": {
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cluster_id": {"type": "string"},
                                    "member_op_ids": {"type": "array", "items": {"type": "string"}},
                                    "rationale": {"type": "string"},
                                },
                                "required": ["cluster_id", "member_op_ids"],
                            },
                        },
                    },
                    "required": ["clusters"],
                },
                "out_path": {"type": "string"},
            },
            "required": ["graph_json", "response"],
        },
    },
]


__all__ = [
    "FUSION_DECISION_TOOLS",
    "xpu_rt_commit_fusion_decision_response",
    "xpu_rt_emit_fusion_decision_request",
]

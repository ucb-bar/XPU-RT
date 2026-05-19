"""Tests for the fusion-decision MCP tool schema and handlers.

These tests do NOT spin up an MCP server — they call the handlers
directly with the same shape an MCP transport would deliver. That
covers the JSON round-trip + planner integration without coupling to
the SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xpu_rt.mcp.session import SessionManager
from xpu_rt.mcp.tools.fusion_decision import (
    FUSION_DECISION_TOOLS,
    xpu_rt_commit_fusion_decision_response,
    xpu_rt_emit_fusion_decision_request,
)


def _write_mlp_chain_graph_json(tmp_path: Path) -> Path:
    """Serialize a 3-node matmul→silu→matmul ContractGraph view."""
    view = {
        "nodes": {
            "n_m1": {
                "op_name": "matmul",
                "input_shapes": [[64, 720], [720, 1440]],
                "output_shapes": [[64, 1440]],
                "dtype": "i8",
            },
            "n_s1": {
                "op_name": "silu",
                "input_shapes": [[64, 1440]],
                "output_shapes": [[64, 1440]],
                "dtype": "i8",
            },
            "n_m2": {
                "op_name": "matmul",
                "input_shapes": [[64, 1440], [1440, 720]],
                "output_shapes": [[64, 720]],
                "dtype": "i8",
            },
        },
        "edges": [
            {"producer_id": "n_m1", "consumer_id": "n_s1", "operand_index": 0,
             "tensor_shape": [64, 1440], "dtype": "i8", "bytes_per_element": 1},
            {"producer_id": "n_s1", "consumer_id": "n_m2", "operand_index": 0,
             "tensor_shape": [64, 1440], "dtype": "i8", "bytes_per_element": 1},
        ],
    }
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(view), encoding="utf-8")
    return p


def test_emit_fusion_decision_request_returns_plan(tmp_path: Path) -> None:
    sm = SessionManager()
    path = _write_mlp_chain_graph_json(tmp_path)
    out = xpu_rt_emit_fusion_decision_request(sm, graph_json=str(path), target_id="gemmini_mx")
    assert out["target_id"] == "gemmini_mx"
    assert out["graph_path"] == str(path)
    plan = out["plan"]
    assert plan["envelope_target"] == "gemmini_mx"
    # The 3-node chain produces ≥ 1 cluster, partitioning all nodes.
    members_total: list[str] = []
    for c in plan["clusters"]:
        assert "cluster_id" in c
        assert "member_op_ids" in c
        members_total.extend(c["member_op_ids"])
    assert set(members_total) == {"n_m1", "n_s1", "n_m2"}


def test_emit_fusion_decision_rejects_missing_file() -> None:
    sm = SessionManager()
    with pytest.raises(FileNotFoundError):
        xpu_rt_emit_fusion_decision_request(sm, graph_json="/nonexistent.json")


def test_commit_fusion_decision_validates_partition(tmp_path: Path) -> None:
    sm = SessionManager()
    path = _write_mlp_chain_graph_json(tmp_path)

    # Good partition: each node in exactly one cluster.
    good = {"clusters": [{"cluster_id": "c0", "member_op_ids": ["n_m1", "n_s1", "n_m2"]}]}
    out = xpu_rt_commit_fusion_decision_response(sm, graph_json=str(path), response=good)
    assert out["ok"] is True
    assert Path(out["override_path"]).is_file()
    assert out["cluster_count"] == 1
    assert out["covered_nodes"] == 3

    # Duplicate node — rejected.
    dup = {"clusters": [
        {"cluster_id": "c0", "member_op_ids": ["n_m1", "n_s1"]},
        {"cluster_id": "c1", "member_op_ids": ["n_s1", "n_m2"]},
    ]}
    out = xpu_rt_commit_fusion_decision_response(sm, graph_json=str(path), response=dup)
    assert out["ok"] is False
    assert "multiple clusters" in out["error"]

    # Missing node — rejected.
    missing = {"clusters": [{"cluster_id": "c0", "member_op_ids": ["n_m1"]}]}
    out = xpu_rt_commit_fusion_decision_response(sm, graph_json=str(path), response=missing)
    assert out["ok"] is False
    assert "do not cover" in out["error"]


def test_fusion_decision_tools_have_valid_descriptors() -> None:
    """Smoke test: the registry list shape must match what the MCP
    server iterates."""
    assert len(FUSION_DECISION_TOOLS) == 2
    for t in FUSION_DECISION_TOOLS:
        assert {"name", "description", "phase", "handler", "input_schema"} <= set(t.keys())
        assert callable(t["handler"])
        assert t["input_schema"]["type"] == "object"


def test_fusion_decision_tools_registered_in_all_tools() -> None:
    """The fusion-decision tools must show up in the top-level
    ALL_TOOLS list so the MCP server actually exposes them."""
    from xpu_rt.mcp.tools import _IN_TREE_TOOLS

    names = {t["name"] for t in _IN_TREE_TOOLS}
    assert "xpu_rt_emit_fusion_decision_request" in names
    assert "xpu_rt_commit_fusion_decision_response" in names

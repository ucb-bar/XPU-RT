"""Lowering tests for Agent IR."""

from __future__ import annotations

from compgen.ir.agent.attrs import EvaluatorKindAttr
from compgen.ir.agent.lower import AgentLoweringOutput, lower_agent
from compgen.ir.agent.ops_claim import ClaimOp, ExpectedProofOp
from compgen.ir.agent.ops_critique import CritiqueOp
from compgen.ir.agent.ops_evidence import BindFactOp, EvidenceSetOp
from compgen.ir.agent.ops_frontier import AlternativeOp, FrontierOp
from compgen.ir.agent.ops_intent import AgentScopeOp, AgentSessionOp
from compgen.ir.agent.ops_memory import MemoryPatternOp
from compgen.ir.agent.ops_protocol import RoleOp
from compgen.ir.agent.ops_synthesis import RequestRewriteOp
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import ModuleOp, Region, StringAttr, SymbolRefAttr
from xdsl.ir import Block


def test_agent_lowering_output_defaults() -> None:
    output = AgentLoweringOutput()
    assert output.request_jobs == []
    assert output.claim_records == []
    assert output.frontier_states == []
    assert output.diagnostics == []


def test_lower_agent_collects_records_by_family() -> None:
    block = Block()
    block.add_op(
        AgentSessionOp.build(
            properties={
                "sym_name": StringAttr("session_main"),
                "objective": StringAttr("latency"),
                "target": StringAttr("cuda_a100"),
                "search_mode": StringAttr("iterative"),
            }
        )
    )
    block.add_op(
        AgentScopeOp.build(
            properties={
                "sym_name": StringAttr("scope_r0"),
                "session_ref": SymbolRefAttr("session_main"),
                "scope_ref": SymbolRefAttr("r0"),
                "scope_kind": StringAttr("region"),
            }
        )
    )
    block.add_op(
        BindFactOp.build(
            properties={
                "sym_name": StringAttr("fact_0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "fact_name": StringAttr("recipe.fact.backend_available"),
            }
        )
    )
    block.add_op(
        EvidenceSetOp.build(
            properties={
                "sym_name": StringAttr("evidence_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "evidence_refs": builtin_dialect.ArrayAttr([SymbolRefAttr("fact_0")]),
            }
        )
    )
    block.add_op(
        RequestRewriteOp.build(
            properties={
                "sym_name": StringAttr("rq_tile_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "family": StringAttr("tile"),
                "evidence_set_ref": SymbolRefAttr("evidence_r0"),
                "output_kind": StringAttr("recipe_candidate"),
                "evaluator": EvaluatorKindAttr("translation_validation"),
            }
        )
    )
    block.add_op(
        ClaimOp.build(
            properties={
                "sym_name": StringAttr("claim_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "kind": StringAttr("correctness"),
                "text": StringAttr("The rewrite preserves semantics."),
            }
        )
    )
    block.add_op(
        ExpectedProofOp.build(
            properties={
                "claim_ref": SymbolRefAttr("claim_r0"),
                "evaluator": EvaluatorKindAttr("translation_validation"),
            }
        )
    )
    block.add_op(
        FrontierOp.build(
            properties={
                "sym_name": StringAttr("frontier_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "objective": StringAttr("latency"),
            }
        )
    )
    block.add_op(
        AlternativeOp.build(
            properties={
                "frontier_ref": SymbolRefAttr("frontier_r0"),
                "target_ref": SymbolRefAttr("cand_tile_r0"),
                "target_kind": StringAttr("recipe_candidate"),
            }
        )
    )
    block.add_op(
        CritiqueOp.build(
            properties={
                "sym_name": StringAttr("critique_r0"),
                "target_ref": SymbolRefAttr("cand_tile_r0"),
                "reason": StringAttr("Needs stronger performance evidence."),
                "severity": StringAttr("medium"),
            }
        )
    )
    block.add_op(
        MemoryPatternOp.build(
            properties={
                "sym_name": StringAttr("memory_tile"),
                "domain": StringAttr("tiling"),
                "pattern": StringAttr("matmul -> tile"),
                "outcome": StringAttr("promote"),
            }
        )
    )
    block.add_op(
        RoleOp.build(
            properties={
                "sym_name": StringAttr("planner"),
                "kind": StringAttr("global_search"),
            }
        )
    )

    output = lower_agent(ModuleOp(Region(block)))

    assert len(output.request_jobs) == 1
    assert output.request_jobs[0]["family"] == "tile"
    assert len(output.claim_records) == 2
    assert len(output.frontier_states) == 2
    assert len(output.critique_records) == 1
    assert len(output.memory_records) == 1
    assert len(output.protocol_records) == 1

"""Round-trip tests for Agent IR."""

from __future__ import annotations

import io

from compgen.ir.agent.attrs import EvaluatorKindAttr, SearchBudgetAttr
from compgen.ir.agent.dialect import Agent
from compgen.ir.agent.ops_claim import ClaimOp
from compgen.ir.agent.ops_critique import CritiqueOp
from compgen.ir.agent.ops_evidence import BindFactOp, EvidenceSetOp
from compgen.ir.agent.ops_frontier import AlternativeOp, FrontierOp
from compgen.ir.agent.ops_intent import AgentScopeOp, AgentSessionOp
from compgen.ir.agent.ops_memory import MemoryPatternOp
from compgen.ir.agent.ops_protocol import RoleOp
from compgen.ir.agent.ops_synthesis import RequestRewriteOp
from xdsl.context import Context
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import ModuleOp, Region, StringAttr, SymbolRefAttr
from xdsl.ir import Block
from xdsl.parser import Parser
from xdsl.printer import Printer


def _round_trip(ops: list) -> None:
    block = Block()
    for op in ops:
        block.add_op(op)
    module = ModuleOp(Region(block))

    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    text1 = buf.getvalue()

    ctx = Context()
    ctx.register_dialect("agent", lambda: Agent)
    ctx.register_dialect("builtin", lambda: builtin_dialect.Builtin)
    parsed = Parser(ctx, text1).parse_module()

    buf2 = io.StringIO()
    Printer(stream=buf2).print_op(parsed)
    text2 = buf2.getvalue()

    assert text1 == text2


def test_agent_round_trip_representative_ops() -> None:
    ops = [
        AgentSessionOp.build(
            properties={
                "sym_name": StringAttr("session_main"),
                "objective": StringAttr("latency"),
                "target": StringAttr("cuda_a100"),
                "search_mode": StringAttr("iterative"),
            }
        ),
        AgentScopeOp.build(
            properties={
                "sym_name": StringAttr("scope_r0"),
                "session_ref": SymbolRefAttr("session_main"),
                "scope_ref": SymbolRefAttr("r0"),
                "scope_kind": StringAttr("region"),
            }
        ),
        BindFactOp.build(
            properties={
                "sym_name": StringAttr("fact_0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "fact_name": StringAttr("recipe.fact.backend_available"),
                "fact_payload": StringAttr("backend=triton"),
            }
        ),
        EvidenceSetOp.build(
            properties={
                "sym_name": StringAttr("evidence_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "evidence_refs": builtin_dialect.ArrayAttr([SymbolRefAttr("fact_0")]),
            }
        ),
        RequestRewriteOp.build(
            properties={
                "sym_name": StringAttr("rq_tile_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "family": StringAttr("tile"),
                "evidence_set_ref": SymbolRefAttr("evidence_r0"),
                "output_kind": StringAttr("recipe_candidate"),
                "search_budget": SearchBudgetAttr(1, 1, 1000),
                "evaluator": EvaluatorKindAttr("translation_validation"),
            }
        ),
        ClaimOp.build(
            properties={
                "sym_name": StringAttr("claim_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "kind": StringAttr("performance"),
                "text": StringAttr("Tile matmul to improve locality."),
            }
        ),
        FrontierOp.build(
            properties={
                "sym_name": StringAttr("frontier_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "objective": StringAttr("latency"),
            }
        ),
        AlternativeOp.build(
            properties={
                "frontier_ref": SymbolRefAttr("frontier_r0"),
                "target_ref": SymbolRefAttr("cand_tile_r0"),
                "target_kind": StringAttr("recipe_candidate"),
            }
        ),
        CritiqueOp.build(
            properties={
                "sym_name": StringAttr("critique_r0"),
                "target_ref": SymbolRefAttr("cand_tile_r0"),
                "reason": StringAttr("Needs stronger legality evidence."),
                "severity": StringAttr("medium"),
            }
        ),
        MemoryPatternOp.build(
            properties={
                "sym_name": StringAttr("memory_tile"),
                "domain": StringAttr("tiling"),
                "pattern": StringAttr("matmul -> tile"),
                "outcome": StringAttr("promote"),
            }
        ),
        RoleOp.build(
            properties={
                "sym_name": StringAttr("planner"),
                "kind": StringAttr("global_search"),
            }
        ),
    ]
    _round_trip(ops)

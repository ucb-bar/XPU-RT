"""Validation tests for Agent IR."""

from __future__ import annotations

from compgen.ir.agent.attrs import EvaluatorKindAttr
from compgen.ir.agent.ops_claim import ClaimOp, ExpectedProofOp
from compgen.ir.agent.ops_evidence import BindFactOp, BindVerificationOp, EvidenceSetOp
from compgen.ir.agent.ops_frontier import CommitOp, FrontierOp
from compgen.ir.agent.ops_intent import AgentScopeOp, AgentSessionOp
from compgen.ir.agent.ops_synthesis import RequestRewriteOp
from compgen.ir.agent.validate import validate_agent_module
from compgen.ir.recipe.ops_candidate import TileOp
from compgen.ir.recipe.ops_scope import RecipeRegionOp
from xdsl.dialects import builtin as builtin_dialect
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, Region, StringAttr, SymbolRefAttr
from xdsl.ir import Block


def _i64(value: int) -> IntegerAttr:
    return IntegerAttr(value, IntegerType(64))


def _recipe_module() -> ModuleOp:
    block = Block()
    block.add_op(
        RecipeRegionOp.build(
            properties={
                "sym_name": StringAttr("r0"),
                "payload_region_id": StringAttr("payload_r0"),
            }
        )
    )
    block.add_op(
        TileOp.build(
            properties={
                "sym_name": StringAttr("cand_tile_r0"),
                "region_ref": SymbolRefAttr("r0"),
                "tile_sizes": builtin_dialect.ArrayAttr([_i64(16), _i64(16), _i64(16)]),
            }
        )
    )
    return ModuleOp(Region(block))


def _base_agent_module(
    *, with_proof: bool = False, with_commit: bool = False, with_verification: bool = False
) -> ModuleOp:
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
    evidence_refs = [SymbolRefAttr("fact_0")]
    if with_verification:
        block.add_op(
            BindVerificationOp.build(
                properties={
                    "sym_name": StringAttr("ver_0"),
                    "scope_ref": SymbolRefAttr("scope_r0"),
                    "verification_key": StringAttr("tv:r0"),
                    "status": StringAttr("passed"),
                }
            )
        )
        evidence_refs.append(SymbolRefAttr("ver_0"))
    block.add_op(
        EvidenceSetOp.build(
            properties={
                "sym_name": StringAttr("evidence_r0"),
                "scope_ref": SymbolRefAttr("scope_r0"),
                "evidence_refs": builtin_dialect.ArrayAttr(evidence_refs),
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
    if with_proof:
        block.add_op(
            ExpectedProofOp.build(
                properties={
                    "claim_ref": SymbolRefAttr("claim_r0"),
                    "evaluator": EvaluatorKindAttr("translation_validation"),
                }
            )
        )
    if with_commit:
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
            CommitOp.build(
                properties={
                    "frontier_ref": SymbolRefAttr("frontier_r0"),
                    "selected_ref": SymbolRefAttr("cand_tile_r0"),
                    "evidence_set_ref": SymbolRefAttr("evidence_r0"),
                }
            )
        )
    return ModuleOp(Region(block))


def test_claim_requiring_proof_fails_without_expected_proof() -> None:
    result = validate_agent_module(_base_agent_module(), recipe_module=_recipe_module())
    assert not result.valid
    assert any("requires an agent.expected_proof" in err.message for err in result.errors)


def test_agent_module_with_expected_proof_passes() -> None:
    result = validate_agent_module(_base_agent_module(with_proof=True), recipe_module=_recipe_module())
    assert result.valid, result.errors


def test_commit_requires_judged_evidence() -> None:
    result = validate_agent_module(
        _base_agent_module(with_proof=True, with_commit=True),
        recipe_module=_recipe_module(),
    )
    assert not result.valid
    assert any("Commit requires an evidence set" in err.message for err in result.errors)


def test_commit_with_verification_evidence_passes() -> None:
    result = validate_agent_module(
        _base_agent_module(with_proof=True, with_commit=True, with_verification=True),
        recipe_module=_recipe_module(),
    )
    assert result.valid, result.errors

"""Deterministic Agent IR seeding around a Recipe IR module."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import structlog
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Block, Operation, Region

from compgen.ir.agent.attrs import FreshnessAttr
from compgen.ir.agent.ops_evidence import BindFactOp, EvidenceSetOp
from compgen.ir.agent.ops_frontier import AlternativeOp, FrontierOp
from compgen.ir.agent.ops_intent import AgentAssumptionOp, AgentScopeOp, AgentSessionOp, AgentUncertaintyOp
from compgen.ir.recipe.ops_scope import AnchorOp, RecipeRegionOp, SegmentOp

log = structlog.get_logger()


def generate_seed_agent(
    recipe_module: ModuleOp,
    target_profile: Any = None,
    objective: str = "latency",
) -> ModuleOp:
    """Generate an Agent IR scaffold from a deterministic Recipe IR seed."""
    session_sym = "session_main"
    global_scope_sym = "scope_session"
    global_evidence_sym = "evidence_session"

    scope_defs: list[tuple[str, str, str]] = []
    scope_by_recipe_sym: dict[str, str] = {}
    scope_kind_by_recipe_sym: dict[str, str] = {}
    evidence_refs_by_scope: dict[str, list[str]] = defaultdict(list)
    alternatives_by_scope: dict[str, list[str]] = defaultdict(list)
    bind_ops: list[Operation] = []

    for op in recipe_module.body.block.ops:
        if isinstance(op, RecipeRegionOp):
            recipe_sym = op.sym_name.data
            scope_sym = _scope_symbol(recipe_sym)
            scope_defs.append((scope_sym, recipe_sym, "region"))
            scope_by_recipe_sym[recipe_sym] = scope_sym
            scope_kind_by_recipe_sym[recipe_sym] = "region"
        elif isinstance(op, SegmentOp):
            recipe_sym = op.sym_name.data
            scope_sym = _scope_symbol(recipe_sym)
            scope_defs.append((scope_sym, recipe_sym, "segment"))
            scope_by_recipe_sym[recipe_sym] = scope_sym
            scope_kind_by_recipe_sym[recipe_sym] = "segment"
        elif isinstance(op, AnchorOp):
            recipe_sym = op.sym_name.data
            scope_sym = _scope_symbol(recipe_sym)
            scope_defs.append((scope_sym, recipe_sym, "anchor"))
            scope_by_recipe_sym[recipe_sym] = scope_sym
            scope_kind_by_recipe_sym[recipe_sym] = "anchor"

    fact_index = 0
    for op in recipe_module.body.block.ops:
        if not op.name.startswith("recipe.fact."):
            continue
        related_recipe_scope = _primary_recipe_scope(op)
        if related_recipe_scope is None or related_recipe_scope not in scope_by_recipe_sym:
            continue
        scope_sym = scope_by_recipe_sym[related_recipe_scope]
        bind_sym = f"fact_{fact_index}"
        fact_index += 1
        bind_ops.append(
            BindFactOp.build(
                properties={
                    "sym_name": StringAttr(bind_sym),
                    "scope_ref": SymbolRefAttr(scope_sym),
                    "fact_name": StringAttr(op.name),
                    "fact_payload": StringAttr(_summarize_recipe_op(op)),
                    "freshness": FreshnessAttr(0, "fresh"),
                }
            )
        )
        evidence_refs_by_scope[scope_sym].append(bind_sym)

    for op in recipe_module.body.block.ops:
        candidate_sym = _recipe_candidate_symbol(op)
        if candidate_sym is None:
            continue
        related_recipe_scope = _primary_recipe_scope(op)
        if related_recipe_scope is None or related_recipe_scope not in scope_by_recipe_sym:
            continue
        scope_sym = scope_by_recipe_sym[related_recipe_scope]
        alternatives_by_scope[scope_sym].append(candidate_sym)

    block = Block()
    block.add_op(
        AgentSessionOp.build(
            properties={
                "sym_name": StringAttr(session_sym),
                "objective": StringAttr(objective),
                "target": StringAttr(_target_name(target_profile)),
                "search_mode": StringAttr("iterative"),
                "constraints": ArrayAttr(
                    [
                        StringAttr("verification_first"),
                        StringAttr("recipe_is_commit_contract"),
                    ]
                ),
            }
        )
    )
    block.add_op(
        AgentScopeOp.build(
            properties={
                "sym_name": StringAttr(global_scope_sym),
                "session_ref": SymbolRefAttr(session_sym),
                "scope_ref": SymbolRefAttr(session_sym),
                "scope_kind": StringAttr("session"),
            }
        )
    )
    block.add_op(
        AgentAssumptionOp.build(
            properties={
                "sym_name": StringAttr("assumption_verification_first"),
                "scope_ref": SymbolRefAttr(global_scope_sym),
                "text": StringAttr("LLM proposals are not truth; correctness comes from evaluators."),
                "status": StringAttr("locked"),
            }
        )
    )

    for scope_sym, recipe_sym, scope_kind in scope_defs:
        block.add_op(
            AgentScopeOp.build(
                properties={
                    "sym_name": StringAttr(scope_sym),
                    "session_ref": SymbolRefAttr(session_sym),
                    "scope_ref": SymbolRefAttr(recipe_sym),
                    "scope_kind": StringAttr(scope_kind),
                }
            )
        )

    for bind_op in bind_ops:
        block.add_op(bind_op)

    block.add_op(
        EvidenceSetOp.build(
            properties={
                "sym_name": StringAttr(global_evidence_sym),
                "scope_ref": SymbolRefAttr(global_scope_sym),
                "evidence_refs": ArrayAttr([]),
            }
        )
    )

    for scope_sym, _, _ in scope_defs:
        block.add_op(
            EvidenceSetOp.build(
                properties={
                    "sym_name": StringAttr(_evidence_symbol(scope_sym)),
                    "scope_ref": SymbolRefAttr(scope_sym),
                    "evidence_refs": ArrayAttr(
                        [SymbolRefAttr(ref) for ref in evidence_refs_by_scope.get(scope_sym, [])]
                    ),
                }
            )
        )
        block.add_op(
            FrontierOp.build(
                properties={
                    "sym_name": StringAttr(_frontier_symbol(scope_sym)),
                    "scope_ref": SymbolRefAttr(scope_sym),
                    "objective": StringAttr(objective),
                }
            )
        )
        alt_refs = alternatives_by_scope.get(scope_sym, [])
        for alt_ref in alt_refs:
            block.add_op(
                AlternativeOp.build(
                    properties={
                        "frontier_ref": SymbolRefAttr(_frontier_symbol(scope_sym)),
                        "target_ref": SymbolRefAttr(alt_ref),
                        "target_kind": StringAttr("recipe_candidate"),
                    }
                )
            )
        block.add_op(
            AgentUncertaintyOp.build(
                properties={
                    "sym_name": StringAttr(f"uncertainty_{scope_sym}"),
                    "scope_ref": SymbolRefAttr(scope_sym),
                    "kind": StringAttr("open_frontier"),
                    "alternatives": ArrayAttr([StringAttr(ref) for ref in alt_refs]),
                }
            )
        )

    log.info("agent.seed.generated", scopes=len(scope_defs), fact_binds=len(bind_ops))
    return ModuleOp(Region(block))


def _scope_symbol(recipe_sym: str) -> str:
    return f"scope_{recipe_sym}"


def _frontier_symbol(scope_sym: str) -> str:
    return f"frontier_{scope_sym.removeprefix('scope_')}"


def _evidence_symbol(scope_sym: str) -> str:
    return f"evidence_{scope_sym.removeprefix('scope_')}"


def _target_name(target_profile: Any) -> str:
    return getattr(target_profile, "name", "unknown")


def _recipe_candidate_symbol(op: Operation) -> str | None:
    if hasattr(op, "sym_name") and isinstance(getattr(op, "sym_name"), StringAttr):
        if op.name.startswith("recipe.") and not op.name.startswith("recipe.region") and not op.name.startswith("recipe.segment") and not op.name.startswith("recipe.anchor") and not op.name.startswith("recipe.guard"):
            return getattr(op, "sym_name").data
    return None


def _primary_recipe_scope(op: Operation) -> str | None:
    for attr_name in ("region_ref", "after_region", "src_region", "region_a"):
        if hasattr(op, attr_name):
            ref = getattr(op, attr_name)
            if isinstance(ref, SymbolRefAttr):
                return ref.root_reference.data
    if hasattr(op, "fuse_regions"):
        fuse_regions = getattr(op, "fuse_regions")
        if isinstance(fuse_regions, ArrayAttr) and fuse_regions.data:
            first = fuse_regions.data[0]
            if isinstance(first, SymbolRefAttr):
                return first.root_reference.data
    return None


def _summarize_recipe_op(op: Operation) -> str:
    fields: list[str] = [op.name]
    for key, attr in op.properties.items():
        if isinstance(attr, StringAttr):
            fields.append(f"{key}={attr.data}")
        elif isinstance(attr, SymbolRefAttr):
            fields.append(f"{key}=@{attr.root_reference.data}")
    return " ".join(fields)


__all__ = ["generate_seed_agent"]

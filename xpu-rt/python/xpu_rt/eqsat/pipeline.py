"""Top-level equality saturation pass for CompGen.

Orchestrates: create eclasses → create egraphs → apply rewrite rules →
assign costs → extract best subprogram → clean up.

Uses xDSL's native ``equivalence`` dialect and transforms.  No external
e-graph libraries are needed.
"""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from xdsl.context import Context
from xdsl.dialects import builtin, equivalence, func
from xdsl.printer import Printer
from xdsl.transforms.eqsat_add_costs import EqsatAddCostsPass
from xdsl.transforms.eqsat_create_eclasses import EqsatCreateEclassesPass
from xdsl.transforms.eqsat_extract import EqsatExtractPass

from compgen.eqsat.config import EqSatConfig

if TYPE_CHECKING:
    from compgen.eqsat.rules.python_rules import EqSatRewriteRule

log = structlog.get_logger()


def _register_dialects(ctx: Context) -> None:
    """Register dialects needed for eqsat."""
    ctx.allow_unregistered = True


def _count_eclasses(module: builtin.ModuleOp) -> int:
    """Count equivalence.class ops in the module."""
    return sum(
        1 for op in module.walk() if isinstance(op, equivalence.AnyClassOp)
    )


def _count_enodes(module: builtin.ModuleOp) -> int:
    """Count total e-nodes (operands across all eclasses)."""
    total = 0
    for op in module.walk():
        if isinstance(op, equivalence.AnyClassOp):
            total += len(op.operands)
    return total


def _count_ops(module: builtin.ModuleOp) -> int:
    """Count non-eqsat operations."""
    return sum(
        1
        for op in module.walk()
        if not isinstance(op, (equivalence.AnyClassOp, equivalence.GraphOp,
                               equivalence.YieldOp, builtin.ModuleOp,
                               func.FuncOp, func.ReturnOp))
    )


def _print_ir(module: builtin.ModuleOp) -> str:
    """Print module to string for debugging."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue()


def create_egraph(module: builtin.ModuleOp) -> None:
    """Wrap a module's functions in e-graph form (eclasses around every value).

    Modifies the module in-place. Only processes functions whose body is a
    single block with no nested regions (xDSL eqsat limitation).
    """
    ctx = Context()
    _register_dialects(ctx)

    # Check if the module has multi-block or nested-region functions that
    # would crash EqsatCreateEclassesPass. If so, extract the safe subset.
    from xdsl.dialects.func import FuncOp as XDSLFuncOp

    has_complex_funcs = False
    for op in module.body.block.ops:
        if isinstance(op, XDSLFuncOp):
            # Check for empty body or nested regions in ops
            if not op.body.blocks:
                has_complex_funcs = True
                break
            for inner_op in op.body.block.ops:
                if inner_op.regions:
                    has_complex_funcs = True
                    break

    if has_complex_funcs:
        # Fall back to manual eclass insertion on the main function's
        # top-level ops only (skip ops with nested regions)
        _create_egraph_safe(module)
    else:
        EqsatCreateEclassesPass().apply(ctx, module)


def _create_egraph_safe(module: builtin.ModuleOp) -> None:
    """Safe eclass creation that handles nested regions by only wrapping
    top-level single-result ops in the main function.
    """
    from xdsl.dialects.func import FuncOp as XDSLFuncOp

    for fn in module.body.block.ops:
        if not isinstance(fn, XDSLFuncOp):
            continue
        if not fn.body.blocks:
            continue

        block = fn.body.block
        ops_to_wrap = []
        for op in block.ops:
            if isinstance(op, func.ReturnOp):
                continue
            if len(op.results) != 1:
                continue
            # Skip ops with nested regions (matmul body, linalg.generic body)
            if op.regions:
                continue
            ops_to_wrap.append(op)

        from xdsl.rewriter import InsertPoint, Rewriter

        for op in ops_to_wrap:
            result = op.results[0]
            eclass_op = equivalence.ClassOp(result)
            Rewriter.insert_op(eclass_op, InsertPoint.after(op))
            result.replace_uses_with_if(
                eclass_op.results[0],
                lambda u: not isinstance(u.operation, equivalence.AnyClassOp),
            )

        # Also wrap block args
        for arg in block.args:
            eclass_op = equivalence.ClassOp(arg)
            Rewriter.insert_op(eclass_op, InsertPoint.at_start(block))
            arg.replace_uses_with_if(
                eclass_op.results[0],
                lambda u: not isinstance(u.operation, equivalence.AnyClassOp),
            )


def apply_rewrite_rules(
    module: builtin.ModuleOp,
    rules: list[EqSatRewriteRule],
    config: EqSatConfig,
) -> dict[str, int]:
    """Apply Python-based rewrite rules to the e-graph.

    Rules add equivalent alternatives to e-classes without destroying
    existing nodes.  Returns a dict of rule_name → match_count.

    Growth is capped: if the e-node count exceeds 10× the initial count,
    rewriting stops to prevent combinatorial explosion.

    Args:
        module: Module with equivalence.class ops.
        rules: List of EqSatRewriteRule instances.
        config: EqSat configuration.

    Returns:
        Dict mapping rule names to number of successful applications.
    """
    stats: dict[str, int] = {}
    initial_enodes = _count_enodes(module)
    max_enodes = max(initial_enodes * 10, 500)

    for iteration in range(config.max_iterations):
        any_matched = False
        for rule in rules:
            count = rule.apply(module)
            stats[rule.name] = stats.get(rule.name, 0) + count
            if count > 0:
                any_matched = True
            # Check growth cap
            if _count_enodes(module) > max_enodes:
                log.debug("eqsat.growth_cap", iteration=iteration, enodes=_count_enodes(module))
                return stats
        if not any_matched:
            log.debug("eqsat.saturated", iteration=iteration)
            break
    return stats


def assign_costs_and_extract(
    module: builtin.ModuleOp,
    config: EqSatConfig,
) -> None:
    """Assign costs to e-graph nodes and extract the cheapest subprogram.

    Modifies the module in-place: after this call, all equivalence.class
    ops are removed and only the cheapest alternative remains.
    """
    ctx = Context()
    _register_dialects(ctx)

    if config.cost_file:
        EqsatAddCostsPass(
            cost_file=config.cost_file, default=config.default_cost
        ).apply(ctx, module)
    else:
        EqsatAddCostsPass(default=config.default_cost).apply(ctx, module)

    EqsatExtractPass().apply(ctx, module)


def run_eqsat_pass(
    module: builtin.ModuleOp,
    config: EqSatConfig | None = None,
    rules: list[EqSatRewriteRule] | None = None,
    cost_dict: dict[str, int] | None = None,
) -> EqSatResult:
    """Run the full equality saturation pass.

    Pipeline:
        1. Wrap values in equivalence classes
        2. Apply rewrite rules (add alternatives, don't destroy)
        3. Assign costs and extract cheapest subprogram

    Args:
        module: xDSL ModuleOp (Payload IR).
        config: EqSat configuration. Uses defaults if None.
        rules: Python rewrite rules. Uses default algebraic rules if None.
        cost_dict: Optional op_name → cost mapping (overrides config.cost_file).

    Returns:
        EqSatResult with statistics about the pass.
    """
    if config is None:
        config = EqSatConfig()

    if rules is None:
        from compgen.eqsat.rules.algebraic import get_default_algebraic_rules
        rules = get_default_algebraic_rules()

    ops_before = _count_ops(module)
    ir_before = _print_ir(module)

    log.info("eqsat.start", ops=ops_before, rules=len(rules))

    # Step 0a: Classify ops (blackbox vs profitable)
    try:
        from compgen.eqsat.blackbox import classify_module, count_blackbox, count_profitable

        classifications = classify_module(module)
        n_profitable = count_profitable(classifications)
        n_blackbox = count_blackbox(classifications)
        log.info(
            "eqsat.classify",
            profitable=n_profitable,
            blackbox=n_blackbox,
            total=len(classifications),
        )
    except Exception:
        log.warning("eqsat.classify_failed", exc_info=True)
        classifications = None
        n_profitable = ops_before

    # Step 0b: Segment if the module is large
    segments = None
    if n_profitable > config.segment_threshold:
        try:
            from compgen.eqsat.segment import segment_module

            segments = segment_module(module, threshold=config.segment_threshold)
            log.info(
                "eqsat.segmented",
                num_segments=len(segments),
                threshold=config.segment_threshold,
            )
        except Exception:
            log.warning("eqsat.segment_failed", exc_info=True)
            segments = None

    # Step 1: Create e-graph structure
    create_egraph(module)
    eclasses_initial = _count_eclasses(module)

    # Step 2: Apply rewrite rules
    rule_stats = apply_rewrite_rules(module, rules, config)
    eclasses_after_rewrite = _count_eclasses(module)
    enodes_after_rewrite = _count_enodes(module)

    log.info(
        "eqsat.after_rewrite",
        eclasses=eclasses_after_rewrite,
        enodes=enodes_after_rewrite,
        rule_stats=rule_stats,
    )

    # Step 2b: Summarize e-graph (before extraction removes eclasses)
    egraph_summary = None
    try:
        from compgen.eqsat.explain import summarize_egraph

        egraph_summary = summarize_egraph(module)
        log.info(
            "eqsat.summary",
            eclasses=egraph_summary.num_eclasses,
            enodes=egraph_summary.num_enodes,
            ambiguous=egraph_summary.ambiguous_eclasses,
        )
    except Exception:
        log.warning("eqsat.summarize_failed", exc_info=True)

    # Step 3: Assign costs and extract
    if cost_dict is not None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(cost_dict, f)
            cost_file = f.name
        config = EqSatConfig(
            **{
                **{
                    field: getattr(config, field)
                    for field in config.__dataclass_fields__
                },
                "cost_file": cost_file,
            }
        )

    assign_costs_and_extract(module, config)

    ops_after = _count_ops(module)
    ir_after = _print_ir(module)
    changed = ir_before != ir_after

    log.info("eqsat.done", ops_before=ops_before, ops_after=ops_after, changed=changed)

    return EqSatResult(
        ops_before=ops_before,
        ops_after=ops_after,
        eclasses_initial=eclasses_initial,
        eclasses_after_rewrite=eclasses_after_rewrite,
        enodes_after_rewrite=enodes_after_rewrite,
        rule_stats=rule_stats,
        changed=changed,
        summary=egraph_summary,
    )


@dataclass(frozen=True)
class EqSatResult:
    """Result of running the eqsat pass."""

    ops_before: int
    ops_after: int
    eclasses_initial: int
    eclasses_after_rewrite: int
    enodes_after_rewrite: int
    rule_stats: dict[str, int]
    changed: bool
    summary: object | None = None  # EGraphSummary when available

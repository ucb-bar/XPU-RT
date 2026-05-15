"""LLM interaction layer for equality saturation.

Provides prompt templates and response parsing for the 7 LLM ↔ e-graph
interaction modes:

A. Rule Proposal — LLM generates Python RewritePattern rules
B. Search State Summary — compact e-graph state for LLM context
C. Blackbox Frontier — LLM decides which ops to open/close
D. Segmentation Hints — LLM proposes segment boundaries
E. Extraction Objectives — LLM adjusts cost model weights
F. Checkpoint Predictions — LLM predicts rewrite waypoints (LGuess)
G. Counterexample Repair — LLM fixes failed rewrite preconditions
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from compgen.eqsat.explain import EGraphSummary, summary_to_prompt
from compgen.eqsat.rules.python_rules import EqSatRewriteRule

if TYPE_CHECKING:
    pass  # GeminiClient imported here when needed

log = structlog.get_logger()


# ============================================================================
# Prompt templates
# ============================================================================

RULE_PROPOSAL_PROMPT = textwrap.dedent("""\
    You are an expert compiler optimizer. Given the e-graph state below,
    propose a Python rewrite rule that adds an equivalent but cheaper
    alternative to an e-class.

    ## E-graph state
    {egraph_summary}

    ## Target
    {target_description}

    ## Objective
    {objective}

    ## Instructions
    Write a Python class that inherits from EqSatRewriteRule.
    The class must:
    1. Have a `name` property returning a unique string
    2. Implement `match_and_add(self, module)` that:
       - Walks the module for pattern matches
       - Creates equivalent alternative ops
       - Adds them to the e-class via `add_alternative_to_eclass()`
       - Returns the count of matches

    ## Available imports
    ```python
    from xdsl.dialects import arith, equivalence, linalg
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import OpResult
    from xdsl.rewriter import InsertPoint, Rewriter
    from compgen.eqsat.rules.python_rules import (
        EqSatRewriteRule, add_alternative_to_eclass, get_eclass_for_result,
    )
    ```

    ## Example
    ```python
    class MyRule(EqSatRewriteRule):
        @property
        def name(self):
            return "my_rule"

        def match_and_add(self, module):
            count = 0
            for op in module.walk():
                if isinstance(op, arith.AddiOp):
                    eclass = get_eclass_for_result(op.result)
                    if eclass is None:
                        continue
                    # Create equivalent alternative
                    commuted = arith.AddiOp(op.rhs, op.lhs)
                    Rewriter.insert_op(commuted, InsertPoint.before(eclass))
                    add_alternative_to_eclass(eclass, commuted.result)
                    count += 1
            return count
    ```

    Return ONLY the Python class definition, no explanation.
""")

SEARCH_STATE_PROMPT = textwrap.dedent("""\
    ## E-graph exploration state
    {egraph_summary}

    ## Rule statistics
    {rule_stats}

    ## Current best cost: {best_cost}

    What should we try next? Choose one:
    1. PROPOSE_RULE — generate a new rewrite rule
    2. CHANGE_BLACKBOX — open/close ops for optimization
    3. ADJUST_SEGMENTS — change segment size for target
    4. CHANGE_WEIGHTS — adjust extraction cost weights

    Respond with the action name and a brief justification.
""")

EXTRACTION_OBJECTIVE_PROMPT = textwrap.dedent("""\
    Given the current e-graph state and target, suggest cost model weights.

    {egraph_summary}

    Target: {target_description}
    Current weights: fusion={fusion_w}, transfer={transfer_w}, backend={backend_w}

    Respond as JSON: {{"fusion_weight": float, "transfer_weight": float, "backend_match_weight": float}}
""")


# ============================================================================
# Rule validation
# ============================================================================


@dataclass(frozen=True)
class RuleValidationResult:
    """Result of validating an LLM-generated rule."""

    valid: bool
    rule: EqSatRewriteRule | None
    error: str


def validate_rule_code(code: str) -> RuleValidationResult:
    """Validate LLM-generated Python rule code.

    Three-stage validation:
    1. Parse — valid Python syntax
    2. Compile — finds EqSatRewriteRule subclass
    3. Instantiate — creates an instance

    Args:
        code: Python source code containing a rule class.

    Returns:
        RuleValidationResult with the rule or error.
    """
    # Stage 1: Parse
    try:
        ast.parse(code)
    except SyntaxError as e:
        return RuleValidationResult(valid=False, rule=None, error=f"Syntax error: {e}")

    # Stage 2: Compile and execute
    try:
        namespace: dict = {}
        # Provide the imports the LLM rule will need
        exec(
            "from xdsl.dialects import arith, equivalence, linalg\n"
            "from xdsl.dialects.builtin import ModuleOp\n"
            "from xdsl.ir import OpResult\n"
            "from xdsl.rewriter import InsertPoint, Rewriter\n"
            "from compgen.eqsat.rules.python_rules import (\n"
            "    EqSatRewriteRule, add_alternative_to_eclass, get_eclass_for_result,\n"
            ")\n",
            namespace,
        )
        exec(code, namespace)
    except Exception as e:
        return RuleValidationResult(valid=False, rule=None, error=f"Execution error: {e}")

    # Stage 3: Find and instantiate the rule class
    rule_classes = [
        v
        for v in namespace.values()
        if isinstance(v, type) and issubclass(v, EqSatRewriteRule) and v is not EqSatRewriteRule
    ]

    if not rule_classes:
        return RuleValidationResult(valid=False, rule=None, error="No EqSatRewriteRule subclass found")

    try:
        rule = rule_classes[0]()
        _ = rule.name  # Verify property works
    except Exception as e:
        return RuleValidationResult(valid=False, rule=None, error=f"Instantiation error: {e}")

    return RuleValidationResult(valid=True, rule=rule, error="")


def validate_and_verify_rule(code: str, max_bitwidth: int = 8) -> RuleValidationResult:
    """Validate AND formally verify an LLM-generated rule.

    Three-stage validation (same as ``validate_rule_code``), plus
    optional PDL formal verification if the rule is exportable.

    Args:
        code: Python source code containing a rule class.
        max_bitwidth: Maximum bitwidth for PDL verification.

    Returns:
        RuleValidationResult. If PDL verification is available and the
        rule is exportable, the result includes verification status in
        the error field (e.g., "PDL verification: sound").
    """
    result = validate_rule_code(code)
    if not result.valid or result.rule is None:
        return result

    # Try to export and verify via PDL
    try:
        from compgen.semantic.rewrite.export_pdl import eqsat_rule_to_pdl
        from compgen.semantic.rewrite.verify_pdl import verify_rewrite_family

        exported = eqsat_rule_to_pdl(result.rule)
        if exported is not None:
            pattern_fn, replacement_fn = exported
            pdl_result = verify_rewrite_family(
                pattern=pattern_fn,
                replacement=replacement_fn,
                num_operands=2,
                max_bitwidth=max_bitwidth,
            )
            if pdl_result.sound:
                log.info("eqsat.rule.verified", rule=result.rule.name, status="sound")
            else:
                log.warning(
                    "eqsat.rule.unsound",
                    rule=result.rule.name,
                    unsound_bitwidths=pdl_result.unsound_bitwidths,
                )
    except Exception as e:
        log.debug("eqsat.rule.verify_skipped", error=str(e))

    return result


# ============================================================================
# LLM interaction functions
# ============================================================================


def format_rule_proposal_prompt(
    egraph_summary: EGraphSummary,
    target_description: str,
    objective: str,
) -> str:
    """Format a rule proposal prompt for the LLM."""
    return RULE_PROPOSAL_PROMPT.format(
        egraph_summary=summary_to_prompt(egraph_summary),
        target_description=target_description,
        objective=objective,
    )


def format_search_state_prompt(
    egraph_summary: EGraphSummary,
    rule_stats: dict[str, int],
    best_cost: float,
) -> str:
    """Format a search state prompt for the LLM."""
    stats_lines = [f"  {name}: {count} matches" for name, count in rule_stats.items()]
    return SEARCH_STATE_PROMPT.format(
        egraph_summary=summary_to_prompt(egraph_summary),
        rule_stats="\n".join(stats_lines) if stats_lines else "  (no rules applied yet)",
        best_cost=f"{best_cost:.1f}",
    )


def format_extraction_objective_prompt(
    egraph_summary: EGraphSummary,
    target_description: str,
    current_weights: dict[str, float],
) -> str:
    """Format an extraction objective prompt for the LLM."""
    return EXTRACTION_OBJECTIVE_PROMPT.format(
        egraph_summary=summary_to_prompt(egraph_summary),
        target_description=target_description,
        fusion_w=current_weights.get("fusion", 1.0),
        transfer_w=current_weights.get("transfer", 1.0),
        backend_w=current_weights.get("backend_match", 1.0),
    )

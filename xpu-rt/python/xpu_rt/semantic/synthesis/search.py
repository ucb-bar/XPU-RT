"""Search over a bounded guard language."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from compgen.semantic.synthesis.dataset import SynthesisExample
from compgen.semantic.synthesis.guard_lang import (
    BoolN,
    BoolOp,
    Cmp,
    CmpOp,
    Const,
    Expr,
    ModEq,
    Var,
    and_,
    eval_guard,
)


@dataclass(frozen=True)
class GuardSearchConfig:
    """Configuration for bounded guard search."""

    max_fragments: int = 8
    max_candidates: int = 128
    profitable_weight: float = 2.0
    complexity_penalty: float = 0.1


@dataclass(frozen=True)
class GuardSearchResult:
    """Result of searching a family-specific guard."""

    promoted_fragments: tuple[Expr, ...]
    sound_fragments: tuple[Expr, ...] = field(default_factory=tuple)
    precise_unsound_fragments: tuple[Expr, ...] = field(default_factory=tuple)
    repaired_fragments: tuple[Expr, ...] = field(default_factory=tuple)
    promoted_score: float = 0.0


def _bool_vars(examples: Iterable[SynthesisExample]) -> list[str]:
    names: set[str] = set()
    for example in examples:
        for name, value in example.env.items():
            if isinstance(value, bool):
                names.add(name)
    return sorted(names)


def _numeric_vars(examples: Iterable[SynthesisExample]) -> list[str]:
    names: set[str] = set()
    for example in examples:
        for name, value in example.env.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                names.add(name)
    return sorted(names)


def _llm_seed_fragments(
    llm_client: Any,
    positives: list[SynthesisExample],
    negatives: list[SynthesisExample],
) -> list[Expr]:
    """Ask LLM to propose guard expression fragments (Unit 10)."""
    try:
        from compgen.agent.prompts.guard_propose import GUARD_PROPOSE_SCHEMA, GuardProposeContext
        from compgen.agent.prompts.guard_propose import format_prompt as fmt_gp
        from compgen.agent.prompts.guard_propose import parse_response as parse_gp
        from compgen.llm.base import GenerationRequest, LLMConfig

        # Collect variable names/types from examples
        var_names: list[str] = []
        var_types: dict[str, str] = {}
        if positives:
            for key, val in positives[0].env.items():
                var_names.append(key)
                var_types[key] = "bool" if isinstance(val, bool) else "int"

        # Summarize examples
        pos_summary = "\n".join(
            f"  {{{', '.join(f'{k}={v}' for k, v in ex.env.items())}}}"
            for ex in positives[:5]
        )
        neg_summary = "\n".join(
            f"  {{{', '.join(f'{k}={v}' for k, v in ex.env.items())}}}"
            for ex in negatives[:5]
        )

        ctx = GuardProposeContext(
            variable_names=var_names,
            variable_types=var_types,
            positive_examples_summary=pos_summary,
            negative_examples_summary=neg_summary,
            num_positives=len(positives),
            num_negatives=len(negatives),
        )
        prompt = fmt_gp(ctx)
        request = GenerationRequest(
            prompt_template=prompt,
            config=LLMConfig(temperature=0.2, max_tokens=800),
        )
        response = llm_client.generate_structured(request, GUARD_PROPOSE_SCHEMA)
        fragments = parse_gp(response.raw_text)
        if not fragments:
            return []

        # Convert fragment dicts to Expr AST
        exprs: list[Expr] = []
        for frag in fragments:
            var = frag.get("var", "")
            op = frag.get("op", "")
            value = frag.get("value")
            if op == "%" and "divisor" in frag:
                exprs.append(ModEq(Var(var), frag["divisor"], frag.get("remainder", 0)))
            elif op == ">=" and value is not None:
                exprs.append(Cmp(CmpOp.GE, Var(var), Const(value)))
            elif op == "<=" and value is not None:
                exprs.append(Cmp(CmpOp.LE, Var(var), Const(value)))
            elif op == "==" and value is not None:
                exprs.append(Cmp(CmpOp.EQ, Var(var), Const(value)))
            elif op == ">" and value is not None:
                exprs.append(Cmp(CmpOp.GT, Var(var), Const(value)))
            elif op == "<" and value is not None:
                exprs.append(Cmp(CmpOp.LT, Var(var), Const(value)))
        return exprs
    except Exception:
        return []


def _seed_fragments(examples: list[SynthesisExample], cfg: GuardSearchConfig) -> list[Expr]:
    atoms: list[Expr] = []
    positives = [example for example in examples if example.safe]
    if not positives:
        return [Const(False)]

    for name in _bool_vars(positives):
        atoms.append(Cmp(CmpOp.EQ, Var(name), Const(True)))

    for name in _numeric_vars(positives):
        values = [example.env[name] for example in positives]
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        atoms.append(Cmp(CmpOp.GE, Var(name), Const(min_value)))
        atoms.append(Cmp(CmpOp.LE, Var(name), Const(max_value)))
        int_values = [int(value) for value in values if isinstance(value, int)]
        if int_values and len(int_values) == len(values):
            for modulus in (2, 4, 8, 16, 32, 64, 128):
                remainders = {value % modulus for value in int_values}
                if len(remainders) == 1:
                    atoms.append(ModEq(Var(name), modulus, next(iter(remainders))))
    return atoms[: cfg.max_candidates]


def _covers(expr: Expr, examples: Iterable[SynthesisExample]) -> int:
    return sum(1 for example in examples if eval_guard(expr, example.env))


def _is_observed_sound(expr: Expr, negatives: Iterable[SynthesisExample]) -> bool:
    return all(not eval_guard(expr, example.env) for example in negatives)


def _score(expr: Expr, examples: list[SynthesisExample], cfg: GuardSearchConfig) -> float:
    score = 0.0
    for example in examples:
        matched = eval_guard(expr, example.env)
        if not matched:
            continue
        if example.safe and example.profitable:
            score += cfg.profitable_weight
        elif example.safe:
            score += 1.0
        else:
            score -= 1000.0
    complexity = len(expr.terms) if isinstance(expr, BoolN) and expr.op == BoolOp.AND else 1
    return score - (cfg.complexity_penalty * complexity)


def _greedy_repair(
    expr: Expr,
    sound_atoms: list[Expr],
    positives: list[SynthesisExample],
    negatives: list[SynthesisExample],
) -> Expr | None:
    repaired = expr
    remaining_negatives = [example for example in negatives if eval_guard(repaired, example.env)]
    while remaining_negatives:
        best_atom = None
        best_reduction = 0
        for atom in sound_atoms:
            candidate = and_(repaired, atom)
            new_negatives = sum(1 for example in remaining_negatives if eval_guard(candidate, example.env))
            reduction = len(remaining_negatives) - new_negatives
            positive_hits = _covers(candidate, positives)
            if reduction > best_reduction and positive_hits > 0:
                best_atom = atom
                best_reduction = reduction
        if best_atom is None:
            return None
        repaired = and_(repaired, best_atom)
        remaining_negatives = [example for example in negatives if eval_guard(repaired, example.env)]
    return repaired if _covers(repaired, positives) > 0 else None


def search_guard_fragments(
    examples: list[SynthesisExample],
    cfg: GuardSearchConfig | None = None,
    *,
    require_profitable: bool = False,
    llm_client: Any = None,
) -> GuardSearchResult:
    """Search for a conjunction of guard fragments.

    The result is intentionally conservative: promoted fragments are always
    observed-sound on the supplied negatives.
    """

    cfg = cfg or GuardSearchConfig()
    if not examples:
        return GuardSearchResult(promoted_fragments=(Const(False),))

    positives = [
        example for example in examples
        if example.safe and (example.profitable if require_profitable else True)
    ]
    negatives = [
        example for example in examples
        if not example.safe or (require_profitable and example.safe and not example.profitable)
    ]
    if not positives:
        return GuardSearchResult(promoted_fragments=(Const(False),))

    atoms = _seed_fragments(positives, cfg)

    # Merge LLM-proposed guard fragments (Unit 10)
    if llm_client is not None:
        llm_atoms = _llm_seed_fragments(llm_client, positives, negatives)
        # Deduplicate by string representation
        existing_strs = {str(a) for a in atoms}
        for la in llm_atoms:
            if str(la) not in existing_strs:
                atoms.append(la)
                existing_strs.add(str(la))

    sound_atoms = [atom for atom in atoms if _is_observed_sound(atom, negatives) and _covers(atom, positives) > 0]
    precise_unsound = [
        atom for atom in atoms
        if _covers(atom, positives) > 0 and not _is_observed_sound(atom, negatives)
    ]

    repaired_exprs: list[Expr] = []
    for atom in precise_unsound:
        repaired = _greedy_repair(atom, sound_atoms, positives, negatives)
        if repaired is not None:
            repaired_exprs.append(repaired)

    candidates: list[Expr] = repaired_exprs + sound_atoms
    if not candidates:
        return GuardSearchResult(
            promoted_fragments=(Const(False),),
            sound_fragments=tuple(sound_atoms[: cfg.max_fragments]),
            precise_unsound_fragments=tuple(precise_unsound[: cfg.max_fragments]),
            repaired_fragments=tuple(repaired_exprs[: cfg.max_fragments]),
        )

    best = max(candidates, key=lambda expr: _score(expr, examples, cfg))
    if isinstance(best, BoolN) and best.op == BoolOp.AND:
        promoted_fragments = best.terms[: cfg.max_fragments]
    else:
        promoted_fragments = (best,)
    return GuardSearchResult(
        promoted_fragments=tuple(promoted_fragments),
        sound_fragments=tuple(sound_atoms[: cfg.max_fragments]),
        precise_unsound_fragments=tuple(precise_unsound[: cfg.max_fragments]),
        repaired_fragments=tuple(repaired_exprs[: cfg.max_fragments]),
        promoted_score=_score(best, examples, cfg),
    )


__all__ = ["GuardSearchConfig", "GuardSearchResult", "search_guard_fragments"]

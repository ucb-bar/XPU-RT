"""Build the propose-request prompt for KernelBlaster v2.

Pulls the per-target :class:`TargetKnowledgeCard` (ISA slice, intrinsics
relevant to the contract's op family, top-K lessons + strategies for
the current :class:`StateVector`, exemplars) and stitches them with the
contract description into a structured prompt the generator can act on.

The prompt is the same shape for the Claude-Code generator and the
Gemini generator — the generator implementation chooses how to
transport it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.kernels.kernelblaster_v2.contract_state import StateVector
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB, StrategyEntry
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory.target_knowledge import (
    ISAInstruction,
    IntrinsicSignature,
    KernelExemplar,
    Lesson,
    TargetKnowledgeCard,
    iter_lessons,
)


GENERATOR_SYSTEM_PROMPT = """\
You are KernelBlaster v2, an XPU-RT-native kernel-generation agent.
Your job is to emit a single kernel source artifact that satisfies the
attached KernelContract and runs efficiently on the named target.

Constraints:
- Honour every field of the contract (dtypes, layout, shapes, target).
- Prefer intrinsics and idioms documented in the Target Card.
- If the contract is infeasible as stated, propose a typed
  ContractFeedback that explains the smallest change that would make
  it feasible — do not silently widen.
- Use the exemplar kernels as concrete starting points; do not copy
  them verbatim unless the contract is identical.
- Return exactly one JSON object matching the response schema. No
  surrounding prose, no markdown fences.

Lessons and strategies surfaced in the prompt are *advice*, not
prescriptions. When two lessons conflict, pick the one with the higher
measured_gain unless the contract makes it inapplicable.
"""

GENERATOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kernel_code": {"type": "string", "description": "Primary source artifact"},
        "language": {
            "type": "string",
            "description": "'c' | 'cpp' | 'cuda' | 'triton' | 'asm' | …",
        },
        "action": {
            "type": "string",
            "description": "Short tag describing the strategy used (e.g. 'tile-K=64', 'use-mvin-stride').",
        },
        "rationale": {"type": "string"},
        "contract_feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "current_value": {"type": "string"},
                    "suggested_value": {"type": "string"},
                    "reason": {"type": "string"},
                    "kind": {"type": "string"},
                    "applies_when": {"type": "string"},
                },
                "required": ["field"],
            },
        },
    },
    "required": ["kernel_code", "action"],
}


@dataclass(frozen=True)
class PromptBundle:
    """Final, transport-ready prompt payload."""

    system: str
    user: str
    schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBuilder:
    """Stitch contract + card + strategy DB into a :class:`PromptBundle`."""

    card: TargetKnowledgeCard
    strategy_db: StrategyDB | None = None

    # ---- top-level entry point ----

    def build(
        self,
        *,
        contract: KernelContract,
        state: StateVector,
        prior_attempts: tuple[dict[str, Any], ...] = (),
        max_lessons: int = 5,
        max_strategies: int = 5,
        max_exemplars: int = 3,
        mega_contract: Any = None,
    ) -> PromptBundle:
        contract_summary = _summarize_contract(contract, state)
        target_summary = _summarize_target(self.card, state, max_exemplars=max_exemplars)
        lessons_md = _format_lessons(self.card, state, max_lessons=max_lessons)
        strategies_md = ""
        if self.strategy_db is not None:
            strategies_md = _format_strategies(
                self.strategy_db.top_for_state(state, limit=max_strategies)
            )
        prior_md = _format_prior_attempts(prior_attempts)

        user_parts: list[str] = [
            "## Contract\n" + contract_summary,
            "## Target\n" + target_summary,
        ]
        # MEGA-aware branch — when the agentic flow (FusionPlanner +
        # MegaContractEmitter) has fused a chain of single-op kernels
        # into one persistent MEGA contract, surface the chain + the
        # internal_events ordering to the generator so it emits a
        # single fused kernel that keeps intermediates resident in
        # scratchpad. Without this branch, the prompt would describe
        # only the *top* contract and the LLM would have no way to
        # know it's supposed to fuse anything.
        if mega_contract is not None:
            user_parts.append("## MEGA fused chain\n" + _summarize_mega_chain(mega_contract))
        if lessons_md:
            user_parts.append("## Lessons learned (relevant to this state)\n" + lessons_md)
        if strategies_md:
            user_parts.append("## Top strategies for this state\n" + strategies_md)
        if prior_md:
            user_parts.append("## Prior attempts in this run\n" + prior_md)
        if mega_contract is not None:
            user_parts.append(
                "Produce ONE fused kernel that runs the whole chain above with all "
                "intermediates held in scratchpad — no DRAM round-trips between "
                "sub-kernels. Honour the internal_events ordering. Preserve the "
                "vanilla KB signature `launch_gpu_implementation(void *output, "
                "void *input_A, void *input_B, int64_t M, int64_t K, int64_t N)`. "
                "Return the JSON object specified by the schema."
            )
        else:
            user_parts.append(
                "Produce one kernel that satisfies the contract. "
                "Return the JSON object specified by the schema."
            )

        return PromptBundle(
            system=GENERATOR_SYSTEM_PROMPT,
            user="\n\n".join(user_parts),
            schema=GENERATOR_RESPONSE_SCHEMA,
            metadata={
                "state_hash": state.hash(),
                "target_id": state.target_id,
                "op_family": state.op_family,
                "is_mega": mega_contract is not None,
                "mega_body_size": len(getattr(mega_contract, "body", ())) if mega_contract is not None else 0,
            },
        )


# ---------------------------------------------------------------------------
# Section formatters
# ---------------------------------------------------------------------------


def _summarize_contract(contract: KernelContract, state: StateVector) -> str:
    lines = [
        f"- region_id: {contract.region_id or '(unset)'}",
        f"- op_family: {state.op_family} (raw: {contract.op_family or '(unset)'})",
        f"- archetype: {state.archetype}",
        f"- dtypes: {', '.join(contract.dtypes) or '(unset)'} -> dtype_class={state.dtype_class}",
        f"- layout: {contract.layout} -> {state.layout_kind}",
        f"- target: {state.target_id} (hardware_key={contract.hardware_key or '(unset)'})",
        f"- objective: {contract.objective}",
        f"- granularity: {state.granularity}",
        f"- shape_signature: {state.shape_signature or '(empty)'}",
    ]
    if contract.input_shapes:
        lines.append("- input_shapes:")
        for shape in contract.input_shapes:
            lines.append(f"    - {list(shape)}")
    if contract.output_shapes:
        lines.append("- output_shapes:")
        for shape in contract.output_shapes:
            lines.append(f"    - {list(shape)}")
    if contract.provider_hints:
        lines.append("- provider_hints: " + json.dumps(contract.provider_hints))
    if contract.constraints:
        constraints_str = ", ".join(f"{k}=…" for k in contract.constraints)
        lines.append(f"- constraints: ({constraints_str})")
    return "\n".join(lines)


def _summarize_target(
    card: TargetKnowledgeCard,
    state: StateVector,
    *,
    max_exemplars: int,
) -> str:
    spec = card.hardware_spec
    parts: list[str] = [
        f"- target_id: {card.target_id}",
        f"- isa_family: {spec.isa_family}",
    ]
    if spec.dataflow_modes:
        parts.append("- dataflow_modes: " + ", ".join(spec.dataflow_modes))
    if spec.memory_tiers:
        tier_strs = [
            f"{t.name}({t.kind}, {t.size_bytes if t.size_bytes else '?'}B)"
            for t in spec.memory_tiers
        ]
        parts.append("- memory_tiers: " + ", ".join(tier_strs))
    if spec.constraints:
        parts.append("- constraints:")
        for c in spec.constraints:
            parts.append(f"    - {c}")

    # Worked-out sizing rules with concrete numbers. We render these in
    # their own block (not inline with `constraints`) so the LLM can't
    # miss the resolved bounds — symbolic constraints with unresolved
    # identifiers were the failure mode that motivated this field. See
    # ``feedback_target_card_derivation_rules`` for the rationale.
    if spec.derivation_rules:
        parts.append("")
        parts.append("### Sizing constraints (worked out — use these numbers directly)")
        for r in spec.derivation_rules:
            unit = f" {r.unit}" if r.unit else ""
            parts.append(f"  * **{r.name}**: {r.symbolic}")
            parts.append(f"      concrete bound: **{int(r.concrete_value) if r.concrete_value.is_integer() else r.concrete_value}{unit}**")
            if r.derivation:
                parts.append(f"      derivation:     {r.derivation}")
            if r.applies_to:
                parts.append(f"      applies to:     {r.applies_to}")
            if r.how_to_apply:
                parts.append(f"      apply as:       {r.how_to_apply}")

    relevant_instructions = _filter_instructions(spec.instructions, state)
    if relevant_instructions:
        parts.append("- relevant_instructions:")
        for ins in relevant_instructions[:20]:
            funct = f" funct={ins.funct_code}" if ins.funct_code is not None else ""
            parts.append(f"    - {ins.mnemonic}{funct}: {ins.summary or ins.signature}")

    relevant_intrinsics = _filter_intrinsics(spec.intrinsics, state)
    if relevant_intrinsics:
        parts.append("- relevant_intrinsics:")
        for itr in relevant_intrinsics[:12]:
            parts.append(f"    - {itr.name}: {itr.c_signature}")

    exemplars = _select_exemplars(card.exemplars, state, max_exemplars=max_exemplars)
    if exemplars:
        parts.append("- exemplars (paths relative to card.exemplars_dir):")
        for ex in exemplars:
            parts.append(
                f"    - {ex.path} (op_family={ex.op_family}, language={ex.language}, "
                f"tags={list(ex.tags)})"
            )
        full_paths = [str(card.exemplars_dir / ex.path) for ex in exemplars]
        parts.append("- exemplars_full_paths: " + json.dumps(full_paths))
    return "\n".join(parts)


def _format_lessons(card: TargetKnowledgeCard, state: StateVector, *, max_lessons: int) -> str:
    rows: list[Lesson] = [
        l
        for l in iter_lessons(card)
        if l.op_family == state.op_family
        and l.dtype_class == state.dtype_class
        and l.layout_kind == state.layout_kind
        and l.archetype == state.archetype
    ]
    if not rows:
        return ""
    # Most-recent-first, then by speedup.
    rows.sort(key=lambda l: (l.timestamp, l.measured_gain), reverse=True)
    rows = rows[:max_lessons]
    lines: list[str] = []
    for l in rows:
        lines.append(
            f"- [{l.timestamp}] action={l.action} measured_gain={l.measured_gain:.2f}"
            + (f" notes={l.notes!r}" if l.notes else "")
        )
    return "\n".join(lines)


def _format_strategies(rows: list[StrategyEntry]) -> str:
    if not rows:
        return ""
    out: list[str] = []
    for r in rows:
        out.append(
            f"- action={r.action} mean_speedup={r.mean_speedup:.2f} "
            f"confidence={r.confidence:.2f} accepted={r.accepted_count}/{r.sample_count}"
        )
    return "\n".join(out)


def _format_prior_attempts(prior_attempts: tuple[dict[str, Any], ...]) -> str:
    if not prior_attempts:
        return ""
    out: list[str] = []
    for i, attempt in enumerate(prior_attempts):
        action = attempt.get("action", "(unknown)")
        accepted = attempt.get("accepted", False)
        speedup = attempt.get("speedup")
        notes = attempt.get("notes", "")
        line = f"- attempt {i + 1}: action={action} accepted={accepted}"
        if speedup is not None:
            line += f" speedup={speedup:.2f}"
        if notes:
            line += f" notes={notes!r}"
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Filtering heuristics
# ---------------------------------------------------------------------------


# Keywords by op-family that pick the relevant subset of an ISA without
# requiring per-target rules. Conservative — if the heuristic returns
# nothing, we fall back to the full list (capped downstream).
_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "matmul": ("matmul", "mvin", "mvout", "preload", "compute", "loop_ws", "mma", "wmma"),
    "conv": ("conv", "loop_conv", "mvin", "mvout"),
    "reduce": ("vred", "reduce", "vsum", "vmax"),
    "softmax": ("vfdiv", "vfexp", "vred", "softmax"),
    "pointwise": ("vadd", "vsub", "vmul", "vfmadd", "ele"),
    "activation": ("vfmax", "relu", "gelu"),
}


def _filter_instructions(
    instructions: tuple[ISAInstruction, ...],
    state: StateVector,
) -> list[ISAInstruction]:
    if not instructions:
        return []
    keywords = _FAMILY_KEYWORDS.get(state.op_family, ())
    if not keywords:
        return list(instructions)
    needles = tuple(k.lower() for k in keywords)
    matches = [
        i
        for i in instructions
        if any(n in (i.mnemonic + " " + i.summary).lower() for n in needles)
    ]
    return matches or list(instructions)


def _filter_intrinsics(
    intrinsics: tuple[IntrinsicSignature, ...],
    state: StateVector,
) -> list[IntrinsicSignature]:
    if not intrinsics:
        return []
    keywords = _FAMILY_KEYWORDS.get(state.op_family, ())
    if not keywords:
        return list(intrinsics)
    needles = tuple(k.lower() for k in keywords)
    matches = [
        i
        for i in intrinsics
        if any(n in (i.name + " " + i.summary).lower() for n in needles)
    ]
    return matches or list(intrinsics)


def _select_exemplars(
    exemplars: tuple[KernelExemplar, ...],
    state: StateVector,
    *,
    max_exemplars: int,
) -> list[KernelExemplar]:
    if not exemplars:
        return []
    preferred = [e for e in exemplars if e.op_family == state.op_family]
    rest = [e for e in exemplars if e not in preferred]
    chosen = preferred + rest
    return chosen[:max_exemplars]


def _summarize_mega_chain(mega: Any) -> str:
    """Render the chain structure of a MEGA :class:`KernelContractV3`.

    The generator reads this to understand which sub-kernels (body[])
    must run, in what order (internal_events), and which inputs/outputs
    cross the fused boundary. The narrative is deliberately compact so
    it doesn't drown the rest of the prompt — the schema-shaped
    details live in the top-level Contract section.
    """
    if mega is None:
        return ""
    lines: list[str] = []
    lines.append(f"- mega op_name: {mega.op_name}")
    lines.append(f"- archetype: {mega.archetype.value}")
    lines.append(f"- granularity: {mega.granularity.value}")
    lines.append(f"- dispatch_model: {mega.orchestration.dispatch.model.value}")
    lines.append(
        f"- body[] length: {len(mega.body)} sub-kernels — all intermediates stay in scratchpad"
    )
    lines.append("- body chain (producer-first):")
    for i, sub in enumerate(mega.body):
        in_shapes = [list(t.shape.dims) for t in sub.io.inputs]
        out_shapes = [list(t.shape.dims) for t in sub.io.outputs]
        lines.append(
            f"    [{i}] {sub.op_name}  archetype={sub.archetype.value}  "
            f"in_shapes={in_shapes}  out_shapes={out_shapes}"
        )
    if mega.internal_events:
        lines.append("- internal_events (sync barriers):")
        for edge in mega.internal_events:
            lines.append(
                f"    {edge.event_name}: body[{edge.producer_idx}] fires; body[{edge.consumer_idx}] waits"
            )
    if mega.metadata:
        cluster_id = mega.metadata.get("cluster_id")
        est = mega.metadata.get("estimated_speedup")
        if cluster_id:
            lines.append(f"- cluster_id: {cluster_id}")
        if est is not None:
            lines.append(f"- planner's predicted speedup vs unfused: {est:.2f}x")
    return "\n".join(lines)

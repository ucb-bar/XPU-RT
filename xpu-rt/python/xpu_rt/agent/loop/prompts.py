"""Prompt-context helpers for the agentic compilation loop.

These functions derive human-readable summaries from Observation /
TargetProfile / history for prompt assembly.  They were previously
methods on AgenticCompilationLoop but used no instance state, so they
read more cleanly as module-level functions that can be tested in
isolation.
"""

from __future__ import annotations

import json
from typing import Any

from compgen.agent.env import Observation
from compgen.agent.serialize import legal_actions_to_dict, observation_to_dict, observation_to_prompt
from compgen.llm.base import Objective, PromptContext
from compgen.targets.schema import TargetProfile

from compgen.agent.loop.records import IterationRecord


def target_summary(target: TargetProfile) -> str:
    device_parts = []
    for device in target.devices:
        max_mem = max((level.size_bytes for level in device.memory_hierarchy), default=0)
        device_parts.append(f"{device.name}: memory={max_mem}B")
    return f"{target.name}\n" + "\n".join(device_parts)


def frontend_summary(obs: Observation) -> str:
    lines = [
        f"graph_breaks={obs.graph_break_count}",
        f"guards={obs.guard_count}",
        f"unsupported_ops={len(obs.unsupported_ops)}",
    ]
    if obs.unsupported_ops:
        lines.append("targets=" + ", ".join(obs.unsupported_ops[:10]))
    return "\n".join(lines)


def analysis_summary(obs: Observation) -> str:
    dossier = obs.analysis_dossier
    if dossier is None:
        return "analysis unavailable"
    repeated = ", ".join(
        f"{name}:{count}" for name, count in sorted(
            dossier.repeated_patterns.items(), key=lambda item: (-item[1], item[0])
        )[:8]
    ) or "(none)"
    lines = [
        f"regions={dossier.total_regions}",
        f"critical_path={list(dossier.critical_path[:8])}",
        f"dynamic_shapes={list(dossier.dynamic_shape_regions[:8])}",
        f"unsupported_targets={list(dossier.unsupported_targets[:8])}",
        f"repeated_patterns={repeated}",
    ]
    for region in dossier.regions[:8]:
        lines.append(
            f"{region.region_id}: kind={region.kind} ai={region.arithmetic_intensity:.2f} "
            f"backends={list(region.backend_viability)} layouts={list(region.layout_candidates)} "
            f"parallel={list(region.parallelizable_with[:5])}"
        )
    return "\n".join(lines)


def unsupported_summary(obs: Observation) -> str:
    if not obs.unsupported_ops:
        return "no unsupported operators"
    return "\n".join(f"- {target}" for target in obs.unsupported_ops)


def pack_summary(obs: Observation) -> str:
    if not obs.active_packs:
        return "no active extension packs"
    lines = [
        f"active={list(obs.active_packs)}",
        f"sealed_surfaces={list(obs.sealed_surfaces[:12])}",
        f"generation_apertures={list(obs.generation_apertures[:12])}",
        f"available_profilers={list(obs.available_profilers[:12])}",
        f"benchmark_targets={list(obs.pack_benchmark_targets[:12])}",
    ]
    return "\n".join(lines)


def integration_branch_summary(obs: Observation) -> str:
    return obs.integration_branch or "no integration branch"


def frontier_summary(obs: Observation, history: list[IterationRecord]) -> str:
    last_action = history[-1].action_type if history else "none"
    return (
        f"step={obs.step_count} budget_remaining={obs.budget_remaining}\n"
        f"best_latency_us={obs.best_latency_us:.3f}\n"
        f"current_latency_us={obs.estimated_total_latency_us:.3f}\n"
        f"last_action={last_action}"
    )


def verification_summary(obs: Observation) -> str:
    if obs.verification is None:
        return "verification unavailable"
    summary = (
        f"tv_passed={obs.verification.tv_passed} "
        f"tv_failed={obs.verification.tv_failed} "
        f"tv_pending={obs.verification.tv_pending}"
    )
    if obs.verification.last_failure_region:
        summary += (
            f"\nlast_failure={obs.verification.last_failure_region}: "
            f"{obs.verification.last_counterexample_summary}"
        )
    return summary


def legal_actions_summary(legal_actions: list[Any]) -> str:
    entries = []
    for item in legal_actions[:12]:
        delta = f"{item.estimated_cost_delta_us:+.2f}us"
        entries.append(f"{item.rank}. {item.action.action_type} {item.action.region_id} {delta} [{item.risk}]")
    return "\n".join(entries) or "(none)"


def kernel_contracts(obs: Observation) -> list[str]:
    dossier = obs.analysis_dossier
    if dossier is None:
        return []
    contracts: list[str] = []
    for region in dossier.regions[:12]:
        contracts.append(
            f"{region.region_id}: backends={','.join(region.backend_viability)} "
            f"layouts={','.join(region.layout_candidates)} "
            f"local_mem_fit={region.local_memory_fit}"
        )
    return contracts


def backend_viability_summary(obs: Observation) -> list[str]:
    dossier = obs.analysis_dossier
    if dossier is None:
        return []
    seen: list[str] = []
    for region in dossier.regions:
        for backend in region.backend_viability:
            if backend not in seen:
                seen.append(backend)
    return seen


def prior_attempts(history: list[IterationRecord]) -> list[str]:
    return [
        f"iter={record.iteration} action={record.action_type} target={record.target} "
        f"improvement={record.improvement_pct:+.2f}% applied={record.applied}"
        for record in history[-8:]
    ]


def evidence_json(obs: Observation, legal_actions: list[Any]) -> str:
    payload = observation_to_dict(obs)
    payload["legal_actions"] = legal_actions_to_dict(legal_actions[:20])
    return json.dumps(payload, sort_keys=True)


def build_prompt_context(
    obs: Observation,
    target: TargetProfile,
    legal_actions: list[Any],
    history: list[IterationRecord] | None = None,
) -> PromptContext:
    return PromptContext(
        model_ir_summary=observation_to_prompt(obs, legal_actions),
        target_profile_summary=target_summary(target),
        available_transforms=sorted({entry["type"] for entry in legal_actions_to_dict(legal_actions)}),
        kernel_contracts=kernel_contracts(obs),
        objective=Objective.LATENCY,
        prior_attempts=prior_attempts(history or []),
        hardware_feedback=verification_summary(obs),
        frontend_diagnostics_summary=frontend_summary(obs),
        analysis_dossier_summary=analysis_summary(obs),
        unsupported_operator_summary=unsupported_summary(obs),
        pack_summary=pack_summary(obs),
        integration_branch_summary=integration_branch_summary(obs),
        frontier_summary=frontier_summary(obs, history or []),
        legal_action_summary=legal_actions_summary(legal_actions),
        evidence_json=evidence_json(obs, legal_actions),
    )

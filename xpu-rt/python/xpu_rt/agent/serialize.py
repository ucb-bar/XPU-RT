"""Agent-efficient serialization of observations and actions.

Converts Observation/Action objects to/from compact formats optimized
for LLM consumption. NOT human-readable MLIR — structured data that
an LLM can parse and act on efficiently.

Two formats:
    - ``to_prompt()`` — compact text block for LLM system/user prompt
    - ``to_dict()`` — structured dict for JSON/tool-use APIs
"""

from __future__ import annotations

from typing import Any

from compgen.agent.env import (
    Action,
    AssignDeviceAction,
    FuseAction,
    LegalAction,
    NoopAction,
    Observation,
    SetDtypeAction,
    StepResult,
    TileAction,
)


def observation_to_prompt(obs: Observation, legal_actions: list[LegalAction] | None = None) -> str:
    """Serialize observation into a compact prompt block for the LLM.

    Format is optimized for token efficiency, not human readability.
    """
    lines: list[str] = []

    # Header
    lines.append(f"OBJ:{obs.objective} STEP:{obs.step_count}/{obs.step_count + obs.budget_remaining}"
                 f" COST:{obs.estimated_total_latency_us:.1f}us BEST:{obs.best_latency_us:.1f}us")
    lines.append(f"DEVICES:{obs.num_devices} [{','.join(obs.device_names)}]")
    lines.append(f"FLOPS:{obs.total_flops:,} BYTES:{obs.total_bytes:,}")
    if obs.graph_break_count or obs.guard_count or obs.unsupported_ops:
        lines.append(
            f"FRONTEND: breaks={obs.graph_break_count} guards={obs.guard_count} "
            f"unsupported={len(obs.unsupported_ops)}"
        )
    if obs.active_packs:
        lines.append(
            f"PACKS: active={','.join(obs.active_packs)} "
            f"sealed={len(obs.sealed_surfaces)} apertures={len(obs.generation_apertures)} "
            f"profilers={','.join(obs.available_profilers[:5]) or '(none)'}"
        )
        if obs.integration_branch:
            lines.append(f"BRANCH: {obs.integration_branch}")
    lines.append("")

    # Regions (compact table)
    lines.append("REGIONS:")
    for r in obs.regions:
        shapes = "x".join(str(s) for s in r.output_shapes[0]) if r.output_shapes else "?"
        bound = "C" if r.is_compute_bound else "M"  # Compute or Memory bound
        dev = f"D{r.device_index}" if r.device_index >= 0 else "D?"
        lines.append(
            f"  {r.region_id}|{r.op_type}|{shapes}|{r.dtype}|"
            f"{r.flops:,}F|{r.estimated_latency_us:.1f}us|{bound}|{dev}"
        )
    lines.append("")

    # Legal actions (top ranked, if provided)
    if legal_actions:
        lines.append(f"ACTIONS({len(legal_actions)}):")
        for la in legal_actions[:20]:  # show top 20
            a = la.action
            delta = f"{la.estimated_cost_delta_us:+.1f}us"
            if isinstance(a, TileAction):
                tile_str = ",".join(str(t) for t in a.tile_sizes)
                lines.append(f"  #{la.rank} TILE {a.region_id} [{tile_str}] {delta} [{la.risk}]")
            elif isinstance(a, AssignDeviceAction):
                lines.append(f"  #{la.rank} PLACE {a.region_id} D{a.device_index} {delta} [{la.risk}]")
            elif isinstance(a, FuseAction):
                lines.append(f"  #{la.rank} FUSE {a.region_id}+{a.target_region_id} {delta} [{la.risk}]")
            elif isinstance(a, SetDtypeAction):
                lines.append(f"  #{la.rank} DTYPE {a.region_id} {a.dtype} {delta} [{la.risk}]")
            elif isinstance(a, NoopAction):
                lines.append(f"  #{la.rank} NOOP {delta}")
            else:
                lines.append(f"  #{la.rank} {a.action_type} {a.region_id} {delta}")
        lines.append("")

    # Verification summary (if available)
    if obs.verification is not None:
        v = obs.verification
        lines.append(
            f"VERIFY: {v.tv_passed}ok {v.tv_failed}fail {v.tv_pending}pending"
        )
        if v.last_failure_region:
            lines.append(f"  FAIL {v.last_failure_region}: \"{v.last_counterexample_summary}\"")
        if v.verified_facts:
            fact_strs = [
                f"{f.region_id}:{f.kind}({f.detail})" for f in v.verified_facts[:5]
            ]
            lines.append(f"  FACTS: {' '.join(fact_strs)}")
        if v.verifiable_op_types:
            lines.append(f"  VERIFIABLE: {','.join(v.verifiable_op_types[:10])}")
        lines.append("")

    if obs.analysis_dossier is not None:
        dossier = obs.analysis_dossier
        lines.append(
            f"DOSSIER: regions={dossier.total_regions} "
            f"critical_path={','.join(dossier.critical_path[:5])}"
        )
        if dossier.unsupported_targets:
            lines.append(f"  UNSUPPORTED: {','.join(dossier.unsupported_targets[:10])}")
        repeated = sorted(dossier.repeated_patterns.items(), key=lambda item: (-item[1], item[0]))
        if repeated:
            top_patterns = " ".join(f"{name}:{count}" for name, count in repeated[:5])
            lines.append(f"  MOTIFS: {top_patterns}")
        lines.append("")

    # Recent history (last 5 steps)
    if obs.history_summary:
        lines.append("HISTORY:")
        for h in obs.history_summary[-5:]:
            status = "OK" if h.was_applied else "SKIP"
            lines.append(
                f"  S{h.step}|{h.action_type}|{h.action_target}|{status}|"
                f"{h.improvement_pct:+.1f}%|{h.error[:30] if h.error else ''}"
            )

    return "\n".join(lines)


def observation_to_dict(obs: Observation) -> dict[str, Any]:
    """Serialize observation to a structured dict (for JSON/tool-use APIs)."""
    result: dict[str, Any] = {
        "objective": obs.objective,
        "step": obs.step_count,
        "budget_remaining": obs.budget_remaining,
        "cost_us": obs.estimated_total_latency_us,
        "best_cost_us": obs.best_latency_us,
        "total_flops": obs.total_flops,
        "graph_break_count": obs.graph_break_count,
        "guard_count": obs.guard_count,
        "unsupported_ops": list(obs.unsupported_ops),
        "packs": {
            "active": list(obs.active_packs),
            "sealed_surfaces": list(obs.sealed_surfaces),
            "generation_apertures": list(obs.generation_apertures),
            "available_profilers": list(obs.available_profilers),
            "benchmark_targets": list(obs.pack_benchmark_targets),
            "integration_branch": obs.integration_branch,
        },
        "num_devices": obs.num_devices,
        "regions": [
            {
                "id": r.region_id,
                "type": r.op_type,
                "shape": r.output_shapes[0] if r.output_shapes else [],
                "flops": r.flops,
                "latency_us": r.estimated_latency_us,
                "bound": "compute" if r.is_compute_bound else "memory",
                "device": r.device_index,
                "dtype": r.dtype,
            }
            for r in obs.regions
        ],
        "analysis": {
            "critical_path": list(obs.analysis_dossier.critical_path),
            "repeated_patterns": dict(obs.analysis_dossier.repeated_patterns),
            "unsupported_targets": list(obs.analysis_dossier.unsupported_targets),
        } if obs.analysis_dossier is not None else None,
    }

    if obs.verification is not None:
        v = obs.verification
        result["verification"] = {
            "tv_passed": v.tv_passed,
            "tv_failed": v.tv_failed,
            "tv_pending": v.tv_pending,
            "last_failure_region": v.last_failure_region,
            "last_counterexample_summary": v.last_counterexample_summary,
            "verified_facts": [
                {"kind": f.kind, "region_id": f.region_id, "confidence": f.confidence, "detail": f.detail}
                for f in v.verified_facts
            ],
            "verifiable_op_types": list(v.verifiable_op_types),
        }

    return result


def legal_actions_to_dict(actions: list[LegalAction]) -> list[dict[str, Any]]:
    """Serialize legal actions for tool-use API."""
    result = []
    for la in actions:
        a = la.action
        entry: dict[str, Any] = {
            "rank": la.rank,
            "type": a.action_type,
            "region_id": a.region_id,
            "delta_us": la.estimated_cost_delta_us,
            "risk": la.risk,
        }
        if isinstance(a, TileAction):
            entry["tile_sizes"] = list(a.tile_sizes)
        elif isinstance(a, AssignDeviceAction):
            entry["device_index"] = a.device_index
        elif isinstance(a, FuseAction):
            entry["target_region_id"] = a.target_region_id
        elif isinstance(a, SetDtypeAction):
            entry["dtype"] = a.dtype
        result.append(entry)
    return result


def parse_action(action_dict: dict[str, Any]) -> Action:
    """Parse an action from a dict (from LLM tool-use response)."""
    atype = action_dict.get("type", "noop")
    rid = action_dict.get("region_id", "")

    if atype == "tile":
        return TileAction(region_id=rid, tile_sizes=tuple(action_dict.get("tile_sizes", [])))
    elif atype == "assign_device":
        return AssignDeviceAction(region_id=rid, device_index=action_dict.get("device_index", 0))
    elif atype == "fuse":
        return FuseAction(region_id=rid, target_region_id=action_dict.get("target_region_id", ""))
    elif atype == "set_dtype":
        return SetDtypeAction(region_id=rid, dtype=action_dict.get("dtype", "f16"))
    elif atype == "noop":
        return NoopAction()
    else:
        return NoopAction()


def result_to_prompt(result: StepResult) -> str:
    """Serialize a step result into a compact feedback string."""
    info = result.info
    status = "APPLIED" if info.action_applied else "REJECTED"
    verify = "PASS" if info.verification_passed else "FAIL"
    return (
        f"RESULT:{status} VERIFY:{verify} "
        f"COST:{info.cost_before_us:.1f}->{info.cost_after_us:.1f}us "
        f"({info.improvement_pct:+.1f}%) "
        f"REWARD:{result.reward:.4f}"
        f"{' ERR:' + info.error if info.error else ''}"
    )


__all__ = [
    "legal_actions_to_dict",
    "observation_to_dict",
    "observation_to_prompt",
    "parse_action",
    "result_to_prompt",
]

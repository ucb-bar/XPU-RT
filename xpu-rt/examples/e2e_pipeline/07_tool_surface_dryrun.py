"""Phase 7 probe — tool surface dry-run against the prototype catalog.

Enumerates every tool in user_perspective/prototypes/tools/ and every
invent-slot in user_perspective/prototypes/invent_slots/. For each:
  - calls the stub impl with plausible dummy arguments;
  - records a transcript entry (one JSONL line per call);
  - for invent-slots, also exercises the baseline_seed and the gate
    against both a good and a deliberately-bad proposal.

Artifacts:
  artifacts/tool_surface_transcript.jsonl     — one line per call
  artifacts/tool_surface_coverage.md          — human-readable summary

This simulates the recorder contract from
analysis/llm_control_boundaries.md before the real compgen.llm.recorder
extension (repo patch P13) lands.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT.parent))

from user_perspective.prototypes.tools import assemble_registry  # noqa: E402
from user_perspective.prototypes.invent_slots.propose_layout_plan import (  # noqa: E402
    propose_baseline_seed as layout_seed,
    verify_layout_plan,
)
from user_perspective.prototypes.invent_slots.propose_fusion import (  # noqa: E402
    propose_baseline_seed as fusion_seed,
    verify_fusion_plan,
)


def _dummy_for_arg(arg: Any) -> Any:
    """Produce a plausible dummy value for a ToolArg based on its dtype."""
    if arg.default is not None:
        return arg.default
    if arg.enum:
        return arg.enum[0]
    mapping = {
        "string": f"dummy_{arg.name}",
        "integer": 1,
        "number": 0.5,
        "bool": True,
        "region_ref": "region_0",
        "region_ref[]": ["region_0"],
        "target_ref": "dummy_target",
        "plan_ref": "dummy_plan",
        "execution_plan_ref": "dummy_plan",
        "artifact_ref": "dummy_artifact",
        "sample_ref": "dummy_sample",
        "object": {},
        "enum": arg.enum[0] if arg.enum else "default",
        "enum_set": [arg.enum[0]] if arg.enum else ["default"],
    }
    return mapping.get(arg.dtype, f"<dummy:{arg.dtype}>")


def _invoke_tool(tool: Any) -> dict[str, Any]:
    args = {a.name: _dummy_for_arg(a) for a in tool.args}
    start = time.perf_counter()
    if tool.impl is None:
        result = {"status": "no_impl", "echoed_args": args}
    else:
        result = tool.impl(**args)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "tool_name": tool.name,
        "phase": tool.phase,
        "kind": tool.kind,
        "wraps_pass": tool.wraps_pass,
        "autocomp_cost_impact": tool.autocomp_cost_impact,
        "args": args,
        "result": result,
        "elapsed_ms": elapsed_ms,
    }


def _exercise_invent_slot(
    slot: Any,
    kind: str,
    regions: list[dict[str, Any]] | None,
    target_resource: dict[str, Any],
) -> list[dict[str, Any]]:
    """For each invent-slot, exercise baseline seed + gate (good and bad)."""
    entries: list[dict[str, Any]] = []

    if slot.name == "propose_layout_plan" and regions:
        good_proposal = layout_seed(regions[0], target_resource)
        good = verify_layout_plan(good_proposal, target_resource)
        bad_proposal = {
            "chosen": {
                "layout": {"alignment_bytes": 7, "tile": [7]},  # not power of 2
            }
        }
        bad = verify_layout_plan(bad_proposal, target_resource)
        entries.append({"slot": slot.name, "seed_source": good_proposal.get("seed_source"),
                        "good_gate": good, "bad_gate": bad})
    elif slot.name == "propose_fusion" and regions:
        good_proposal = fusion_seed(regions, target_resource)
        good = verify_fusion_plan(
            good_proposal,
            cost_budget={"max_peak_live_bytes": 10**12},
            target_resource=target_resource,
        )
        bad_proposal = {"chosen": {"fusion_spec": {"grouped_regions": [], "target_family": ""}}}
        bad = verify_fusion_plan(
            bad_proposal,
            cost_budget={"max_peak_live_bytes": 10**12},
            target_resource=target_resource,
        )
        entries.append({"slot": slot.name, "seed_source": good_proposal.get("seed_source"),
                        "good_gate": good, "bad_gate": bad})
    else:
        # Other invent-slots: record stub exercise only.
        entries.append({
            "slot": slot.name, "seed_source": "not_exercised_in_stub",
            "good_gate": {"status": "stub_skipped"},
            "bad_gate": {"status": "stub_skipped"},
        })
    return entries


def run(target: str) -> int:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "tool_surface_transcript.jsonl"
    coverage_path = out_dir / "tool_surface_coverage.md"

    target_path = ROOT / "configs" / "targets" / f"{target}.v2.yaml"
    target_doc = yaml.safe_load(target_path.read_text())

    # Load a couple of regions from the smolvla region inventory if present.
    regions: list[dict[str, Any]] | None = None
    inv_path = (ROOT / "artifacts" / "smolvla_slice" / "stage_3_analysis"
                / target / "region_inventory.json")
    if inv_path.exists():
        regions = json.loads(inv_path.read_text())["regions"]

    registry = assemble_registry()

    transcript_lines: list[str] = []
    per_phase_counts: dict[int, dict[str, int]] = {}
    tool_successes = 0
    tool_failures: list[str] = []
    invent_successes = 0
    invent_failures: list[str] = []

    for phase, group in sorted(registry.items()):
        per_phase_counts[phase] = {
            "tools": len(group["tools"]),
            "invent_slots": len(group["invent_slots"]),
        }
        for name, tool in sorted(group["tools"].items()):
            try:
                entry = _invoke_tool(tool)
                entry["success"] = True
                tool_successes += 1
            except Exception as e:   # noqa: BLE001
                entry = {
                    "tool_name": tool.name,
                    "phase": tool.phase,
                    "kind": tool.kind,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                tool_failures.append(name)
            transcript_lines.append(json.dumps(entry, sort_keys=True))

        for name, slot in sorted(group["invent_slots"].items()):
            try:
                slot_entries = _exercise_invent_slot(slot, "invent_slot", regions, target_doc)
                for se in slot_entries:
                    transcript_lines.append(json.dumps({
                        "slot_name": slot.name,
                        "phase": slot.phase,
                        "kind": "invent_slot",
                        "gate": slot.gate,
                        "exercise": se,
                        "success": True,
                    }, sort_keys=True))
                invent_successes += 1
            except Exception as e:   # noqa: BLE001
                transcript_lines.append(json.dumps({
                    "slot_name": slot.name,
                    "phase": slot.phase,
                    "kind": "invent_slot",
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }, sort_keys=True))
                invent_failures.append(name)

    transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    # Coverage summary
    lines = [
        "# Tool-surface dry-run coverage",
        "",
        f"Target resource: `{target}`",
        f"Regions loaded from: `{inv_path.relative_to(ROOT) if inv_path.exists() else 'none'}`",
        "",
        "## Counts",
        "",
        "| Phase | Tools | Invent-slots |",
        "|---|---:|---:|",
    ]
    total_tools = total_slots = 0
    for phase in sorted(per_phase_counts):
        c = per_phase_counts[phase]
        total_tools += c["tools"]
        total_slots += c["invent_slots"]
        lines.append(f"| Phase {phase} | {c['tools']} | {c['invent_slots']} |")
    lines.append(f"| **Total** | **{total_tools}** | **{total_slots}** |")
    lines += ["",
              f"- Tool invocations succeeded: **{tool_successes}/{total_tools}**",
              f"- Invent-slot exercises succeeded: **{invent_successes}/{total_slots}**",
              ""]
    if tool_failures:
        lines.append(f"- Tool failures: `{tool_failures}`")
    if invent_failures:
        lines.append(f"- Invent-slot failures: `{invent_failures}`")
    lines += ["",
              "## Artifacts",
              "",
              f"- transcript: `{transcript_path.relative_to(ROOT)}` ({len(transcript_lines)} lines)",
              f"- coverage: `{coverage_path.relative_to(ROOT)}`",
              ""]
    coverage_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nTool-surface dry-run complete — target={target}")
    print(f"  tools      {tool_successes}/{total_tools} invoked successfully")
    print(f"  invent-slots {invent_successes}/{total_slots} exercised successfully")
    if tool_failures or invent_failures:
        print(f"  FAILURES: tools={tool_failures}  slots={invent_failures}")
    print(f"  -> {transcript_path.relative_to(ROOT)}")
    print(f"  -> {coverage_path.relative_to(ROOT)}")
    return 0 if (not tool_failures and not invent_failures) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="openq_5165rb")
    args = p.parse_args()
    return run(args.target)


if __name__ == "__main__":
    sys.exit(main())

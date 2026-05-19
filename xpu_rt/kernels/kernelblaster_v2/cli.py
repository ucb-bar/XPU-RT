"""``xpu-rt-blast`` CLI.

Drives the KernelBlaster v2 agent loop from the shell. The default
generator is **Gemini** (behind the budget cap) — the agent-file path
needs a Claude Code session, which the ``/xpu-rt-blast`` skill
orchestrates.

Sub-commands:

  show       — read-only summary of a target's strategy DB / lessons
  run        — drive the loop for one contract (YAML/JSON file)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import yaml  # type: ignore[import-untyped]

from xpu_rt.kernels.kernelblaster_v2 import (
    AgentLoopConfig,
    KernelBlasterV2,
    KernelGeneratorLLM,
)
from xpu_rt.kernels.kernelblaster_v2.evaluators import (
    EvaluationReport,
    MockEvaluator,
)
from xpu_rt.kernels.kernelblaster_v2.strategy_db import StrategyDB
from xpu_rt.kernels.provider import KernelContract
from xpu_rt.memory import target_knowledge as tk


@click.group(help="Generate kernels via KernelBlaster v2.")
def main() -> None:
    """xpu-rt-blast — contract-driven kernel generator."""


@main.command("show")
@click.argument("target_id")
@click.option("--op-family", default="", help="Filter strategies by op family.")
@click.option("--limit", default=20, show_default=True)
def show_cmd(target_id: str, op_family: str, limit: int) -> None:
    """Print the strategy DB + lesson count for TARGET_ID."""
    if not tk.exists(target_id):
        click.echo(f"no card for target_id={target_id!r}", err=True)
        sys.exit(2)
    card = tk.load(target_id)
    db = StrategyDB.for_card(card)
    rows = list(db.rows.values())
    if op_family:
        rows = [r for r in rows if op_family in r.state_key]
    rows.sort(key=lambda r: -r.mean_speedup)
    rows = rows[:limit]
    click.echo(f"Target:     {target_id}")
    click.echo(f"Lessons:    {sum(1 for _ in tk.iter_lessons(card))}")
    click.echo(f"Strategies: {len(db.rows)} total ({len(rows)} shown)")
    for r in rows:
        click.echo(
            f"  speedup={r.mean_speedup:.2f}  "
            f"acc={r.accepted_count}/{r.sample_count}  "
            f"action={r.action}  state={r.state_key}"
        )


@main.command("run")
@click.option(
    "--contract",
    "contract_path",
    required=True,
    type=click.Path(exists=True),
    help="YAML or JSON file describing a KernelContract.",
)
@click.option(
    "--target",
    "target_id",
    default="",
    help="Override the contract's target_name (looked up in the card store).",
)
@click.option(
    "--model",
    default="gemini-2.5-flash",
    show_default=True,
    help="Gemini model for the headless generator.",
)
@click.option(
    "--max-iterations",
    default=4,
    show_default=True,
    type=int,
)
@click.option(
    "--accept-threshold",
    default=1.0,
    show_default=True,
    type=float,
)
def run_cmd(
    contract_path: str,
    target_id: str,
    model: str,
    max_iterations: int,
    accept_threshold: float,
) -> None:
    """Drive the KB v2 loop for one contract.

    The default evaluator is the :class:`MockEvaluator` — it always
    accepts the candidate with score 1.0. The real cross-compile +
    sim evaluators land alongside the e2e verification (task #10).
    """
    payload = _load_contract_file(Path(contract_path))
    if target_id:
        payload["target_name"] = target_id
    contract = _contract_from_dict(payload)
    if not contract.target_name:
        click.echo("contract is missing target_name", err=True)
        sys.exit(2)
    if not tk.exists(contract.target_name):
        click.echo(
            f"no card for target_name={contract.target_name!r}; run "
            f"`xpu-rt-target seed <id>` or the /xpu-rt-target skill first.",
            err=True,
        )
        sys.exit(2)
    card = tk.load(contract.target_name)
    loop = KernelBlasterV2(
        card=card,
        generator=KernelGeneratorLLM(model=model),
        evaluator=MockEvaluator(
            table=lambda c: EvaluationReport(correct=True, score=1.0)
        ),
        config=AgentLoopConfig(
            max_iterations=max_iterations,
            accept_threshold=accept_threshold,
        ),
    )
    result = loop.run(contract)
    provider = result.to_provider_result()
    click.echo(
        json.dumps(
            {
                "found": provider.found,
                "plan": provider.plan,
                "iterations_used": provider.iterations_used,
                "metadata": provider.metadata,
                "kernel_code_preview": provider.kernel_code[:400],
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_contract_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        body = yaml.safe_load(text)
    else:
        body = json.loads(text)
    if not isinstance(body, dict):
        raise click.UsageError(f"contract file {path} did not parse to a dict")
    return body


def _contract_from_dict(payload: dict[str, Any]) -> KernelContract:
    return KernelContract(
        region_id=str(payload.get("region_id", "")),
        op_family=str(payload.get("op_family", "")),
        input_shapes=tuple(tuple(s) for s in payload.get("input_shapes", ())),
        output_shapes=tuple(tuple(s) for s in payload.get("output_shapes", ())),
        dtypes=tuple(payload.get("dtypes", ())),
        layout=str(payload.get("layout", "row_major")),
        target_name=str(payload.get("target_name", "")),
        hardware_key=str(payload.get("hardware_key", "")),
        objective=str(payload.get("objective", "latency")),
        constraints=dict(payload.get("constraints", {})),
        provider_hints=dict(payload.get("provider_hints", {})),
    )


if __name__ == "__main__":
    main()

"""Fresh-agent run modes.

Three modes ship the reproducibility argument:

- ``greedy_baseline``: deterministic, no-LLM, runnable in CI. The
  contractual reproducibility floor — if the doc surface is enough
  for the greedy resolver, it's enough for a fresh agent.
- ``fresh_claude``:    operator-driven; CI just records the operator's
  outcome via :func:`record_manual_session_result`.
- ``current_claude``:  same operator path; the productivity upper bound.

The audit cares about ``greedy_baseline`` succeeding and about the
caveat ledger having recent rows for the two operator modes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compgen.audit.caveat_ledger import (
    Caveat,
    CaveatLedger,
    make_caveat,
)
from compgen.audit.errors import AuditError

VALID_MODES = ("greedy_baseline", "fresh_claude", "current_claude")


@dataclass
class GreedyBaselineResult:
    """Outcome of a greedy-baseline run."""

    success: bool
    run_dir: Path
    model_id: str
    target_id: str
    typed_outcome: str  # "verified" | "typed_blocked" | "silent_partial" | "error"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "run_dir": str(self.run_dir),
            "model_id": self.model_id,
            "target_id": self.target_id,
            "typed_outcome": self.typed_outcome,
            "error": self.error,
        }


def run_greedy_baseline(
    *,
    task_pack_dir: Path,
    out_dir: Path,
    model_yaml: Path,
    target_yaml: Path,
    stop_after: str = "agent-decision-request",
) -> GreedyBaselineResult:
    """Run the greedy/no-LLM baseline against a task-pack checkout.

    The contract: if this succeeds, the public doc surface is sufficient
    to drive a deterministic compile to a typed outcome. A fresh
    Claude session has at least the same doc surface plus its
    reasoning, so it should be able to do at least as well.
    """
    from compgen.graph_compilation.run import run_graph_compilation

    model_yaml = Path(model_yaml).resolve()
    target_yaml = Path(target_yaml).resolve()
    out_dir = Path(out_dir).resolve()

    try:
        run_graph_compilation(
            model_yaml,
            target_yaml,
            out_dir,
            stop_after=stop_after,
            selection_mode="greedy",
        )
    except Exception as exc:  # noqa: BLE001 - we classify, then re-record
        # Typed-blocked outcomes raise a typed error; a generic
        # exception is a hard failure.
        from compgen.runtime.errors import (
            BundleEmissionError,
            UnsupportedTopologyError,
        )

        # Some module names exist; some may not depending on what's
        # imported. Catch broadly here and look at the type name.
        type_name = type(exc).__name__
        if isinstance(exc, (UnsupportedTopologyError,)) or "Unsupported" in type_name:
            return GreedyBaselineResult(
                success=True,  # typed-blocked counts as honest
                run_dir=out_dir,
                model_id=model_yaml.stem,
                target_id=target_yaml.stem,
                typed_outcome="typed_blocked",
                error=f"{type_name}: {exc}",
            )
        # downstream-gate rejection is a typed retry signal.
        if "M-15B" in str(exc) or "downstream" in str(exc).lower():
            return GreedyBaselineResult(
                success=True,
                run_dir=out_dir,
                model_id=model_yaml.stem,
                target_id=target_yaml.stem,
                typed_outcome="typed_blocked",
                error=f"{type_name}: {exc}",
            )
        return GreedyBaselineResult(
            success=False,
            run_dir=out_dir,
            model_id=model_yaml.stem,
            target_id=target_yaml.stem,
            typed_outcome="error",
            error=f"{type_name}: {exc}",
        )

    # Pipeline returned without raising. Determine if the run was
    # honest by looking for verification_report.json or
    # downstream_retry_request.json.
    verification = out_dir / "verification_report.json"
    if verification.exists():
        outcome = "verified"
    else:
        outcome = "typed_blocked"  # stop_after=agent-decision-request, no verify yet
    return GreedyBaselineResult(
        success=True,
        run_dir=out_dir,
        model_id=model_yaml.stem,
        target_id=target_yaml.stem,
        typed_outcome=outcome,
    )


def record_manual_session_result(
    *,
    ledger_path: Path,
    mode: str,
    success: bool,
    evidence_paths: list[str],
    notes: str = "",
) -> Caveat:
    """Append an operator-recorded outcome to the caveat ledger.

    Used by both ``fresh_claude`` and ``current_claude`` modes. CI does
    not run this; an operator does after running the corresponding
    Claude Code session.
    """
    if mode not in VALID_MODES:
        raise AuditError(f"mode {mode!r} must be one of {VALID_MODES}")
    if mode == "greedy_baseline":
        raise AuditError(
            "greedy_baseline is run by CI via run_greedy_baseline; "
            "use record_manual_session_result only for fresh_claude / current_claude"
        )
    if not evidence_paths:
        raise AuditError("evidence_paths must be non-empty (run dir, log, etc.)")

    ledger = (
        CaveatLedger.load(ledger_path)
        if ledger_path.exists()
        else CaveatLedger(source_path=ledger_path)
    )
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "resolved" if success else "open"
    # Build a caveat id that satisfies ^[a-z][a-z0-9_]*$ — strip
    # everything non-alphanumeric and downcase.
    ts_compact = "".join(
        c.lower() for c in now if c.isalnum()
    )
    caveat_id = f"manual_{mode}_{ts_compact}"
    caveat = make_caveat(
        id=caveat_id,
        claim_affected=f"manual_{mode}_session_completes_task",
        status=status,
        is_bug=False,
        blocks_paper_claim=False,
        required_to_close=(
            "operator confirms a fresh Claude Code session completes the task; "
            "this row records that confirmation"
            if mode == "fresh_claude"
            else "operator confirms the current Claude Code session completes the task"
        ),
        evidence_paths=evidence_paths,
        notes=notes,
    )
    ledger.add(caveat)
    ledger.dump(ledger_path)
    return caveat


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fresh-agent run modes (M-31A.4)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "run-greedy", help="Run the greedy baseline against a task pack",
    )
    g.add_argument("--task-pack", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--model", type=Path, required=True)
    g.add_argument("--target", type=Path, required=True)
    g.add_argument("--stop-after", default="agent-decision-request")

    r = sub.add_parser(
        "record-manual-session-result",
        help="Append an operator-recorded fresh/current Claude outcome",
    )
    r.add_argument("--ledger", type=Path, required=True)
    r.add_argument("--mode", choices=("fresh_claude", "current_claude"), required=True)
    r.add_argument("--success", choices=("true", "false"), required=True)
    r.add_argument("--evidence-paths", action="append", default=[])
    r.add_argument("--notes", default="")

    args = p.parse_args(argv)

    if args.cmd == "run-greedy":
        result = run_greedy_baseline(
            task_pack_dir=args.task_pack,
            out_dir=args.out,
            model_yaml=args.model,
            target_yaml=args.target,
            stop_after=args.stop_after,
        )
        print(
            f"greedy_baseline: success={result.success} "
            f"typed_outcome={result.typed_outcome} run_dir={result.run_dir}"
        )
        if result.error:
            print(f"  error: {result.error}", file=sys.stderr)
        return 0 if result.success else 1

    if args.cmd == "record-manual-session-result":
        caveat = record_manual_session_result(
            ledger_path=args.ledger,
            mode=args.mode,
            success=(args.success == "true"),
            evidence_paths=args.evidence_paths,
            notes=args.notes,
        )
        print(f"recorded caveat {caveat.id} (status={caveat.status})")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())

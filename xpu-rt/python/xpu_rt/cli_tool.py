"""``compgen tool`` subcommand group.

Generic CLI dispatcher over the ToolCard registry. Every registered
tool is callable through:

::

    compgen tool list
    compgen tool describe <tool_id>
    compgen tool run <tool_id> --input <input.json> --out <out_dir>
    compgen tool audit-promotion

The CLI is the only authoritative T1 surface — :mod:`compgen.tools`
itself enforces input/output schemas, but a tool only counts as T1 in
the maturity audit once it is callable from a shell. The
``compgen-tool`` console script (see ``pyproject.toml``) is a thin
alias for ``compgen tool``.

Hard rules (enforced):

* All output is JSON on stdout, one object per command. No
  free-form text. Failures exit non-zero with a typed JSON error
  payload so scripts (and the fresh-agent grader) can parse
  outcomes without screen-scraping.
* The CLI never accepts a request inline as flags. Requests come
  from a JSON file (``--input``) or stdin (``--stdin``). This keeps
  inputs hashable and replayable, which is what grades against.
* Maturity is *informational* in the CLI — ``compgen tool run`` will
  happily execute a T0 tool. The audit is what blocks promotion
  past a tool's evidence ceiling.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import click

from compgen.tools.errors import (
    ToolCardError,
    ToolEntrypointError,
    ToolInputSchemaError,
    ToolOutputSchemaError,
    ToolRunError,
)
from compgen.tools.tool_card import ToolCard
from compgen.tools.tool_registry import iter_tool_cards
from compgen.tools.tool_runner import ToolRunner


def _emit_json(payload: dict[str, Any]) -> None:
    """Write ``payload`` as canonical JSON to stdout + newline."""

    click.echo(json.dumps(payload, sort_keys=True, indent=2))


def _emit_error(exit_code: int, error_type: str, **fields: Any) -> click.exceptions.Exit:
    """Emit a typed error payload and return a non-zero ``ClickExit``.

    Callers should ``raise`` the returned exception so Click handles
    the exit code without printing usage.
    """

    _emit_json({"status": "error", "error_type": error_type, **fields})
    return click.exceptions.Exit(exit_code)


def _load_cards(root: Path | None) -> dict[str, ToolCard]:
    """Load every card under ``root`` (or the shipped cards dir).

    A malformed card raises immediately — hard rule. The CLI maps that
    to exit code 4 (registry violation) so CI can pin on it.
    """

    cards: dict[str, ToolCard] = {}
    for card in iter_tool_cards(root):
        if card.tool_id in cards:
            raise ToolCardError(
                f"duplicate tool_id {card.tool_id!r} in registry "
                f"(check cards under {root or '<default>'})"
            )
        cards[card.tool_id] = card
    return cards


@click.group("tool")
@click.option(
    "--cards-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Override the cards directory (defaults to "
    "python/compgen/tools/cards/). Useful for testing or driving "
    "user-extension cards.",
)
@click.pass_context
def tool(ctx: click.Context, cards_root: Path | None) -> None:
    """Run, list, and inspect ToolCard-declared CompGen tools."""

    ctx.ensure_object(dict)
    ctx.obj["cards_root"] = cards_root


@tool.command("list")
@click.option(
    "--phase",
    type=str,
    default=None,
    help="Filter by tool phase (env_probe, graph_analysis, ...).",
)
@click.option(
    "--maturity",
    type=str,
    default=None,
    help="Filter by minimum maturity rung (e.g. ``T3`` returns T3+).",
)
@click.pass_obj
def tool_list(obj: dict[str, Any], phase: str | None, maturity: str | None) -> None:
    """List every registered ToolCard as a JSON array."""

    try:
        cards = _load_cards(obj.get("cards_root"))
    except ToolCardError as exc:
        raise _emit_error(4, "registry_violation", message=str(exc)) from exc

    rows: list[dict[str, Any]] = []
    min_idx = 0
    if maturity is not None:
        from compgen.tools.tool_card import MATURITY_LEVELS

        if maturity not in MATURITY_LEVELS:
            raise _emit_error(
                2,
                "invalid_filter",
                message=f"unknown maturity {maturity!r}; must be one of {list(MATURITY_LEVELS)}",
            )
        min_idx = MATURITY_LEVELS.index(maturity)

    for card in sorted(cards.values(), key=lambda c: c.tool_id):
        if phase is not None and card.phase != phase:
            continue
        if card.maturity_index < min_idx:
            continue
        rows.append(
            {
                "tool_id": card.tool_id,
                "maturity": card.maturity,
                "phase": card.phase,
                "entrypoints": {
                    "python": card.entrypoints.python,
                    "cli": card.entrypoints.cli,
                    "mcp": card.entrypoints.mcp,
                },
                "description": card.description.strip().splitlines()[0] if card.description else "",
            }
        )

    _emit_json({"status": "ok", "count": len(rows), "tools": rows})


@tool.command("describe")
@click.argument("tool_id", type=str)
@click.pass_obj
def tool_describe(obj: dict[str, Any], tool_id: str) -> None:
    """Print the full ToolCard body (JSON) for ``tool_id``."""

    try:
        cards = _load_cards(obj.get("cards_root"))
    except ToolCardError as exc:
        raise _emit_error(4, "registry_violation", message=str(exc)) from exc
    card = cards.get(tool_id)
    if card is None:
        raise _emit_error(
            3,
            "unknown_tool_id",
            tool_id=tool_id,
            available=sorted(cards),
        )
    _emit_json({"status": "ok", "tool": card.to_dict()})


@tool.command("run")
@click.argument("tool_id", type=str)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON request file. Mutually exclusive with --stdin.",
)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    default=False,
    help="Read the JSON request from stdin instead of --input.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where result.json + trace.jsonl + tool artifacts are written.",
)
@click.pass_obj
def tool_run(
    obj: dict[str, Any],
    tool_id: str,
    input_path: Path | None,
    use_stdin: bool,
    out_dir: Path,
) -> None:
    """Execute a tool by id with the request loaded from --input or --stdin.

    Exit codes:

    ====  =====================================
    0     status=ok or status=blocked
    1     status=error (tool reported error)
    2     CLI argument problem
    3     unknown tool_id
    4     registry / card schema violation
    5     input did not validate against input_schema
    6     output did not validate against output_schema
    7     entrypoint missing / not callable
    8     entrypoint raised mid-execution
    ====  =====================================
    """

    if input_path is None and not use_stdin:
        raise _emit_error(
            2, "missing_request", message="one of --input PATH or --stdin is required"
        )
    if input_path is not None and use_stdin:
        raise _emit_error(
            2, "ambiguous_request", message="--input and --stdin are mutually exclusive"
        )

    try:
        if use_stdin:
            raw = sys.stdin.read()
        else:
            raw = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _emit_error(2, "input_io_error", message=str(exc)) from exc
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _emit_error(2, "input_json_decode_error", message=str(exc)) from exc
    if not isinstance(request, dict):
        raise _emit_error(
            2,
            "input_must_be_object",
            message=f"request must be a JSON object; got {type(request).__name__}",
        )

    try:
        cards = _load_cards(obj.get("cards_root"))
    except ToolCardError as exc:
        raise _emit_error(4, "registry_violation", message=str(exc)) from exc

    card = cards.get(tool_id)
    if card is None:
        raise _emit_error(
            3,
            "unknown_tool_id",
            tool_id=tool_id,
            available=sorted(cards),
        )

    try:
        result = ToolRunner().run(card, request=request, out_dir=out_dir)
    except ToolInputSchemaError as exc:
        raise _emit_error(5, "input_schema_violation", tool_id=tool_id, message=str(exc)) from exc
    except ToolOutputSchemaError as exc:
        raise _emit_error(6, "output_schema_violation", tool_id=tool_id, message=str(exc)) from exc
    except ToolEntrypointError as exc:
        raise _emit_error(7, "entrypoint_error", tool_id=tool_id, message=str(exc)) from exc
    except ToolRunError as exc:
        raise _emit_error(
            8,
            "entrypoint_raised",
            tool_id=tool_id,
            message=str(exc),
            traceback=traceback.format_exception_only(type(exc), exc)[-1].strip(),
        ) from exc

    payload = result.to_dict()
    _emit_json(payload)
    # ``status=error`` from the tool itself (not a runner-side schema
    # failure) exits 1 so a CI gate can short-circuit.
    if result.status == "error":
        raise click.exceptions.Exit(1)


@tool.command("audit-promotion")
@click.pass_obj
def tool_audit_promotion(obj: dict[str, Any]) -> None:
    """Run the T0→T7 promotion audit over every registered tool.

    This subcommand is a thin pass-through to
    ``compgen.audit.tool_promotion``. Until lands, it
    emits ``status=not_implemented`` so the CLI surface is stable but
    honestly says nothing is checked yet.
    """

    try:
        from compgen.audit.tool_promotion import run_tool_promotion_audit
    except ImportError:
        _emit_json(
            {
                "status": "not_implemented",
                "message": (
                    "compgen.audit.tool_promotion (M-92) is not yet "
                    "implemented; this command is a stable surface that "
                    "will dispatch to it when it lands."
                ),
            }
        )
        return

    cards = _load_cards(obj.get("cards_root"))
    report = run_tool_promotion_audit(cards=list(cards.values()))  # type: ignore[name-defined]
    _emit_json(report.to_dict())
    if report.violations:
        raise click.exceptions.Exit(1)


__all__ = ["tool"]

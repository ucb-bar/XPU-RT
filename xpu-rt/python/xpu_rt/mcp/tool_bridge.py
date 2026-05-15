"""MCP bridge for ToolCard-declared tools.

Every :class:`compgen.tools.ToolCard` whose ``entrypoints.mcp`` is
non-empty is registered as an MCP tool through this bridge. The
bridge is intentionally *thin*: it takes the card, builds the
:class:`compgen.tools.ToolRunner`-backed handler, and produces the
exact ``{name, description, phase, handler, input_schema}`` dict
shape the existing :mod:`compgen.mcp.tools` package consumes.

Hard rules (enforced by the bridge + checked by the T5 gate):

1. **No unique business logic in the MCP layer.** The bridge handler
   calls ``ToolRunner.run`` and returns its result. Any computation
   beyond that belongs in the Python entrypoint declared on the card.

2. **MCP input_schema is bit-equal to the ToolCard input_schema.**
   The MCP tool surfaces the same contract the CLI runner enforces;
   if they ever diverge the audit fails. This is the schema-equivalence
   gate the plan calls for.

3. **out_dir is bridge-managed.** MCP callers do not see ``out_dir``
   in the input_schema. The bridge creates a unique
   ``.compgen/mcp_tool_runs/<tool_id>/<timestamp>/`` directory per
   invocation so result + trace are durable and replayable but the
   caller never has to think about paths.

4. **Bridge handlers must not raise.** Every exception is translated
   to a typed ``status=error`` MCP response so the model sees a
   structured payload rather than an SDK-level crash.
"""

from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from compgen.tools.errors import (
    ToolEntrypointError,
    ToolInputSchemaError,
    ToolOutputSchemaError,
    ToolRunError,
)
from compgen.tools.tool_card import ToolCard
from compgen.tools.tool_registry import iter_tool_cards
from compgen.tools.tool_runner import ToolRunner

# Root for per-invocation out_dirs created by the bridge. Lives under
# .compgen/ so it is gitignored (the repo's .gitignore already covers
# .compgen/) and shared across the user's session.
MCP_TOOL_RUNS_ROOT = Path(".compgen") / "mcp_tool_runs"


def _bridge_out_dir(tool_id: str) -> Path:
    """Return a unique out_dir for one MCP invocation of ``tool_id``.

    The path includes a millisecond timestamp so repeated calls do not
    collide and so the audit can replay them in order.
    """

    stamp = f"{int(time.time() * 1000)}"
    return MCP_TOOL_RUNS_ROOT / tool_id / stamp


def _make_handler(card: ToolCard) -> Callable[..., dict[str, Any]]:
    """Build the MCP handler ``(sm, **kwargs) -> dict`` for ``card``.

    The handler closes over the card so the bridge does not need a
    runtime dispatch table — Python's regular import-time binding does
    the job. The ``sm`` argument is accepted for signature
    compatibility with the rest of :mod:`compgen.mcp.tools` but the
    bridge does not consume it; ToolCard tools are stateless w.r.t.
    the MCP session.
    """

    def handler(sm: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG001
        # ``out_dir`` is a bridge-owned escape hatch: callers normally
        # do not supply it, but tests / replay grading can pin it.
        explicit = kwargs.pop("out_dir", None)
        out_dir = Path(explicit) if explicit is not None else _bridge_out_dir(card.tool_id)
        try:
            result = ToolRunner().run(card, request=kwargs, out_dir=out_dir)
        except ToolInputSchemaError as exc:
            return {
                "status": "error",
                "error_type": "input_schema_violation",
                "tool_id": card.tool_id,
                "message": str(exc),
            }
        except ToolOutputSchemaError as exc:
            return {
                "status": "error",
                "error_type": "output_schema_violation",
                "tool_id": card.tool_id,
                "message": str(exc),
            }
        except ToolEntrypointError as exc:
            return {
                "status": "error",
                "error_type": "entrypoint_error",
                "tool_id": card.tool_id,
                "message": str(exc),
            }
        except ToolRunError as exc:
            return {
                "status": "error",
                "error_type": "entrypoint_raised",
                "tool_id": card.tool_id,
                "message": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            # Hard rule 4: never crash the MCP transport.
            return {
                "status": "error",
                "error_type": "bridge_internal_error",
                "tool_id": card.tool_id,
                "message": str(exc),
                "traceback": traceback.format_exception_only(type(exc), exc)[-1].strip(),
            }
        return result.to_dict()

    handler.__name__ = f"compgen_tool_bridge__{card.tool_id}"
    handler.__doc__ = card.description.strip().splitlines()[0] if card.description else card.tool_id
    return handler


def _phase_for_mcp(card: ToolCard) -> str:
    """Translate the ToolCard phase to the MCP phase taxonomy.

    The existing MCP package uses ``lifecycle | inspect | transform |
    job``; the ToolCard phase taxonomy uses
    ``env_probe | graph_analysis | recipe_authoring | kernel_codegen |
    extension_authoring | evidence``. The bridge maps the latter to
    the closest concept in the former so the existing
    ``list_phase_tools`` grouping is preserved.
    """

    return {
        "env_probe": "inspect",
        "graph_analysis": "inspect",
        "recipe_authoring": "transform",
        "kernel_codegen": "transform",
        "extension_authoring": "transform",
        "evidence": "inspect",
    }.get(card.phase, "inspect")


def make_mcp_tool_dict(card: ToolCard) -> dict[str, Any]:
    """Produce the MCP tool dict for one ToolCard.

    The returned dict is shaped to match the existing
    ``_IN_TREE_TOOLS`` aggregation in
    :mod:`compgen.mcp.tools.__init__`.

    Raises
    ------
    ValueError
        If the card's ``entrypoints.mcp`` is empty — the bridge is only
        for cards that declare an MCP surface.
    """

    if not card.entrypoints.mcp:
        raise ValueError(
            f"tool {card.tool_id!r} declares no entrypoints.mcp; "
            f"the bridge is only for T5+ cards"
        )
    return {
        "name": card.entrypoints.mcp,
        "description": card.description.strip().splitlines()[0]
        if card.description
        else card.tool_id,
        "phase": _phase_for_mcp(card),
        "handler": _make_handler(card),
        # Deep-copy so a downstream mutation of the MCP-registered schema
        # cannot retroactively affect the ToolCard's input_schema.
        "input_schema": copy.deepcopy(card.input_schema),
        # Provenance — useful for the evidence pack so the cross-
        # surface matrix can mark which MCP tools came from cards.
        "_card_tool_id": card.tool_id,
    }


def bridge_tools(cards_root: Path | None = None) -> list[dict[str, Any]]:
    """Discover every ToolCard with an MCP entrypoint and return its
    MCP tool dict. Imported by :mod:`compgen.mcp.tools.__init__` at
    aggregation time."""

    out: list[dict[str, Any]] = []
    for card in iter_tool_cards(cards_root):
        if not card.entrypoints.mcp:
            continue
        out.append(make_mcp_tool_dict(card))
    return out


__all__ = [
    "MCP_TOOL_RUNS_ROOT",
    "bridge_tools",
    "make_mcp_tool_dict",
]

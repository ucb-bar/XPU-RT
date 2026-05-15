"""Pass-tool registry + invocation.

Discovers ``PassToolCard`` YAML files under
``python/compgen/pass_tools/cards/`` (plus any registered from
user extensions), resolves their ``entrypoint`` strings, and
exposes ``apply_pass_tool`` for typed invocation.

The pass-tool function signature is::

    def run(*, contract, **kwargs) -> PassToolResult: ...

Pass tools never mutate Payload IR directly. The registry
enforces this at result-validation time: if a ``PassToolResult``'s
``recipe_delta`` is empty when status is ``proposal``, or if
``status`` is outside the typed enum, the call raises
:class:`PassToolResultError`.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from compgen.pass_tools.pass_tool_result import (
    PassToolResult,
    PassToolResultError,
)
from compgen.pass_tools.pass_tool_types import (
    PassToolCard,
    PassToolCardError,
)


def _cards_root() -> Path:
    return Path(__file__).resolve().parent / "cards"


class PassToolRegistryError(RuntimeError):
    """Raised on registry-level failures."""

    def __init__(self, *, tool_id: str, reason: str, detail: str = "") -> None:
        self.tool_id = tool_id
        self.reason = reason
        self.detail = detail
        super().__init__(
            f"pass_tool_registry_error: tool_id={tool_id!r} reason={reason!r}"
            + (f" detail={detail!r}" if detail else "")
        )


@dataclass
class PassToolRegistry:
    """In-memory registry of pass-tool cards."""

    cards: dict[str, PassToolCard]

    def tool_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.cards.keys()))

    def card_for(self, tool_id: str) -> PassToolCard:
        try:
            return self.cards[tool_id]
        except KeyError as exc:
            raise PassToolRegistryError(
                tool_id=tool_id, reason="unknown_tool_id"
            ) from exc


def iter_pass_tool_cards(root: Path | None = None) -> Iterator[PassToolCard]:
    base = root or _cards_root()
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise PassToolCardError(
                f"pass-tool card {path} must be a YAML mapping"
            )
        yield PassToolCard.from_dict(body, source=path)


def build_pass_tool_registry(
    root: Path | None = None,
    *,
    extra_cards: tuple[PassToolCard, ...] = (),
) -> PassToolRegistry:
    """Discover shipped pass-tool cards under ``root`` and merge in
    ``extra_cards`` from user extensions."""

    cards: dict[str, PassToolCard] = {}
    for card in iter_pass_tool_cards(root):
        cards[card.tool_id] = card
    for card in extra_cards:
        if card.tool_id in cards:
            raise PassToolRegistryError(
                tool_id=card.tool_id, reason="duplicate_tool_id"
            )
        cards[card.tool_id] = card
    return PassToolRegistry(cards=cards)


def _split_entrypoint(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise PassToolRegistryError(
            tool_id="<unknown>", reason="bad_entrypoint_syntax", detail=spec
        )
    module_path, _, symbol = spec.partition(":")
    if not module_path or not symbol:
        raise PassToolRegistryError(
            tool_id="<unknown>", reason="bad_entrypoint_syntax", detail=spec
        )
    return module_path, symbol


def resolve_entrypoint(card: PassToolCard) -> Callable[..., PassToolResult]:
    """Resolve ``card.entrypoint`` to its callable.

    Raises :class:`PassToolRegistryError` with a typed reason on
    failure — never a raw ImportError or AttributeError.
    """

    try:
        module_path, symbol = _split_entrypoint(card.entrypoint)
    except PassToolRegistryError as exc:
        raise PassToolRegistryError(
            tool_id=card.tool_id,
            reason="bad_entrypoint_syntax",
            detail=exc.detail,
        ) from exc

    try:
        mod = importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError) as exc:
        raise PassToolRegistryError(
            tool_id=card.tool_id,
            reason="module_not_importable",
            detail=module_path,
        ) from exc

    try:
        fn = getattr(mod, symbol)
    except AttributeError as exc:
        raise PassToolRegistryError(
            tool_id=card.tool_id,
            reason="symbol_not_in_module",
            detail=symbol,
        ) from exc

    if not callable(fn):
        raise PassToolRegistryError(
            tool_id=card.tool_id,
            reason="entrypoint_not_callable",
            detail=type(fn).__name__,
        )
    return fn


def apply_pass_tool(
    registry: PassToolRegistry,
    tool_id: str,
    /,
    **kwargs: Any,
) -> PassToolResult:
    """Invoke the named pass tool and validate its typed result.

    The pass-tool function must accept keyword arguments and
    return a :class:`PassToolResult`. The registry validates that:

    * the returned object is a ``PassToolResult``;
    * its ``tool_id`` matches the registry entry;
    * the typed status / recipe_delta discipline holds (already
      enforced by ``PassToolResult.__post_init__``);
    * every op in ``recipe_delta`` is in the card's
      ``allowed_recipe_ops`` set (if the set is non-empty).

    Raises :class:`PassToolResultError` on any violation — pass-
    tool authors cannot smuggle out-of-bound Recipe-IR ops.
    """

    card = registry.card_for(tool_id)
    fn = resolve_entrypoint(card)
    result = fn(**kwargs)

    if not isinstance(result, PassToolResult):
        raise PassToolResultError(
            f"pass tool {tool_id!r} returned {type(result).__name__}, "
            f"expected PassToolResult"
        )
    if result.tool_id != tool_id:
        raise PassToolResultError(
            f"pass tool {tool_id!r} returned result.tool_id="
            f"{result.tool_id!r}; tool_id mismatch"
        )

    if card.allowed_recipe_ops and result.status == "proposal":
        allowed = set(card.allowed_recipe_ops)
        for op in result.recipe_delta:
            if op["op"] not in allowed:
                raise PassToolResultError(
                    f"pass tool {tool_id!r} emitted recipe op "
                    f"{op['op']!r} not in card.allowed_recipe_ops="
                    f"{sorted(allowed)}"
                )
    return result

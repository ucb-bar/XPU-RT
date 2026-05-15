"""ToolCard discovery + loading.

Cards live under ``python/xpu_rt/tools/cards/*.yaml``. Loading
mirrors :func:`xpu_rt.providers.card_loader.iter_provider_cards`
so the audit and CLI runner can iterate over all known tools without
caring about how cards were authored.

User extensions contribute cards through the extension manifest
; those cards flow through
:mod:`xpu_rt.extensions.registry` rather than this loader, but the
:class:`xpu_rt.tools.ToolCard` schema is shared.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from xpu_rt.tools.errors import ToolCardError
from xpu_rt.tools.tool_card import ToolCard


def tool_cards_root() -> Path:
    """Return the in-repo default cards directory.

    Kept as a function (not a module-level constant) so monkeypatched
    test fixtures can resolve a clean path without import-time
    side-effects.
    """

    return Path(__file__).resolve().parent / "cards"


def _load_yaml(path: Path) -> dict[str, Any]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ToolCardError(
            f"tool card {path} must be a YAML mapping; "
            f"got {type(body).__name__}"
        )
    return body


def load_tool_card(path: Path) -> ToolCard:
    """Load a single ToolCard YAML file.

    Raises :class:`xpu_rt.tools.errors.ToolCardError` on schema
    violation. The file's parent is recorded as the card's source for
    error messages.
    """

    return ToolCard.from_dict(_load_yaml(path), source=path)


def iter_tool_cards(root: Path | None = None) -> Iterator[ToolCard]:
    """Iterate over every ToolCard found under ``root``.

    Order is deterministic (alphabetical by filename) so audits and
    evidence packs produce byte-stable output.

    Cards that fail schema validation raise immediately — there is no
    silent skip. A malformed card is by definition unaudited and
    must not be hidden.
    """

    base = root or tool_cards_root()
    if not base.is_dir():
        return
    for path in sorted(base.glob("*.yaml")):
        yield load_tool_card(path)

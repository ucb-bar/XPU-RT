"""Built-in tool implementations shipped with CompGen.

Each module in this package implements one tool's Python entrypoint
(``run(request, *, out_dir) -> dict``) referenced from the matching
YAML card under ``python/compgen/tools/cards/``.

The first inhabitant — ``echo`` — exists to keep the runner exercised
in tests without coupling to any heavy compiler subsystem. Real
hot-path tools (provider probe, agent-decision request/commit,
extension authoring) land through →.
"""

from __future__ import annotations

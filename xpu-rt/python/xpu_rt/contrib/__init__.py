"""Local-extension → upstream-PR workflow.

A user with a productive extension under ``~/.compgen/extensions/`` can
run ``compgen contrib draft --slot my_fusion`` to:

1. Create a ``contrib/<slot>`` branch.
2. Copy the extension file into
   ``python/compgen/agent/invent_slots/contrib/<slot>.py``.
3. Synthesise a regression test from the ``_state.json`` invocation
   log (see :mod:`compgen.agent.extensions`).
4. Run pytest on the new test.
5. Commit locally and print a ``gh pr create`` command. The contrib
   module NEVER pushes to a remote.

Public API: :func:`draft_pr`, :func:`list_extensions`,
:func:`status`.
"""

from __future__ import annotations

from compgen.contrib.draft import (
    ContribDraftResult,
    ContribExtension,
    draft_pr,
    list_extensions,
    status,
)

__all__ = [
    "ContribDraftResult",
    "ContribExtension",
    "draft_pr",
    "list_extensions",
    "status",
]

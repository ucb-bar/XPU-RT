"""Subprocess bootstrap (follow-up).

When a CompGen adapter spawns a subprocess (e.g. KernelBlaster's
``run_RL.py``), prepending the parent directory of this file onto
``PYTHONPATH`` causes the child Python interpreter to auto-import
this ``sitecustomize`` module at startup, which then installs the
``google.genai`` instrumentation in the subprocess. That way the
spawned process's Gemini API calls flow into the same
``.compgen/gemini_usage/events.jsonl`` as the parent's.

Disable via ``COMPGEN_DISABLE_GEMINI_INSTRUMENTATION=1``. Best-
effort: never raises (a failing bootstrap must not crash the
spawned process).
"""

from __future__ import annotations

import os


def _install_compgen_instrumentation_in_subprocess() -> None:
    if os.environ.get("COMPGEN_DISABLE_GEMINI_INSTRUMENTATION", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        from compgen.observability.gemini_usage import (
            install_genai_instrumentation,
            install_openai_instrumentation,
        )

        install_genai_instrumentation()
        install_openai_instrumentation()
    except Exception:  # noqa: BLE001
        pass


_install_compgen_instrumentation_in_subprocess()

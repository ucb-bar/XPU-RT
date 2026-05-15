"""Reference ToolCard implementation — ``echo``.

A minimal, real entrypoint that keeps :class:`compgen.tools.tool_runner.ToolRunner`
exercised end-to-end:

* takes a ``text`` field (and optional ``count`` repetitions);
* writes a real artifact under ``out_dir``;
* returns a typed status, the written line, and the artifact path.

This is *not* a stub. It does real I/O and is used as the
deterministic fixture for the unit tests + the promotion
audit's "every registered tool runs" suite.

Negative-control tools (``echo_returns_bad_status``,
``echo_writes_outside_out_dir``) live alongside this module so the
runner's hard-rule branches have positive coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(request: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Echo ``text`` (optionally ``count`` times) into ``out_dir/echo.txt``.

    The request shape is fixed by ``echo.yaml`` (input_schema); this
    function trusts the runner to have validated it.
    """

    text = str(request["text"])
    count = int(request.get("count", 1))
    if count < 1:
        # The schema permits count>=1; this is the runtime fallback.
        # We do NOT silently coerce — we surface the invariant.
        return {
            "status": "error",
            "lines_written": 0,
            "artifacts": [],
            "reason": "count must be >= 1",
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "echo.txt"
    body = "\n".join([text] * count) + "\n"
    artifact.write_text(body, encoding="utf-8")

    return {
        "status": "ok",
        "lines_written": count,
        "artifacts": [str(artifact.resolve())],
    }


def returns_bad_status(request: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Negative-control entrypoint: violates ``output_schema.status``.

    Used by ``tests/tools/test_tool_runner.py`` to assert
    :class:`compgen.tools.errors.ToolOutputSchemaError` fires.
    """

    return {"status": "definitely_not_a_real_status", "lines_written": 0, "artifacts": []}


def writes_outside_out_dir(
    request: dict[str, Any], *, out_dir: Path
) -> dict[str, Any]:
    """Negative-control entrypoint: writes outside the declared root.

    Returns an artifact path under ``/tmp/escape_<pid>`` (which is
    *not* under ``out_dir`` and not in any of echo.yaml's
    ``writes.allowed_roots``). The runner must reject this.
    """

    import os

    artifact = Path(f"/tmp/escape_{os.getpid()}_{id(request)}.txt")
    artifact.write_text("escaped", encoding="utf-8")
    return {
        "status": "ok",
        "lines_written": 1,
        "artifacts": [str(artifact)],
    }


def crashes(request: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Negative-control entrypoint: raises mid-execution."""

    raise RuntimeError("intentional crash for negative-control test")

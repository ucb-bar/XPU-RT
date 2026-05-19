"""Backwards-compat shim — Gemmini-pinned wrapper around
:mod:`xpu_rt.spike_harness.compile_server`.

Use ``uv run python -m xpu_rt.spike_harness.compile_server --target
gemmini_mx --port 8201`` directly for new scripts. This shim exists
only so existing CLI invocations and import sites keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from xpu_rt.spike_harness.compile_server import (  # noqa: F401
    CompilationResult,
    make_app,
)
from xpu_rt.spike_harness.targets import resolve_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8201)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = make_app(resolve_target("gemmini_mx"))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


# Module-level APP for `uvicorn xpu_rt.kb_gemmini.compile_server:APP` users.
APP = make_app(resolve_target("gemmini_mx"))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["APP", "CompilationResult", "main"]

"""Backwards-compat shim — Saturn-pinned wrapper around
:mod:`xpu_rt.spike_harness.compile_server`."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from xpu_rt.spike_harness.compile_server import CompilationResult, make_app  # noqa: F401
from xpu_rt.spike_harness.targets import resolve_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8301)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = make_app(resolve_target("saturn_opu_v128"))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


APP = make_app(resolve_target("saturn_opu_v128"))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["APP", "CompilationResult", "main"]

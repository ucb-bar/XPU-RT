#!/usr/bin/env python3
"""Tamper a saved torch tensor file in place.

Used by graph_capture stage acceptance scripts to verify that ``replay-goldens``
rejects a tampered ``golden_outputs.pt``. By default we add a 1.0
perturbation to every numeric tensor in the saved object — that is
guaranteed to break exact-match replay without producing NaN/inf.

Usage::

    python scripts/dev/tamper_tensor.py --path /tmp/run/00_graph_capture/golden_outputs.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _perturb(obj):  # type: ignore[no-untyped-def]
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj + 1.0
        return obj + 1
    if isinstance(obj, list):
        return [_perturb(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_perturb(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _perturb(v) for k, v in obj.items()}
    return obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tamper a torch.save'd tensor file in place.")
    parser.add_argument("--path", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.path.exists():
        raise SystemExit(f"error: {args.path} does not exist")
    obj = torch.load(args.path, weights_only=False)
    perturbed = _perturb(obj)
    torch.save(perturbed, args.path)
    print(f"tampered: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

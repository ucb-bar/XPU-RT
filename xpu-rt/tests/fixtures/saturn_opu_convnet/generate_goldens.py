"""Regenerate golden_inputs.pt / golden_outputs.pt for the ConvNet fixture.

Run once from a developer workstation::

    uv run python tests/fixtures/saturn_opu_convnet/generate_goldens.py

The goldens feed both the CompGen compile-path differential test and the
Zephyr/Spike/FireSim end-to-end tests, so they must be stable and
reproducible. Seed is pinned in :mod:`model`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tests.fixtures.saturn_opu_convnet.model import build_model, default_inputs


def main() -> None:
    out_dir = Path(__file__).parent
    model = build_model()
    inputs = default_inputs()
    with torch.no_grad():
        outputs = model(*inputs)
    torch.save(inputs, out_dir / "golden_inputs.pt")
    torch.save(outputs, out_dir / "golden_outputs.pt")
    print(f"Wrote {out_dir / 'golden_inputs.pt'} and {out_dir / 'golden_outputs.pt'}")


if __name__ == "__main__":
    main()

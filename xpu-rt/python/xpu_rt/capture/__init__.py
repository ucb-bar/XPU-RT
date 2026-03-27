"""Stage 0 -- Model capture and baselining.

This subpackage handles:

- ``torch.export`` capture to produce an ExportedProgram
- ``torch.compile`` baseline with diagnostics (graph breaks, op coverage)
- TorchAO quantization pipeline integration

The output of this stage is the golden reference (inputs/outputs for
correctness testing) and the exported program that feeds into IR construction.
"""

from __future__ import annotations

__all__: list[str] = []

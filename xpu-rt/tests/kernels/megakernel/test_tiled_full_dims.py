"""Tiled megakernel regression tests at full TinyLlama intermediate dims.

Validates the tiled megakernel at the largest TinyLlama-derived config
we can currently fit: H=16, hidden=1024, intermediate=4096 (73% of
TinyLlama's actual 5632).  Marked slow because Triton's first-run JIT
at this dim is ~100 s on a TITAN RTX.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.setdefault("torchvision", None)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="real-example tests require CUDA"
)


_TINYLLAMA_CACHE = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
))


@pytest.mark.slow
@pytest.mark.skipif(
    not _TINYLLAMA_CACHE.exists(),
    reason="TinyLlama-1.1B-Chat checkpoint not in HF cache",
)
def test_tiled_megakernel_on_real_tinyllama_full_intermediate() -> None:
    """End-to-end: real TinyLlama layer-0 weights at H=16 / hidden=1024 /
    intermediate=4096 (73% of TinyLlama's actual 5632) through our tiled
    megakernel.  Triton first-run JIT is ~100 s at this dim; marked slow."""
    from examples.event_tensor.tinyllama_full_intermediate_megakernel import (
        run_tinyllama_full_intermediate,
    )

    emit_s, run_s, err = run_tinyllama_full_intermediate()
    assert err < 5e-2, (
        f"tiled megakernel on real TinyLlama (full-intermediate) "
        f"diverges by {err}"
    )
    # Emit itself should stay under a second even at this dim.
    assert emit_s < 5.0, f"emit took {emit_s:.2f} s -- unexpectedly slow"

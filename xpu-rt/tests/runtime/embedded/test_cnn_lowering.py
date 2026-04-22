"""Numerical correctness for the CNN lowering.

Compiles the lowered C on the HOST (not the cross toolchain), loads
it via ctypes, and verifies the output matches torch eager within
float32 tolerance. The Spike run uses the exact same C source, so a
passing host test is strong evidence the Spike run is also correct
modulo the cross compiler.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # fixtures

import numpy as np
import torch

from compgen.runtime.embedded.cnn_lowering import lower_cnn_to_c
from tests.fixtures.saturn_opu_convnet.model import build_model, default_inputs


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host cc available")
def test_lowered_convnet_matches_torch(tmp_path: Path) -> None:
    model = build_model()
    inputs = default_inputs()
    with torch.no_grad():
        expected = model(*inputs).detach().cpu().numpy().astype(np.float32).ravel()

    lowered = lower_cnn_to_c(model, sample_input_shape=(3, 64, 64))
    assert lowered.input_bytes == 49152
    assert lowered.output_bytes == 64
    assert lowered.num_params > 100_000  # ConvNet has hundreds of thousands of params

    # Drop the emitted forward + the foundational runtime headers/sources
    # into tmp_path and compile. The foundational runtime lives on disk
    # at runtime/include + runtime/src; we copy from there.
    import shutil as _shutil
    runtime_root = Path(__file__).resolve().parents[3] / "runtime"
    (tmp_path / "compgen").mkdir()
    for name in ("types.h", "arena.h", "ops.h"):
        _shutil.copy2(runtime_root / "include" / "compgen" / name, tmp_path / "compgen" / name)
    for name in ("arena.c", "ops.c"):
        _shutil.copy2(runtime_root / "src" / name, tmp_path / name)
    (tmp_path / "compgen_model_forward.c").write_text(lowered.forward_c_source)

    # Inline the blob as raw bytes in a C object.
    blob_c = tmp_path / "model_blob.c"
    rows = []
    for start in range(0, len(lowered.weights_blob), 32):
        chunk = lowered.weights_blob[start : start + 32]
        rows.append("    " + ", ".join(f"0x{b:02x}" for b in chunk))
    body = ",\n".join(rows)
    blob_c.write_text(
        "#include <stddef.h>\n#include <stdint.h>\n"
        "__attribute__((aligned(16)))\n"
        "const uint8_t compgen_model_blob[] = {\n"
        + body + "\n};\n"
        "const size_t compgen_model_blob_size = sizeof(compgen_model_blob);\n"
    )

    # Shared-library wrapper that exposes the forward for ctypes.
    (tmp_path / "shim.c").write_text(
        """
        #include <stdlib.h>
        #include "compgen/types.h"
        cg_status_t compgen_model_forward(const float *, float *, void *, size_t);
        void compgen_run(const float *in, float *out) {
            static unsigned char arena[4*1024*1024];  /* 4 MiB host arena */
            (void)compgen_model_forward(in, out, arena, sizeof(arena));
        }
        """
    )

    so_path = tmp_path / "libconvnet.so"
    subprocess.run(
        ["cc", "-O2", "-std=c17", "-Wall", "-Werror", "-fPIC", "-shared",
         str(tmp_path / "arena.c"),
         str(tmp_path / "ops.c"),
         str(tmp_path / "compgen_model_forward.c"),
         str(tmp_path / "model_blob.c"),
         str(tmp_path / "shim.c"),
         "-o", str(so_path),
         f"-I{tmp_path}"],
        check=True, capture_output=True, text=True,
    )

    lib = ctypes.CDLL(str(so_path))
    lib.compgen_run.restype = None
    lib.compgen_run.argtypes = [ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float)]

    x = inputs[0].detach().cpu().numpy().astype(np.float32).ravel()
    y = np.zeros(16, dtype=np.float32)
    lib.compgen_run(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )

    # The C conv sums channels in a different inner order than torch's
    # MKL conv, so expect ~1e-4 float32 drift — not zero.
    max_abs = float(np.max(np.abs(y - expected)))
    max_rel = float(np.max(np.abs(y - expected) / (np.abs(expected) + 1e-6)))
    assert np.allclose(y, expected, rtol=1e-3, atol=1e-3), (
        f"mismatch: max_abs={max_abs:.3e} max_rel={max_rel:.3e}\n"
        f"  expected={expected}\n  got     ={y}"
    )

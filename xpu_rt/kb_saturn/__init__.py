"""Saturn / OPU vector-unit KB harness (RVV 1.0 + zvl128b).

Sibling to :mod:`xpu_rt.kb_gemmini`. Same FastAPI-server pair shape,
same per-shape templates + driver protocol — only the toolchain
options change:

  * compile: ``riscv64-unknown-linux-gnu-gcc`` with
    ``-march=rv64gcv_zvl128b`` (RVV 1.0 + 128-bit vectors per Saturn's
    OPU profile). The OPU-specific asm-macro intrinsics
    (``OPMVINBCAST``, ``VOPACC``) in Saturn's own benchmark headers
    are not emitted — the LLM-driven kernels target the stable RVV
    1.0 intrinsic set instead. That is the "best-effort" KB-on-OPU
    path; a follow-up could surface Saturn's OPU header into the
    Target Card.

  * run: stock ``spike --isa=rv64gcv_zvl128b pk`` — Saturn's vector
    unit is RVV 1.0 compliant, so the standard Spike binary executes
    every vmacc / vle / vse the kernel emits.

  * cycle count: standard ``read_csr(mcycle)`` (per
    ``chipyard/generators/saturn/benchmarks/common/util.h``).
"""

from __future__ import annotations

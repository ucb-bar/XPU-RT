"""RISC-V / RoCC evaluator backed by Spike (+ optional Gemmini extension).

Compiles a candidate kernel against a generated test harness, runs the
result on ``spike --extension=gemmini pk``, parses ``mismatches=N/M``
and ``cycles=N`` from stdout, and returns an :class:`EvaluationReport`.

Conventions the candidate kernel must follow (enforced by the prompt
builder; see ``prompt_builder.py``):

  * The kernel exposes a function named ``kernel_under_test`` with the
    arg list ``(<input ptrs in contract order>, <output ptrs in
    contract order>)``. For matmul: ``kernel_under_test(const int8_t
    *A, const int8_t *B, int32_t *C)`` — exactly the same shape as the
    Spike+RVV demo that landed in plan 1.
  * The kernel must compile under ``riscv64-unknown-linux-gnu-gcc``
    with the chipyard gemmini-rocc-tests include paths on -I.
  * Inputs are statically sized via ``contract.input_shapes`` /
    ``contract.output_shapes`` — symbolic shapes are not yet
    supported (will raise ``NotImplementedError`` so the agent loop
    sees a deterministic failure rather than a silent skip).

Op families implemented today:

  * ``matmul`` (int8 × int8 → int32) — the Phase-A target.

Op families that explicitly raise :class:`NotImplementedError`
(documented at the call site, not silently routed to "skip"):

  * ``conv``, ``reduce``, ``softmax``, ``pointwise``, ``activation``.

The evaluator is best-effort: a stderr line from the compile step
becomes ``EvaluationReport.compile_log``; a spike crash becomes
``runtime_log``. Either way the report's ``correct`` is False and
``score`` is 0.0 so the agent loop bumps the strategy DB and tries
a different action next round.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xpu_rt.kernels.kernelblaster_v2.evaluators.base import (
    EvaluationReport,
    Evaluator,
)
from xpu_rt.kernels.kernelblaster_v2.generators import ProposeResponse
from xpu_rt.kernels.provider import KernelContract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults — overridable via env so CI / alternative chipyard checkouts work.
# ---------------------------------------------------------------------------

DEFAULT_CONDA_ROOT = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools")
DEFAULT_GEMMINI_ROOT = Path("/scratch2/agustin/chipyard/generators/gemmini")
DEFAULT_TIMEOUT_S = 60

ENV_CONDA_ROOT = "XPU_RT_RISCV_CONDA_ROOT"
ENV_GEMMINI_ROOT = "XPU_RT_CHIPYARD_GEMMINI_ROOT"
ENV_SPIKE = "XPU_RT_SPIKE_BIN"
ENV_PK = "XPU_RT_PK_BIN"
ENV_CC = "XPU_RT_RISCV_CC"


def _conda_root() -> Path:
    return Path(os.environ.get(ENV_CONDA_ROOT, str(DEFAULT_CONDA_ROOT)))


def _gemmini_root() -> Path:
    return Path(os.environ.get(ENV_GEMMINI_ROOT, str(DEFAULT_GEMMINI_ROOT)))


def _spike_bin() -> Path:
    env = os.environ.get(ENV_SPIKE)
    return Path(env) if env else _conda_root() / "bin" / "spike"


def _pk_bin() -> Path:
    env = os.environ.get(ENV_PK)
    return Path(env) if env else _conda_root() / "riscv64-unknown-elf" / "bin" / "pk"


def _cc_bin() -> Path:
    env = os.environ.get(ENV_CC)
    return Path(env) if env else _conda_root() / "bin" / "riscv64-unknown-linux-gnu-gcc"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


_MISMATCH_RE = re.compile(r"^\s*mismatches\s*=\s*(\d+)\s*/\s*(\d+)\s*$", re.MULTILINE)
_CYCLES_RE = re.compile(r"^\s*cycles\s*=\s*(\d+)\s*$", re.MULTILINE)
# Additional Gemmini counters surfaced for KB-vanilla's bottleneck-state
# classifier — populated by the matmul harness alongside `cycles=`.
_EXE_ACTIVE_RE = re.compile(r"^\s*exe_active_cycles\s*=\s*(\d+)\s*$", re.MULTILINE)
_LOAD_DMA_WAIT_RE = re.compile(r"^\s*load_dma_wait_cycles\s*=\s*(\d+)\s*$", re.MULTILINE)
_SCRATCHPAD_A_WAIT_RE = re.compile(r"^\s*scratchpad_a_wait_cycles\s*=\s*(\d+)\s*$", re.MULTILINE)


def _parse_output(stdout: str) -> tuple[int | None, int | None, int | None]:
    """Return ``(mismatches, total, cycles)`` from harness stdout."""
    miss_total: tuple[int, int] | None = None
    m = _MISMATCH_RE.search(stdout)
    if m:
        miss_total = (int(m.group(1)), int(m.group(2)))
    cycles: int | None = None
    c = _CYCLES_RE.search(stdout)
    if c:
        cycles = int(c.group(1))
    if miss_total is None:
        return None, None, cycles
    return miss_total[0], miss_total[1], cycles


def _parse_counter_extras(stdout: str) -> dict[str, int]:
    """Parse the extra Gemmini counter lines into a flat dict.

    Returns the subset of {exe_active_cycles, load_dma_wait_cycles,
    scratchpad_a_wait_cycles} present in stdout. Missing counters are
    omitted (callers handle the absence). Used by the report builder
    and by KB-vanilla's bottleneck-state classifier.
    """
    out: dict[str, int] = {}
    for key, pat in (
        ("exe_active_cycles", _EXE_ACTIVE_RE),
        ("load_dma_wait_cycles", _LOAD_DMA_WAIT_RE),
        ("scratchpad_a_wait_cycles", _SCRATCHPAD_A_WAIT_RE),
    ):
        m = pat.search(stdout)
        if m:
            out[key] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Harness generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MatmulShape:
    M: int
    K: int
    N: int
    dtype_in: str  # "i8"
    dtype_acc: str  # "i32"


def _resolve_matmul_shape(contract: KernelContract) -> _MatmulShape:
    if len(contract.input_shapes) != 2 or len(contract.output_shapes) != 1:
        raise NotImplementedError(
            f"CRiscvEvaluator matmul harness requires exactly 2 inputs + 1 output; "
            f"got {len(contract.input_shapes)} / {len(contract.output_shapes)} from "
            f"region_id={contract.region_id!r}"
        )
    A, B = contract.input_shapes
    (C,) = contract.output_shapes
    if not (len(A) == 2 and len(B) == 2 and len(C) == 2):
        raise NotImplementedError(
            f"CRiscvEvaluator currently only handles 2D matmul; got shapes {A}, {B} -> {C}"
        )
    if A[1] != B[0] or A[0] != C[0] or B[1] != C[1]:
        raise NotImplementedError(
            f"matmul shape inconsistency: A={A} B={B} C={C}"
        )
    M, K, N = A[0], A[1], B[1]
    # Only the simple ("i8", "i8", "i32") flavour is wired today.
    dtypes = tuple(d.lower() for d in contract.dtypes)
    if "i8" not in dtypes and "int8" not in dtypes:
        raise NotImplementedError(
            f"CRiscvEvaluator matmul harness expects an i8 input dtype; "
            f"got dtypes={contract.dtypes}"
        )
    return _MatmulShape(M=M, K=K, N=N, dtype_in="i8", dtype_acc="i32")


_MATMUL_HARNESS = r"""
// Auto-generated by xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv.
// Do not edit.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "include/gemmini.h"
#include "include/gemmini_counter.h"

#define M {M}
#define K {K}
#define N {N}

extern void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C);

// Deterministic LCG so the test is reproducible bit-for-bit.
static uint32_t _lcg(uint32_t *s) {{
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}}

static void fill_i8(int8_t *p, int n, uint32_t seed) {{
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) p[i] = (int8_t)((_lcg(&s) >> 16) & 0xFF);
}}

static void scalar_ref(const int8_t *A, const int8_t *B, int32_t *C) {{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {{
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A[m*K + k] * (int32_t)B[k*N + n];
            C[m*N + n] = acc;
        }}
}}

int main(void) {{
    // Inputs and outputs live in static storage so spike+pk doesn't
    // need a huge stack — these arrays can be MB-sized.
    static int8_t  A[M*K];
    static int8_t  B[K*N];
    static int32_t Cref[M*N];
    static int32_t Ctest[M*N];

    fill_i8(A, M*K, 0xC0FFEE);
    fill_i8(B, K*N, 0xBEEF0042);

    scalar_ref(A, B, Cref);

    // Reset + configure 4 counter slots so the harness emits a
    // richer NCU-substitute trace KB-vanilla's bandit can classify
    // into a bottleneck state (memory_bandwidth / compute_throughput
    // / scratchpad_pressure / pipeline_hazard). KB-v2 ignores the
    // extras; the parser tolerates them.
    gemmini_flush(0);
    counter_configure(0, MAIN_LD_ST_EX_CYCLES);
    counter_configure(1, EXE_ACTIVE_CYCLE);
    counter_configure(2, LOAD_DMA_WAIT_CYCLE);
    counter_configure(3, SCRATCHPAD_A_WAIT_CYCLE);
    counter_snapshot_reset();
    int32_t cycles_before = counter_read(0);
    int32_t exe_before    = counter_read(1);
    int32_t dma_wait_before = counter_read(2);
    int32_t spad_wait_before = counter_read(3);
    kernel_under_test(A, B, Ctest);
    gemmini_fence();
    counter_snapshot_take();
    int32_t cycles_after = counter_read(0);
    int32_t exe_after    = counter_read(1);
    int32_t dma_wait_after = counter_read(2);
    int32_t spad_wait_after = counter_read(3);
    int32_t cycles = cycles_after - cycles_before;
    int32_t exe_active_cycles    = exe_after - exe_before;
    int32_t load_dma_wait_cycles = dma_wait_after - dma_wait_before;
    int32_t scratchpad_a_wait_cycles = spad_wait_after - spad_wait_before;

    int mismatches = 0;
    int total = M * N;
    int first_idx = -1;
    int32_t first_ref = 0, first_got = 0;
    for (int i = 0; i < total; ++i) {{
        if (Cref[i] != Ctest[i]) {{
            if (first_idx < 0) {{ first_idx = i; first_ref = Cref[i]; first_got = Ctest[i]; }}
            ++mismatches;
        }}
    }}

    printf("M=%d K=%d N=%d ops=%d\n", M, K, N, M*N*K*2);
    printf("mismatches=%d/%d\n", mismatches, total);
    if (first_idx >= 0)
        printf("first_diff_at=%d ref=%d got=%d\n", first_idx, first_ref, first_got);
    printf("cycles=%d\n", cycles);
    printf("exe_active_cycles=%d\n", exe_active_cycles);
    printf("load_dma_wait_cycles=%d\n", load_dma_wait_cycles);
    printf("scratchpad_a_wait_cycles=%d\n", scratchpad_a_wait_cycles);
    return mismatches == 0 ? 0 : 1;
}}
"""


_SATURN_MATMUL_HARNESS = r"""
// Auto-generated by xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv.
// Saturn / RVV variant — no Gemmini headers; cycles via mcycle CSR.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define M {M}
#define K {K}
#define N {N}

extern void kernel_under_test(const int8_t *A, const int8_t *B, int32_t *C);

static uint32_t _lcg(uint32_t *s) {{
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}}

static void fill_i8(int8_t *p, int n, uint32_t seed) {{
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) p[i] = (int8_t)((_lcg(&s) >> 16) & 0xFF);
}}

static void scalar_ref(const int8_t *A, const int8_t *B, int32_t *C) {{
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {{
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)A[m*K + k] * (int32_t)B[k*N + n];
            C[m*N + n] = acc;
        }}
}}

static inline uint64_t read_mcycle(void) {{
    uint64_t c;
    asm volatile ("rdcycle %0" : "=r"(c));
    return c;
}}

int main(void) {{
    static int8_t  A[M*K];
    static int8_t  B[K*N];
    static int32_t Cref[M*N];
    static int32_t Ctest[M*N];

    fill_i8(A, M*K, 0xC0FFEE);
    fill_i8(B, K*N, 0xBEEF0042);

    scalar_ref(A, B, Cref);

    uint64_t c0 = read_mcycle();
    kernel_under_test(A, B, Ctest);
    uint64_t c1 = read_mcycle();
    uint64_t cycles = c1 - c0;

    int mismatches = 0;
    int total = M * N;
    int first_idx = -1;
    int32_t first_ref = 0, first_got = 0;
    for (int i = 0; i < total; ++i) {{
        if (Cref[i] != Ctest[i]) {{
            if (first_idx < 0) {{ first_idx = i; first_ref = Cref[i]; first_got = Ctest[i]; }}
            ++mismatches;
        }}
    }}

    printf("M=%d K=%d N=%d ops=%d\n", M, K, N, M*N*K*2);
    printf("mismatches=%d/%d\n", mismatches, total);
    if (first_idx >= 0)
        printf("first_diff_at=%d ref=%d got=%d\n", first_idx, first_ref, first_got);
    printf("cycles=%llu\n", (unsigned long long)cycles);
    return mismatches == 0 ? 0 : 1;
}}
"""


def _generate_harness(contract: KernelContract, *, target_id: str = "gemmini") -> str:
    op_family = contract.op_family.lower()
    if op_family in ("matmul", "mm", "gemm", "bmm", "linear"):
        shape = _resolve_matmul_shape(contract)
        t = target_id.lower()
        if t.startswith("saturn") or t.startswith("opu"):
            return _SATURN_MATMUL_HARNESS.format(M=shape.M, K=shape.K, N=shape.N)
        return _MATMUL_HARNESS.format(M=shape.M, K=shape.K, N=shape.N)
    raise NotImplementedError(
        f"CRiscvEvaluator does not yet generate a harness for op_family="
        f"{contract.op_family!r}; matmul is the only Phase-A target. "
        f"Implementing this op family is a follow-up task."
    )


# ---------------------------------------------------------------------------
# Toolchain availability
# ---------------------------------------------------------------------------


class _ToolchainMissing(RuntimeError):
    """Raised when riscv64-unknown-linux-gnu-gcc, spike, or pk is not present."""


def _check_toolchain() -> None:
    missing = []
    for name, path in (("cc", _cc_bin()), ("spike", _spike_bin()), ("pk", _pk_bin())):
        if not path.is_file() or not os.access(path, os.X_OK):
            missing.append(f"{name}={path}")
    if missing:
        raise _ToolchainMissing(
            "RISC-V toolchain not found: " + ", ".join(missing)
            + f". Set {ENV_CONDA_ROOT}/{ENV_SPIKE}/{ENV_PK}/{ENV_CC} or install chipyard's riscv-tools conda env."
        )


def _check_gemmini_extension() -> bool:
    """Best-effort probe for ``spike --extension=gemmini`` support."""
    try:
        proc = subprocess.run(
            [str(_spike_bin()), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "--extension=" in (proc.stdout + proc.stderr)


def _gemmini_include_args() -> list[str]:
    """Include flags so the harness picks up gemmini.h + gemmini_counter.h."""
    root = _gemmini_root() / "software" / "gemmini-rocc-tests"
    return [
        f"-I{root}",
        f"-I{root}/include",
        f"-I{root}/riscv-tests",
        f"-I{root}/riscv-tests/env",
    ]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


@dataclass
class CRiscvEvaluator:
    """Evaluate a candidate C kernel on ``spike --extension=gemmini pk``.

    Args:
        contract: The contract the candidate is supposed to satisfy.
            Used to size the harness and decide which op_family stencil
            to instantiate.
        keep_workdir: When True, the tmp workdir is preserved for
            postmortem inspection. Defaults to False (cleaned up).
        timeout_s: Wall-clock seconds before the spike subprocess is
            killed. Default: 60s — Spike is fast; if a candidate takes
            longer it almost certainly has an infinite loop.
        require_gemmini_extension: When True (default), the evaluator
            confirms ``spike --extension=gemmini`` is available and
            raises ``_ToolchainMissing`` otherwise. Tests can set this
            to False to exercise the build/run path on plain RVV.
        score_for_correct: Baseline score returned for any correct
            candidate before any cycle-derived weighting. The agent
            loop picks "best so far" by score, so a sensible default
            is ``1.0`` and the evaluator divides into it.
    """

    contract: KernelContract
    keep_workdir: bool = False
    timeout_s: int = DEFAULT_TIMEOUT_S
    require_gemmini_extension: bool = True
    score_for_correct: float = 1.0
    name: str = "c_riscv"
    # Cross-target dispatch. Default preserves the Gemmini path
    # exactly (so existing callers + tests keep working). When
    # ``target_id`` is ``"saturn_opu_v128"`` (or any id starting with
    # ``"saturn"``/``"opu"``) the evaluator switches to RVV 1.0 +
    # zvl128b — different ``-march``, different Spike flag (``--isa``
    # not ``--extension``), no Gemmini header include paths.
    target_id: str = "gemmini"

    # ---- target dispatch -------------------------------------------------

    def _is_saturn(self) -> bool:
        t = self.target_id.lower()
        return t.startswith("saturn") or t.startswith("opu")

    def _spike_flag(self) -> tuple[str, ...]:
        """Spike flag(s) prepended to the run command for this target.

        Gemmini uses ``--extension=gemmini`` (RoCC custom-3); Saturn
        uses ``--isa=rv64gcv_zvl128b_zicntr`` (RVV 1.0 + 128-bit
        vectors + the ``zicntr`` cycle counter extension). OPU
        outer-product-unit instructions are picked up automatically
        when ``XPU_RT_SPIKE_BIN`` points at the Saturn-OPU Spike
        fork (https://github.com/CobbledSteel/riscv-isa-sim/tree/saturn-opu-extension)
        — no extension flag, it's baked into the binary.
        """
        if self._is_saturn():
            return ("--isa=rv64gcv_zvl128b_zicntr",)
        return ("--extension=gemmini",)

    def _march_flags(self) -> tuple[str, ...]:
        """``-march`` + ``-Wa,-march`` flags for this target's compile."""
        if self._is_saturn():
            # zicntr exposes the mcycle CSR — required for the kb_saturn
            # harness's read_csr(mcycle) to assemble.
            return (
                "-march=rv64gcv_zvl128b_zicntr",
                "-Wa,-march=rv64gcv_zvl128b_zicntr",
            )
        return ("-march=rv64gc", "-Wa,-march=rv64gc")

    def _target_include_args(self) -> tuple[str, ...]:
        """Header search paths; Gemmini needs the rocc-tests includes,
        Saturn just uses ``<riscv_vector.h>`` from the toolchain."""
        if self._is_saturn():
            return ()
        return tuple(_gemmini_include_args())

    # ---- entry point ------------------------------------------------------

    def evaluate(self, candidate: ProposeResponse) -> EvaluationReport:
        # Cheap availability check first so the agent loop sees an
        # honest failure (not a silent compile error) when the
        # toolchain isn't present.
        try:
            _check_toolchain()
        except _ToolchainMissing as exc:
            return EvaluationReport(
                correct=False,
                score=0.0,
                compile_log=str(exc),
                metadata={"reason": "toolchain_missing"},
            )

        # The require_gemmini_extension gate only applies on the
        # Gemmini target — Saturn uses Spike's stock ``--isa=`` flag
        # so the extension probe is irrelevant.
        if self.require_gemmini_extension and not self._is_saturn() and not _check_gemmini_extension():
            return EvaluationReport(
                correct=False,
                score=0.0,
                compile_log="spike --extension=gemmini not supported by this build",
                metadata={"reason": "no_gemmini_extension"},
            )

        try:
            harness_src = _generate_harness(self.contract, target_id=self.target_id)
        except NotImplementedError as exc:
            return EvaluationReport(
                correct=False,
                score=0.0,
                compile_log=str(exc),
                metadata={"reason": "harness_unsupported_op_family"},
            )

        workdir = Path(
            tempfile.mkdtemp(prefix=f"xpu_rt_eval_{self.contract.region_id or 'r'}_")
        )
        try:
            return self._run_in_workdir(workdir, candidate, harness_src)
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    # ---- internal --------------------------------------------------------

    def _run_in_workdir(
        self,
        workdir: Path,
        candidate: ProposeResponse,
        harness_src: str,
    ) -> EvaluationReport:
        kernel_path = workdir / "kernel.c"
        harness_path = workdir / "harness.c"
        elf_path = workdir / "kernel.elf"
        kernel_path.write_text(candidate.kernel_code or "// (empty candidate)\n")
        harness_path.write_text(harness_src)

        compile_cmd = [
            str(_cc_bin()),
            "-std=gnu99",
            "-O2",
            "-static",
            "-fno-common",
            "-fno-builtin-printf",
            "-DBAREMETAL=0",
            *self._march_flags(),
            *self._target_include_args(),
            str(kernel_path),
            str(harness_path),
            "-o",
            str(elf_path),
            "-lm",
            "-lgcc",
        ]
        compile_proc = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=self.timeout_s
        )
        if compile_proc.returncode != 0:
            return EvaluationReport(
                correct=False,
                score=0.0,
                compile_log=(compile_proc.stdout + compile_proc.stderr)[-4000:],
                metadata={"reason": "compile_failed", "cmd": " ".join(compile_cmd)},
            )

        spike_cmd = [
            str(_spike_bin()),
            *self._spike_flag(),
            str(_pk_bin()),
            str(elf_path),
        ]
        try:
            run_proc = subprocess.run(
                spike_cmd, capture_output=True, text=True, timeout=self.timeout_s
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluationReport(
                correct=False,
                score=0.0,
                runtime_log=f"spike timed out after {self.timeout_s}s: {exc}",
                metadata={"reason": "spike_timeout"},
            )

        stdout = run_proc.stdout or ""
        stderr = run_proc.stderr or ""
        mismatches, total, cycles = _parse_output(stdout)
        # Surface the 3 extra counters (exe_active, load_dma_wait,
        # scratchpad_a_wait) the harness emits when running on
        # Gemmini. Saturn / non-Gemmini paths emit only `cycles=`
        # so this returns an empty dict and the bottleneck-state
        # classifier degrades to "unknown" gracefully.
        counter_extras = _parse_counter_extras(stdout)

        if mismatches is None or total is None:
            return EvaluationReport(
                correct=False,
                score=0.0,
                runtime_log=(stdout + stderr)[-4000:],
                metadata={
                    "reason": "no_mismatch_line",
                    "exit": run_proc.returncode,
                    **counter_extras,
                },
            )

        correct = mismatches == 0
        score = self.score_for_correct if correct else 0.0
        return EvaluationReport(
            correct=correct,
            score=score,
            cycles=cycles,
            runtime_log=stdout[-4000:],
            diff_summary=(
                f"{mismatches}/{total} mismatches"
                + (f"; cycles={cycles}" if cycles is not None else "")
            ),
            metadata={
                "mismatches": mismatches,
                "total": total,
                "spike_exit": run_proc.returncode,
                "cmd": " ".join(spike_cmd),
                **counter_extras,
            },
        )


__all__ = [
    "CRiscvEvaluator",
    "DEFAULT_GEMMINI_ROOT",
    "DEFAULT_CONDA_ROOT",
    "ENV_CC",
    "ENV_CONDA_ROOT",
    "ENV_GEMMINI_ROOT",
    "ENV_PK",
    "ENV_SPIKE",
]

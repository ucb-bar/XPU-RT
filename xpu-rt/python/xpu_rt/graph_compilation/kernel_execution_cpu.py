""".CPU sub-track MLIR-derived C codegen + cffi compile + run.

Emits a self-contained C function for tiled matmul with the
SetTileParams candidate's tile baked in. Compiles via ``cffi`` (which
invokes the host C compiler), runs the resulting callable on
synthesized fp32 inputs, times via ``time.perf_counter_ns``, and
compares numerical output against eager ``torch.matmul``.

The emitted C is straightforward: an outer triple-loop over tiles,
inner triple-loop over the tile's matmul. No SIMD intrinsics; no
threading. The point of .CPU is to prove the
**MLIR → C → compiled binary → measured timing** pipeline works
end-to-end with real numerical equivalence — not to be fast.

Hard non-goals:
No MLIR-to-C lowering generality. .CPU only handles the
  matmul-tile-loop subset that emits for SetTileParams.
No SIMD / OpenMP / parallelism. territory.
- No libcompgen_rt-specific runtime calls. The cffi-compiled .so is
  loaded via cffi's own dlopen; libcompgen_rt is referenced in the
  artifact for documentation only.

Best-effort: cffi missing, gcc missing, compile failure, run failure
all emit typed ``compile_status`` / ``run_status`` codes; never raises.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# C codegen
# --------------------------------------------------------------------------- #


def _emit_c_source(
    *, candidate_id: str, region_id: str,
    M: int, N: int, K: int, tM: int, tN: int, tK: int,
    fn_name: str,
) -> str:
    """Emit a C function ``void <fn_name>(const float* A, const float* B,
    float* C)`` that computes C = A @ B for fixed M, N, K with tiled
    structure tM, tN, tK. Row-major; no aliasing assumed.

    The tile loops mirror 's `transformed_payload.real.mlir` outer
    nest; the inner micro-matmul is a straight (i, j, k) triple-loop
    that accumulates into a per-tile fp32 accumulator before storing.
    Accumulation order matches eager when tile_K == K (single
    K-iteration). For tile_K < K the C accumulation order matches
    the `_tiled_matmul_eval` Python pattern.
    """
    return f"""\
/* M-19.CPU generated tiled matmul kernel.
 *
 * candidate_id: {candidate_id}
 * region_id:    {region_id}
 * matmul_shape: M={M} N={N} K={K}
 * tile:         tM={tM} tN={tN} tK={tK}
 *
 * Numerically equivalent to the M-16 Python `_tiled_matmul_eval`
 * for the same inputs and the same tile dims; row-major fp32.
 */

#include <string.h>

void {fn_name}(const float* __restrict__ A,
{' ' * (len(fn_name) + 6)}const float* __restrict__ B,
{' ' * (len(fn_name) + 6)}float* __restrict__ C) {{
    const int M = {M};
    const int N = {N};
    const int K = {K};
    const int tM = {tM};
    const int tN = {tN};
    const int tK = {tK};

    /* Initialize C. Source MLIR uses tensor.empty() then iter_args
     * threading, so logical initial state is zero. */
    memset(C, 0, sizeof(float) * (size_t)M * (size_t)N);

    /* Triple loop over tiles (i, j, k) mirrors M-11B's scf.for nest. */
    for (int i = 0; i < M; i += tM) {{
        const int ii_max = (i + tM <= M) ? tM : (M - i);
        for (int j = 0; j < N; j += tN) {{
            const int jj_max = (j + tN <= N) ? tN : (N - j);
            for (int k = 0; k < K; k += tK) {{
                const int kk_max = (k + tK <= K) ? tK : (K - k);
                /* Inner micro-matmul: standard ijk loop on the tile. */
                for (int ii = 0; ii < ii_max; ++ii) {{
                    for (int jj = 0; jj < jj_max; ++jj) {{
                        float acc = C[(i + ii) * N + (j + jj)];
                        for (int kk = 0; kk < kk_max; ++kk) {{
                            acc += A[(i + ii) * K + (k + kk)]
                                 * B[(k + kk) * N + (j + jj)];
                        }}
                        C[(i + ii) * N + (j + jj)] = acc;
                    }}
                }}
            }}
        }}
    }}
}}
"""


def _classify_error(max_abs: float, max_rel: float, atol: float, rtol: float) -> str:
    if max_abs == 0.0 and max_rel == 0.0:
        return "discharged_compiled_bit_equality"
    if max_abs <= atol and max_rel <= rtol:
        return "discharged_tolerance_eps"
    return "fail_outside_tolerance"


# --------------------------------------------------------------------------- #
# Run + measure
# --------------------------------------------------------------------------- #


def run_cpu_track(*, out_dir: Path, common: dict[str, Any]) -> Path:
    """Emit C source → cffi compile → load .so → run + time → compare
    vs eager torch.matmul. Returns the path to the emitted
    ``compiled_kernel_run_cpu.json`` artifact. Best-effort; never
    raises."""
    artifact_path = out_dir / "compiled_kernel_run_cpu.json"
    base = {
        **common,
        "track": "cpu_compgen_rt",
        "c_source_path": None,
        "c_source_sha256": None,
        "compiler": "",
        "compile_command": "",
        "compiled_lib_path": None,
        "device": {"kind": "cpu", "name": ""},
        "measured_us_per_iter": None,
        "measured_us_stddev": None,
        "bandwidth_gbps": None,
        "flops_per_s": None,
        "numerical": {
            "max_abs_error": None,
            "max_rel_error": None,
            "refinement_status": "not_run",
        },
        "compile_status": "not_run",
        "run_status": "not_run",
        "note": "",
        "known_limitations": [
            "no SIMD; no parallelism (single-thread inner loop)",
            "single batch size; no autotune",
            "uses cffi as the compiler-driver (delegates to host gcc/clang)",
            "matmul-tile-loop subset only (no general MLIR→C codegen yet)",
        ],
        "generated_at_utc": _utcnow(),
    }

    # 1. cffi availability.
    try:
        from cffi import FFI
    except ImportError as exc:
        base["compile_status"] = "cffi_unavailable"
        base["note"] = f"cffi missing: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    # 2. compiler availability (best-effort probe).
    cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if cc is None:
        base["compile_status"] = "compiler_unavailable"
        base["note"] = "no gcc/clang/cc on PATH"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path
    base["compiler"] = Path(cc).name

    # 3. torch availability for inputs + reference.
    try:
        import torch
    except ImportError as exc:
        base["compile_status"] = "torch_unavailable"
        base["note"] = f"torch missing: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    M = int(common["matmul_shape"]["M"])
    N = int(common["matmul_shape"]["N"])
    K = int(common["matmul_shape"]["K"])
    tM = int(common["tile"]["M"])
    tN = int(common["tile"]["N"])
    tK = int(common["tile"]["K"])
    region_id = str(common["region_id"])
    candidate_id = str(common["candidate_id"])
    safe_region = "".join(
        c if c.isalnum() or c == "_" else "_" for c in region_id
    ) or "matmul"
    fn_name = f"compgen_m19_matmul_{safe_region}"

    # 4. Emit C source.
    c_source = _emit_c_source(
        candidate_id=candidate_id, region_id=region_id,
        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK, fn_name=fn_name,
    )
    src_path = out_dir / f"cpu_kernel_{safe_region}.c"
    src_path.write_text(c_source, encoding="utf-8")
    base["c_source_path"] = str(src_path.relative_to(out_dir.parent))
    base["c_source_sha256"] = _sha256_text(c_source)

    # 5. cffi compile.
    try:
        ffi = FFI()
        ffi.cdef(
            f"void {fn_name}(const float* A, const float* B, float* C);"
        )
        # Use set_source + compile to produce a real .so. tmpdir keeps
        # build artifacts isolated.
        build_dir = out_dir / f"cffi_build_{safe_region}"
        build_dir.mkdir(parents=True, exist_ok=True)
        ffi.set_source(
            f"_compgen_m19_cpu_{safe_region}",
            c_source,
            extra_compile_args=["-O2", "-fno-fast-math"],
        )
        compiled_so = ffi.compile(tmpdir=str(build_dir), verbose=False)
        base["compiled_lib_path"] = str(Path(compiled_so).relative_to(out_dir.parent))
        base["compile_command"] = (
            f"cffi {Path(cc).name} -O2 -fno-fast-math (set_source/compile)"
        )
    except Exception as exc:  # noqa: BLE001
        base["compile_status"] = "compile_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["compile_status"] = "compiled"

    # 6. Load compiled extension dynamically.
    try:
        import importlib.util as _importlib_util
        ext_name = f"_compgen_m19_cpu_{safe_region}"
        spec = _importlib_util.spec_from_file_location(ext_name, compiled_so)
        if spec is None or spec.loader is None:
            raise ImportError("could not spec compiled cpu kernel module")
        ext = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        kernel_fn = getattr(ext.lib, fn_name)
    except Exception as exc:  # noqa: BLE001
        base["compile_status"] = "compile_failed"
        base["note"] = f"loading failed: {type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    # 7. Run + measure.
    try:
        g = torch.Generator()
        g.manual_seed(0xC0DE19)
        A_t = torch.randn(M, K, dtype=torch.float32, generator=g).contiguous()
        B_t = torch.randn(K, N, dtype=torch.float32, generator=g).contiguous()
        C_t = torch.zeros(M, N, dtype=torch.float32).contiguous()

        # Use cffi's cast-from-buffer to get raw pointers.
        a_ptr = ext.ffi.cast("float*", A_t.data_ptr())
        b_ptr = ext.ffi.cast("float*", B_t.data_ptr())
        c_ptr = ext.ffi.cast("float*", C_t.data_ptr())

        warmup = int(common.get("warmup", 4))
        for _ in range(warmup):
            kernel_fn(a_ptr, b_ptr, c_ptr)

        iters = int(common.get("iterations", 32))
        per_iter_us: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            kernel_fn(a_ptr, b_ptr, c_ptr)
            per_iter_us.append((time.perf_counter_ns() - t0) / 1000.0)

        mean_us = sum(per_iter_us) / len(per_iter_us) if per_iter_us else 0.0
        if len(per_iter_us) > 1:
            var = sum((x - mean_us) ** 2 for x in per_iter_us) / (
                len(per_iter_us) - 1
            )
            stddev_us = var ** 0.5
        else:
            stddev_us = 0.0
        base["measured_us_per_iter"] = mean_us
        base["measured_us_stddev"] = stddev_us
        bytes_per_iter = (M * K + K * N + M * N) * 4
        if mean_us > 0:
            base["bandwidth_gbps"] = (bytes_per_iter / 1e9) / (mean_us / 1e6)
        flops_per_iter = 2 * M * N * K
        if mean_us > 0:
            base["flops_per_s"] = flops_per_iter / (mean_us / 1e6)

        base["run_status"] = "ok"

        # Numerical check.
        ref = torch.matmul(A_t, B_t)
        diff = (ref - C_t).abs()
        max_abs = float(diff.max().item())
        denom = ref.abs().clamp(min=1e-12)
        max_rel = float((diff / denom).max().item())
        base["numerical"] = {
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "refinement_status": _classify_error(
                max_abs, max_rel, atol=1e-5, rtol=1e-4,
            ),
        }
    except Exception as exc:  # noqa: BLE001
        base["run_status"] = "run_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"

    artifact_path.write_text(
        json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
    )
    return artifact_path

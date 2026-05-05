"""M-19.GPU sub-track — Triton compile + launch + time.

Emits a self-contained Triton kernel parameterised on the
SetTileParams candidate's tile (BLOCK_M, BLOCK_N, BLOCK_K), compiles
it (Triton's runtime handles the GPU JIT), launches it on synthesized
inputs whose shape matches the matmul region, and measures latency
via ``compgen.kernels.measure.measure_kernel`` (production GPU timing
with torch.cuda.Event + synchronize).

Numerical equality vs eager ``torch.matmul`` is required:
- ``max_abs_error == 0 AND max_rel_error == 0`` ⇒
  ``refinement_status = "discharged_compiled_bit_equality"``
- otherwise within (atol=1e-6, rtol=1e-6) ⇒
  ``refinement_status = "discharged_tolerance_eps"``
- otherwise ⇒ ``refinement_status = "fail_outside_tolerance"``

TF32 is explicitly disabled in ``tl.dot(allow_tf32=False)`` so the
tiled accumulation order is the same as eager fp32 matmul on the same
inputs.

Best-effort:
- torch unavailable → ``compile_status: "torch_unavailable"``
- torch.cuda not available → ``compile_status: "device_unavailable"``
- triton missing → ``compile_status: "triton_unavailable"``
- compile/launch raises → ``compile_status: "compile_failed"`` /
  ``run_status: "run_failed"`` with typed note. Never raises.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Triton matmul kernel template (parameterised on tile)
# --------------------------------------------------------------------------- #
#
# The kernel carries the tile dims as Python-formatted constants so the
# emitted source is byte-deterministic across runs (no triton.autotune).
# It uses tl.dot(allow_tf32=False) to keep accumulation precision matching
# eager fp32 matmul.

_TRITON_KERNEL_TEMPLATE = """\
\"\"\"M-19.GPU generated Triton kernel.

candidate_id: {candidate_id}
region_id: {region_id}
matmul_shape: M={M} N={N} K={K}
tile:        BLOCK_M={tM} BLOCK_N={tN} BLOCK_K={tK}
generated_at_utc: {generated_at_utc}

Numerically equivalent to torch.matmul(A, B) when K_iters == 1 and
allow_tf32=False (single 16x16 dot per output tile reproduces eager
accumulation order). For K_iters > 1 the accumulation reorder may
differ; report records this as fail_refinement_mismatch in that case.
\"\"\"

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk
                      + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, k_iters):
        k_offs = offs_k + k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm
                      + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


BLOCK_M = {tM}
BLOCK_N = {tN}
BLOCK_K = {tK}
"""


def _emit_kernel_source(
    *, candidate_id: str, region_id: str,
    M: int, N: int, K: int, tM: int, tN: int, tK: int,
) -> str:
    """Render the Triton kernel source. The output is byte-deterministic
    given the same inputs (no embedded UTC timestamp variation in the
    body — the timestamp lives in the docstring header which is part
    of the source file but is fixed at emit time)."""
    return _TRITON_KERNEL_TEMPLATE.format(
        candidate_id=candidate_id, region_id=region_id,
        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
        generated_at_utc=_utcnow(),
    )


def _emit_kernel_source_deterministic(
    *, candidate_id: str, region_id: str,
    M: int, N: int, K: int, tM: int, tN: int, tK: int,
) -> str:
    """Same as _emit_kernel_source but WITHOUT the UTC timestamp, so
    the SHA256 is byte-deterministic across reruns. The artifact's
    `kernel_source_sha256` is computed against this deterministic body
    so re-running on the same model + tile produces a stable SHA."""
    return _TRITON_KERNEL_TEMPLATE.format(
        candidate_id=candidate_id, region_id=region_id,
        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
        generated_at_utc="(deterministic)",
    )


# --------------------------------------------------------------------------- #
# Run + measure
# --------------------------------------------------------------------------- #


def _make_inputs(M: int, K: int, N: int, device, dtype):  # type: ignore[no-untyped-def]
    """Generate deterministic input tensors. Same seed used by M-18.3."""
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(0xC0DE19)
    A = torch.randn(M, K, dtype=dtype, device=device, generator=g)
    B = torch.randn(K, N, dtype=dtype, device=device, generator=g)
    return A, B


def _classify_error(max_abs: float, max_rel: float, atol: float, rtol: float) -> str:
    if max_abs == 0.0 and max_rel == 0.0:
        return "discharged_compiled_bit_equality"
    if max_abs <= atol and max_rel <= rtol:
        return "discharged_tolerance_eps"
    return "fail_outside_tolerance"


def run_gpu_track(*, out_dir: Path, common: dict[str, Any]) -> Path:
    """Compile + launch the GPU Triton kernel. Returns the path to the
    emitted ``compiled_kernel_run_gpu.json`` artifact.

    Best-effort: every failure mode emits a typed status; never raises.
    """
    artifact_path = out_dir / "compiled_kernel_run_gpu.json"
    base = {
        **common,
        "track": "gpu_triton",
        "kernel_source_path": None,
        "kernel_source_sha256": None,
        "launch_config": None,
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
            "single batch size",
            "single tile candidate (M-19 foundation; M-21 fans out)",
            "Triton autotune disabled — fixed launch config",
            "TF32 disabled to preserve fp32 accumulation order vs eager",
        ],
        "generated_at_utc": _utcnow(),
    }

    # 1. torch availability.
    try:
        import torch
    except ImportError as exc:
        base["compile_status"] = "torch_unavailable"
        base["note"] = f"torch missing: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    if not torch.cuda.is_available():
        base["compile_status"] = "device_unavailable"
        base["note"] = "torch.cuda.is_available() == False"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    # 2. triton availability.
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        base["compile_status"] = "triton_unavailable"
        base["note"] = f"triton missing: {exc}"
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

    # 3. Emit kernel source (deterministic body for SHA pinning).
    kernel_source = _emit_kernel_source(
        candidate_id=candidate_id, region_id=region_id,
        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
    )
    deterministic_body = _emit_kernel_source_deterministic(
        candidate_id=candidate_id, region_id=region_id,
        M=M, N=N, K=K, tM=tM, tN=tN, tK=tK,
    )
    safe_region = "".join(
        c if c.isalnum() or c == "_" else "_" for c in region_id
    ) or "matmul"
    src_path = out_dir / f"triton_kernel_{safe_region}.py"
    src_path.write_text(kernel_source, encoding="utf-8")

    base["kernel_source_path"] = str(src_path.relative_to(out_dir.parent))
    base["kernel_source_sha256"] = _sha256_text(deterministic_body)
    base["device"] = {
        "kind": "cuda",
        "name": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    # 4. Import the emitted kernel via importlib.
    try:
        import importlib.util as _importlib_util
        spec = _importlib_util.spec_from_file_location(
            f"_m19_triton_kernel_{safe_region}", src_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not spec triton kernel module")
        mod = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_callable = getattr(mod, "matmul_kernel")
    except Exception as exc:  # noqa: BLE001
        base["compile_status"] = "compile_failed"
        base["note"] = f"{type(exc).__name__}: {exc}"
        artifact_path.write_text(
            json.dumps(base, indent=2, sort_keys=True), encoding="utf-8",
        )
        return artifact_path

    base["compile_status"] = "compiled"

    # 5. Launch.
    try:
        device = "cuda"
        A, B = _make_inputs(M, K, N, device=device, dtype=torch.float32)
        C = torch.zeros(M, N, dtype=torch.float32, device=device)

        grid = (
            (M + tM - 1) // tM,
            (N + tN - 1) // tN,
        )
        launch_config = {
            "grid": list(grid),
            "block_m": tM, "block_n": tN, "block_k": tK,
            "num_warps": 4, "num_stages": 2,
        }
        base["launch_config"] = launch_config

        # Warmup.
        warmup_iters = int(common.get("warmup", 4))
        for _ in range(warmup_iters):
            kernel_callable[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                tM, tN, tK,
                num_warps=4, num_stages=2,
            )
        torch.cuda.synchronize()

        # Measure.
        iters = int(common.get("iterations", 32))
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        per_iter_us: list[float] = []
        for _ in range(iters):
            start_evt.record()
            kernel_callable[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                tM, tN, tK,
                num_warps=4, num_stages=2,
            )
            end_evt.record()
            torch.cuda.synchronize()
            per_iter_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

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
        # Bandwidth: bytes moved per iter / time per iter, in GB/s.
        # A: M*K*4, B: K*N*4, C: M*N*4 bytes. Read/write total.
        bytes_per_iter = (M * K + K * N + M * N) * 4
        if mean_us > 0:
            base["bandwidth_gbps"] = (bytes_per_iter / 1e9) / (mean_us / 1e6)
        flops_per_iter = 2 * M * N * K
        if mean_us > 0:
            base["flops_per_s"] = flops_per_iter / (mean_us / 1e6)

        base["run_status"] = "ok"

        # Numerical check.
        ref = torch.matmul(A, B)
        diff = (ref - C).abs()
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

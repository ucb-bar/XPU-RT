"""Kernel Lifetime Evidence (register pressure, occupancy,
shared memory, optional ncu dynamic counters).

Fills the gap row 3 (compiled_lifetime) explicitly leaves open
in its honest non-claims. Two complementary data sources:

1. **Triton CompiledKernel introspection** (PRIMARY; always available,
   no perms required). Re-runs each /Triton kernel and
   captures from the resulting ``triton.compiler.CompiledKernel``:

   - ``n_regs``                   register count per thread
   - ``n_spills``                 register spills (0 = no spilling)
   - ``metadata.shared``          shared memory per block in bytes
   - ``metadata.num_warps``       launch config
   - ``metadata.num_stages``      pipeline depth
   - ``metadata.target.arch``     compute capability (e.g. sm75)
   - ``len(asm['ptx'])``          PTX size (proxy for code size)
   - ``len(asm['cubin'])``        cubin size (binary size)
   - ``theoretical_occupancy``    derived from sm-target limits

2. **ncu (Nsight Compute) dynamic counters** (OPTIONAL; admin-only).
   When ``RmProfilingAdminOnly=0`` (or run as root), ncu attempts to
   collect dynamic counters: SM efficiency, achieved occupancy,
   memory throughput, cache hit rates. When admin-only is set
   (typical user environment), emits typed ``ncu_admin_only``.

The combination flips row 3 (compiled_lifetime) from
``ready_for_m24_1`` → ``ready`` when Triton introspection succeeds
(no admin needed). Dynamic counters are an additive refinement.

Hard non-goals:
- No new measurement (kernels are re-run, not regenerated).
- No compiler-core imports.
No mutation of ////source artifacts.
fp32 only (matches //).
- Best-effort everywhere: missing torch / triton / kernel source /
  ncu blocked → typed unavailable. Never raises.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# SM-target limits (used for theoretical occupancy)
# --------------------------------------------------------------------------- #
# Reference: NVIDIA CUDA Programming Guide, Compute Capability table.
# Per-SM resource limits on Turing (sm75), Ampere (sm80/86), Ada (sm89),
# Hopper (sm90). Conservative; "max threads per SM" is the limiting
# factor for most matmul kernels.
_SM_LIMITS: dict[int, dict[str, int]] = {
    75: {  # Turing (TITAN RTX, T4, RTX 20-series)
        "max_threads_per_sm": 1024,
        "max_blocks_per_sm": 16,
        "max_warps_per_sm": 32,
        "registers_per_sm": 65536,
        "shared_mem_per_sm_bytes": 65536,
        "warp_size": 32,
    },
    80: {  # Ampere SM80 (A100)
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_warps_per_sm": 64,
        "registers_per_sm": 65536,
        "shared_mem_per_sm_bytes": 167936,
        "warp_size": 32,
    },
    86: {  # Ampere SM86 (RTX 30-series)
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 16,
        "max_warps_per_sm": 48,
        "registers_per_sm": 65536,
        "shared_mem_per_sm_bytes": 102400,
        "warp_size": 32,
    },
    89: {  # Ada (RTX 40-series)
        "max_threads_per_sm": 1536,
        "max_blocks_per_sm": 24,
        "max_warps_per_sm": 48,
        "registers_per_sm": 65536,
        "shared_mem_per_sm_bytes": 102400,
        "warp_size": 32,
    },
    90: {  # Hopper (H100)
        "max_threads_per_sm": 2048,
        "max_blocks_per_sm": 32,
        "max_warps_per_sm": 64,
        "registers_per_sm": 65536,
        "shared_mem_per_sm_bytes": 233472,
        "warp_size": 32,
    },
}


def _theoretical_occupancy(
    *,
    arch: int,
    n_regs: int,
    shared_mem: int,
    threads_per_block: int,
) -> dict[str, Any]:
    """Compute theoretical SM occupancy from kernel attributes + SM
    limits. Returns dict with active_warps_per_sm, active_blocks_per_sm,
    occupancy_fraction (0.0–1.0). Falls back gracefully when arch is
    unknown."""
    limits = _SM_LIMITS.get(arch)
    if limits is None:
        return {
            "occupancy_fraction": None,
            "active_warps_per_sm": None,
            "active_blocks_per_sm": None,
            "limit": "unknown_arch",
            "arch": arch,
        }

    warp_size = limits["warp_size"]
    warps_per_block = max(1, (threads_per_block + warp_size - 1) // warp_size)

    # Block count limited by:
    # 1. max_blocks_per_sm (architectural cap)
    # 2. registers: registers_per_sm / (n_regs * threads_per_block)
    # 3. shared mem: shared_mem_per_sm_bytes / shared_mem (if non-zero)
    # 4. warps: max_warps_per_sm / warps_per_block
    cap_blocks_arch = limits["max_blocks_per_sm"]
    cap_blocks_warps = (
        limits["max_warps_per_sm"] // warps_per_block
        if warps_per_block > 0 else 0
    )
    cap_blocks_regs = (
        limits["registers_per_sm"] // (n_regs * threads_per_block)
        if (n_regs > 0 and threads_per_block > 0) else cap_blocks_arch
    )
    cap_blocks_shmem = (
        limits["shared_mem_per_sm_bytes"] // shared_mem
        if shared_mem > 0 else cap_blocks_arch
    )

    blocks = min(
        cap_blocks_arch, cap_blocks_warps,
        cap_blocks_regs, cap_blocks_shmem,
    )
    blocks = max(0, blocks)
    active_warps = blocks * warps_per_block

    # Identify the limiting factor.
    candidates = (
        ("arch_block_cap", cap_blocks_arch),
        ("warp_count", cap_blocks_warps),
        ("registers", cap_blocks_regs),
        ("shared_mem", cap_blocks_shmem),
    )
    limit = min(candidates, key=lambda c: c[1])[0]

    return {
        "occupancy_fraction": (
            active_warps / limits["max_warps_per_sm"]
            if limits["max_warps_per_sm"] > 0 else None
        ),
        "active_warps_per_sm": active_warps,
        "active_blocks_per_sm": blocks,
        "warps_per_block": warps_per_block,
        "limit": limit,
        "arch": arch,
        "sm_limits_used": limits,
    }


# --------------------------------------------------------------------------- #
# Triton kernel introspection
# --------------------------------------------------------------------------- #


def _introspect_triton_kernel(
    *,
    kernel_src_path: Path,
    matmul_shape: tuple[int, int, int],
    tile: tuple[int, int, int],
) -> dict[str, Any]:
    """Re-run an -emitted Triton kernel once to obtain the
    ``CompiledKernel`` object and extract static attributes. Returns
    a dict with register_pressure, register_spills, shared_memory_bytes,
    theoretical_occupancy, etc."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return {"introspection_status": "torch_unavailable",
                "reason": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"introspection_status": "cuda_unavailable"}
    try:
        import triton  # noqa: F401
    except ImportError:
        return {"introspection_status": "triton_unavailable"}
    try:
        import importlib.util
    except ImportError as exc:  # pragma: no cover
        return {"introspection_status": "importlib_unavailable",
                "reason": f"{type(exc).__name__}: {exc}"}

    try:
        spec = importlib.util.spec_from_file_location(
            f"_m24_1_kernel_{kernel_src_path.stem}", kernel_src_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not spec triton kernel module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_callable = getattr(mod, "matmul_kernel", None)
        if kernel_callable is None:
            kernel_callable = getattr(mod, "fused_kernel", None)
        if kernel_callable is None:
            return {"introspection_status": "kernel_not_found",
                    "reason": (
                        "neither matmul_kernel nor fused_kernel "
                        "exported by this module"
                    )}
    except Exception as exc:  # noqa: BLE001
        return {"introspection_status": "import_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    M, N, K = matmul_shape
    tM, tN, tK = tile
    device = "cuda"

    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(0xC0DE241)
        A = torch.randn(M, K, dtype=torch.float32,
                        device=device, generator=gen)
        B = torch.randn(K, N, dtype=torch.float32,
                        device=device, generator=gen)
        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        grid = ((M + tM - 1) // tM, (N + tN - 1) // tN)
        # Try matmul_kernel signature first (layout).
        try:
            compiled = kernel_callable[grid](
                A, B, C,
                M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                tM, tN, tK,
                num_warps=4, num_stages=2,
            )
        except TypeError:
            # Fall back to fused_kernel signature (layout).
            n_elems = M * N
            bias_len = N
            BLOCK = max(16, ((N + 15) // 16) * 16)
            grid_fused = ((n_elems + BLOCK - 1) // BLOCK,)
            b_arg = torch.zeros(bias_len, dtype=torch.float32, device=device)
            compiled = kernel_callable[grid_fused](
                A, b_arg, C,
                n_elems, bias_len,
                BLOCK=BLOCK,
            )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return {"introspection_status": "launch_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    if compiled is None:
        return {"introspection_status": "compile_returned_none"}

    n_regs = int(getattr(compiled, "n_regs", 0) or 0)
    n_spills = int(getattr(compiled, "n_spills", 0) or 0)
    metadata = getattr(compiled, "metadata", None)
    shared = 0
    num_warps = 0
    num_stages = 0
    arch = 0
    warp_size = 32
    if metadata is not None:
        shared = int(getattr(metadata, "shared", 0) or 0)
        num_warps = int(getattr(metadata, "num_warps", 0) or 0)
        num_stages = int(getattr(metadata, "num_stages", 0) or 0)
        target = getattr(metadata, "target", None)
        if target is not None:
            arch = int(getattr(target, "arch", 0) or 0)
            warp_size = int(getattr(target, "warp_size", 32) or 32)

    threads_per_block = num_warps * warp_size if num_warps > 0 else 0
    occ = _theoretical_occupancy(
        arch=arch, n_regs=n_regs,
        shared_mem=shared,
        threads_per_block=threads_per_block,
    )

    asm = getattr(compiled, "asm", {}) or {}
    ptx_source = asm.get("ptx") if isinstance(asm, dict) else None
    cubin_bytes = asm.get("cubin") if isinstance(asm, dict) else None
    ptx_size_bytes = len(ptx_source) if ptx_source is not None else None
    cubin_size_bytes = (
        len(cubin_bytes) if cubin_bytes is not None else None
    )

    return {
        "introspection_status": "introspected",
        "register_pressure": n_regs,
        "register_spills": n_spills,
        "shared_memory_bytes": shared,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "threads_per_block": threads_per_block,
        "warp_size": warp_size,
        "target_arch": arch,
        "ptx_size_bytes": ptx_size_bytes,
        "cubin_size_bytes": cubin_size_bytes,
        "theoretical_occupancy": occ,
        "data_source": "triton.compiler.CompiledKernel",
    }


# --------------------------------------------------------------------------- #
# ncu (Nsight Compute) availability + collection
# --------------------------------------------------------------------------- #


def _ncu_available() -> tuple[bool, str, str | None]:
    """Return (available, reason, ncu_path).

    Available iff ncu is on PATH OR at the standard CUDA install
    location, AND the kernel does not enforce admin-only profiling
    counters (``RmProfilingAdminOnly=0``).
    """
    candidates = [
        "ncu",
        "/usr/local/cuda/bin/ncu",
        "/usr/local/cuda-12.6/bin/ncu",
        "/usr/local/cuda-12/bin/ncu",
    ]
    ncu_path: str | None = None
    for cand in candidates:
        resolved = shutil.which(cand) if not cand.startswith("/") else (
            cand if Path(cand).exists() else None
        )
        if resolved is not None:
            ncu_path = resolved
            break
    if ncu_path is None:
        return False, "ncu binary not on PATH or in /usr/local/cuda*/bin", None

    # Probe the admin-only flag.
    params_path = Path("/proc/driver/nvidia/params")
    admin_only = None
    if params_path.exists():
        try:
            for line in params_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("RmProfilingAdminOnly:"):
                    admin_only = int(line.split(":")[1].strip())
                    break
        except (OSError, ValueError):
            pass

    if admin_only == 1 and os.geteuid() != 0:
        return (
            False,
            f"ncu present at {ncu_path} but RmProfilingAdminOnly=1 "
            f"and not root (need root or kernel param "
            f"RmProfilingAdminOnly=0)",
            ncu_path,
        )

    return True, f"ncu available at {ncu_path}", ncu_path


def _collect_ncu_dynamic_metrics(
    *,
    ncu_path: str,
    kernel_src_path: Path,
    matmul_shape: tuple[int, int, int],
    tile: tuple[int, int, int],
) -> dict[str, Any]:
    """Spawn a tiny driver process and wrap it under ncu to collect
    dynamic counters: sm__throughput, l1tex__throughput, etc.
    Best-effort: any failure → typed unavailable. Never raises."""
    M, N, K = matmul_shape
    tM, tN, tK = tile
    driver = (
        "import importlib.util, sys, torch\n"
        f"src = {str(kernel_src_path)!r}\n"
        f"M, N, K = {M}, {N}, {K}\n"
        f"tM, tN, tK = {tM}, {tN}, {tK}\n"
        "spec = importlib.util.spec_from_file_location('m241_kernel', src)\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "kernel = getattr(mod, 'matmul_kernel', None) or "
        "getattr(mod, 'fused_kernel', None)\n"
        "if kernel is None: sys.exit(1)\n"
        "g = torch.Generator(device='cuda'); g.manual_seed(0xC0DE241)\n"
        "A = torch.randn(M, K, dtype=torch.float32, device='cuda', generator=g)\n"
        "B = torch.randn(K, N, dtype=torch.float32, device='cuda', generator=g)\n"
        "C = torch.zeros(M, N, dtype=torch.float32, device='cuda')\n"
        "grid = ((M+tM-1)//tM, (N+tN-1)//tN)\n"
        "try:\n"
        "    kernel[grid](A, B, C, M, N, K,\n"
        "        A.stride(0), A.stride(1), B.stride(0), B.stride(1),\n"
        "        C.stride(0), C.stride(1), tM, tN, tK,\n"
        "        num_warps=4, num_stages=2)\n"
        "except TypeError:\n"
        "    n = M*N; BL = max(16, ((N+15)//16)*16)\n"
        "    bias = torch.zeros(N, dtype=torch.float32, device='cuda')\n"
        "    kernel[((n+BL-1)//BL,)](A, bias, C, n, N, BLOCK=BL)\n"
        "torch.cuda.synchronize()\n"
    )

    # NOTE: ncu does NOT accept the `--` separator before the program;
    # without it, options are unambiguously parsed.
    cmd = [
        ncu_path,
        "--launch-count", "1",
        "--metrics",
        ",".join((
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
            "launch__block_size",
            "launch__waves_per_multiprocessor",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "smsp__cycles_active.avg",
        )),
        "--print-summary", "per-kernel",
        "python3", "-c", driver,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=120,
            check=False,
            env={**os.environ},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ncu_status": "ncu_subprocess_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    stderr_text = proc.stderr or ""
    stdout_text = proc.stdout or ""

    # Common error: ERR_NVGPUCTRPERM (admin-only).
    if "ERR_NVGPUCTRPERM" in stderr_text:
        return {
            "ncu_status": "ncu_admin_only",
            "reason": (
                "ERR_NVGPUCTRPERM — performance counters require "
                "root or RmProfilingAdminOnly=0"
            ),
            "stderr_tail": stderr_text[-300:],
        }
    if proc.returncode != 0:
        return {
            "ncu_status": "ncu_subprocess_nonzero",
            "reason": f"returncode={proc.returncode}",
            "stderr_tail": stderr_text[-300:],
        }

    # Best-effort parse — ncu's per-kernel output isn't trivially
    # JSON-able. We capture the raw output and surface a summary.
    return {
        "ncu_status": "ncu_collected",
        "stdout_tail": stdout_text[-1000:],
        "stderr_tail": stderr_text[-300:],
    }


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelLifetimeResult:
    overall: str  # "ok" | "no_regions" | "not_run"
    out_dir: Path
    report_path: Path
    region_count: int
    introspected_count: int
    ncu_collected_count: int


def _kernel_src_for_region(
    *, run_dir: Path, region_id: str,
) -> Path | None:
    """Find the /-emitted Triton source file for a region."""
    base = run_dir / "02_graph_analysis" / "kernel_execution"
    candidates = [
        base / "regions" / region_id / f"triton_kernel_{region_id}.py",
        base / f"triton_kernel_{region_id}.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_kernel_lifetime_evidence(
    run_dir: Path,
) -> KernelLifetimeResult:
    """Build kernel-lifetime evidence layer. Best-effort;
    never raises."""
    run_dir = Path(run_dir).resolve()
    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "kernel_lifetime"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "kernel_lifetime_evidence_report.json"

    # Source: 's region list (every region with compiled evidence).
    cb = _read_json(
        ga / "compiled_bottleneck" / "compiled_bottleneck_report.json"
    )
    if cb is None or cb.get("overall") != "ok":
        body = {
            "schema_version": "kernel_lifetime_evidence_report_v1",
            "overall": "not_run",
            "reason": "M-22 compiled_bottleneck not_run",
            "regions": [],
            "ncu_availability": dict(zip(
                ("available", "reason", "ncu_path"),
                _ncu_available(),
            )),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return KernelLifetimeResult(
            overall="not_run", out_dir=out_dir, report_path=report_path,
            region_count=0, introspected_count=0, ncu_collected_count=0,
        )

    ncu_avail, ncu_reason, ncu_path = _ncu_available()

    regions_out: list[dict[str, Any]] = []
    introspected_count = 0
    ncu_collected_count = 0
    for r in cb.get("regions", []) or []:
        if r.get("model_status") != "ok":
            continue
        rid = str(r.get("region_id") or "")
        if not rid:
            continue
        sh = r.get("matmul_shape") or {}
        t = r.get("tile") or {}
        try:
            shape = (int(sh["M"]), int(sh["N"]), int(sh["K"]))
            tile = (int(t["M"]), int(t["N"]), int(t["K"]))
        except (KeyError, ValueError, TypeError):
            continue

        kernel_src = _kernel_src_for_region(run_dir=run_dir, region_id=rid)
        if kernel_src is None:
            regions_out.append({
                "region_id": rid,
                "candidate_id": r.get("candidate_id"),
                "introspection_status": "kernel_source_missing",
                "reason": (
                    "no triton_kernel_<region>.py under "
                    "kernel_execution/ or kernel_execution/regions/<rid>/"
                ),
            })
            continue

        triton_data = _introspect_triton_kernel(
            kernel_src_path=kernel_src,
            matmul_shape=shape, tile=tile,
        )
        if triton_data.get("introspection_status") == "introspected":
            introspected_count += 1

        ncu_data: dict[str, Any] = {"ncu_status": "ncu_unavailable",
                                    "reason": ncu_reason}
        if ncu_avail and ncu_path is not None:
            ncu_data = _collect_ncu_dynamic_metrics(
                ncu_path=ncu_path,
                kernel_src_path=kernel_src,
                matmul_shape=shape, tile=tile,
            )
            if ncu_data.get("ncu_status") == "ncu_collected":
                ncu_collected_count += 1

        regions_out.append({
            "region_id": rid,
            "candidate_id": r.get("candidate_id"),
            "matmul_shape": {"M": shape[0], "N": shape[1], "K": shape[2]},
            "tile": {"M": tile[0], "N": tile[1], "K": tile[2]},
            "kernel_source_path": str(kernel_src.relative_to(run_dir)),
            "triton_introspection": triton_data,
            "ncu_evidence": ncu_data,
        })

    overall = "ok" if regions_out else "no_regions"

    body = {
        "schema_version": "kernel_lifetime_evidence_report_v1",
        "overall": overall,
        "ncu_availability": {
            "available": ncu_avail, "reason": ncu_reason, "ncu_path": ncu_path,
        },
        "regions": regions_out,
        "summary": {
            "region_count": len(regions_out),
            "introspected_count": introspected_count,
            "ncu_collected_count": ncu_collected_count,
        },
        "known_limitations": [
            "static kernel attributes (registers, shared mem, "
            "occupancy) come from triton.compiler.CompiledKernel "
            "introspection — deterministic, no admin needed",
            "dynamic counters (SM throughput, achieved occupancy, "
            "cache hit rates) require ncu with non-admin "
            "perf-counter access (RmProfilingAdminOnly=0)",
            "fp32 only",
            "single launch config per kernel (num_warps=4, num_stages=2 "
            "fixed by M-19/M-23 templates)",
            "theoretical_occupancy uses architectural per-SM limits; "
            "achieved occupancy may differ due to launch geometry",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )

    return KernelLifetimeResult(
        overall=overall, out_dir=out_dir, report_path=report_path,
        region_count=len(regions_out),
        introspected_count=introspected_count,
        ncu_collected_count=ncu_collected_count,
    )

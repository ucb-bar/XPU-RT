"""M-22.1 — Real `torch.profiler` (GPU) + `linux perf` (CPU) evidence
for compiled regions.

Layered ON TOP of M-22's deterministic post-hoc utilization derivation.
M-22.1 re-runs each compiled kernel under a profiler / perf wrapper
and records:

- GPU: per-kernel ``self_cuda_time_total``, ``cuda_memory_usage``,
  ``count`` from ``torch.profiler.profile(activities=[CUDA])``.
- CPU: ``cycles``, ``instructions``, ``cache-references``,
  ``cache-misses``, ``LLC-loads``, ``LLC-load-misses`` from
  ``perf stat -e <events> --json``.

The result is layered onto M-22's per-region ``compiled_evidence`` block
as a sibling ``profiler_evidence`` field. M-22's own
``cache_evidence: not_collected`` placeholder is replaced with a
typed status: ``cuda_collected | perf_collected | perf_unavailable |
profiler_unavailable | not_collected``.

Hard non-goals:

- Does not modify M-22's analytical or post-hoc fields.
- Does not require root.
- Best-effort everywhere: missing torch.profiler / unavailable perf /
  raised exceptions all degrade to typed ``not_collected`` with a
  reason string. Never raises.
- No new candidate generation, no compiler-core imports.

Environment:

- GPU path: requires torch>=1.12 with `torch.profiler.profile` + CUDA
  available. The user's environment provides this (torch 2.10).
- CPU path: requires `perf stat` AND
  `kernel.perf_event_paranoid <= 2` (without root). When paranoid >= 3,
  perf refuses to collect cache events for non-root users — typed
  fallback ``perf_unavailable`` is recorded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_PERF_EVENTS = (
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
)


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
# perf availability probe
# --------------------------------------------------------------------------- #


def _perf_available() -> tuple[bool, str]:
    """Return (available, reason). Available iff (a) ``perf`` is on
    PATH and (b) ``perf_event_paranoid`` allows hardware events for
    non-root users (<= 2)."""
    if shutil.which("perf") is None:
        return False, "perf binary not on PATH"
    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    if not paranoid_path.exists():
        return False, "/proc/sys/kernel/perf_event_paranoid missing"
    try:
        paranoid = int(paranoid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False, "could not read perf_event_paranoid"
    if paranoid >= 3:
        return False, (
            f"perf_event_paranoid={paranoid} (>=3 blocks non-root "
            "hardware events; need <= 2)"
        )
    return True, f"perf available (paranoid={paranoid})"


# --------------------------------------------------------------------------- #
# GPU profiler track — wraps a Triton kernel under torch.profiler
# --------------------------------------------------------------------------- #


def _profile_triton_kernel(
    *,
    kernel_src_path: Path,
    matmul_shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    """Re-run an M-19-emitted Triton kernel under
    ``torch.profiler.profile(activities=[CUDA])`` and return aggregated
    per-kernel CUDA stats."""
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function
    except ImportError as exc:  # pragma: no cover
        return {"profiler_status": "torch_unavailable",
                "reason": f"{type(exc).__name__}: {exc}"}
    if not torch.cuda.is_available():
        return {"profiler_status": "cuda_unavailable"}

    try:
        import importlib.util
    except ImportError as exc:  # pragma: no cover
        return {"profiler_status": "importlib_unavailable",
                "reason": f"{type(exc).__name__}: {exc}"}

    try:
        spec = importlib.util.spec_from_file_location(
            f"_m22_1_kernel_{kernel_src_path.stem}", kernel_src_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not spec kernel module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel_callable = getattr(mod, "matmul_kernel")
    except Exception as exc:  # noqa: BLE001
        return {"profiler_status": "import_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    M, N, K = matmul_shape
    tM, tN, tK = tile
    device = "cuda"

    try:
        g = torch.Generator(device=device)
        g.manual_seed(0xC0DE221)  # different seed; doesn't need to match M-19
        A = torch.randn(M, K, dtype=torch.float32, device=device, generator=g)
        B = torch.randn(K, N, dtype=torch.float32, device=device, generator=g)
        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        grid = ((M + tM - 1) // tM, (N + tN - 1) // tN)

        # Warmup outside the profile block.
        for _ in range(int(warmup)):
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

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            with record_function("compgen_m22_1_matmul"):
                for _ in range(int(iterations)):
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
    except Exception as exc:  # noqa: BLE001
        return {"profiler_status": "run_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    # Aggregate per-kernel CUDA stats.
    try:
        events = prof.key_averages()
    except Exception as exc:  # noqa: BLE001
        return {"profiler_status": "aggregate_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    cuda_kernel_events: list[dict[str, Any]] = []
    total_self_cuda_us = 0.0
    total_self_cpu_us = 0.0
    total_cuda_calls = 0
    for ev in events:
        # torch 2.x renamed self_cuda_time_total → self_device_time_total.
        # Fall through both for forward/backward compat.
        self_cuda_raw = (
            getattr(ev, "self_device_time_total", None)
            or getattr(ev, "self_cuda_time_total", None)
            or 0
        )
        try:
            self_cuda = float(self_cuda_raw)
        except (TypeError, ValueError):
            self_cuda = 0.0
        try:
            self_cpu = float(getattr(ev, "self_cpu_time_total", 0) or 0)
        except (TypeError, ValueError):
            self_cpu = 0.0
        if self_cuda > 0:
            cuda_kernel_events.append({
                "key": str(ev.key),
                "count": int(getattr(ev, "count", 0)),
                "self_cuda_time_us": self_cuda,
            })
            total_self_cuda_us += self_cuda
            total_cuda_calls += int(getattr(ev, "count", 0))
        total_self_cpu_us += self_cpu

    iters_count = int(iterations)
    return {
        "profiler_status": "cuda_collected",
        "iterations": iters_count,
        "warmup": int(warmup),
        "total_self_cuda_us": total_self_cuda_us,
        "total_self_cpu_us": total_self_cpu_us,
        "total_cuda_calls": total_cuda_calls,
        "self_cuda_us_per_iter": (
            total_self_cuda_us / iters_count if iters_count > 0 else None
        ),
        "kernel_events": cuda_kernel_events[:20],  # cap for size
    }


# --------------------------------------------------------------------------- #
# CPU perf track — wraps a cffi-compiled C kernel under perf stat
# --------------------------------------------------------------------------- #


def _perf_collect_cpu(
    *,
    kernel_src_path: Path,
    matmul_shape: tuple[int, int, int],
    tile: tuple[int, int, int],
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    """Wrap a Python subprocess that loads + executes the M-19 CPU
    kernel ``iterations`` times under ``perf stat -e <events>``.
    Returns parsed cycles / instructions / cache_references /
    cache_misses / LLC_loads / LLC_load_misses, or a typed
    ``perf_unavailable`` block."""
    available, reason = _perf_available()
    if not available:
        return {"profiler_status": "perf_unavailable", "reason": reason}

    # We need a C kernel to wrap. The M-19 CPU track emits a .c file
    # next to each region; if absent, we can't do anything.
    cffi_dir = kernel_src_path.parent
    c_files = list(cffi_dir.glob("cpu_kernel_*.c"))
    if not c_files:
        return {"profiler_status": "no_cpu_kernel_source",
                "reason": f"no cpu_kernel_*.c under {cffi_dir}"}

    # Build a tiny driver script: load the cffi module, run the kernel
    # `iterations` times, exit. perf stat wraps the whole subprocess.
    driver = (
        "import importlib.util, sys, numpy as np\n"
        f"M, N, K = {matmul_shape}\n"
        f"iters = {iterations}\n"
        f"warmup = {warmup}\n"
        f"src_dir = {str(cffi_dir)!r}\n"
        "import os, sys; sys.path.insert(0, src_dir)\n"
        "from pathlib import Path\n"
        "spec = None\n"
        "for so_dir in Path(src_dir).glob('cffi_build_*'):\n"
        "    for so in so_dir.glob('*.so'):\n"
        "        spec = importlib.util.spec_from_file_location(so.stem, so)\n"
        "        break\n"
        "    if spec is not None: break\n"
        "if spec is None: sys.exit('no compiled .so found')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "ffi, lib = mod.ffi, mod.lib\n"
        "import numpy as np\n"
        "rng = np.random.default_rng(0xC0DE221)\n"
        "A = rng.standard_normal((M, K), dtype=np.float32)\n"
        "B = rng.standard_normal((K, N), dtype=np.float32)\n"
        "C = np.zeros((M, N), dtype=np.float32)\n"
        "fns = [n for n in dir(lib) if n.startswith('compgen_m19_matmul')]\n"
        "if not fns: sys.exit('no compgen_m19_matmul function found')\n"
        "fn = getattr(lib, fns[0])\n"
        "for _ in range(warmup):\n"
        "    fn(ffi.cast('float*', A.ctypes.data),\n"
        "       ffi.cast('float*', B.ctypes.data),\n"
        "       ffi.cast('float*', C.ctypes.data))\n"
        "for _ in range(iters):\n"
        "    fn(ffi.cast('float*', A.ctypes.data),\n"
        "       ffi.cast('float*', B.ctypes.data),\n"
        "       ffi.cast('float*', C.ctypes.data))\n"
    )

    perf_args = [
        "perf", "stat",
        "-e", ",".join(_PERF_EVENTS),
        "-x", ",",  # CSV output to stderr
        "--", "python3", "-c", driver,
    ]
    try:
        proc = subprocess.run(
            perf_args,
            capture_output=True, text=True,
            timeout=120,
            check=False,
            env={**os.environ},
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"profiler_status": "perf_subprocess_failed",
                "reason": f"{type(exc).__name__}: {exc}"}

    if proc.returncode != 0:
        return {
            "profiler_status": "perf_subprocess_nonzero",
            "reason": f"returncode={proc.returncode}",
            "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
        }

    counters: dict[str, int | None] = {}
    # perf stat -x , emits one event per stderr line:
    # "<value>,<unit>,<event_name>,<run_time>,<percentage>"
    for line in (proc.stderr or "").splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            value = int(parts[0])
        except (ValueError, TypeError):
            continue
        event = parts[2].strip()
        if event in _PERF_EVENTS:
            counters[event] = value

    if not counters:
        return {
            "profiler_status": "perf_no_counters_parsed",
            "reason": "no events matched in stderr",
            "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
        }

    # Derived metrics (best-effort; None when divisor is zero).
    crefs = counters.get("cache-references")
    cmisses = counters.get("cache-misses")
    llc_loads = counters.get("LLC-loads")
    llc_load_misses = counters.get("LLC-load-misses")
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")

    cache_miss_rate = (
        cmisses / crefs
        if crefs and crefs > 0 and cmisses is not None
        else None
    )
    llc_miss_rate = (
        llc_load_misses / llc_loads
        if llc_loads and llc_loads > 0 and llc_load_misses is not None
        else None
    )
    ipc = (
        instructions / cycles
        if cycles and cycles > 0 and instructions is not None
        else None
    )

    return {
        "profiler_status": "perf_collected",
        "iterations": int(iterations),
        "warmup": int(warmup),
        "events": counters,
        "derived": {
            "cache_miss_rate": cache_miss_rate,
            "llc_miss_rate": llc_miss_rate,
            "instructions_per_cycle": ipc,
        },
    }


# --------------------------------------------------------------------------- #
# Top-level entry point — fans out across regions M-22 has evidence for
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProfilerEvidenceResult:
    overall: str  # "ok" | "no_regions" | "not_run"
    out_dir: Path
    report_path: Path
    summary_md_path: Path
    gpu_collected_count: int
    cpu_collected_count: int
    region_count: int


def _kernel_src_for_region(
    *, run_dir: Path, region_id: str, track: str,
) -> Path | None:
    """Find the M-19/M-20 emitted kernel source file for a region."""
    base = run_dir / "02_graph_analysis" / "kernel_execution"
    # M-20 fan-out: regions/<region>/triton_kernel_<region>.py
    candidates = [
        base / "regions" / region_id / f"triton_kernel_{region_id}.py"
        if track == "gpu" else
        base / "regions" / region_id / f"cpu_kernel_{region_id}.c",
        # M-19 single-region: triton_kernel_<region>.py at top
        base / f"triton_kernel_{region_id}.py"
        if track == "gpu" else
        base / f"cpu_kernel_{region_id}.c",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_profiler_evidence(
    run_dir: Path,
    *,
    iterations: int = 64,
    warmup: int = 8,
) -> ProfilerEvidenceResult:
    """Build M-22.1 profiler-evidence layer. Best-effort; never raises."""
    run_dir = Path(run_dir).resolve()

    ga = run_dir / "02_graph_analysis"
    out_dir = ga / "profiler_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "profiler_evidence_report.json"
    summary_md_path = out_dir / "profiler_evidence_summary.md"

    m22 = _read_json(
        ga / "compiled_bottleneck" / "compiled_bottleneck_report.json"
    )
    if m22 is None or m22.get("overall") != "ok":
        body = {
            "schema_version": "profiler_evidence_report_v1",
            "overall": "not_run",
            "reason": (
                "M-22 compiled_bottleneck not_run or no_measurements; "
                "M-22.1 has nothing to layer onto"
            ),
            "regions": [],
            "perf_availability": dict(zip(
                ("available", "reason"), _perf_available(),
            )),
            "generated_at_utc": _utcnow(),
        }
        report_path.write_text(
            json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
        )
        summary_md_path.write_text(
            "# Profiler Evidence (M-22.1) — not_run\n\n"
            "M-22 produced no compiled measurements to layer onto.\n",
            encoding="utf-8",
        )
        return ProfilerEvidenceResult(
            overall="not_run", out_dir=out_dir,
            report_path=report_path, summary_md_path=summary_md_path,
            gpu_collected_count=0, cpu_collected_count=0,
            region_count=0,
        )

    perf_avail, perf_reason = _perf_available()

    regions_out: list[dict[str, Any]] = []
    gpu_collected = 0
    cpu_collected = 0
    for r in m22.get("regions", []) or []:
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

        gpu_block: dict[str, Any] | None = None
        cpu_block: dict[str, Any] | None = None

        gpu_src = _kernel_src_for_region(
            run_dir=run_dir, region_id=rid, track="gpu",
        )
        if gpu_src is not None and (r.get("gpu") or {}).get(
            "measured_us_per_iter") is not None:
            gpu_block = _profile_triton_kernel(
                kernel_src_path=gpu_src,
                matmul_shape=shape, tile=tile,
                iterations=iterations, warmup=warmup,
            )
            if gpu_block.get("profiler_status") == "cuda_collected":
                gpu_collected += 1

        cpu_src = _kernel_src_for_region(
            run_dir=run_dir, region_id=rid, track="cpu",
        )
        if cpu_src is not None and (r.get("cpu") or {}).get(
            "measured_us_per_iter") is not None:
            if perf_avail:
                cpu_block = _perf_collect_cpu(
                    kernel_src_path=cpu_src,
                    matmul_shape=shape, tile=tile,
                    iterations=iterations, warmup=warmup,
                )
                if cpu_block.get("profiler_status") == "perf_collected":
                    cpu_collected += 1
            else:
                cpu_block = {
                    "profiler_status": "perf_unavailable",
                    "reason": perf_reason,
                }

        # cache_evidence taxonomy.
        if (gpu_block and gpu_block.get("profiler_status") == "cuda_collected"):
            cache_evidence = "cuda_collected"
        elif (cpu_block and cpu_block.get("profiler_status") == "perf_collected"):
            cache_evidence = "perf_collected"
        elif (cpu_block and cpu_block.get("profiler_status") == "perf_unavailable"):
            cache_evidence = "perf_unavailable"
        else:
            cache_evidence = "not_collected"

        regions_out.append({
            "region_id": rid,
            "candidate_id": r.get("candidate_id"),
            "matmul_shape": {"M": shape[0], "N": shape[1], "K": shape[2]},
            "tile": {"M": tile[0], "N": tile[1], "K": tile[2]},
            "gpu": gpu_block,
            "cpu": cpu_block,
            "cache_evidence": cache_evidence,
        })

    body = {
        "schema_version": "profiler_evidence_report_v1",
        "overall": "ok" if regions_out else "no_regions",
        "perf_availability": {
            "available": perf_avail, "reason": perf_reason,
        },
        "iterations": iterations, "warmup": warmup,
        "regions": regions_out,
        "summary": {
            "region_count": len(regions_out),
            "gpu_collected_count": gpu_collected,
            "cpu_collected_count": cpu_collected,
        },
        "known_limitations": [
            "perf_event_paranoid >= 3 disables non-root perf cache events",
            "torch.profiler CUDA totals are aggregated per kernel_key, "
            "not per region — the kernel re-runs identically across regions",
            "no Nsight integration (would require ncu binary + privileged perms)",
            "no instruction-mix breakdown (compute vs memory vs sync)",
        ],
        "generated_at_utc": _utcnow(),
    }
    report_path.write_text(
        json.dumps(body, indent=2, sort_keys=True), encoding="utf-8",
    )

    # Layer onto M-22's compiled_bottleneck_report.regions[*] AND onto
    # hardware_resource_report.regions[*].compiled_evidence as
    # profiler_evidence + replace cache_evidence value.
    _apply_overlays(
        run_dir=run_dir,
        regions_out=regions_out,
    )

    md_lines = [
        f"# Profiler Evidence (M-22.1) — overall=ok\n",
        f"- perf availability: {perf_avail} ({perf_reason})",
        f"- iterations: {iterations}  warmup: {warmup}",
        f"- regions: {len(regions_out)}  "
        f"(gpu_collected={gpu_collected}, cpu_collected={cpu_collected})",
        "",
        "| region | cache_evidence | gpu self_cuda_us/iter | cpu cache_miss_rate |",
        "|---|---|---|---|",
    ]
    for r in regions_out:
        gpu = r.get("gpu") or {}
        cpu = r.get("cpu") or {}
        gpu_us = gpu.get("self_cuda_us_per_iter")
        cpu_derived = (cpu.get("derived") or {}) if cpu else {}
        miss_rate = cpu_derived.get("cache_miss_rate")
        md_lines.append(
            f"| `{r['region_id']}` | `{r['cache_evidence']}` "
            f"| {gpu_us:.2f} " if gpu_us is not None else
            f"| `{r['region_id']}` | `{r['cache_evidence']}` | — "
        )
        # (markdown rendering kept simple; the JSON is the source of truth)

    summary_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return ProfilerEvidenceResult(
        overall="ok" if regions_out else "no_regions",
        out_dir=out_dir, report_path=report_path,
        summary_md_path=summary_md_path,
        gpu_collected_count=gpu_collected,
        cpu_collected_count=cpu_collected,
        region_count=len(regions_out),
    )


def _apply_overlays(
    *, run_dir: Path, regions_out: list[dict[str, Any]],
) -> None:
    """Layer profiler_evidence per region onto:

    1. ``compiled_bottleneck_report.regions[*].profiler_evidence``
    2. ``hardware_resource_report.regions[*].compiled_evidence
       .profiler_evidence`` AND
       ``hardware_resource_report.regions[*].compiled_evidence
       .cache_evidence`` (replacing the M-22 ``not_collected`` value).
    """
    by_region = {r["region_id"]: r for r in regions_out}

    cb_path = (
        run_dir / "02_graph_analysis" / "compiled_bottleneck"
        / "compiled_bottleneck_report.json"
    )
    if cb_path.exists():
        try:
            doc = json.loads(cb_path.read_text(encoding="utf-8"))
            for r in doc.get("regions", []) or []:
                rid = r.get("region_id") or ""
                ev = by_region.get(rid)
                if ev is not None and r.get("model_status") == "ok":
                    r["profiler_evidence"] = {
                        "gpu": ev.get("gpu"),
                        "cpu": ev.get("cpu"),
                        "cache_evidence": ev.get("cache_evidence"),
                    }
                    # Replace the M-22 placeholder value.
                    r["cache_evidence"] = ev.get("cache_evidence")
            cb_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass

    hrr_path = (
        run_dir / "02_graph_analysis" / "readiness"
        / "hardware_resource_report.json"
    )
    if hrr_path.exists():
        try:
            doc = json.loads(hrr_path.read_text(encoding="utf-8"))
            for r in doc.get("regions", []) or []:
                rid = r.get("region_id") or ""
                ev = by_region.get(rid)
                if ev is not None and r.get("compiled_evidence") is not None:
                    r["compiled_evidence"]["profiler_evidence"] = {
                        "gpu": ev.get("gpu"),
                        "cpu": ev.get("cpu"),
                        "cache_evidence": ev.get("cache_evidence"),
                    }
                    r["compiled_evidence"]["cache_evidence"] = (
                        ev.get("cache_evidence")
                    )
            hrr_path.write_text(
                json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8",
            )
        except (OSError, json.JSONDecodeError):
            pass

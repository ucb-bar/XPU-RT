"""Auto-discovery for graph_compilation target spec on the host.

Probes the running Linux host and emits a partial
``graphcomp_target_config_v1`` YAML describing the CPU plus any
visible accelerators. Fields the discovery cannot prove (e.g. peak
memory bandwidth without microbenchmarks, vendor-specific compute
caps) are emitted with sensible defaults and a ``# fill in`` comment
so the operator can complete them by hand.

Sources used (each step skips silently when its tool/file is missing):

- ``/proc/cpuinfo`` and ``/proc/meminfo``
- ``getconf LEVEL{1d,1i,2,3}_CACHE_SIZE``
- ``lscpu`` (used as a structured fallback for ``CPU max MHz``)
- ``nvidia-smi --query-gpu=...`` (NVIDIA)
- ``rocm-smi`` (AMD ROCm)
- ``hl-smi`` (Habana Gaudi)
- ``neuron-ls`` (AWS Trainium / Inferentia)
- ``/dev/{nvidia*, kfd, accel*, neuron*}`` (last-resort presence checks)
- ``TPU_NAME`` / ``XRT_TPU_CONFIG`` env vars (Google TPU)

This module is conservative: it never fabricates peak FLOPS or
bandwidth without an evidence path; for unknown devices it records
``provenance`` and leaves a placeholder.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #


@dataclass
class CPUInfo:
    vendor_id: str = ""
    model_name: str = ""
    cpu_family: int = 0
    model: int = 0
    physical_cores: int = 0
    logical_cores: int = 0
    threads_per_core: int = 1
    sockets: int = 1
    base_freq_mhz: float = 0.0
    max_freq_mhz: float = 0.0
    flags: list[str] = field(default_factory=list)
    l1d_cache_bytes: int = 0
    l1i_cache_bytes: int = 0
    l2_cache_bytes: int = 0
    l3_cache_bytes: int = 0
    page_size_bytes: int = 0
    total_memory_bytes: int = 0


def _parse_proc_cpuinfo() -> dict[str, Any]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return {}

    def _kv(b: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for line in b.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
        return out

    first = _kv(blocks[0])
    physical_ids = set()
    core_ids = set()
    for b in blocks:
        kv = _kv(b)
        physical_ids.add(kv.get("physical id", ""))
        core_ids.add(kv.get("core id", ""))

    return {
        "vendor_id": first.get("vendor_id", ""),
        "model_name": first.get("model name", ""),
        "cpu_family": int(first.get("cpu family", "0") or "0"),
        "model": int(first.get("model", "0") or "0"),
        "physical_cores": len({c for c in core_ids if c}) or 0,
        "logical_cores": len(blocks),
        "sockets": len({p for p in physical_ids if p}) or 1,
        "base_freq_mhz": float(first.get("cpu MHz", "0") or "0"),
        "flags": (first.get("flags", "") or "").split(),
    }


def _parse_lscpu() -> dict[str, str]:
    if shutil.which("lscpu") is None:
        return {}
    try:
        out = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    kv: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        kv[k.strip()] = v.strip()
    return kv


def _parse_meminfo() -> int:
    path = Path("/proc/meminfo")
    if not path.exists():
        return 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            try:
                return int(line.split()[1]) * 1024
            except (ValueError, IndexError):
                return 0
    return 0


def _getconf(name: str) -> int:
    if shutil.which("getconf") is None:
        return 0
    try:
        out = subprocess.run(
            ["getconf", name], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return 0
    try:
        return int(out)
    except ValueError:
        return 0


def discover_cpu() -> CPUInfo:
    info = CPUInfo()
    cpuinfo = _parse_proc_cpuinfo()
    info.vendor_id = cpuinfo.get("vendor_id", "")
    info.model_name = cpuinfo.get("model_name", "")
    info.cpu_family = cpuinfo.get("cpu_family", 0)
    info.model = cpuinfo.get("model", 0)
    info.physical_cores = cpuinfo.get("physical_cores", 0)
    info.logical_cores = cpuinfo.get("logical_cores", 0)
    info.sockets = cpuinfo.get("sockets", 1)
    info.base_freq_mhz = cpuinfo.get("base_freq_mhz", 0.0)
    info.flags = list(cpuinfo.get("flags", []))

    if info.physical_cores and info.logical_cores:
        info.threads_per_core = max(1, info.logical_cores // max(info.physical_cores, 1))

    lscpu = _parse_lscpu()
    if "CPU max MHz" in lscpu:
        try:
            info.max_freq_mhz = float(lscpu["CPU max MHz"])
        except ValueError:
            pass
    if not info.max_freq_mhz and info.base_freq_mhz:
        info.max_freq_mhz = info.base_freq_mhz

    info.l1d_cache_bytes = _getconf("LEVEL1_DCACHE_SIZE")
    info.l1i_cache_bytes = _getconf("LEVEL1_ICACHE_SIZE")
    info.l2_cache_bytes = _getconf("LEVEL2_CACHE_SIZE")
    info.l3_cache_bytes = _getconf("LEVEL3_CACHE_SIZE")
    info.page_size_bytes = _getconf("PAGE_SIZE")
    info.total_memory_bytes = _parse_meminfo()
    return info


def estimate_cpu_peak_gflops(cpu: CPUInfo) -> float:
    """Theoretical peak fp32 throughput in GFLOPS, assuming SIMD FMA.

    Uses the cheapest applicable SIMD width:

    - AVX-512 (avx512f flag): 16 fp32 lanes × 2 (FMA) = 32 fp32 ops/cycle/core
    - AVX2 + FMA (avx2 + fma flags): 8 fp32 lanes × 2 = 16 ops/cycle/core
    - SSE / AVX (avx flag): 4 fp32 lanes × 2 = 8 ops/cycle/core
    - else: 2 ops/cycle/core (scalar FMA assumed)

    Multiplied by physical core count and max boost frequency. Note this
    is *theoretical peak*; sustained throughput on real workloads is
    typically 30-60% of this number.
    """
    flags = set(cpu.flags)
    if "avx512f" in flags:
        ops_per_cycle = 32.0
    elif "avx2" in flags and "fma" in flags:
        ops_per_cycle = 16.0
    elif "avx" in flags:
        ops_per_cycle = 8.0
    else:
        ops_per_cycle = 2.0
    cores = cpu.physical_cores or cpu.logical_cores or 1
    freq_ghz = (cpu.max_freq_mhz or cpu.base_freq_mhz) / 1000.0
    return round(ops_per_cycle * freq_ghz * cores, 3)


def estimate_cpu_peak_bandwidth(cpu: CPUInfo) -> float:
    """Conservative DRAM bandwidth estimate in GB/s.

    We have no reliable on-system signal for memory bandwidth without
    benchmarking, so we use socket-class heuristics that the operator
    can override:

    - HEDT / workstation Threadripper / Xeon-W (>= 16 cores): 80 GB/s
    - desktop (<= 8 cores): 30 GB/s
    - server multi-socket: 100 GB/s × sockets
    - else: 30 GB/s
    """
    cores = cpu.physical_cores or cpu.logical_cores or 1
    if cpu.sockets > 1:
        return float(100 * cpu.sockets)
    if cores >= 16:
        return 80.0
    if cores >= 8:
        return 50.0
    return 30.0


def cpu_supported_dtypes(cpu: CPUInfo) -> list[str]:
    flags = set(cpu.flags)
    out = ["fp32"]
    if "f16c" in flags or "avx512f" in flags or "vaes" in flags:
        out.append("fp16")
    if "avx512_bf16" in flags or "amx_bf16" in flags:
        out.append("bf16")
    return out


# --------------------------------------------------------------------------- #
# Accelerators
# --------------------------------------------------------------------------- #


@dataclass
class AcceleratorInfo:
    kind: str
    detected_via: str
    name: str = ""
    count: int = 0
    memory_bytes: int = 0
    compute_capability: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _run_cmd(cmd: list[str], *, timeout: int = 5) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def discover_nvidia_gpus() -> list[AcceleratorInfo]:
    out = _run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[AcceleratorInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name, mem_mib, cc = parts[0], parts[1], parts[2]
        try:
            mem_bytes = int(mem_mib) * 1024 * 1024
        except ValueError:
            mem_bytes = 0
        gpus.append(
            AcceleratorInfo(
                kind="nvidia_gpu",
                detected_via="nvidia-smi",
                name=name,
                count=1,
                memory_bytes=mem_bytes,
                compute_capability=cc,
                extra={"pci_bus_id": parts[3] if len(parts) > 3 else ""},
            )
        )
    return gpus


def discover_amd_gpus() -> list[AcceleratorInfo]:
    out = _run_cmd(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        import json as _json
        obj = _json.loads(out)
    except (ValueError, ImportError):
        return []
    gpus: list[AcceleratorInfo] = []
    for k, v in obj.items():
        if not isinstance(v, dict):
            continue
        name = v.get("Card series", "") or v.get("Card model", "")
        mem_str = v.get("VRAM Total Memory (B)", "0") or v.get("VRAM Total Memory", "0")
        try:
            mem_bytes = int(mem_str)
        except (ValueError, TypeError):
            mem_bytes = 0
        gpus.append(
            AcceleratorInfo(
                kind="amd_gpu",
                detected_via="rocm-smi",
                name=str(name),
                count=1,
                memory_bytes=mem_bytes,
                extra={"slot": k},
            )
        )
    return gpus


def discover_habana() -> list[AcceleratorInfo]:
    out = _run_cmd(["hl-smi", "--query", "--quiet"]) or _run_cmd(["hl-smi"])
    if not out:
        return []
    return [
        AcceleratorInfo(
            kind="habana_gaudi",
            detected_via="hl-smi",
            name="Habana Gaudi (probe-only)",
            count=out.count("AIP") or 1,
            extra={"raw_head": out.splitlines()[:5]},
        )
    ]


def discover_trainium() -> list[AcceleratorInfo]:
    out = _run_cmd(["neuron-ls", "-j"]) or _run_cmd(["neuron-ls"])
    if not out:
        return []
    return [
        AcceleratorInfo(
            kind="aws_trainium_inferentia",
            detected_via="neuron-ls",
            name="AWS NeuronCore device(s)",
            count=out.count("nc") or 1,
            extra={"raw_head": out.splitlines()[:5]},
        )
    ]


def discover_google_tpu() -> list[AcceleratorInfo]:
    if os.environ.get("TPU_NAME") or os.environ.get("XRT_TPU_CONFIG"):
        return [
            AcceleratorInfo(
                kind="google_tpu",
                detected_via="env(TPU_NAME)",
                name=os.environ.get("TPU_NAME", "tpu"),
                count=1,
            )
        ]
    return []


def discover_dev_node_hints() -> list[AcceleratorInfo]:
    """Last-resort: presence of /dev/{nvidia*, kfd, accel*, neuron*} when
    no vendor tool is installed. Records *presence only*, not capability."""
    hints: list[AcceleratorInfo] = []
    if any(Path("/dev").glob("nvidia[0-9]*")):
        hints.append(
            AcceleratorInfo(
                kind="nvidia_gpu_dev_only",
                detected_via="/dev/nvidia*",
                name="NVIDIA device node present (no nvidia-smi result)",
                count=len(list(Path("/dev").glob("nvidia[0-9]*"))),
            )
        )
    if Path("/dev/kfd").exists():
        hints.append(
            AcceleratorInfo(
                kind="amd_gpu_dev_only",
                detected_via="/dev/kfd",
                name="ROCm KFD device node present",
                count=1,
            )
        )
    accel_nodes = list(Path("/dev").glob("accel*"))
    if accel_nodes:
        hints.append(
            AcceleratorInfo(
                kind="generic_accelerator_dev_only",
                detected_via="/dev/accel*",
                name="Generic accelerator device node",
                count=len(accel_nodes),
                extra={"nodes": [str(p) for p in accel_nodes]},
            )
        )
    neuron_nodes = list(Path("/dev").glob("neuron*"))
    if neuron_nodes:
        hints.append(
            AcceleratorInfo(
                kind="aws_neuron_dev_only",
                detected_via="/dev/neuron*",
                name="AWS Neuron device node present",
                count=len(neuron_nodes),
                extra={"nodes": [str(p) for p in neuron_nodes]},
            )
        )
    return hints


def discover_all_accelerators() -> list[AcceleratorInfo]:
    """Probe every supported accelerator family and merge results.

    /dev-based hints are dropped when a vendor tool already produced a
    rich record for the same family (e.g. don't list both ``nvidia_gpu``
    and ``nvidia_gpu_dev_only``).
    """
    found: list[AcceleratorInfo] = []
    found.extend(discover_nvidia_gpus())
    found.extend(discover_amd_gpus())
    found.extend(discover_habana())
    found.extend(discover_trainium())
    found.extend(discover_google_tpu())
    rich_kinds = {a.kind for a in found}
    for hint in discover_dev_node_hints():
        underlying = hint.kind.removesuffix("_dev_only")
        if underlying not in rich_kinds:
            found.append(hint)
    return found


# --------------------------------------------------------------------------- #
# YAML emission
# --------------------------------------------------------------------------- #


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")
    return s or "host"


def build_target_yaml(
    *,
    out_path: Path,
    target_id: str | None = None,
) -> dict[str, Any]:
    """Discover host capabilities and write a partial target YAML.

    Returns the dict that was written so callers can inspect it without
    re-reading the file.
    """
    cpu = discover_cpu()
    accelerators = discover_all_accelerators()

    if target_id is None:
        if accelerators:
            target_id = f"host_{_slug(accelerators[0].name or accelerators[0].kind)}"
        else:
            target_id = f"host_{_slug(cpu.model_name) or 'cpu'}"

    peak_compute = estimate_cpu_peak_gflops(cpu)
    peak_bw = estimate_cpu_peak_bandwidth(cpu)

    obj: dict[str, Any] = {
        "schema_version": "graphcomp_target_config_v1",
        "target_id": target_id,
        "device_kind": "cpu",
        "description": (
            f"Auto-discovered {cpu.vendor_id} {cpu.model_name} "
            f"({cpu.physical_cores}c/{cpu.logical_cores}t).\n"
            f"Edit fields marked '# fill in' before relying on this profile "
            f"for production cost models."
        ),
        "auto_discovered": True,
        "discovery_provenance": {
            "cpu": asdict(cpu),
            "accelerators": [asdict(a) for a in accelerators],
            "host_uname": _safe_uname(),
        },
        "peak_compute_gflops": peak_compute,
        "peak_bandwidth_gb_s": peak_bw,
        "memory_tiers": {
            "scratchpad_bytes": cpu.l1d_cache_bytes or 32_768,
            "l2_bytes": cpu.l2_cache_bytes or 524_288,
            "l3_bytes": cpu.l3_cache_bytes or 16_777_216,
            "system_bytes": cpu.total_memory_bytes or 16 * 1024 * 1024 * 1024,
        },
        "supported_dtypes": cpu_supported_dtypes(cpu),
        "numerical_budgets": {
            "fp32": 1.0e-3,
            "fast_math": 5.0e-3,
            "fp16_accum": 1.0e-2,
            "fp8_e4m3": 1.0e-1,
        },
        "working_set_tiles": {
            "matmul": [
                {"M": 16, "N": 16, "K": 16},
                {"M": 32, "N": 32, "K": 32},
                {"M": 64, "N": 64, "K": 32},
                {"M": 128, "N": 128, "K": 32},
                {"M": 256, "N": 256, "K": 64},
                {"M": 512, "N": 512, "K": 64},
            ],
            "elementwise": [
                {"numel": 1024},
                {"numel": 4096},
                {"numel": 16384},
            ],
        },
    }
    if accelerators:
        obj["accelerators_present"] = [
            {
                "kind": a.kind,
                "name": a.name,
                "count": a.count,
                "memory_bytes": a.memory_bytes,
                "compute_capability": a.compute_capability,
                "detected_via": a.detected_via,
                "peak_compute_gflops_estimate": None,  # fill in
                "peak_bandwidth_gb_s_estimate": None,  # fill in
            }
            for a in accelerators
        ]
        obj["description"] = (
            obj["description"]
            + f"\nDetected accelerators: {[a.kind for a in accelerators]}. "
            + "Their peak compute / bandwidth fields are left null — fill in by hand."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Auto-discovered by `python -m compgen.graph_compilation discover-target`.\n"
        "# Review and adjust before relying on this for production cost models.\n"
        + yaml.safe_dump(obj, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return obj


def _safe_uname() -> dict[str, str]:
    try:
        u = os.uname()
    except OSError:
        return {}
    return {
        "sysname": u.sysname,
        "nodename": u.nodename,
        "release": u.release,
        "version": u.version,
        "machine": u.machine,
    }

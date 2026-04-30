"""Auto-detected backend selection for the ETC dispatch path.

The agentic-compilation contract: a PyPI user (or their agent)
calls ``compile_model(model, target="auto")`` with no flags and
expects CompGen to figure out everything — which NVRTC to use,
which cuBLASDx precision, which tile shape, which SM tag, etc.

Every flag the matcher accepts is something the user shouldn't
need to know about. This module probes the local device + reachable
libraries and emits a :class:`BackendChoice` the matcher consumes
in place of those flags.

Per bridge #074-#099, every round-trip to bwell exposed an
implicit assumption baked into the codebase. Each one of those
assumptions becomes a probe here:

- cuBLASDx headers reachable? → ``cublasdx_available``
- libcudacxx + CUTLASS headers reachable? → deps for cuBLASDx
- cu13 NVRTC reachable? → ``cu13_nvrtc_available``
- Target arch from probe → maps to NVRTC ``--gpu-architecture`` +
  cuBLASDx ``SM<...>`` tag
- Tile shape from arch → 64×64×16 on Blackwell (mma.sync at #095),
  32×32×32 elsewhere

The probe is deterministic for a given (target, library snapshot)
tuple. First call takes ~50 ms; cached for the rest of the
process. Pass ``force_refresh=True`` to bypass the cache (e.g.
after pip-installing nvidia-mathdx mid-session).

Public API:

    from compgen.runtime.autotune import probe_device, BackendChoice

    choice = probe_device(target="auto")
    # → BackendChoice(target_arch="sm_100", use_cublasdx=True, ...)

The ``rationale`` field is populated for the audit-via-MCP story:
the agent can ask "why did the compiler pick X for op Y?" and get
a string grounded in the probe's decision tree, not folk knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendChoice:
    """Auto-detected backend configuration for the ETC dispatch path.

    Replaces the four manual flags the matcher used to require:
    ``prefer_cublasdx_for_linears``, ``cublasdx_precision``,
    ``target_arch``, ``use_cu13_nvrtc``. The matcher consumes a
    single :class:`BackendChoice` instead.

    Attributes:
        target_arch: NVRTC ``--gpu-architecture`` form. Either
            probed (``"sm_100"`` etc.) or pass-through when the
            caller specified an explicit arch for cross-compilation.
        cublasdx_available: All three of cuBLASDx + libcudacxx +
            CUTLASS headers are reachable for an NVRTC compile.
        cu13_nvrtc_available: ``libnvrtc.so.13`` is reachable
            (either via ``nvidia.cu13`` torch dep or
            ``nvidia.cuda_nvrtc`` standalone).
        use_cublasdx_for_linears: Final decision — emit cuBLASDx
            bodies for linear ops. True when the target hardware
            is Blackwell-class AND all libraries reach.
        cublasdx_precision: ``"fp32"`` | ``"bf16_fp32"``. bf16 +
            fp32 accumulator is selected for Blackwell (mma.sync /
            tcgen05 path); fp32 SIMT for older arches.
        use_cu13_nvrtc: Final decision — route NVRTC compile through
            cu13 instead of cu12. True when the target arch needs
            ``__CUDA_ARCH__`` > 900 (Blackwell sm_100/sm_120).
        cublasdx_sm: Integer ``SM<...>`` template tag for cuBLASDx.
            Mapped from ``target_arch`` per
            :func:`compgen.runtime.lowering.fx_to_megakernel._arch_to_cublasdx_sm`.
        tile_m, tile_n, tile_k: Per-task tile shape.
            64×64×16 for cuBLASDx (mma.sync trigger per #095),
            32×32×32 for the hand_rolled_fmaf path.
        rationale: Plain-English explanation of why each flag took
            its final value. Goes into the bundle's decision log so
            the agent can audit without re-running the probe.
        target_origin: ``"probed"`` (CudaDeviceProbe ran) or
            ``"explicit"`` (caller passed an arch string) or
            ``"fallback"`` (no probe + no arg → defaulted).
    """

    target_arch: str
    cublasdx_available: bool
    cu13_nvrtc_available: bool
    use_cublasdx_for_linears: bool
    cublasdx_precision: str
    use_cu13_nvrtc: bool
    cublasdx_sm: int
    tile_m: int
    tile_n: int
    tile_k: int
    rationale: str
    target_origin: str
    library_paths: dict[str, str | None] = field(default_factory=dict)
    # Wave 1.6 — cluster-launch dimensions for vendors that support
    # multi-block-per-task cooperation (NVIDIA cluster-launch on
    # sm_90+). When ``supports_clusters`` is True the universal
    # compile path passes ``(cluster_dim_x, cluster_dim_y,
    # cluster_dim_z)`` to ``compute_static_schedule(cluster_dim=...)``;
    # otherwise stays at single-block tasks. ``None`` means
    # ``supports_clusters`` is False — vendors without the
    # primitive (CPU, AMD pre-CDNA3, etc.).
    supports_clusters: bool = False
    cluster_dim_x: int | None = None
    cluster_dim_y: int | None = None
    cluster_dim_z: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the bundle's compile_context.json + the
        decision log surfaced through MCP.

        Tile shape is exposed BOTH as the combined ``tile_shape``
        list AND as individual ``tile_m / tile_n / tile_k`` ints —
        per bridge #102, the agent's audit query was reading the
        per-axis names and seeing ``None`` even though the values
        were set internally. Both forms surface so either query
        path works.
        """
        return {
            "target_arch": self.target_arch,
            "target_origin": self.target_origin,
            "cublasdx_available": self.cublasdx_available,
            "cu13_nvrtc_available": self.cu13_nvrtc_available,
            "use_cublasdx_for_linears": self.use_cublasdx_for_linears,
            "cublasdx_precision": self.cublasdx_precision,
            "use_cu13_nvrtc": self.use_cu13_nvrtc,
            "cublasdx_sm": self.cublasdx_sm,
            "tile_shape": [self.tile_m, self.tile_n, self.tile_k],
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "supports_clusters": self.supports_clusters,
            "cluster_dim": (
                [self.cluster_dim_x, self.cluster_dim_y, self.cluster_dim_z] if self.cluster_dim_x is not None else None
            ),
            "rationale": self.rationale,
            "library_paths": dict(self.library_paths),
        }


_PROBE_CACHE: dict[str, BackendChoice] = {}


def probe_device(
    target: str = "auto",
    *,
    force_refresh: bool = False,
) -> BackendChoice:
    """Probe the local device + reachable libraries; return a
    :class:`BackendChoice` ready for the matcher.

    Args:
        target: ``"auto"`` (probe via :class:`CudaDeviceProbe` when
            available, fall back to ``sm_100`` paper-faithful
            default) or an explicit arch string like ``"sm_100"``,
            ``"sm_90"`` for cross-compilation. The arch is what
            NVRTC will see; cuBLASDx's ``SM<...>`` tag is mapped
            from it.
        force_refresh: Bypass the in-process cache. Use when pip-
            installing libraries mid-session (e.g. installing
            nvidia-mathdx after probe_device already ran).

    Returns:
        :class:`BackendChoice`. Never raises — every probe failure
        gracefully falls back to a safe default (e.g. cuBLASDx not
        reachable → use_cublasdx_for_linears=False, hand_rolled_fmaf
        path).

    The function is deterministic for a given (target, library
    snapshot) tuple — calling twice with the same arguments returns
    the same choice.
    """
    cache_key = target if not force_refresh else f"{target}@refresh"
    if cache_key in _PROBE_CACHE and not force_refresh:
        return _PROBE_CACHE[cache_key]

    arch, origin = _resolve_target_arch(target)
    cublasdx_ok, cu13_ok, lib_paths = _probe_libraries()

    is_blackwell = _is_blackwell(arch)
    # Decision tree — the entire reason this module exists.
    use_cu13_nvrtc = bool(is_blackwell and cu13_ok)
    use_cublasdx = bool(cublasdx_ok and use_cu13_nvrtc)
    if use_cublasdx and is_blackwell:
        precision = "bf16_fp32"
        tile = (64, 64, 16)
    elif use_cublasdx:
        # Hopper / older — fp32 cuBLASDx, smaller tile is fine since
        # tensor-core engagement isn't the win there.
        precision = "fp32"
        tile = (32, 32, 32)
    else:
        # Hand-rolled fmaf fallback. Precision field is meaningless
        # in this branch; we pin "fp32" for serialization stability.
        precision = "fp32"
        tile = (32, 32, 32)

    cublasdx_sm = _arch_to_cublasdx_sm(arch)

    # Wave 1.6 — cluster-launch decision. NVIDIA sm_90+ has the
    # cooperative cluster primitive; Hopper (sm_90) has it too but
    # we only enable for Blackwell since that's where bridge #108's
    # data validates the perf impact. The cluster shape (2, 1, 1) =
    # 2-block clusters is the conservative starting point — bigger
    # clusters need careful smem accounting + the body must
    # cluster-cooperate. (4, 1, 1) and (8, 1, 1) are tunable later.
    supports_clusters = bool(is_blackwell)
    if supports_clusters:
        cluster_dim_x: int | None = 2
        cluster_dim_y: int | None = 1
        cluster_dim_z: int | None = 1
    else:
        cluster_dim_x = cluster_dim_y = cluster_dim_z = None

    rationale = _format_rationale(
        arch=arch,
        origin=origin,
        cublasdx_ok=cublasdx_ok,
        cu13_ok=cu13_ok,
        use_cublasdx=use_cublasdx,
        use_cu13_nvrtc=use_cu13_nvrtc,
        precision=precision,
        tile=tile,
        cublasdx_sm=cublasdx_sm,
    )

    choice = BackendChoice(
        target_arch=arch,
        cublasdx_available=cublasdx_ok,
        cu13_nvrtc_available=cu13_ok,
        use_cublasdx_for_linears=use_cublasdx,
        cublasdx_precision=precision,
        use_cu13_nvrtc=use_cu13_nvrtc,
        cublasdx_sm=cublasdx_sm,
        tile_m=tile[0],
        tile_n=tile[1],
        tile_k=tile[2],
        supports_clusters=supports_clusters,
        cluster_dim_x=cluster_dim_x,
        cluster_dim_y=cluster_dim_y,
        cluster_dim_z=cluster_dim_z,
        rationale=rationale,
        target_origin=origin,
        library_paths=lib_paths,
    )
    _PROBE_CACHE[cache_key] = choice
    return choice


# ---------------------------------------------------------------------------
# Internal probes
# ---------------------------------------------------------------------------


def _resolve_target_arch(target: str) -> tuple[str, str]:
    """Resolve the user's ``target`` to a concrete NVRTC arch.

    Probe order (per bridge #102 — bwell hit ``fallback`` even on
    real hardware because :class:`CudaDeviceProbe` needs
    ``libcompgen_rt-cuda.so`` which the wheel doesn't always ship):

    1. ``CudaDeviceProbe`` via the native HAL — most reliable when
       the .so is present.
    2. ``torch.cuda.get_device_capability()`` — works on any
       Blackwell/Hopper host with torch + CUDA. The agent's host
       always has torch.
    3. ``"sm_100"`` fallback — paper-faithful default.

    Returns ``(arch, origin)`` where ``origin`` is ``"probed"``
    (HAL), ``"probed_torch"`` (torch fallback), ``"explicit"``
    (caller passed an arch), or ``"fallback"`` (no probe worked).
    """
    if target != "auto":
        return target, "explicit"

    # 1. Try the native HAL first — most reliable.
    try:
        from compgen.runtime.native.cuda import (
            CudaDeviceProbe,
            CudaUnavailableError,
        )

        probe = CudaDeviceProbe()
        cc_major = int(probe.compute_capability_major)
        cc_minor = int(probe.compute_capability_minor)
        return f"sm_{cc_major}{cc_minor}", "probed"
    except (CudaUnavailableError, ImportError, AttributeError):
        pass
    except Exception:  # noqa: BLE001
        pass

    # 2. Torch fallback — every Blackwell user has torch via
    # compgen[cuda] or torch>=2.6, which means torch.cuda is
    # reachable when the GPU exists. Per #102: this fixes the
    # "target_origin=fallback on real hardware" gap.
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            return f"sm_{cc_major}{cc_minor}", "probed_torch"
    except Exception:  # noqa: BLE001
        pass

    # 3. Paper-faithful default.
    return "sm_100", "fallback"


def _probe_libraries() -> tuple[bool, bool, dict[str, str | None]]:
    """Probe cuBLASDx (+ deps) + cu13 NVRTC reachability.

    Returns ``(cublasdx_ok, cu13_nvrtc_ok, lib_paths)``. Every
    individual library path is captured (None when unreachable) so
    the rationale + decision log show exactly what was probed.
    """
    paths: dict[str, str | None] = {
        "cublasdx_include": None,
        "libcudacxx_include": None,
        "cutlass_include": None,
        "cu13_nvrtc_lib": None,
    }
    try:
        from compgen.runtime.native.cuda import (
            _resolve_cu13_nvrtc_lib_path,
            discover_cublasdx_include,
            discover_cutlass_include,
            discover_libcudacxx_include,
        )

        paths["cublasdx_include"] = discover_cublasdx_include()
        paths["libcudacxx_include"] = discover_libcudacxx_include()
        paths["cutlass_include"] = discover_cutlass_include()
        paths["cu13_nvrtc_lib"] = _resolve_cu13_nvrtc_lib_path()
    except Exception:  # noqa: BLE001
        # Probe is best-effort; never raises.
        pass

    cublasdx_ok = all(paths[k] is not None for k in ("cublasdx_include", "libcudacxx_include", "cutlass_include"))
    cu13_ok = paths["cu13_nvrtc_lib"] is not None
    return cublasdx_ok, cu13_ok, paths


def _is_blackwell(arch: str) -> bool:
    """sm_100 / sm_120 (datacenter + workstation Blackwell). The
    cu13-NVRTC + tensor-core path applies to these and only these."""
    a = arch.lower().lstrip("sm_").rstrip("a")
    return a in {"100", "120"}


def _arch_to_cublasdx_sm(arch: str) -> int:
    """Re-export of the lowering matcher's mapping. Keeps the SM tag
    selection in one place; the matcher imports back from here when
    consuming the BackendChoice."""
    from compgen.runtime.lowering.fx_to_megakernel import (
        _arch_to_cublasdx_sm as _impl,
    )

    return _impl(arch)


def _format_rationale(
    *,
    arch: str,
    origin: str,
    cublasdx_ok: bool,
    cu13_ok: bool,
    use_cublasdx: bool,
    use_cu13_nvrtc: bool,
    precision: str,
    tile: tuple[int, int, int],
    cublasdx_sm: int,
) -> str:
    """One-string explanation of every decision the probe made.

    Goes into the bundle's compile_context.json + the agent-facing
    audit query so the rationale survives across the wheel ship +
    bridge round-trip without needing to re-run the probe.
    """
    lines = [
        f"target={arch} (origin={origin})",
        f"cublasdx headers reachable: {cublasdx_ok}",
        f"cu13 NVRTC reachable:        {cu13_ok}",
        f"→ use_cu13_nvrtc:            {use_cu13_nvrtc}",
        f"→ use_cublasdx_for_linears:  {use_cublasdx}",
    ]
    if use_cublasdx:
        lines.append(f"→ precision={precision}, tile={tile[0]}×{tile[1]}×{tile[2]}, SM<{cublasdx_sm}>")
        if precision == "bf16_fp32":
            lines.append("  (bf16+fp32-acc on Blackwell engages mma.sync per #095)")
        else:
            lines.append("  (fp32 SIMT path; tensor cores not engaged)")
    else:
        if not cublasdx_ok:
            lines.append("  → fall back to hand_rolled_fmaf (cuBLASDx headers not reachable)")
        elif not cu13_ok:
            lines.append(
                "  → fall back to hand_rolled_fmaf (cu13 NVRTC not reachable; "
                "without it Blackwell's __CUDA_ARCH__ stays at 900 and "
                "cuBLASDx silently SIMTs per #089)"
            )
        else:
            lines.append("  → fall back to hand_rolled_fmaf (non-Blackwell target)")
    return "\n".join(lines)


def _clear_probe_cache_for_tests() -> None:
    """Test-only helper. Clears the in-process cache so a unit test
    can re-run the probe with monkeypatched library availability."""
    _PROBE_CACHE.clear()

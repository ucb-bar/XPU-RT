"""Map NVRTC ``--gpu-architecture`` flag to cuBLASDx ``SM<...>`` tag.

The cuBLASDx 0.4.0 (mathdx 25.6.0) dispatch tables ship for SM<700>,
<720>, <750>, <800>, <860>, <870>, <890>, <900>, <1000>. Picking
the wrong tag silently SIMTs the kernel (per bridge #087/#089).

Owned by ``targets/gpu/nvidia/common/`` because the mapping
applies to every NVIDIA arch — Blackwell, Hopper, Ampere, etc. —
even though the Blackwell leaf is what most paper-shape work uses.

Wave 1.14 moves this here from
``runtime/lowering/fx_to_megakernel.py``. The original location
re-exports for one round so existing imports keep working.
"""

from __future__ import annotations


def arch_to_cublasdx_sm(target_arch: str) -> int:
    """Map NVRTC arch to cuBLASDx ``SM<...>`` integer tag.

    Examples:
        ``"sm_90"`` / ``"sm_90a"`` → 900   Hopper
        ``"sm_100"`` / ``"sm_100a"`` → 1000 Blackwell B100/B200
        ``"sm_120"`` / ``"sm_120a"`` → 1000 (workstation Blackwell —
                                             cuBLASDx 0.4.0 doesn't
                                             ship SM<1200>, falls
                                             back to SM<1000>)
        older → mapped to nearest supported

    Unknown / future arches default to ``SM<1000>`` since that's
    Blackwell-correct for the paper-faithful hardware. The
    user-side ``target_arch`` is unaffected — NVRTC still gets
    the real arch flag; only the cuBLASDx tag is mapped.
    """
    a = target_arch.lower().lstrip("sm_").rstrip("a")
    table = {
        "70": 700,
        "72": 720,
        "75": 750,
        "80": 800,
        "86": 860,
        "87": 870,
        "89": 890,
        "90": 900,
        "100": 1000,
        "120": 1000,  # workstation Blackwell falls back to SM<1000>
    }
    return table.get(a, 1000)

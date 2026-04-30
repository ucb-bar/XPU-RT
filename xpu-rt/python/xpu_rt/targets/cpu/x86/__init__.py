"""x86 CPU vendor entry — AVX-512 / AVX2 family.

Wave 1.15 — second backend that validates the abstraction holds
outside GPU-land. The unified target hierarchy under
``targets/{class}/{vendor}/{arch}/`` claims to generalize; this
package is the load-bearing test of that claim.

Stubs for now (scalar fmaf C++, no SIMD intrinsics) — the C++
auto-vectorizer extracts AVX-512 / AVX2 throughput from the loops.
Future arch-leaves under ``x86/avx512/``, ``x86/avx2/`` would
specialize with intrinsics.
"""

from __future__ import annotations

from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter
from compgen.targets.cpu.x86.cost import X86CostModel
from compgen.targets.cpu.x86.probe import X86Probe
from compgen.targets.cpu.x86.runtime import X86Runtime
from compgen.targets.registry import register_target


def _register_x86() -> None:
    probe = X86Probe()
    register_target(
        target_class="cpu",
        vendor="x86",
        arch="",  # vendor-common; arch leaves under avx512/, avx2/ etc.
        probe=probe,
        body_emitter=X86BodyEmitter(),
        runtime=X86Runtime(),
        cost_model=X86CostModel(arch=probe.device_arch()),
        rationale=(
            "x86 CPU vendor (Wave 1.15 stub). JIT path is a system "
            "clang/gcc invocation + ctypes.CDLL of the resulting .so. "
            "No event tensors over PCIe — intra-CPU sync is "
            "effectively free; the megakernel collapses to a serial "
            "task chain. Bodies are scalar-fmaf C++; the compiler's "
            "auto-vectorizer extracts AVX-512/AVX2 throughput. Per "
            "the unified target hierarchy: see "
            "docs/architecture/target-hierarchy-inventory.md."
        ),
        registration_path="in_tree",
        metadata={
            "vendor_url": "https://www.intel.com / https://www.amd.com",
            "jit_toolchain": "clang/gcc + ctypes.CDLL",
            "memory_model": "cache-coherent SMP",
            "supports_clusters": False,
            "supports_tensor_cores": False,
            "default_tile_shape": [32, 32, 32],
            "preferred_precision": "fp32",
            "detected_arch": probe.device_arch(),
            "cxx_compiler": probe.library_paths()["cxx_compiler"],
        },
    )


_register_x86()

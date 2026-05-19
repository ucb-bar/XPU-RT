"""Cross-compile a XPU-RT bundle for riscv64 bare-metal (Spike+HTIF).

Phase C v1 — focuses on the minimum useful surface:

  * one or more XNNPACK-generated C kernels under
    ``<bundle>/generated_kernels/xnnpack/``;
  * each kernel exposes the per-kernel ABI from
    ``xpu_rt/drivers/xnnpack/xnnpack_bridge.h`` (``xpu_rt_kernel_init``
    / ``xpu_rt_kernel_run`` / ``xpu_rt_kernel_destroy``);
  * a driver source generated from
    :mod:`xpu_rt.runtime.cross_compile.driver_template` that runs the
    regions in ``execution_plan.yaml`` order, stages
    ``golden_inputs.pt`` into the first region's input buffer, and
    emits the final output + a 64-bit checksum to HTIF stdout.

The build then links those C sources against the pre-built
``libxpu_rt_static.a`` (produced from
``runtime/native/libxpu_rt/CMakeLists.txt`` with the
``riscv64-spike-rvv`` toolchain). Output is one ``program.elf`` in the
bundle, runnable directly under
``spike --isa=rv64gcv <bundle>/program.elf``.

Phases D/E reuse this orchestrator unchanged; only the driver
template's region-call generation grows to handle more op kinds.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import torch
import yaml

from xpu_rt.graph_compilation.region_dossier import CrossCompileConfig

log = structlog.get_logger(__name__)

_DRIVER_TEMPLATE = (
    Path(__file__).parent / "driver_template.c.tmpl"
).read_text(encoding="utf-8")


class CrossCompileError(RuntimeError):
    """Typed error raised when cross-compilation fails.

    The ``reason`` attribute is one of:

      * ``"cross_gcc_missing"``    — toolchain binaries not found
      * ``"libxpu_rt_missing"``    — pre-built static lib not located
      * ``"no_kernels_to_compile"`` — bundle has no generated_kernels
      * ``"cmake_configure_failed"`` — cmake -B step returned non-zero
      * ``"link_failed"``          — cmake --build step returned non-zero
      * ``"bridge_header_missing"`` — xnnpack_bridge.h not found
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


@dataclass(frozen=True)
class CrossCompileResult:
    """Typed result returned by :func:`cross_compile_riscv64_bundle`."""

    status: str  # "ok" | "skipped" | "failed"
    elf_path: Path | None
    staging_dir: Path | None
    cmake_log_path: Path | None
    reason: str | None = None


def _find_libxpu_rt_static(repo_root: Path) -> Path | None:
    """Look for a pre-built riscv64 libxpu_rt_static.a in known build dirs."""
    candidates = [
        repo_root / "build" / "riscv-spike" / "libxpu_rt_static.a",
        repo_root / "build" / "rt-riscv-spike" / "libxpu_rt_static.a",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _find_companion_archives(libxpu_rt: Path) -> list[Path]:
    """Find sibling .a archives libxpu_rt_static needs at exe link time.

    libxpu_rt_static is a thin archive of just our HAL sources; the
    XNNPACK + pthreadpool symbols it references live in separate
    static libs side-by-side in the build dir.
    """
    build_dir = libxpu_rt.parent
    found: list[Path] = []
    # Search order matters for single-pass linkers: dependants first.
    for stem in (
        "libxpu_rt_static.a",  # the HAL itself (must come first)
        # XNNPACK splits its static lib into many sub-archives; only
        # the ones libxpu_rt actually pulls in.
        "libXNNPACK.a",
        "libxnnpack-microkernels-prod.a",
        "libpthreadpool.a",
    ):
        # Recurse into _deps/ tree under build_dir for the libs CMake
        # placed under subdirectories.
        for cand in [build_dir / stem, *build_dir.rglob(stem)]:
            if cand.is_file() and cand not in found:
                found.append(cand)
                break
    return found


def _collect_xnnpack_kernels(bundle_dir: Path) -> list[dict[str, Any]]:
    """Find generated_kernels/xnnpack/*.c + their metadata.

    Returns a list of dicts ordered by filename, each:
      ``{"name": <stem>, "source": <Path to .c>, "metadata": <dict>}``.

    The XnnpackProvider writes metadata to a shared
    ``kernel_metadata.json`` in the same artifact dir (one file per
    propose() call, overwritten when multiple kernels are emitted
    into the same dir). v1 bundles have one kernel so this is
    sufficient; multi-kernel bundles are a Phase C.1 follow-up.
    """
    xnn_dir = bundle_dir / "generated_kernels" / "xnnpack"
    if not xnn_dir.is_dir():
        return []
    shared_meta_path = xnn_dir / "kernel_metadata.json"
    shared_meta: dict[str, Any] = {}
    if shared_meta_path.is_file():
        try:
            shared_meta = json.loads(shared_meta_path.read_text(encoding="utf-8"))
        except Exception:
            shared_meta = {}
    rows: list[dict[str, Any]] = []
    for src in sorted(xnn_dir.glob("*.c")):
        # Prefer a per-kernel <stem>.json if it exists (future-proof for
        # the multi-kernel layout); fall back to the shared metadata.
        per_kernel_path = src.with_suffix(".json")
        meta: dict[str, Any] = dict(shared_meta)
        if per_kernel_path.is_file():
            try:
                meta.update(json.loads(per_kernel_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        rows.append({"name": src.stem, "source": src, "metadata": meta})
    return rows


def _load_golden_inputs_flat(bundle_dir: Path) -> tuple[list[float], int]:
    """Load ``golden_inputs.pt`` and return its flat fp32 contents.

    The MVP supports only fp32 tensors; non-fp32 inputs raise
    :class:`CrossCompileError`.
    """
    gi_path = bundle_dir / "golden_inputs.pt"
    if not gi_path.is_file():
        return [], 0
    raw = torch.load(gi_path, map_location="cpu", weights_only=False)
    if isinstance(raw, torch.Tensor):
        tensors: list[torch.Tensor] = [raw]
    elif isinstance(raw, (list, tuple)):
        tensors = list(raw)
    else:
        raise CrossCompileError(
            "no_kernels_to_compile",
            f"unexpected golden_inputs.pt type: {type(raw)}",
        )
    flat: list[float] = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype != torch.float32:
            raise CrossCompileError(
                "no_kernels_to_compile",
                f"v1 cross-compile only supports fp32 inputs; got {t.dtype}",
            )
        flat.extend(float(x) for x in t.detach().contiguous().flatten().tolist())
    return flat, len(flat)


def _emit_golden_inputs_c(flat: list[float], dst: Path) -> None:
    """Emit ``golden_inputs.c`` with the input data as a const array."""
    lines = [
        "/* Auto-generated by xpu_rt.runtime.cross_compile.riscv64_bare. */",
        "#include <stddef.h>",
        "",
        f"const size_t golden_inputs_count = {len(flat)};",
        "const float golden_inputs_data[] = {",
    ]
    if not flat:
        lines.append("    0.0f")
    else:
        for i in range(0, len(flat), 8):
            chunk = flat[i : i + 8]
            row = ", ".join(_fmt_float(x) for x in chunk)
            lines.append(f"    {row},")
    lines.append("};")
    lines.append("")
    dst.write_text("\n".join(lines), encoding="utf-8")


def _fmt_float(x: float) -> str:
    """Emit a float literal as raw bits via a union — exact preservation."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return f"({_float_via_bits_literal(bits)})"


def _float_via_bits_literal(bits: int) -> str:
    # ``__builtin_bit_cast`` is a C++ feature; for plain C we emit the
    # decimal representation, which is fine for fp32 round-trip when
    # the value is < 2^24 and not subnormal. For the v1 model
    # (graph_break_mlp) every input is in [-10, 10], so decimal is
    # losslessly representable at 9 significant digits.
    f = struct.unpack("<f", struct.pack("<I", bits))[0]
    return f"{f!r}f"


def _render_driver(
    *,
    model_id: str,
    target_id: str,
    kernels: list[dict[str, Any]],
    final_output_count: int,
    final_output_dtype_bytes: int,
) -> str:
    """Render the C driver source from ``driver_template.c.tmpl``.

    v1: regions execute sequentially with a static input buffer for
    region 0 (seeded from `golden_inputs_data`) and an output buffer
    per region. The MVP wires each region's output directly to the
    next region's input (so the bundle's regions must be linear).
    """
    n_regions = len(kernels)
    region_includes: list[str] = []
    region_buffers: list[str] = []
    region_calls: list[str] = []

    for i, k in enumerate(kernels):
        meta = k["metadata"]
        # MVP shape: assume each region is an FC of (in_c, out_c),
        # batch=1, all fp32. The shape is read from the kernel's
        # `kInitShape` constants in its emitted C, but for v1 we use
        # the metadata if present, else a sane default.
        shape = meta.get("xnn_create_shape") or meta.get("shape") or [4, 3]
        in_c = int(shape[0]) if len(shape) >= 1 else 4
        out_c = int(shape[1]) if len(shape) >= 2 else 3
        region_buffers.append(
            f"static float region{i}_input[{in_c}] __attribute__((aligned(16)));\n"
            f"static const size_t region{i}_input_count = {in_c};\n"
            f"static float region{i}_output[{out_c}] __attribute__((aligned(16)));\n"
            f"static const size_t region{i}_output_count = {out_c};"
        )
        # Forward-declare the per-region symbols from its .c.
        region_includes.append(
            f"/* region {i} from {k['name']}.c */\n"
            f"extern int  region{i}_init(void);\n"
            f"extern int  region{i}_run(const void* const* inputs, size_t n_inputs,\n"
            f"                          void* const* outputs, size_t n_outputs,\n"
            f"                          const int64_t* runtime_shape, size_t n_runtime_shape);\n"
            f"extern void region{i}_destroy(void);"
        )
        # If not the first region, copy previous region's output into
        # this region's input.
        wire = (
            ""
            if i == 0
            else (
                f"        if (region{i - 1}_output_count > region{i}_input_count) {{\n"
                f"            puts(\"FAIL: region{i - 1}->region{i} wire size mismatch\");\n"
                f"            return 10 + {i};\n"
                f"        }}\n"
                f"        memcpy(region{i}_input, region{i - 1}_output,\n"
                f"               region{i - 1}_output_count * sizeof(float));\n"
            )
        )
        region_calls.append(
            f"    /* --- region {i} ({k['name']}) --- */\n"
            f"    {{\n"
            f"{wire}"
            f"        const int64_t rshape[] = {{ 1 }};\n"
            f"        const void* inputs[1]  = {{ region{i}_input }};\n"
            f"        void*       outputs[1] = {{ region{i}_output }};\n"
            f"        if (region{i}_init() != 0) {{\n"
            f"            puts(\"FAIL: region{i}_init\");\n"
            f"            return 20 + {i};\n"
            f"        }}\n"
            f"        if (region{i}_run(inputs, 1, outputs, 1, rshape, 1) != 0) {{\n"
            f"            puts(\"FAIL: region{i}_run\");\n"
            f"            return 30 + {i};\n"
            f"        }}\n"
            f"    }}"
        )

    final_idx = n_regions - 1
    txt = _DRIVER_TEMPLATE
    txt = txt.replace("%%MODEL_ID%%", model_id)
    txt = txt.replace("%%TARGET_ID%%", target_id)
    txt = txt.replace("%%N_REGIONS%%", str(n_regions))
    txt = txt.replace("%%BUNDLE_HASH%%", "n/a")
    txt = txt.replace("%%REGION_INCLUDES%%", "\n".join(region_includes))
    txt = txt.replace("%%REGION_BUFFERS%%", "\n\n".join(region_buffers))
    txt = txt.replace("%%REGION_CALLS%%", "\n\n".join(region_calls))
    txt = txt.replace("%%FINAL_OUTPUT_VAR%%", f"region{final_idx}_output")
    txt = txt.replace("%%FINAL_OUTPUT_COUNT%%", f"region{final_idx}_output_count")
    return txt


def _rename_kernel_symbols(src: Path, dst: Path, region_idx: int) -> None:
    """Rewrite the kernel's ABI symbols to be region-prefixed.

    The XnnpackProvider emits each kernel with the same exported
    symbol names (``xpu_rt_kernel_init``, ``..._run``, ``..._destroy``).
    When multiple regions land in the same final ELF the symbols
    collide at link time. We rename each region's copy to
    ``regionN_{init,run,destroy}`` via simple text substitution.
    """
    text = src.read_text(encoding="utf-8")
    for canonical, prefixed in (
        ("xpu_rt_kernel_init",    f"region{region_idx}_init"),
        ("xpu_rt_kernel_run",     f"region{region_idx}_run"),
        ("xpu_rt_kernel_destroy", f"region{region_idx}_destroy"),
    ):
        text = text.replace(canonical, prefixed)
    dst.write_text(text, encoding="utf-8")


def _render_staging_cmakelists(
    *,
    target_id: str,
    cross: CrossCompileConfig,
    kernel_basenames: list[str],
    libxpu_rt_path: Path,
    archives: list[Path],
    repo_root: Path,
) -> str:
    """Render the staging dir's top-level CMakeLists.txt."""
    sources = ["driver.c", "golden_inputs.c"] + kernel_basenames
    sources_block = "\n    ".join(sources)
    bridge_include = repo_root / "runtime" / "native" / "libxpu_rt" / "src"
    bridge_include_pub = repo_root / "runtime" / "native" / "libxpu_rt" / "include"
    # GCC's --start-group / --end-group lets the single-pass linker
    # iterate until fixed-point so cross-archive references (xpu_rt →
    # XNNPACK → pthreadpool → libm) resolve regardless of order.
    archive_block = "\n    ".join(f'"{p}"' for p in archives)
    return f"""# Auto-generated by xpu_rt.runtime.cross_compile.riscv64_bare.
cmake_minimum_required(VERSION 3.16)
project(compgen_bundle C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(program
    {sources_block}
)
target_include_directories(program PRIVATE
    ${{CMAKE_SOURCE_DIR}}
    {bridge_include_pub}
    {bridge_include}
)
target_link_libraries(program PRIVATE
    -Wl,--start-group
    {archive_block}
    -Wl,--end-group
    stdc++ m)
set_target_properties(program PROPERTIES SUFFIX ".elf")
"""


def cross_compile_riscv64_bundle(
    bundle_dir: Path,
    *,
    target_id: str,
    cross: CrossCompileConfig,
    repo_root: Path,
    model_id: str = "bundle",
) -> CrossCompileResult:
    """Cross-compile the bundle's generated kernels into a single ELF.

    The output ELF is written to ``<bundle>/program.elf``. Intermediate
    files land under ``<bundle>/cross_compile_riscv64/``. All failures
    are mapped to a typed :class:`CrossCompileError` (reason in the
    instance's ``.reason`` field).
    """
    bundle_dir = Path(bundle_dir).resolve()
    staging_dir = bundle_dir / "cross_compile_riscv64"
    build_dir = staging_dir / "build"
    cmake_log_path = staging_dir / "cmake.log"

    # Locate libxpu_rt_static.a — must have been built by the
    # riscv-spike CMake configuration before this stage runs.
    lib = _find_libxpu_rt_static(repo_root)
    if lib is None:
        raise CrossCompileError(
            "libxpu_rt_missing",
            "no riscv64 libxpu_rt_static.a found under build/riscv-spike/. "
            "Run the libxpu_rt cross-compile first.",
        )

    # Locate the cross-toolchain file.
    toolchain_file_rel = cross.cmake_toolchain_file
    toolchain_file = (
        Path(toolchain_file_rel)
        if Path(toolchain_file_rel).is_absolute()
        else repo_root / toolchain_file_rel
    )
    if not toolchain_file.is_file():
        raise CrossCompileError(
            "cross_gcc_missing",
            f"toolchain file not found: {toolchain_file}",
        )

    # Collect generated XNNPACK kernels.
    kernels = _collect_xnnpack_kernels(bundle_dir)
    if not kernels:
        return CrossCompileResult(
            status="skipped",
            elf_path=None,
            staging_dir=None,
            cmake_log_path=None,
            reason="no XNNPACK kernels in bundle/generated_kernels/",
        )

    log.info(
        "cross_compile: staging riscv64 build",
        target=target_id,
        n_kernels=len(kernels),
        libxpu_rt=str(lib),
    )

    # Fresh staging dir.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # Copy + rename each kernel's symbols.
    kernel_basenames: list[str] = []
    for i, k in enumerate(kernels):
        dst_name = f"region{i}_{k['name']}.c"
        _rename_kernel_symbols(k["source"], staging_dir / dst_name, i)
        kernel_basenames.append(dst_name)

    # Stage the bridge header at the path the kernel source uses
    # (the XnnpackProvider emits `#include "xpu_rt/drivers/xnnpack/
    # xnnpack_bridge.h"` — a virtual path that doesn't exist as-is in
    # the libxpu_rt source tree, only after install). Copy the header
    # into the staging dir so the include resolves cleanly under any
    # toolchain.
    bridge_src = (
        repo_root / "runtime" / "native" / "libxpu_rt" / "src" /
        "drivers" / "xnnpack" / "xnnpack_bridge.h"
    )
    if not bridge_src.is_file():
        raise CrossCompileError(
            "bridge_header_missing",
            f"xnnpack_bridge.h not found at {bridge_src}",
        )
    staged_bridge_dir = staging_dir / "xpu_rt" / "drivers" / "xnnpack"
    staged_bridge_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bridge_src, staged_bridge_dir / "xnnpack_bridge.h")

    # Emit driver.c.
    driver_src = _render_driver(
        model_id=model_id,
        target_id=target_id,
        kernels=kernels,
        final_output_count=0,  # filled at runtime via region_N_output_count
        final_output_dtype_bytes=4,
    )
    (staging_dir / "driver.c").write_text(driver_src, encoding="utf-8")

    # Emit golden_inputs.c from golden_inputs.pt.
    flat_inputs, _ = _load_golden_inputs_flat(bundle_dir)
    _emit_golden_inputs_c(flat_inputs, staging_dir / "golden_inputs.c")

    # Emit CMakeLists.
    archives = _find_companion_archives(lib)
    cmakelists_text = _render_staging_cmakelists(
        target_id=target_id,
        cross=cross,
        kernel_basenames=kernel_basenames,
        libxpu_rt_path=lib,
        archives=archives,
        repo_root=repo_root,
    )
    (staging_dir / "CMakeLists.txt").write_text(cmakelists_text, encoding="utf-8")

    # Configure + build.
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake = shutil.which("cmake") or "/usr/bin/cmake"

    configure_cmd = [
        cmake,
        "-S",
        str(staging_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain_file}",
    ]
    log.info("cross_compile: cmake configure", cmd=configure_cmd)
    with cmake_log_path.open("w", encoding="utf-8") as fp:
        rc = subprocess.run(
            configure_cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            cwd=str(staging_dir),
            env={**os.environ},
        )
    if rc.returncode != 0:
        raise CrossCompileError(
            "cmake_configure_failed",
            f"cmake configure failed; log at {cmake_log_path}",
        )

    build_cmd = [cmake, "--build", str(build_dir), "--", "program"]
    log.info("cross_compile: cmake build", cmd=build_cmd)
    with cmake_log_path.open("a", encoding="utf-8") as fp:
        rc = subprocess.run(
            build_cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            cwd=str(staging_dir),
            env={**os.environ},
        )
    if rc.returncode != 0:
        raise CrossCompileError(
            "link_failed",
            f"cmake build failed; log at {cmake_log_path}",
        )

    elf_built = build_dir / "program.elf"
    if not elf_built.is_file():
        raise CrossCompileError(
            "link_failed",
            f"expected ELF not produced at {elf_built}",
        )

    elf_out = bundle_dir / "program.elf"
    shutil.copy2(elf_built, elf_out)
    log.info("cross_compile: ELF written", path=str(elf_out))

    return CrossCompileResult(
        status="ok",
        elf_path=elf_out,
        staging_dir=staging_dir,
        cmake_log_path=cmake_log_path,
    )

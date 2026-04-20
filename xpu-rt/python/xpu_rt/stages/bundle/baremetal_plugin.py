"""Bundle-stage plugin: ukernel_runtime targets get a buildable C project.

Takes the post-recipe Payload IR + a ``TargetProfile`` for a
ukernel_runtime target (e.g. ``openq_5165rb``) and writes a complete
Hexagon-style C project under ``<bundle_dir>/baremetal/``:

  baremetal/
    kernels/<func_sym>.c       — one per ``func.func`` in the payload IR
    npu_driver_ext.h           — extra prototypes for our emitted helpers
    npu_driver_ext.c           — host-runnable implementations (memcpy + libm)
    memory_map.h               — from BaremetalEmitter scaffold
    npu_driver.h / npu_driver.c
    weights.h
    main.c                     — dispatch driver
    linker.ld
    Makefile

The agent's ``apply_recipe`` mutates the Payload IR before this plugin
runs, so a different proposal really does produce different
``kernels/*.c`` bytes. That's the whole point of P5.2 + P5.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.dialects.func import FuncOp

from compgen.runtime.baremetal.c_codegen import (
    GeneratedCFunction,
    emit_module,
    emit_npu_driver_extension_c,
    emit_npu_driver_extension_h,
)
from compgen.runtime.baremetal.emitter import BaremetalEmitter
from compgen.runtime.memory_layout import BufferRef
from compgen.runtime.program_builder import DeviceKernel, ModelProgram
from compgen.targets.schema import TargetProfile

log = structlog.get_logger()


@dataclass(frozen=True)
class BaremetalBundleResult:
    output_dir: Path
    program_name: str
    kernel_files: list[Path]
    extension_header: Path
    extension_source: Path
    makefile: Path


def _func_byte_size(func: FuncOp, *, default_inputs: int = 1024) -> int:
    """Conservative byte-size estimate for buffer planning."""
    total = 0
    for in_t in func.function_type.inputs:
        if isinstance(in_t, TensorType):
            n = 1
            for d in in_t.get_shape():
                d_int = int(d) if d > 0 else 1
                n *= d_int
            total += n * 4  # f32 default
    return max(total, default_inputs)


def _device_kernel_for(g: GeneratedCFunction, *, byte_size: int) -> DeviceKernel:
    """Wrap one emitted C function as a DeviceKernel for the BaremetalEmitter."""
    in_buf = BufferRef(
        name=f"{g.c_name}_in",
        shape=(byte_size // 4,),
        dtype="f32",
        size_bytes=byte_size,
        persistent=False,
    )
    out_buf = BufferRef(
        name=f"{g.c_name}_out",
        shape=(byte_size // 4,),
        dtype="f32",
        size_bytes=byte_size,
        persistent=False,
    )
    return DeviceKernel(
        kernel_id=g.c_name,
        pattern_id=g.pattern_id,
        device="npu",
        code=g.source,
        language="c",
        inputs=[in_buf],
        outputs=[out_buf],
    )


def write_baremetal_bundle(
    module: ModuleOp,
    target: TargetProfile,
    output_dir: Path,
) -> BaremetalBundleResult:
    """Emit the buildable C project for ``module`` into ``output_dir``.

    Args:
        module: Post-recipe Payload IR (typically ``env.payload_module``
            after :func:`compgen.mcp.tools.recipe_apply.apply_recipe` has
            run).
        target: The active TargetProfile (used for the program name +
            future device-specific tuning).
        output_dir: Directory to write the C project under.

    Returns:
        :class:`BaremetalBundleResult` with concrete paths the caller
        can list / SHA / parse-check.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Walk the module → emit one C function per func.func.
    funcs = emit_module(
        module,
        file_header=(f'/* CompGen-emitted kernel for target {target.name}. */\n#include "../npu_driver_ext.h"\n'),
    )

    # 2. Build a ModelProgram so BaremetalEmitter has something to chew on.
    builder_kernels: list[DeviceKernel] = []
    only_definitions = [g for g in funcs if g.is_definition]
    if not only_definitions:
        # No body to emit — populate at least one kernel so the emitter
        # doesn't produce an empty project.
        only_definitions = funcs[:1]
    for g in only_definitions:
        if g.is_definition:
            byte_size = 4096
        else:
            byte_size = 0
        builder_kernels.append(_device_kernel_for(g, byte_size=byte_size))

    program = ModelProgram(
        name=target.name,
        host_kernels=[],
        device_kernels=builder_kernels,
        execution_order=[k.kernel_id for k in builder_kernels],
        memory_layout=None,  # left None — emitter falls back to a stub map
        initialization=[],
        weight_data={},
    )

    emitter = BaremetalEmitter(deployment="bare_metal")
    emitter.emit(program, output_dir)

    # 3. Layer in our richer codegen alongside the scaffolded driver:
    #    - kernels/*.c gets re-overwritten with our full bodies (the
    #      BaremetalEmitter writes one per execution_order entry already,
    #      but its code = kernel.code, so the contents already match.)
    #    - npu_driver_ext.h / .c carry the per-aten + linalg helpers our
    #      emitted bodies call into.
    ext_h = output_dir / "npu_driver_ext.h"
    ext_h.write_text(emit_npu_driver_extension_h(funcs, model_name=target.name))
    ext_c = output_dir / "npu_driver_ext.c"
    ext_c.write_text(emit_npu_driver_extension_c(model_name=target.name))

    # 4. Patch the Makefile to compile npu_driver_ext.c too.
    mk = output_dir / "Makefile"
    if mk.exists():
        text = mk.read_text()
        if "npu_driver_ext.c" not in text:
            patched = text.replace(
                "npu_driver.c",
                "npu_driver.c npu_driver_ext.c",
                1,
            )
            # Also link libm for sqrtf/expf/tanhf.
            if "-lm" not in patched:
                patched = patched.rstrip() + "\n\nLDLIBS += -lm\n"
            mk.write_text(patched)

    kernel_dir = output_dir / "kernels"
    kernel_files = sorted(kernel_dir.glob("*.c")) if kernel_dir.exists() else []

    return BaremetalBundleResult(
        output_dir=output_dir,
        program_name=target.name,
        kernel_files=kernel_files,
        extension_header=ext_h,
        extension_source=ext_c,
        makefile=mk,
    )


__all__ = [
    "BaremetalBundleResult",
    "write_baremetal_bundle",
]

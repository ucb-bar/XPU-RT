"""Tests for the embedded C ABI runtime emitter."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from compgen.runtime.embedded import EmbeddedOptions, emit_embedded


def _opts(**kwargs) -> EmbeddedOptions:
    return EmbeddedOptions(**kwargs)


def test_emits_four_artifacts(tmp_path: Path) -> None:
    result = emit_embedded(tmp_path, options=_opts(input_bytes=256, output_bytes=64))
    assert result.output_dir == tmp_path.resolve()
    for path in [
        result.header,
        result.blob_source,
        result.runtime_source,
        result.makefile,
    ]:
        assert path.exists(), f"missing emitted artifact: {path}"


def test_header_declares_stable_c_abi(tmp_path: Path) -> None:
    result = emit_embedded(tmp_path, options=_opts(input_bytes=16, output_bytes=8))
    header = result.header.read_text()
    # Entry points.
    for symbol in ("compgen_init", "compgen_invoke", "compgen_shutdown"):
        assert f"int {symbol}" in header or f"void {symbol}" in header
    # PAL callbacks.
    for pal in ("compgen_pal_log", "compgen_pal_time_ns", "compgen_pal_abort"):
        assert pal in header
    # Input/output size macros track the options.
    assert "COMPGEN_MODEL_INPUT_BYTES  ((size_t)16u)" in header
    assert "COMPGEN_MODEL_OUTPUT_BYTES ((size_t)8u)" in header


def test_blob_encodes_bytes(tmp_path: Path) -> None:
    payload = bytes(range(32))
    result = emit_embedded(tmp_path, model_blob=payload)
    source = result.blob_source.read_text()
    # Blob lives in .rodata (plain aligned array). Named sections caused
    # orphan-section placement failures during Zephyr bring-up.
    assert "compgen_model_blob[]" in source
    assert "aligned(16)" in source
    for byte in payload:
        assert f"0x{byte:02x}" in source
    assert "compgen_model_blob_size" in source


def test_blob_is_nonempty_even_when_model_blob_empty(tmp_path: Path) -> None:
    result = emit_embedded(tmp_path, model_blob=b"")
    # The emitter emits a single 0x00 so the symbol is not zero-length
    # and stays in the annotated section.
    assert re.search(r"compgen_model_blob\[]\s*=\s*{\s*0x00", result.blob_source.read_text())


def test_runtime_handles_null_and_misaligned_arena(tmp_path: Path) -> None:
    result = emit_embedded(tmp_path)
    rt = result.runtime_source.read_text()
    # Null arena, zero size, and misalignment all return -22 (-EINVAL).
    assert "return -22" in rt
    # invoke fails before init with -1.
    assert "return -1" in rt
    # Output size is capacity-checked against macro.
    assert "COMPGEN_MODEL_OUTPUT_BYTES" in rt


def test_makefile_uses_configured_toolchain(tmp_path: Path) -> None:
    result = emit_embedded(
        tmp_path,
        options=_opts(cross_compiler="riscv64-zephyr-elf-gcc", archiver="llvm-ar"),
    )
    mk = result.makefile.read_text()
    assert "CC      ?= riscv64-zephyr-elf-gcc" in mk
    assert "AR      ?= llvm-ar" in mk
    assert "libcompgen_model.a" in mk
    # Source list covers the two emitted translation units.
    assert "SRCS := compgen_model.c model_blob.c" in mk


def _have_host_cc() -> bool:
    return shutil.which("cc") is not None


def test_emits_ukernel_sources_and_contracts(tmp_path: Path) -> None:
    """When ukernels are supplied, the emitter drops them into kernels/."""
    from compgen.kernels.provider import KernelContract
    from compgen.kernels.providers.exo_riscv_opu import emit_kernels

    ks = emit_kernels(
        KernelContract(
            op_family="matmul",
            dtypes=("int8", "int8", "int32"),
            target_name="saturn-opu-v128d64",
            hardware_key="saturn-opu-v128d64",
            constraints={
                "inner_tile": [16, 16, 128],
                # xopu in features → VOPACC fast-path variant.
                "features": ["v", "xopu"],
            },
        )
    )
    result = emit_embedded(tmp_path, ukernels=ks)
    assert len(result.ukernel_sources) == 1
    assert result.ukernel_sources[0].name == "mmt4d_s8s8s32_16x16x128_xopu.c"
    assert result.kernel_contracts is not None
    contracts = result.kernel_contracts.read_text()
    assert "mmt4d_s8s8s32_16x16x128_xopu" in contracts
    # Makefile SRCS picked up the kernel source.
    mk = result.makefile.read_text()
    assert "kernels/mmt4d_s8s8s32_16x16x128_xopu.c" in mk
    # Header now declares the ukernel.
    hdr = result.header.read_text()
    assert "compgen_mmt4d_s8s8s32_16x16x128_xopu" in hdr


@pytest.mark.skipif(not _have_host_cc(), reason="no host cc available")
def test_emitted_sources_compile_on_host(tmp_path: Path) -> None:
    """Sanity: the emitted C compiles on the host (sans +v / +xopu flags).

    Catches obvious template drift — missing semicolons, bad casts,
    header/source mismatch — without requiring a RISC-V cross compiler.
    """
    result = emit_embedded(
        tmp_path,
        options=_opts(input_bytes=32, output_bytes=16),
        model_blob=b"\x01\x02\x03\x04",
    )

    obj_rt = tmp_path / "compgen_model.o"
    obj_blob = tmp_path / "model_blob.o"
    for src, obj in [(result.runtime_source, obj_rt), (result.blob_source, obj_blob)]:
        subprocess.run(
            [
                "cc",
                "-std=c17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                str(src),
                "-o",
                str(obj),
                f"-I{tmp_path}",
            ],
            check=True,
        )
    assert obj_rt.exists() and obj_blob.exists()

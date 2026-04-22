"""Tests for the Zephyr overlay generator."""

from __future__ import annotations

from pathlib import Path

import pytest
from compgen.extensions.zephyr import ZephyrOverlayOptions, emit_overlay


def _make_fake_bundle(root: Path) -> Path:
    """Produce a minimal bundle directory with the three required artifacts."""
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "libcompgen_model.a").write_bytes(b"!<arch>\n")  # valid ar header byte
    (bundle / "model_blob.c").write_text(
        "const unsigned char compgen_model_blob[] = {0};\nconst unsigned compgen_model_blob_size = 1;\n"
    )
    (bundle / "compgen_model.h").write_text(
        "#pragma once\n"
        "int compgen_init(void*, unsigned long);\n"
        "int compgen_invoke(const void*, unsigned long, void*, unsigned long);\n"
        "void compgen_shutdown(void);\n"
    )
    return bundle


def _make_fake_zephyr_root(root: Path) -> Path:
    zephyr = root / "zephyr-chipyard-sw"
    (zephyr / "samples").mkdir(parents=True)
    return zephyr


def test_emit_overlay_writes_all_artifacts(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    zephyr = _make_fake_zephyr_root(tmp_path)

    result = emit_overlay(bundle, zephyr, ZephyrOverlayOptions(sample_name="ut_sample"))

    assert result.paths.root == (zephyr / "samples" / "ut_sample").resolve()
    for path in [
        result.paths.cmake_lists,
        result.paths.prj_conf,
        result.paths.custom_sections_ld,
        result.paths.main_c,
        result.paths.compgen_lib,
        result.paths.model_blob,
        result.paths.model_header,
    ]:
        assert path.exists(), f"overlay did not write {path}"

    # Build command references the default board and sample name.
    assert "west build" in result.build_command
    assert "ut_sample" in result.build_command
    assert "spike_riscv64" in result.build_command


def test_cmake_lists_links_compgen_model(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    zephyr = _make_fake_zephyr_root(tmp_path)
    result = emit_overlay(bundle, zephyr)
    cmake = result.paths.cmake_lists.read_text()
    assert "find_package(Zephyr" in cmake
    assert "compgen_model" in cmake
    assert "libcompgen_model.a" in cmake
    # The overlay no longer emits a named-section linker snippet; named
    # sections caused orphan-placement failures on spike_riscv64.
    assert "zephyr_linker_sources(DATA_SECTIONS" not in cmake


def test_prj_conf_respects_smp_option(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    zephyr = _make_fake_zephyr_root(tmp_path)

    single = emit_overlay(bundle, zephyr, ZephyrOverlayOptions(sample_name="single"))
    multi = emit_overlay(
        bundle,
        zephyr,
        ZephyrOverlayOptions(sample_name="multi", smp=True, mp_max_num_cpus=4),
    )
    assert "CONFIG_SMP=n" in single.paths.prj_conf.read_text()
    assert "CONFIG_SMP=y" in multi.paths.prj_conf.read_text()
    assert "CONFIG_MP_MAX_NUM_CPUS=4" in multi.paths.prj_conf.read_text()


def test_main_c_exercises_compgen_c_abi(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    zephyr = _make_fake_zephyr_root(tmp_path)
    result = emit_overlay(bundle, zephyr)
    main = result.paths.main_c.read_text()
    for symbol in ["compgen_init", "compgen_invoke", "compgen_shutdown", "compgen_pal_log"]:
        assert symbol in main
    # Arena + input/output live in plain BSS — aligned but not in a
    # named linker section (see overlay.py for why).
    assert "aligned(16)" in main
    assert "compgen_arena" in main
    # printf/sys_reboot path: matches Zephyr hello_world's HTIF-flushing
    # pattern that we verified works on spike_riscv64.
    assert "sys_reboot(SYS_REBOOT_COLD)" in main
    assert "printf" in main


def test_custom_sections_ld_keeps_expected_sections(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    zephyr = _make_fake_zephyr_root(tmp_path)
    result = emit_overlay(bundle, zephyr)
    ld = result.paths.custom_sections_ld.read_text()
    for section in (
        "input_data_sec",
        "compgen_model_sec",
        "compgen_arena_sec",
        "compgen_input_sec",
    ):
        assert section in ld


def test_missing_bundle_artifact_raises(tmp_path: Path) -> None:
    bundle = tmp_path / "incomplete"
    bundle.mkdir()
    (bundle / "libcompgen_model.a").write_bytes(b"!<arch>\n")
    zephyr = _make_fake_zephyr_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing required artifact"):
        emit_overlay(bundle, zephyr)


def test_invalid_zephyr_root_raises(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    with pytest.raises(FileNotFoundError, match="samples/"):
        emit_overlay(bundle, tmp_path / "not-a-zephyr-tree")


def test_copies_optional_input_blob(tmp_path: Path) -> None:
    bundle = _make_fake_bundle(tmp_path)
    (bundle / "input_blob.c").write_text("/* golden input bytes */\n")
    zephyr = _make_fake_zephyr_root(tmp_path)
    result = emit_overlay(bundle, zephyr)
    assert result.paths.input_blob is not None
    assert result.paths.input_blob.exists()
    assert result.paths.input_blob.read_text() == "/* golden input bytes */\n"

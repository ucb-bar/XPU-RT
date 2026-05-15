"""Tests for ``runtime.embedded.e2e_spike``.

End-to-end Saturn OPU ConvNet pipeline driver. Produces an embedded
bundle + a Zephyr overlay from a fixture module. We exercise the
public entry point on the in-tree fixture without invoking the
Zephyr SDK or Spike — just assert the files that Python is
responsible for producing land on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from compgen.runtime.embedded.e2e_spike import (
    E2ESpikeArtifacts,
    build_e2e_artifacts,
)


def test_build_e2e_artifacts_produces_bundle_and_overlay(tmp_path: Path) -> None:
    """The builder must deterministically produce a bundle dir and a
    Zephyr overlay under the sample name the caller requests."""
    bundle_dir = tmp_path / "bundle"
    zephyr_root = tmp_path / "zephyr"
    zephyr_root.mkdir()
    (zephyr_root / "samples").mkdir()

    artifacts = build_e2e_artifacts(
        model_fixture_module="tests.fixtures.saturn_opu_convnet",
        bundle_dir=bundle_dir,
        zephyr_root=zephyr_root,
        sample_name="unit_test_convnet",
    )
    assert isinstance(artifacts, E2ESpikeArtifacts)
    assert artifacts.bundle_dir == bundle_dir
    assert artifacts.bundle_dir.is_dir()
    # Zephyr overlay lands in samples/<sample_name>/.
    overlay = zephyr_root / "samples" / "unit_test_convnet"
    assert overlay.is_dir()
    # main.c must be present — that's the emitter's primary job.
    assert (overlay / "src" / "main.c").is_file() or (overlay / "main.c").is_file()


def test_build_e2e_artifacts_accepts_golden_input(tmp_path: Path) -> None:
    """When ``golden_input_path`` is supplied, the golden bytes come
    from that file (not the fixture's ``default_inputs()``). We pin
    this because the Phase-1 bundle contract's ``golden_inputs.pt``
    feeds this slot on a real compile."""
    bundle_dir = tmp_path / "bundle"
    zephyr_root = tmp_path / "zephyr"
    zephyr_root.mkdir()
    (zephyr_root / "samples").mkdir()

    # Build a plausible-shape golden input (3, 64, 64) and save.
    golden_path = tmp_path / "golden.pt"
    torch.manual_seed(0)
    golden = torch.randn(1, 3, 64, 64)
    torch.save(golden, golden_path)

    artifacts = build_e2e_artifacts(
        model_fixture_module="tests.fixtures.saturn_opu_convnet",
        bundle_dir=bundle_dir,
        zephyr_root=zephyr_root,
        sample_name="unit_test_convnet_golden",
        golden_input_path=golden_path,
    )
    # Golden bytes must be the loaded tensor's bytes (float32 planar).
    assert len(artifacts.golden_input_bytes) > 0
    assert isinstance(artifacts.golden_input_bytes, (bytes, bytearray))


def test_build_e2e_rejects_missing_fixture(tmp_path: Path) -> None:
    """Rejecting an unknown fixture module is the right failure —
    the old fallback used to silently produce an empty bundle."""
    with pytest.raises((ModuleNotFoundError, ImportError)):
        build_e2e_artifacts(
            model_fixture_module="tests.fixtures.definitely_not_there",
            bundle_dir=tmp_path / "bundle",
            zephyr_root=tmp_path / "zephyr",
        )
